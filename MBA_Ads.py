# ============================================================
# Recomendador de Combos — painel simples + deep-dive opcional
# Métrica: Score = Lift × log(1 + Pedidos A+B)
# Guardrails fixos: AB≥30; P(B|A)≥1%; P(B)≥0,01%; Lift≥1,2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recomendador de Combos", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
h1, h2, h3, h4 { color:#0b0b0b !important; font-weight:800 !important; }
table td, table th { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------- CONFIG ----------
PARQUET_PATH = "/Users/joaonolasco/Downloads/total_orders_2025.parquet"
COL_ITEM = "product_title"
TOP_N = 200

# Guardrails hardcoded
MIN_AB   = 30       # pedidos A+B
MIN_PBA  = 0.01     # 1% (P(B|A))
MIN_PB   = 0.0001   # 0,01% (P(B))
MIN_LIFT = 1.2      # 1.2x

# ---------- LOAD ----------
@st.cache_data(show_spinner=False)
def load_base(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["order_id", COL_ITEM]).drop_duplicates(subset=["order_id", COL_ITEM])
    return df

df = load_base(PARQUET_PATH)

# ---------- FUNÇÕES ----------
def aplicar_filtros(df: pd.DataFrame, f_data, f_hub, f_cat, f_supplier) -> pd.DataFrame:
    inicio, fim = pd.to_datetime(f_data[0]), pd.to_datetime(f_data[1])
    mask = df["data"].between(inicio, fim)
    if f_hub:
        mask &= df["hub"].isin(f_hub)
    if f_cat:
        mask &= df["customer_category"].isin(f_cat)
    if f_supplier and "supplier" in df.columns:
        mask &= df["supplier"].isin(f_supplier)
    return df.loc[mask].copy()

def calcula_recs(df_sub: pd.DataFrame, produto_a: str, top_n=TOP_N):
    if not produto_a:
        return pd.DataFrame(), pd.DataFrame(), ""

    total_ped = df_sub["order_id"].nunique()
    if total_ped == 0:
        return pd.DataFrame(), pd.DataFrame(), ""

    pedidos_A = df_sub.loc[df_sub[COL_ITEM] == produto_a, "order_id"].unique()
    n_A = len(pedidos_A)
    if n_A == 0:
        return pd.DataFrame(), pd.DataFrame(), ""

    # universo com A
    df_in = df_sub[df_sub["order_id"].isin(pedidos_A)]

    # contagem de pedidos que contêm A+B (por B)
    contagem_AB = (
        df_in[df_in[COL_ITEM] != produto_a]
        .groupby(COL_ITEM)["order_id"].nunique()
        .sort_values(ascending=False)
    )

    # suportes (globais)
    support_A = n_A / total_ped                                 # P(A)
    support_B = (df_sub.groupby(COL_ITEM)["order_id"].nunique() / total_ped)  # P(B)
    support_AB = contagem_AB / total_ped                        # P(A∩B)
    support_B = support_B.reindex(contagem_AB.index).fillna(0)

    # P(B|A) = P(A∩B) / P(A)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_B_dado_A = (support_AB / support_A).replace([np.inf, -np.inf], 0).fillna(0)

    # Lift = P(B|A)/P(B)
    p_B_safe = support_B.clip(lower=1e-12)
    lift = (p_B_dado_A / p_B_safe).replace([np.inf, -np.inf], 0).fillna(0)

    # Monta dataframe
    recs = pd.DataFrame({
        "Produto recomendado": contagem_AB.index,
        "Pedidos A+B": contagem_AB.values.astype(int),
        "P(B|A) (%)": (p_B_dado_A.values * 100),
        "P(B) (%)": (support_B.values * 100),
        "Incremento (pp)": (p_B_dado_A.values - support_B.values) * 100,  # pontos percentuais
        "Lift": lift.values,
        "Pedidos com A": n_A,
        "Pedidos totais": total_ped
    })

    # Score
    recs["log(1+AB)"] = np.log1p(recs["Pedidos A+B"])
    recs["Score"] = recs["Lift"] * recs["log(1+AB)"]

    # Guardrails
    recs = recs[
        (recs["Pedidos A+B"] >= MIN_AB) &
        (recs["P(B|A) (%)"] >= MIN_PBA * 100) &
        (recs["P(B) (%)"]   >= MIN_PB  * 100) &
        (recs["Lift"]       >= MIN_LIFT)
    ]
    if recs.empty:
        return recs, recs, ""

    # Ordena por Score e cria ranking (1,2,3…)
    recs = recs.sort_values(["Score", "Lift", "Pedidos A+B"], ascending=False).head(top_n)
    recs["Ranking"] = recs["Score"].rank(method="dense", ascending=False).astype(int)

    # Formatação visual (deep-dive continua com score numérico)
    for c in ["P(B|A) (%)", "P(B) (%)", "Incremento (pp)"]:
        recs[c] = recs[c].round(2)
    recs["Lift"] = recs["Lift"].round(2)
    recs["log(1+AB)"] = recs["log(1+AB)"].round(3)
    recs["Score"] = recs["Score"].round(3)

    # Painel enxuto — mostra ranking em vez do score
    grid = recs[["Produto recomendado", "P(B|A) (%)", "Ranking"]].copy()
    grid = grid.rename(columns={"Produto recomendado": "Produto",
                                "P(B|A) (%)": "Compra conjunta (%)",
                                "Ranking": "Afinidade (ranking)"})

    # Documentação (vai para o expander)
    doc = f"""
**Como calculamos (álgebra):**

- P(A)     = pedidos_com_A / pedidos_totais
- P(B)     = pedidos_com_B / pedidos_totais
- P(B|A)   = pedidos_com_A_e_B / pedidos_com_A
- Lift     = P(B|A) / P(B)
- Incremento (pp) = (P(B|A) - P(B)) × 100
- Score    = Lift × ln(1 + Pedidos_A+B)

**Guardrails fixos (filtros internos):**
- Pedidos_A+B ≥ {MIN_AB}
- P(B|A) ≥ {MIN_PBA:.2%}
- P(B)   ≥ {MIN_PB:.2%}
- Lift   ≥ {MIN_LIFT:.2f}

**Ordenação final:** decrescente por Score (desempate por Lift e volume).
"""
    return recs, grid, doc

# ---------- SIDEBAR ----------
st.sidebar.header("Filtros")
data_min, data_max = df["data"].min().date(), df["data"].max().date()
f_data = st.sidebar.date_input("Período", value=(data_min, data_max))
f_hub = st.sidebar.multiselect("Hub", sorted(df["hub"].dropna().unique()))
f_cat = st.sidebar.multiselect("Categoria de Cliente", sorted(df["customer_category"].dropna().unique()))
f_supplier = st.sidebar.multiselect("Fornecedor", sorted(df["supplier"].dropna().unique())) if "supplier" in df.columns else []

# ---------- UI ----------
st.title("✅ Recomendador de Combos")
st.caption("Mostramos só o essencial: **Produto**, **Compra conjunta (%)** e **Afinidade (ranking)**. "
           "Abra o deep-dive para a matemática e auditoria.")

df_prev = aplicar_filtros(df, (data_min, data_max), f_hub, f_cat, f_supplier)
itens = sorted(df_prev[COL_ITEM].unique().tolist())
produto_a = st.selectbox("Escolha o produto base (A)", itens if itens else ["(nenhum encontrado)"])

if st.button("🔍 Gerar"):
    with st.spinner("Calculando recomendações…"):
        df_sub = aplicar_filtros(df, f_data, f_hub, f_cat, f_supplier)
        recs_full, grid, doc_math = calcula_recs(df_sub, produto_a)

    n_orders = df_sub["order_id"].nunique()
    n_items  = df_sub[COL_ITEM].nunique()
    st.write(f"**Pedidos no recorte:** {n_orders:,} • **Itens distintos:** {n_items:,}")

    if grid is None or grid.empty:
        st.warning("Nenhum item passou pelos limites fixos. (AB≥30 • P(B|A)≥1% • P(B)≥0,01% • Lift≥1,2)")
    else:
        st.subheader("🛒 Produtos recomendados")
        st.markdown(f"### **{produto_a}**")

        # Tabela principal — ranking no lugar do score
        st.dataframe(grid, use_container_width=True, hide_index=True)

        # Deep-dive (documentação + tabela completa com Score)
        with st.expander("📋 Ver detalhes (diagnóstico do score)"):
            st.markdown("## 📘 Documentação de cálculos")
            st.markdown(doc_math)

            deep_cols = [
                "Produto recomendado", "Ranking", "Pedidos A+B", "log(1+AB)", "Lift", "Score",
                "P(B|A) (%)", "P(B) (%)", "Incremento (pp)", "Pedidos com A", "Pedidos totais"
            ]
            st.dataframe(
                recs_full[deep_cols].rename(columns={"Produto recomendado": "Produto"}),
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "📥 Baixar CSV (completo)",
                data=recs_full.to_csv(index=False).encode("utf-8"),
                file_name="recomendacoes_combos.csv",
                mime="text/csv"
            )
else:
    st.info("Escolha o produto A e clique em **Gerar** para ver a lista de recomendados.")
