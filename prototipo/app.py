"""
Protótipo de apoio à decisão — vendas aprovadas previstas por vendedor (Olist)
TCC: previsão para pequenos vendedores da Olist (coorte de 870 vendedores)

ENTRADA: as saídas da execução indicada em EXECUCAO no config.yaml
    saidas/previsoes_teste.csv          previsões semanais do teste, h = 1 (notebook 05)
    saidas/acumulado_4_semanas.csv      previsões das próximas 4 semanas, recursivas (notebook 11; opcional)
    artefatos/selecao_modelo.json       modelo escolhido na validação (notebook 04)
    artefatos/avaliacao_teste.json      modelos elegíveis (notebook 05)

Rodar (da raiz do projeto):
    .venv\\Scripts\\python -m pip install -r prototipo\\requirements.txt
    .venv\\Scripts\\python -m streamlit run prototipo\\app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

RAIZ = Path(__file__).resolve().parents[1]
MODELO_REF = "Naive"
MIN_SEMANAS_FRASE = 5          # mínimo de comparações sem empate para mostrar "X de cada 10"

st.set_page_config(page_title="Vendas aprovadas previstas — protótipo", layout="wide")


# ------------------------------------------------------------------ execução e dados
def _ler_json(p: Path):
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:            # arquivos gravados no Windows antes da correção de codificação
        return json.loads(raw.decode("cp1252"))


@st.cache_data
def execucao():
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    base = RAIZ / "execucoes" / cfg["EXECUCAO"]
    sel, aval = base / "artefatos" / "selecao_modelo.json", base / "artefatos" / "avaliacao_teste.json"
    return {
        "nome": cfg["EXECUCAO"], "saidas": base / "saidas",
        "escolhido": _ler_json(sel)["modelo_escolhido"] if sel.is_file() else None,
        "elegiveis": _ler_json(aval)["ELIGIBLE_FINAL"] if aval.is_file() else None,
    }


@st.cache_data
def carregar_semanal(fonte):
    """CSV da v6 (target_week, target_next_week, pred_*) ou formato antigo (week, y_true, y_pred_*)."""
    df = pd.read_csv(fonte)
    if "target_week" in df.columns:
        df = df.rename(columns={"target_week": "week", "target_next_week": "y_true"})
        df = df.rename(columns={c: "y_pred_" + c.removeprefix("pred_") for c in df.columns if c.startswith("pred_")})
    df["week"] = pd.to_datetime(df["week"])
    df["seller_id"] = df["seller_id"].astype(str)
    keep = ["seller_id", "week", "y_true"] + [c for c in df.columns if c.startswith("y_pred_")]
    return df[keep].sort_values(["seller_id", "week"]).reset_index(drop=True)


@st.cache_data
def carregar_acumulado(caminho):
    if not Path(caminho).is_file():
        return None
    df = pd.read_csv(caminho, parse_dates=["origem", "semana_inicio", "semana_fim"])
    df["seller_id"] = df["seller_id"].astype(str)
    return df.sort_values(["seller_id", "origem"]).reset_index(drop=True)


def ler_csv_opcional(caminho):
    return pd.read_csv(caminho) if Path(caminho).is_file() else None


EX = execucao()
st.sidebar.header("Dados")
st.sidebar.caption(f"Execução: **{EX['nome']}**")
up = st.sidebar.file_uploader("Outro CSV de previsões semanais (opcional)", type="csv")
try:
    df = carregar_semanal(up if up is not None else EX["saidas"] / "previsoes_teste.csv")
except FileNotFoundError:
    st.info(f"Não encontrei as previsões em {EX['saidas']}. Rode os notebooks até o 05 ou envie um CSV na barra lateral.")
    st.stop()
acum = carregar_acumulado(EX["saidas"] / "acumulado_4_semanas.csv")

cols_pred = [c for c in df.columns if c.startswith("y_pred_")]
nome = {c: c.removeprefix("y_pred_") for c in cols_pred}
MODELO = EX["escolhido"] or "LightGBM_MAE_Exog"
col_mod, col_ref = f"y_pred_{MODELO}", f"y_pred_{MODELO_REF}"
if col_mod not in df.columns:
    st.error(f"Previsões do modelo {MODELO} não encontradas. Modelos no CSV: {', '.join(nome.values())}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Limites deste protótipo**\n\n"
    "- Funciona sobre o período de teste (jan–ago/2018), não sobre datas reais futuras.\n"
    "- Mostra **vendas aprovadas**, não dinheiro disponível: o repasse vem depois e com descontos.\n"
    "- Previsão pontual; a faixa é empírica (± erro médio já observado), não um intervalo estatístico.\n"
    "- 4 semanas: previsão recursiva, com incerteza maior que a da semana seguinte.\n"
    "- Não foi avaliado com usuários."
)


# ------------------------------------------------------------------ apoio
def frase_confianca(e_modelo, e_ref, ref_txt):
    """'X de cada 10' com comparações anteriores, sem contar empates (ex.: semanas sem venda previstas como zero)."""
    e_modelo, e_ref = np.asarray(e_modelo, float), np.asarray(e_ref, float)
    dif = ~np.isclose(e_modelo, e_ref)
    n = int(dif.sum())
    if n < MIN_SEMANAS_FRASE:
        return "Ainda há poucas semanas anteriores para dizer com que frequência esta previsão acerta mais."
    x = round(10 * float((e_modelo[dif] < e_ref[dif]).mean()))
    return (f"Nas {n} comparações anteriores (sem contar empates), esta previsão ficou mais perto do realizado "
            f"do que {ref_txt} em **{x} de cada 10**.")


def faixa(erros_vendedor, erros_coorte):
    if len(erros_vendedor) >= 4:
        return float(np.mean(erros_vendedor)), "erro já observado deste vendedor"
    if len(erros_coorte):
        return float(np.mean(erros_coorte)), "erro já observado da coorte"
    return np.nan, "sem períodos anteriores"


def brl(v, casas=2):
    s = f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


# ------------------------------------------------------------------ tela principal
st.title("Vendas aprovadas previstas")
st.warning("**Isto não é dinheiro disponível.** É a previsão das vendas com pagamento aprovado. "
           "O repasse da Olist acontece depois da venda e com descontos (comissão e frete). "
           "Use como sinal de quanto deve entrar, não como saldo.")

c1, c2 = st.columns([2, 1])
vendedor = c1.selectbox("Vendedor", df["seller_id"].unique())
opcoes = ["Próxima semana"] + (["Próximas 4 semanas"] if acum is not None else [])
periodo = c2.radio("Período", opcoes, horizontal=True,
                   help=None if acum is not None else "Rode o notebook 11 para habilitar 4 semanas.")
d = df[df["seller_id"] == vendedor].reset_index(drop=True)

if periodo == "Próxima semana":
    semanas = d["week"].dt.date.tolist()
    semana = st.select_slider("Semana prevista", options=semanas, value=semanas[-1],
                              format_func=lambda s: s.strftime("%d/%m/%Y"))
    i = semanas.index(semana)
    hist = d.iloc[:i]                                      # semanas anteriores (realizado já conhecido)
    prev, real = float(d.loc[i, col_mod]), float(d.loc[i, "y_true"])
    ant = df[df["week"] < d.loc[i, "week"]]
    erro, origem_faixa = faixa(np.abs(hist["y_true"] - hist[col_mod]), np.abs(ant["y_true"] - ant[col_mod]))
    ref_val = float(hist["y_true"].tail(8).mean()) if len(hist) else np.nan
    ref_lbl = "Média semanal das 8 semanas anteriores"
    real_lbl = "Vendido na semana (conferência)"
    frase = (frase_confianca(np.abs(hist["y_true"] - hist[col_mod]), np.abs(hist["y_true"] - hist[col_ref]),
                             "repetir a semana passada") if col_ref in d else "")
    pergunta = "Quanto tenho para pagar nesta semana (R$)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["week"], y=d["y_true"], name="Vendido", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=d["week"], y=d[col_mod], name="Previsão", mode="lines", line=dict(dash="dash")))
    if col_ref in d:
        fig.add_trace(go.Scatter(x=d["week"], y=d[col_ref], name="Repetir a semana anterior",
                                 mode="lines", line=dict(dash="dot"), opacity=0.5))
    fig.add_vline(x=d.loc[i, "week"], line_width=1, line_color="gray")
else:
    a = acum[acum["seller_id"] == vendedor].reset_index(drop=True)
    origens = a["origem"].dt.date.tolist()
    o = st.select_slider("Hoje é o fim da semana de", options=origens, value=origens[-1],
                         format_func=lambda s: s.strftime("%d/%m/%Y"))
    j = origens.index(o)
    lin = a.loc[j]
    st.caption(f"Previsão para {lin['semana_inicio']:%d/%m} a {lin['semana_fim'] + pd.Timedelta(days=6):%d/%m/%Y}")
    prev, real = float(lin["previsto_acum"]), float(lin["real_acum"])
    # só janelas de 4 semanas já encerradas na data de hoje
    fechadas = a[a["semana_fim"] <= lin["origem"]]
    fech_coorte = acum[acum["semana_fim"] <= lin["origem"]]
    erro, origem_faixa = faixa(np.abs(fechadas["real_acum"] - fechadas["previsto_acum"]),
                               np.abs(fech_coorte["real_acum"] - fech_coorte["previsto_acum"]))
    ref_val, ref_lbl = float(lin["mm4_acum"]), "Vendido nas últimas 4 semanas"
    real_lbl = "Vendido nas 4 semanas (conferência)"
    frase = frase_confianca(np.abs(fechadas["real_acum"] - fechadas["previsto_acum"]),
                            np.abs(fechadas["real_acum"] - fechadas["naive_acum"]),
                            "repetir 4 vezes a última semana")
    pergunta = "Quanto tenho para pagar nestas 4 semanas (R$)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["week"], y=d["y_true"], name="Vendido", mode="lines+markers"))
    semanas_h = [lin["origem"] + pd.Timedelta(weeks=h) for h in range(1, 5)]
    fig.add_trace(go.Scatter(x=semanas_h, y=[lin[f"previsto_h{h}"] for h in range(1, 5)], name="Previsão (4 semanas)",
                             mode="lines+markers", line=dict(dash="dash")))
    fig.add_vrect(x0=lin["semana_inicio"] - pd.Timedelta(days=3), x1=lin["semana_fim"] + pd.Timedelta(days=3),
                  fillcolor="gray", opacity=0.12, line_width=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vendas aprovadas previstas", brl(prev))
m2.metric("Faixa provável", "—" if np.isnan(erro) else f"{brl(max(prev - erro, 0), 0)} – {brl(prev + erro, 0)}",
          help=f"Previsão ± {origem_faixa} (média dos erros em períodos anteriores).")
m3.metric(ref_lbl, "—" if np.isnan(ref_val) else brl(ref_val))
m4.metric(real_lbl, brl(real))
if frase:
    st.markdown(frase)

fig.update_layout(height=380, yaxis_title="R$ por semana", margin=dict(t=10), legend=dict(orientation="h", y=-0.18))
st.plotly_chart(fig)

st.subheader("Vai dar para pagar?")
compromissos = st.number_input(pergunta, min_value=0.0, step=50.0)
if compromissos > 0:
    pessimista = prev if np.isnan(erro) else max(prev - erro, 0.0)
    if np.isnan(erro):
        st.info("Ainda não há períodos anteriores para estimar a faixa; a comparação usa só a previsão.")
    if pessimista >= compromissos:
        st.success(f"Mesmo no limite de baixo da faixa ({brl(pessimista)}), as vendas previstas superam o que você tem a pagar.")
    elif prev >= compromissos:
        st.warning(f"A previsão supera o que você tem a pagar, mas o limite de baixo da faixa ({brl(pessimista)}) não. "
                   "Considere guardar uma reserva.")
    else:
        st.error(f"Atenção: as vendas previstas ({brl(prev)}) ficam abaixo do que você tem a pagar ({brl(compromissos)}).")
    st.caption("Lembre que vendas aprovadas não são dinheiro na conta: o repasse vem depois e com descontos.")
st.caption("A previsão costuma ficar abaixo do que de fato é vendido, principalmente somando várias semanas. "
           "Compare também com o que você vendeu no período anterior.")

# ------------------------------------------------------------------ detalhes técnicos
with st.expander("Detalhes técnicos"):
    st.markdown(f"**Modelo:** {MODELO} (escolhido na validação) · **Semana seguinte:** previsão direta (h = 1) · "
                "**4 semanas:** previsão recursiva com o mesmo modelo, variáveis macro congeladas na data da previsão.")
    mostrar = [c for c in cols_pred if EX["elegiveis"] is None or nome[c] in EX["elegiveis"]]
    linhas = []
    for c in mostrar:
        err = df["y_true"] - df[c]
        tot = df.groupby("week")[["y_true", c]].sum()
        linhas.append({"Modelo": nome[c] + ("  ◀ escolhido" if nome[c] == MODELO else ""),
                       "MAE vendedor-semana (R$)": np.mean(np.abs(err)),
                       "WAPE vendedor-semana": np.sum(np.abs(err)) / np.sum(np.abs(df["y_true"])),
                       "WAPE total semanal": np.sum(np.abs(tot["y_true"] - tot[c])) / np.sum(tot["y_true"]),
                       "Viés do total": tot[c].sum() / tot["y_true"].sum() - 1})
    st.markdown("**Semana seguinte (teste, jan–ago/2018)**")
    st.dataframe(pd.DataFrame(linhas).sort_values("WAPE vendedor-semana").style.format(
        {"MAE vendedor-semana (R$)": "R$ {:,.2f}", "WAPE vendedor-semana": "{:.1%}", "WAPE total semanal": "{:.1%}",
         "Viés do total": "{:+.1%}"}), hide_index=True)
    excl = [nome[c] for c in cols_pred if c not in mostrar]
    if excl:
        st.caption(f"Fora da tabela por instabilidade na validação: {', '.join(excl)}.")
    met4 = ler_csv_opcional(EX["saidas"] / "acumulado_metricas.csv")
    por_h = ler_csv_opcional(EX["saidas"] / "acumulado_erro_por_horizonte.csv")
    if met4 is not None:
        st.markdown("**Próximas 4 semanas (acumulado, recursivo)**")
        st.dataframe(met4.style.format({"MAE_vendedor_R$": "R$ {:,.2f}", "WAPE_vendedor": "{:.1%}",
                                        "vies_total_%": "{:+.1f}%", "%_vendedor_origens_subestimadas": "{:.1f}%",
                                        "WAPE_total_coorte": "{:.1%}", "skill_vs_naive4": "{:+.1%}"}), hide_index=True)
    if por_h is not None:
        st.dataframe(por_h.style.format({"MAE_R$": "R$ {:,.2f}", "WAPE": "{:.1%}", "vies_%": "{:+.1f}%"}),
                     hide_index=True)
    st.caption("Números oficiais do trabalho (MASE, IC e Diebold-Mariano): tabela_final_resultados.csv.")
