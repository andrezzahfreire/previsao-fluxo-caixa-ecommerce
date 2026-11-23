"""Gera os notebooks finos a partir das definições abaixo (fonte única do texto e do código das células).

Uso: python ferramentas/gerar_notebooks.py   (a partir da raiz do projeto)
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''# === SETUP (idêntico em todos os notebooks; só muda NOTEBOOK) ===
NOTEBOOK = "{nome}"
RAIZ = "/content/drive/MyDrive/tcc_fluxo"      # Colab. Localmente: caminho da pasta do projeto (ou deixe None)

import os, sys
try:
    from google.colab import drive
    drive.mount("/content/drive")
    !pip install -q -r {{RAIZ}}/requirements.txt
except ImportError:                              # fora do Colab
    RAIZ = os.environ.get("TCC_RAIZ") or None
    if RAIZ is None:
        from pathlib import Path
        RAIZ = str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.yaml").is_file()))
sys.path.insert(0, f"{{RAIZ}}/src")
%load_ext autoreload
%autoreload 2

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tcc_fluxo import iniciar
C, A = iniciar(RAIZ, notebook=NOTEBOOK)'''

FIM = '''A.finalizar()'''


def nb(nome, titulo, intro, cells):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
    c = [nbf.v4.new_markdown_cell(f"# {titulo}\n\n{intro}"),
         nbf.v4.new_code_cell(SETUP.format(nome=nome))]
    for kind, src in cells:
        c.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    c.append(nbf.v4.new_code_cell(FIM))
    n.cells = c
    nbf.write(n, OUT / f"{nome}.ipynb")
    print("ok", nome)


# ============================================================================ 01
nb("01_alvo", "01 — Construção do alvo (seções 1–5 do v5)",
   "**Lê:** CSVs do Olist em `dados_brutos/olist/`.  \n"
   "**Grava:** `order_seller_revenue`, `weekly_observed`, `weeks_info`, `corte.json`, `auditoria_alvo.json`.  \n"
   "Nenhum modelo é ajustado aqui.",
   [
    ("code", '''from tcc_fluxo import alvo, graficos

orders, payments, items = alvo.ler_olist(C.caminho("dados_olist"))
tab, status_sem_aprov = alvo.diagnostico_bruto(orders, payments, items)
display(tab)
print("Status dos pedidos sem order_approved_at:")
display(status_sem_aprov)'''),
    ("md", '''### Estratégia para `order_approved_at` ausente
Regra fixada antes de olhar modelos: se a proporção de ausentes for ≤ `APPROVED_MISSING_MAX_SHARE`, esses pedidos são excluídos e a semana é definida só por `order_approved_at`; caso contrário toda a base usaria `order_purchase_timestamp`.'''),
    ("code", '''data = alvo.escolher_coluna_data(orders, C.APPROVED_MISSING_MAX_SHARE)
DATE_COL, DATE_STRATEGY = data["DATE_COL"], data["DATE_STRATEGY"]
print(f"Estratégia: {DATE_STRATEGY} | ausentes: {data['n_missing']} ({data['share_missing']:.3%})")
print("Alvo:", data["TARGET_NAME"])'''),
    ("md", "## 2–3. Pagamento aprovado por pedido-vendedor e agregação semanal (seg–dom)"),
    ("code", '''payment_by_order, item_by_seller_order, osr, resumo_filtro = alvo.receita_pedido_vendedor(
    orders, payments, items, DATE_STRATEGY)
print(resumo_filtro)
weekly_observed = alvo.agregar_semanal(osr, DATE_COL)
print(f"Linhas vendedor-semana observadas: {len(weekly_observed):,} | vendedores: {weekly_observed['seller_id'].nunique():,}")'''),
    ("md", "## 4. Auditorias obrigatórias e reconciliação\nSe algum `assert` falhar, a construção do alvo está errada e o notebook para aqui."),
    ("code", '''recon = alvo.auditorias(orders, payments, items, payment_by_order, item_by_seller_order,
                       osr, weekly_observed, DATE_STRATEGY)
display(recon)
print("Auditorias e reconciliação: OK")
display(alvo.composicao_status(osr))'''),
    ("md", "## 5. Corte temporal: última semana completa"),
    ("code", '''corte = alvo.corte_test_end(osr, weekly_observed, DATE_COL, C.TAIL_GAP_DAYS, C.TAIL_MAX_ORDERS, C.TRAIN_TARGET_END)
TEST_END = corte["TEST_END"]
print(f"Último registro na base: {corte['daily'].index.max().date()}")
print(f"Dias tratados como resíduo: {[d.date() for d in corte['residual_days']]} ({corte['residual_orders']} pedidos)")
print(f"Fim regular: {corte['last_regular_day'].date()} | semanas removidas: {[w.date() for w in corte['removed_weeks']]}")
print(f"TEST_END: {TEST_END.date()} (seg) a {(TEST_END + pd.Timedelta(days=6)).date()} (dom)")
wi = corte["weeks_info"]
display(wi.tail(8))

prev8 = wi.loc[:TEST_END - pd.Timedelta(weeks=1), "pedidos"].tail(8)
print(f"Pedidos na semana {TEST_END.date()}: {wi.loc[TEST_END, 'pedidos']:,} = "
      f"{wi.loc[TEST_END, 'pedidos'] / prev8.median():.0%} da mediana das 8 semanas anteriores")
fig = graficos.total_semanal_observado(wi, TEST_END)
A.figura(fig, "total_semanal_observado.png"); plt.show()'''),
    ("md", "### Gravação dos artefatos"),
    ("code", '''weekly_observed = weekly_observed.loc[weekly_observed["week_start"] <= TEST_END].copy()
A.salvar_df(osr, "order_seller_revenue")
A.salvar_df(weekly_observed, "weekly_observed")
A.salvar_df(wi.reset_index(), "weeks_info")
A.salvar_json({"TEST_END": TEST_END, "last_regular_day": corte["last_regular_day"],
               "residual_days": [str(d.date()) for d in corte["residual_days"]],
               "removed_weeks": [str(w.date()) for w in corte["removed_weeks"]],
               **data}, "corte")
A.salvar_json({"filtro": resumo_filtro, "reconciliacao": recon["Valor"].to_dict()}, "auditoria_alvo")'''),
   ])

# ============================================================================ 02
nb("02_exogenas", "02 — Variáveis exógenas (seção 5.1 do v5)",
   "**Lê:** `corte.json` (01) e o cache SGS em `dados_brutos/macro_cache/` (baixa da API do BCB só se faltar).  \n"
   "**Grava:** `exogenas_semanais`, `macro_mensal`, `calendario_eventos`, `exogenas_listas.json`.  \n"
   "**Feriados:** fonte definida em `FERIADOS_FONTE` no config (BrasilAPI com cache em `dados_brutos/brasilapi_feriados/`, "
   "ou cálculo pela Páscoa). Datas comerciais são sempre calculadas.\n\n"
   "O cache fica fora da pasta da execução de propósito: o BCB revisa séries (ex.: IBC-Br), e reutilizar o mesmo "
   "arquivo garante que todas as execuções usam a mesma vintage. Registre a data do download no texto do TCC.",
   [
    ("code", '''from tcc_fluxo import exogenas

corte = A.carregar_json("corte")
TEST_END = pd.Timestamp(corte["TEST_END"])
macro_long, macro_wide = exogenas.montar_macro(C.MACRO_SERIES, C.EXOGENOUS_API_START, C.EXOGENOUS_API_END,
                                               C.caminho("macro_cache"))
FONTE_FERIADOS = C.get("FERIADOS_FONTE", "calculado")
events = exogenas.calendario_eventos(range(C.PANEL_START.year - 1, TEST_END.year + 2), fonte_feriados=FONTE_FERIADOS,
                                     cache_dir=C.caminho("brasilapi_feriados"))
print(f"Feriados: fonte = {FONTE_FERIADOS}")
exog, listas = exogenas.exogenas_semanais(events, macro_wide, C.MACRO_SERIES, C.PANEL_START, TEST_END,
                                          C.MACRO_LAG_MONTHS)
print(f"Macro: {len(C.MACRO_SERIES)} séries, {len(macro_wide)} meses, atraso de {C.MACRO_LAG_MONTHS} meses")
print(f"Calendário: {len(listas['CALENDAR_FEATURES'])} atributos, {len(events)} datas | exógenas: {len(listas['EXOGENOUS_FEATURES'])}")
display(exog.head())'''),
    ("code", '''A.salvar_df(exog, "exogenas_semanais")
A.salvar_df(macro_long, "macro_mensal")
A.salvar_df(events, "calendario_eventos")
cache_hash = {p.name: __import__("tcc_fluxo.io", fromlist=["x"]).sha256_arquivo(p)
              for p in sorted(C.caminho("macro_cache").glob("*.csv"))}
feriados_hash = ({p.name: __import__("tcc_fluxo.io", fromlist=["x"]).sha256_arquivo(p)
                  for p in sorted(C.caminho("brasilapi_feriados").glob("*.json"))} if FONTE_FERIADOS == "brasilapi" else {})
A.salvar_json({**listas, "fonte": C.MACRO_SOURCE, "fonte_feriados": FONTE_FERIADOS, "feriados_cache_sha256": feriados_hash, "series_sgs": C.MACRO_SERIES,
               "lag_macro_meses": C.MACRO_LAG_MONTHS, "cache_sha256": cache_hash,
               "n_meses_macro": len(macro_wide), "n_semanas_exogenas": len(exog), "n_eventos": len(events)},
              "exogenas_listas")
A.tabela(exog, "calendario_macro_semanal.csv")'''),
   ])

# ============================================================================ 03
nb("03_painel", "03 — Coorte, painel, atributos e divisão temporal (seções 6–11 do v5)",
   "**Lê:** `weekly_observed`, `weeks_info`, `corte.json` (01); `exogenas_semanais`, `exogenas_listas.json` (02).  \n"
   "**Grava:** `coorte`, `panel_base`, `painel` (com a coluna `conjunto` = treino/teste/fora), `painel_spec.json`, `divisao.json`.\n\n"
   "Este notebook **constrói** o teste mas não avalia nada nele. Daqui em diante, as linhas de teste só saem por "
   "`A.carregar_teste()`, que exige a seleção congelada.",
   [
    ("code", '''from tcc_fluxo import painel as P

corte = A.carregar_json("corte")
TEST_END = pd.Timestamp(corte["TEST_END"])
weekly_observed = A.carregar_df("weekly_observed")
wi = A.carregar_df("weeks_info").set_index("week_start")
exog = A.carregar_df("exogenas_semanais")
listas = A.carregar_json("exogenas_listas")

print("Semanas com aprovações antes de/até o início do painel:")
display(wi.loc[:C.PANEL_START + pd.Timedelta(weeks=3), ["pedidos", "valor"]])
pre = weekly_observed["week_start"] < C.PANEL_START
print(f"Registros vendedor-semana anteriores a {C.PANEL_START.date()} descartados: {pre.sum():,}")'''),
    ("md", "## 6. Coorte (somente informação até `TRAIN_END`)"),
    ("code", '''elig, cohort, COHORT_LABEL, observed_train = P.coorte(
    weekly_observed, C.PANEL_START, C.TRAIN_END, C.MIN_CALENDAR_WEEKS, C.MIN_SALES_WEEKS,
    C.USE_MEI_PROXY, C.MEI_MONTHLY_PROXY_LIMIT)
print("Critério:", COHORT_LABEL)
print(f"Vendedores com venda no treino: {len(elig):,} | elegíveis: {len(cohort):,} | excluídos: {len(elig) - len(cohort):,}")
fora = set(weekly_observed.loc[weekly_observed["week_start"] > C.TRAIN_END, "seller_id"]) - set(elig.index)
print(f"Vendedores com venda no teste e sem venda no treino (fora da coorte por construção): {len(fora):,}")
display(cohort[["avg_weekly_revenue_train", "avg_monthly_revenue_train", "revenue_train", "sales_weeks",
                "calendar_weeks"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T)'''),
    ("md", "## 7. Painel vendedor-semana (zeros preenchidos, sem interpolação)"),
    ("code", '''panel_base = P.montar_panel_base(cohort, weekly_observed, TEST_END, C.PANEL_START)
print(f"Linhas: {len(panel_base):,} | vendedores: {panel_base['seller_id'].nunique():,} | "
      f"semanas: {panel_base['week_start'].nunique()}")
print(f"Semanas com receita zero: {panel_base['revenue'].eq(0).mean():.1%}")'''),
    ("md", "## 8–10. Atributos sem vazamento, calendário da semana-alvo e alvo (h = 1 semana)"),
    ("code", '''panel = P.attach_exogenous_features(panel_base, exog)
EXOG = listas["EXOGENOUS_FEATURES"]
ALL_NUMERIC = P.BASE_NUMERIC_FEATURES + EXOG
assert ((panel["target_week"] - panel["week_start"]).dropna() == pd.Timedelta(days=7)).all()
assert panel.loc[panel["target_week"].notna(), EXOG].notna().all().all()

feature_cols = ALL_NUMERIC + ["target_week", "target_week_of_year"]
_, changed = P.teste_vazamento(panel_base, panel, exog, feature_cols, C.LEAK_TEST_CUT, C.RANDOM_SEED)
print(f"Atributos até {C.LEAK_TEST_CUT.date()} idênticos após perturbar o futuro: OK "
      f"(controle: {changed:.0%} dos alvos após o corte mudaram)")'''),
    ("md", "## 11. Divisão temporal pela data do alvo"),
    ("code", '''painel, resumo = P.dividir(panel, ALL_NUMERIC, C.TRAIN_TARGET_END, TEST_END)
print(pd.Series(resumo))
profile = P.perfis_treino(cohort)
print(profile.groupby("perfil", observed=True)["freq_venda_treino"].agg(["min", "max", "size"]))'''),
    ("code", '''A.salvar_df(cohort.reset_index(), "coorte")
A.salvar_df(profile.reset_index(), "perfis_treino")
A.salvar_df(panel_base, "panel_base")
A.salvar_df(painel, "painel")
A.salvar_json({"BASE_NUMERIC_FEATURES": P.BASE_NUMERIC_FEATURES, "EXOGENOUS_FEATURES": EXOG,
               "CALENDAR_FEATURES": listas["CALENDAR_FEATURES"], "HOLIDAY_FEATURES": listas["HOLIDAY_FEATURES"],
               "COMMERCIAL_FEATURES": listas["COMMERCIAL_FEATURES"], "MACRO_FEATURES": listas["MACRO_FEATURES"],
               "CAT_FEATURES": P.CAT_FEATURES, "TARGET": P.TARGET, "COHORT_LABEL": COHORT_LABEL}, "painel_spec")
A.salvar_json({**resumo, "TEST_END": TEST_END, "teste_vazamento_corte": C.LEAK_TEST_CUT,
               "teste_vazamento_controle": changed}, "divisao")'''),
    ("md", """### 6.1 Porte da coorte (descritivo — não exclui ninguém)
Quantos vendedores passam do limite de faturamento do MEI, usando a receita média mensal do treino
(proporcional aos meses de atividade em 2017). A receita é a proxy do alvo, não o faturamento fiscal.
Esta célula só lê o artefato `coorte`: pode ser rodada sozinha (depois da célula de setup) sem refazer o painel."""),
    ("code", '''from tcc_fluxo import painel as P
coorte_art = A.carregar_df("coorte")
LIMITES_MEI = {"MEI a partir de 2018 (R$ 81 mil/ano)": C.MEI_MONTHLY_PROXY_LIMIT,   # LC 155/2016
               "MEI em 2017 (R$ 60 mil/ano)": 5000.0}                              # limite vigente nos dados de treino
porte = P.resumo_porte(coorte_art, LIMITES_MEI)
A.tabela(porte, "porte_coorte_mei.csv"); display(porte)
A.salvar_json(porte.to_dict("records"), "porte_coorte")'''),
   ])

# ============================================================================ 04
nb("04_validacao_selecao", "04 — Validação temporal e seleção (seções 13–16 do v5)",
   "**Lê:** `painel` (só linhas de treino), `panel_base` até `TRAIN_TARGET_END`, `painel_spec.json`.  \n"
   "**Grava:** `cv_detalhe`, `resumo_validacao`, **`selecao_modelo.json`** (com o hash do painel).\n\n"
   "Nenhuma linha de teste é carregada. A seleção é congelada aqui e destrava o notebook 05.",
   [
    ("code", '''from tcc_fluxo import cv, modelos, Especificacao

spec = Especificacao.de_json(A.carregar_json("painel_spec"), C)
train = A.carregar_treino()
hist = A.carregar_historico_treino()
BOOST, RIDGE_NAMES, CV_MODELS = modelos.lista_modelos(C.RIDGE_ALPHAS)
cv_origins, dropped = cv.origens_cv(C)
print("Modelos na validação:", CV_MODELS)
print(f"Origens completas ({len(cv_origins)}): {[o.date() for o in cv_origins]} | removidas: {dropped or 'nenhuma'}")'''),
    ("md", "## 14. Validação temporal (somente dentro do treino)"),
    ("code", '''cv_detail, ridge_q = cv.rodar_cv(train, hist, cv_origins, CV_MODELS, spec, C)
A.salvar_df(cv_detail, "cv_detalhe")
A.tabela(cv_detail, "validacao_temporal_detalhada.csv")'''),
    ("md", "## 15. Estabilidade do Ridge\nInstável se, em alguma origem, gerar previsão não finita ou acima do maior alvo do treino da própria origem."),
    ("code", '''display(ridge_q)
ridge_stab, RIDGE_NAME, RIDGE_ELIGIBLE = cv.estabilidade_ridge(cv_detail, RIDGE_NAMES)
display(ridge_stab)
print(f"Ridge {'estável: ' + RIDGE_NAME if RIDGE_ELIGIBLE else 'REMOVIDO da seleção (documentado: ' + RIDGE_NAME + ')'}")'''),
    ("md", "## 16. Resumo da validação e seleção congelada"),
    ("code", '''resumo = cv.resumo_validacao(cv_detail, BOOST, RIDGE_NAME, RIDGE_ELIGIBLE)
A.salvar_df(resumo, "resumo_validacao")
A.tabela(resumo, "resumo_validacao.csv")
display(resumo.sort_values("MASE_vend_media"))

selection = cv.selecionar(resumo, cv_origins, C, RIDGE_NAME, RIDGE_ELIGIBLE, painel_sha256=A.sha("painel"))
selection["boost_models"] = BOOST
A.salvar_json(selection, "selecao_modelo")
print({k: selection[k] for k in ["modelo_escolhido", "criterio", "valor_criterio", "modelos_em_empate_pratico"]})'''),
   ])

# ============================================================================ 05
nb("05_avaliacao_teste", "05 — Avaliação única no teste (seções 17–22 e 25 do v5)",
   "**Lê:** `selecao_modelo.json`, `painel` (treino + **teste via trava**), `panel_base` até `TRAIN_TARGET_END`, "
   "`coorte`, `perfis_treino`, `resumo_validacao`, `order_seller_revenue`.  \n"
   "**Grava:** `previsoes_teste`, métricas, IC bootstrap, Diebold-Mariano, perfis, sensibilidade, tabela final.\n\n"
   "Este é o **único** acesso confirmatório ao teste. A escolha não muda depois daqui.",
   [
    ("code", '''from tcc_fluxo import avaliacao as AV, inferencia as INF, metricas as M, graficos, Especificacao

sel = A.carregar_json("selecao_modelo")
CHOSEN = sel["modelo_escolhido"]
spec_json = A.carregar_json("painel_spec")
spec = Especificacao.de_json(spec_json, C)
TEST_END = pd.Timestamp(A.carregar_json("corte")["TEST_END"])
train = A.carregar_treino()
test = A.carregar_teste(finalidade="confirmatoria")
hist = A.carregar_historico_treino()
resumo_val = A.carregar_df("resumo_validacao")

BOOST = sel["boost_models"]
RIDGE_NAME, RIDGE_ELIGIBLE = sel["ridge_alpha_documentado"], sel["ridge_elegivel"]
FINAL = ["Naive", "MediaMovel4", RIDGE_NAME, *BOOST]
ELIGIBLE = ["Naive", "MediaMovel4", *BOOST] + ([RIDGE_NAME] if RIDGE_ELIGIBLE else [])
print("Modelo escolhido na validação:", CHOSEN)'''),
    ("md", "## 17. Reestimação com todo o treino e previsão do teste"),
    ("code", '''test, comparison, pred_checks = AV.prever_teste(train, test, FINAL, ELIGIBLE, spec, spec.exog)
display(pred_checks)
A.salvar_df(comparison, "previsoes_teste")
A.tabela(comparison, "previsoes_teste.csv")

det = AV.auditoria_determinismo(train, test, BOOST, spec, C.DETERMINISM_TOL)
A.tabela(det, "auditoria_determinismo.csv"); display(det)
assert det.empty or det["deterministico"].all(), "previsões não reproduzíveis no mesmo ambiente"'''),
    ("code", '''scale_final = M.seller_scales(hist, C.TRAIN_TARGET_END)
agg_scale_final = M.aggregate_scale(hist, C.TRAIN_TARGET_END)
R = AV.metricas_teste(comparison, FINAL, ELIGIBLE, CHOSEN, scale_final, agg_scale_final)
A.tabela(R["vendedor_semana"], "metricas_teste_vendedor_semana.csv")
A.tabela(R["per_seller"], "metricas_por_vendedor.csv")
A.tabela(R["seller_summary"], "resumo_metricas_por_vendedor.csv")
A.tabela(R["agregado"], "metricas_teste_agregado.csv")
print("TESTE — nível vendedor-semana"); display(R["vendedor_semana"])
print("TESTE — total semanal da coorte"); display(R["agregado"])'''),
    ("md", '''### Viés do total semanal (acréscimo da v6)
Uma perda MAE estima a **mediana** condicional. Em séries com muitas semanas zeradas, a mediana fica abaixo da média e a soma das previsões dos vendedores subestima o total. Esta tabela mostra o tamanho desse efeito no total da coorte.'''),
    ("code", '''vies = AV.vies_agregado(R["aggregate"], ELIGIBLE)
A.tabela(vies, "vies_total_semanal.csv"); display(vies)'''),
    ("md", "## 18–19. Intervalos bootstrap e Diebold-Mariano (HLN)"),
    ("code", '''ci = INF.intervalos_bootstrap(R["per_seller"], R["aggregate"], FINAL, C.N_BOOT, C.RANDOM_SEED)
A.tabela(ci, "intervalos_confianca.csv")
display(ci.loc[ci["metrica"].isin(["skill vs Naive (agrupado)", "MASE médio por vendedor",
                                   "proporção de vendedores com ganho", "skill vs Naive"])])

dm = INF.tabela_dm(comparison, R["aggregate"], FINAL, CHOSEN, C.N_BOOT, C.RANDOM_SEED, C.MIN_WEEKS_DM)
for nivel in dm["nivel"].unique():                 # acréscimo v6: Holm dentro de cada nível
    msk = dm["nivel"].eq(nivel)
    dm.loc[msk, "p_holm"] = INF.holm(dm.loc[msk, "p_valor_hln"].to_numpy(float))
A.tabela(dm, "teste_diebold_mariano.csv"); display(dm)'''),
    ("md", "## 20. Desempenho por perfil de vendedor"),
    ("code", '''profile = A.carregar_df("perfis_treino").set_index("seller_id")
prof = AV.metricas_por_perfil(R["per_seller"], profile, CHOSEN, ELIGIBLE)
A.tabela(prof, "metricas_por_perfil.csv")
display(prof.loc[prof["modelo"].isin(ELIGIBLE)])'''),
    ("md", "## 21. Gráficos"),
    ("code", '''fig = graficos.agregado_teste(R["aggregate"], CHOSEN); A.figura(fig, "agregado_teste.png"); plt.show()
fig = graficos.comparacao_modelos(resumo_val, ci, FINAL, ELIGIBLE, CHOSEN, RIDGE_NAME, RIDGE_ELIGIBLE)
A.figura(fig, "comparacao_modelos.png"); plt.show()'''),
    ("md", "## 22. Sensibilidade (não altera a escolha)"),
    ("code", '''osr = A.carregar_df("order_seller_revenue")
cohort_idx = A.carregar_df("coorte")["seller_id"]
sens, share_nd, n_sem = AV.sensibilidade(comparison, ELIGIBLE, TEST_END, scale_final, agg_scale_final, osr, cohort_idx)
print(f"Teste SEM a semana-alvo {TEST_END.date()} ({n_sem} semanas):"); display(sens)
print(f"Participação de pedidos não 'delivered' no alvo do teste (coorte): {share_nd:.2%}")
A.tabela(sens, "sensibilidade_sem_ultima_semana.csv")'''),
    ("md", "## 25. Tabela final"),
    ("code", '''final_table = AV.tabela_final(FINAL, ELIGIBLE, CHOSEN, resumo_val, ci, R["seller_summary"], R["agregado"])
A.tabela(final_table, "tabela_final_resultados.csv"); display(final_table)
A.salvar_json({"FINAL_MODELS": FINAL, "ELIGIBLE_FINAL": ELIGIBLE, "CHOSEN_MODEL": CHOSEN,
               "scale_final": scale_final.dropna().to_dict(), "agg_scale_final": agg_scale_final}, "avaliacao_teste")'''),
   ])

# ============================================================================ 06
nb("06_diagnostico", "06 — Diagnóstico dos erros no teste (seção 23 do v5)",
   "**Lê:** `previsoes_teste` (05), `selecao_modelo.json`, `perfis_treino`, `panel_base` até `TRAIN_TARGET_END`, teste (trava, finalidade *diagnostico*).  \n"
   "**Grava:** tabelas e figuras `diagnostico_*`.\n\n"
   "**Correção em relação ao v5:** o modelo do diagnóstico principal é o escolhido em `selecao_modelo.json` "
   "(parâmetro `DIAG_MODELO_PRINCIPAL` do config). No v5 ele estava fixo em `CatBoost_MAE`, embora a seleção "
   "tenha escolhido `LightGBM_MAE_Exog`.",
   [
    ("code", '''from tcc_fluxo import diagnostico as D, metricas as M, graficos

sel = A.carregar_json("selecao_modelo")
MAIN = sel["modelo_escolhido"] if C.DIAG_MODELO_PRINCIPAL == "escolhido" else C.DIAG_MODELO_PRINCIPAL
comparison = A.carregar_df("previsoes_teste")
test = A.carregar_teste(finalidade="diagnostico")
hist = A.carregar_historico_treino()
profile = A.carregar_df("perfis_treino").set_index("seller_id")
scale_final = M.seller_scales(hist, C.TRAIN_TARGET_END)

MODELS = [m for m in D.ALL_DIAG if f"pred_{m}" in comparison.columns]
FOCUS = [m for m in D.FOCUS if m in MODELS]
assert MAIN in MODELS, f"{MAIN} sem previsões no teste"
thr = D.limiares(hist)
print(f"Modelo do diagnóstico principal: {MAIN}")
print(f"Pico > R$ {thr['PEAK_THRESHOLD']:,.2f} | sinal ≥ R$ {thr['SIGNAL_THRESHOLD']:,.2f} (P90/P10 da receita positiva no treino)")
diag = D.base_diagnostico(comparison, profile, test, MODELS, thr)'''),
    ("md", "### 23.1 Diagnóstico resumido de todos os modelos"),
    ("code", '''geral = D.diagnostico_geral(diag, MODELS, thr["SIGNAL_THRESHOLD"])
A.tabela(geral, "diagnostico_geral_modelos.csv")
display(geral.set_index("Modelo").style.format("{:,.2f}"))'''),
    ("md", "### 23.2 Semanas sem × com receita e faixas de valor"),
    ("code", '''diag["grupo"] = np.where(diag["y"] > 0, "com receita", "sem receita")
zp = D.group_table(diag, "grupo", list(dict.fromkeys([MAIN, "Naive"])))
faixas = D.group_table(diag, "faixa_real", list(dict.fromkeys([m for m in ["LightGBM_MAE", "LightGBM_MAE_Exog", "CatBoost_MAE",
                                                         "CatBoost_MAE_Exog", MAIN, "Naive", "CatBoost_Tweedie"] if m in MODELS])))
A.tabela(zp, "diagnostico_principal_zero_positivo.csv"); A.tabela(faixas, "diagnostico_faixas_valor.csv")
display(zp); display(faixas)
z = diag["y"] == 0
print(f"Em semanas sem receita, {MAIN} prevê exatamente 0 em {diag.loc[z, f'pred_{MAIN}'].eq(0).mean():.1%} "
      f"(Naive: {diag.loc[z, 'pred_Naive'].eq(0).mean():.1%}).")'''),
    ("md", "### 23.3 Erros por vendedor"),
    ("code", '''sel_models = list(dict.fromkeys(FOCUS + [MAIN]))
ds = D.por_vendedor(diag, MAIN, sel_models, scale_final)
A.tabela(ds, "diagnostico_principal_por_vendedor.csv")
sk = ds["skill_vs_naive"].dropna()
print(f"Vendedores com viés negativo no total do teste: {(ds[f'vies_{MAIN}'] < 0).mean():.1%}")
print(f"Vendedores em que {MAIN} erra menos que o Naive: {(sk > 0).sum()} de {len(sk)} ({(sk > 0).mean():.1%})")
print(f"Vendedores sem nenhuma venda no teste: {int((ds['real_total'] == 0).sum())}")
conc = ds.sort_values(f"erro_abs_total_{MAIN}", ascending=False)
tot = conc[f"erro_abs_total_{MAIN}"].sum()
print(f"Concentração: 10 vendedores = {conc[f'erro_abs_total_{MAIN}'].head(10).sum() / tot:.1%} do erro; "
      f"10% dos vendedores = {conc[f'erro_abs_total_{MAIN}'].head(int(np.ceil(0.1 * len(conc)))).sum() / tot:.1%}")
fig = graficos.erro_por_vendedor(ds, MAIN); A.figura(fig, "diagnostico_erro_por_vendedor.png"); plt.show()'''),
    ("md", "### 23.4–23.5 Erros por perfil e por semana"),
    ("code", '''dp, zero_ref = D.por_perfil(diag, MODELS, thr["SIGNAL_THRESHOLD"])
A.tabela(dp, "diagnostico_por_perfil.csv")
print("Referência trivial — MAE de prever zero sempre:"); display(zero_ref.to_frame().T)
display(dp.loc[dp["modelo"].isin(FOCUS)])

dw = D.por_semana(diag, MODELS)
A.tabela(dw, "diagnostico_por_semana.csv")
print(f"Semanas em que {MAIN} erra menos que o Naive: {(dw[f'MAE_{MAIN}'] < dw['MAE_Naive']).sum()} de {len(dw)}")
print(f"Semanas com viés negativo de {MAIN}: {(dw[f'vies_{MAIN}'] < 0).sum()} de {len(dw)}")
fig = graficos.erro_por_semana(dw, FOCUS); A.figura(fig, "diagnostico_erro_por_semana.png"); plt.show()'''),
    ("md", "### 23.6–23.7 Maiores erros, concentração e picos"),
    ("code", '''cols = ["seller_id", "perfil", "week_start", "target_week", "revenue_current", "rev_mean_4", "zero_weeks_prev_4",
        "y"] + [f"pred_{m}" for m in FOCUS] + [f"err_{MAIN}", "pico"]
top = diag.nlargest(20, f"abs_{MAIN}")[list(dict.fromkeys(cols))].rename(columns={"y": "real"})
A.tabela(top, "diagnostico_maiores_erros.csv")
print(f"Dos 20 maiores erros de {MAIN}: {(top[f'err_{MAIN}'] < 0).sum()} subestimações, {int(top['pico'].sum())} em picos")
A.tabela(D.concentracao(diag, MODELS), "diagnostico_concentracao_erros.csv")

picos = D.peak_table(diag, diag["pico"], "global", MODELS, thr["PEAK_THRESHOLD"], thr["seller_peak_thr"])
picos_rel = D.peak_table(diag, diag["pico_relativo"], "relativo ao vendedor", MODELS, thr["PEAK_THRESHOLD"], thr["seller_peak_thr"])
A.tabela(picos, "diagnostico_picos.csv"); A.tabela(picos_rel, "diagnostico_picos_relativos.csv")
display(picos.drop(columns="criterio").set_index("modelo"))
order_pk = [m for m in ["LightGBM_MAE", "LightGBM_MAE_Exog", "CatBoost_MAE", "CatBoost_MAE_Exog", "CatBoost_MAE_raw",
                        "HistGB_MAE", "MediaMovel4", "Naive", "HistGB_Poisson", "CatBoost_Tweedie"] if m in MODELS]
fig = graficos.picos(picos, order_pk, thr["PEAK_THRESHOLD"], int(diag["pico"].sum()))
A.figura(fig, "diagnostico_picos.png"); plt.show()'''),
   ])

# ============================================================================ 07
nb("07_experimento1", "07 — Experimento 1: atributos de ocorrência e intensidade (seção 24 do v5)",
   "**Lê:** treino, `panel_base` até `TRAIN_TARGET_END`, `painel` até `CV_FIRST_ORIGIN` (correlações), `selecao_modelo.json`.  \n"
   "**Grava:** `limites_prototipo.json` (congelado antes da validação), resultados por origem/perfil, "
   "`experimento1_modelo_escolhido.json`. O teste só é carregado se alguma variante for escolhida.\n\n"
   "**Status para o texto:** a ablação é feita sobre o `CatBoost_MAE` (mesma perda e hiperparâmetros), não sobre o "
   "modelo escolhido na seção 16. Como as hipóteses (ocorrência, picos) surgiram do diagnóstico do teste "
   "(notebook 06), um eventual resultado no teste é **exploratório**, não confirmatório.",
   [
    ("code", '''from tcc_fluxo import experimento1 as E, cv, Especificacao
from tcc_fluxo.metricas import seller_scales

spec = Especificacao.de_json(A.carregar_json("painel_spec"), C)
sel = A.carregar_json("selecao_modelo")
train = A.carregar_treino()
hist = A.carregar_historico_treino()
cv_origins, _ = cv.origens_cv(C)
assert len(cv_origins) == 6

extra = E.build_extra_features(hist)                     # causal: idêntico ao cálculo sobre a série completa
print(f"{len(E.EXP1_ALL)} atributos novos para {len(extra):,} linhas")'''),
    ("md", "### 24.1 Verificações dos atributos (vazamento, redundâncias, exemplo conceitual)"),
    ("code", '''rng = np.random.default_rng(C.RANDOM_SEED)
pert = hist.copy()
fut = pert["week_start"] > C.LEAK_TEST_CUT
pert.loc[fut, "revenue"] = pert.loc[fut, "revenue"] * rng.uniform(0.5, 3.0, fut.sum()) + 1000.0
pert.loc[fut, "orders"] = pert.loc[fut, "orders"] + 7
changed, check = E.verificar_atributos(extra, E.build_extra_features(pert), C.LEAK_TEST_CUT, hist)
print(f"Sem vazamento: OK (controle: {changed:.0%} dos atributos mudam depois do corte) | exemplo: {check}")

extra_cols = extra.drop(columns="zero_streak")
train_x = train.merge(extra_cols, on=["seller_id", "week_start"], how="left", validate="one_to_one")
assert len(train_x) == len(train)'''),
    ("md", "### 24.2 Decisões congeladas antes da validação"),
    ("code", '''assert np.allclose(train_x["pos_sum_4"], train_x[["revenue_current", "rev_lag_1", "rev_lag_2", "rev_lag_3"]].sum(axis=1))
assert np.allclose(train_x["max_4"], train_x[["revenue_current", "rev_lag_1", "rev_lag_2", "rev_lag_3"]].max(axis=1))
corr_df = (hist.loc[hist["week_start"] <= C.CV_FIRST_ORIGIN, ["seller_id", "week_start"]]
           .merge(A.carregar_painel(C.CV_FIRST_ORIGIN)[["seller_id", "week_start"] + spec.numeric],
                  on=["seller_id", "week_start"])
           .merge(extra_cols, on=["seller_id", "week_start"]))
INT_RED, l1 = E.reduce_block(E.EXP1_BLOCKS["intensidade"], "intensidade_red", corr_df, spec.numeric, C.CORR_MAX)
PIC_RED, l2 = E.reduce_block(E.EXP1_BLOCKS["picos"], "picos_red", corr_df, spec.numeric, C.CORR_MAX)
INTPIC_RED, l3 = E.reduce_block(INT_RED + PIC_RED, "intensidade_picos_red", corr_df, spec.numeric, C.CORR_MAX)
A.tabela(pd.DataFrame(l1 + l2 + l3), "experimento1_reducao_atributos.csv")

LIMITS, cal_df = E.calibrar_limites(train, hist, C)
A.tabela(cal_df, "limites_calibracao_pre_cv.csv"); display(cal_df)
ROUNDS = E.rodadas(INT_RED, PIC_RED, INTPIC_RED, spec.exog)
frozen, LIMITS_HASH = E.congelar_decisoes(LIMITS, ROUNDS, cv_origins, C)
A.salvar_json(frozen, "limites_prototipo")
print(f"Decisões congeladas; sha256={LIMITS_HASH[:16]}…")'''),
    ("md", "### 24.3 Validação nas seis origens"),
    ("code", '''perfil_val, exp1_long, EXP1_MODELS = E.validar(train_x, hist, cv_origins, ROUNDS, spec, C)
A.salvar_df(perfil_val, "experimento1_perfil_validacao")
A.tabela(perfil_val, "experimento1_perfil_validacao.csv")
cv_detail = A.carregar_df("cv_detalhe")
ref_cv = cv_detail.loc[cv_detail["modelo"] == "CatBoost_MAE", "MASE_vend_media"].to_numpy()
base_cv = perfil_val.query("rodada == 'base' and perfil == 'todos'")["MASE_vend_media"].to_numpy()
assert np.allclose(ref_cv, base_cv), (ref_cv, base_cv)
print("Rodada 'base' reproduz exatamente o CatBoost_MAE da validação principal: OK")'''),
    ("md", "### 24.4 Ganho por perfil"),
    ("code", '''prof_mean, prof_delta = E.ganho_por_perfil(perfil_val, ROUNDS)
A.tabela(prof_delta, "experimento1_ganho_por_perfil.csv")
display(prof_delta.set_index(["rodada", "perfil"]))
display(E.perguntas(prof_delta))'''),
    ("md", "### 24.5 Critérios por perfil e conformidade individual"),
    ("code", '''seller_c, comp_o, comp_s, crit_perfil, limits_all = E.conformidade(exp1_long, perfil_val, prof_mean, ROUNDS,
                                                                    cv_origins, LIMITS, C)
A.tabela(seller_c, "experimento1_conformidade_vendedor.csv")
A.tabela(comp_s, "experimento1_conformidade_vendedor_resumo.csv")
A.tabela(crit_perfil, "experimento1_criterios_por_perfil.csv")
display(crit_perfil[["rodada", "perfil", "MAE_ok", "MAE_positiva_ok", "P90_ok", "P95_ok", "vies_abs_ok",
                     "origens_dentro_dos_limites", "pct_vendedores_dentro_limites_media", "passou"]])
print("Passa nos três perfis:", limits_all.to_dict())'''),
    ("md", "### 24.6 Critérios globais, de picos e bootstrap das diferenças"),
    ("code", '''boot_df, glob_crit, glob, glob_wape = E.criterios_globais(exp1_long, perfil_val, prof_mean, ROUNDS, C)
A.tabela(boot_df, "experimento1_diferencas_bootstrap.csv")
A.tabela(glob_crit, "experimento1_criterios_globais.csv", index=True)
display(glob_crit)'''),
    ("md", "### 24.7 Seleção (somente validação)"),
    ("code", '''frozen_now = A.carregar_json("limites_prototipo")
assert E.hash_decisoes(frozen_now) == LIMITS_HASH, "limites alterados após o congelamento"
elig, exp1_choice = E.selecionar(ROUNDS, glob, glob_crit, limits_all, crit_perfil, LIMITS_HASH, cv_origins, C)
exp1_choice["modelo_principal_secao16"] = sel["modelo_escolhido"]
A.salvar_json(exp1_choice, "experimento1_modelo_escolhido")
display(elig)
print(exp1_choice["decisao"])
sens, el, tied, hip = E.sensibilidade_sem_limites(ROUNDS, glob, glob_crit, C)
print(f"[informativo] sem os limites por perfil: elegíveis {el}; empate {tied}; escolha hipotética {hip}")'''),
    ("md", "### 24.8 Teste — somente se a validação escolheu uma variante"),
    ("code", '''EXP1_CHOSEN = exp1_choice["modelo_escolhido"]
if EXP1_CHOSEN in (None, "base"):
    print("Teste do Experimento 1 NÃO executado:", "nenhum candidato elegível" if EXP1_CHOSEN is None else "a validação manteve a base")
else:
    from tcc_fluxo import diagnostico as D, metricas as M
    test = A.carregar_teste(finalidade="exploratoria")
    serie = A.carregar_serie_completa(finalidade="exploratoria")
    extra_full = E.build_extra_features(serie).drop(columns="zero_streak")
    train_x2 = train.merge(extra_full, on=["seller_id", "week_start"], how="left", validate="one_to_one")
    test_x = test.merge(extra_full, on=["seller_id", "week_start"], how="left", validate="one_to_one")
    comparison = A.carregar_df("previsoes_teste")
    profile = A.carregar_df("perfis_treino").set_index("seller_id")
    exp1_test, resumo_t = E.testar_escolhido(EXP1_CHOSEN, ROUNDS, train_x2, test_x, comparison, profile,
                                             M.seller_scales(hist, C.TRAIN_TARGET_END), D.limiares(hist), LIMITS, spec, C)
    A.tabela(exp1_test, "experimento1_teste.csv"); display(exp1_test)
    print("Diferença de MAE no teste (nova − base):", resumo_t)'''),
   ])

# ============================================================================ 08a
nb("08a_secundaria_estatisticos", "08a — Análise secundária: ARIMA/SARIMA(X) e Prophet",
   "**Lê:** série completa (trava, finalidade *secundaria*), `exogenas_semanais`, `previsoes_teste`.  \n"
   "**Grava:** `secundaria_pred_vendedor_estat`, `secundaria_pred_agregado`.\n\n"
   "Não participa da seleção. Ordens escolhidas por AIC **no treino**; previsão de 1 passo no teste. "
   "Por vendedor: parâmetros do treino aplicados com filtro de Kalman (sem reestimar). Agregado: reestimado a cada semana.",
   [
    ("code", '''from tcc_fluxo import secundaria as S

spec_json = A.carregar_json("painel_spec")
TEST_END = pd.Timestamp(A.carregar_json("corte")["TEST_END"])
exog = A.carregar_df("exogenas_semanais")
serie = A.carregar_serie_completa(finalidade="secundaria")
comparison = A.carregar_df("previsoes_teste")
test_weeks = np.sort(comparison["target_week"].unique())

MACRO = spec_json["MACRO_FEATURES"]
sec_exog, EXOG_COLS, faixa = S.preparar_exogenas(exog, MACRO, C.SEC_EXOG_CAL, C.PANEL_START, C.TRAIN_TARGET_END, TEST_END)
assert pd.DatetimeIndex(np.sort(serie["week_start"].unique())).isin(sec_exog.index).all()
A.tabela(faixa, "secundaria_faixa_macro.csv", index=True); display(faixa)'''),
    ("code", '''par = S.parametros(C, EXOG_COLS)
ts_pred, specs, spec_resumo = S.rodar_arima_vendedores(serie, sec_exog, test_weeks, C.TRAIN_TARGET_END, par, C.SEC_N_JOBS)
A.salvar_df(ts_pred, "secundaria_pred_vendedor_estat")
A.tabela(specs, "secundaria_especificacoes_vendedor.csv"); A.tabela(spec_resumo, "secundaria_especificacoes_resumo.csv")
display(spec_resumo)'''),
    ("code", '''agg_pred, agg_series, info = S.rodar_agregado(serie, sec_exog, EXOG_COLS, test_weeks, C.TRAIN_TARGET_END, C)
A.salvar_df(agg_pred, "secundaria_pred_agregado")
A.salvar_json({"agregado": info, "parametros": par}, "secundaria_configuracao")
print(info)'''),
   ])

# ============================================================================ 08b
nb("08b_secundaria_lstm", "08b — Análise secundária: LSTM (global, univariada e multivariada)",
   "**Lê:** treino, teste (trava, *secundaria*), série completa, `exogenas_semanais`.  \n"
   "**Grava:** `secundaria_pred_lstm`.\n\n"
   "Separado do 08a porque usa PyTorch (no Colab, pode ativar GPU sem afetar os outros notebooks). "
   "Épocas escolhidas num holdout interno do **treino** (últimas `LSTM_HOLDOUT_WEEKS` semanas-alvo).",
   [
    ("code", '''from tcc_fluxo import secundaria as S

spec_json = A.carregar_json("painel_spec")
TEST_END = pd.Timestamp(A.carregar_json("corte")["TEST_END"])
exog = A.carregar_df("exogenas_semanais")
train = A.carregar_treino()
test = A.carregar_teste(finalidade="secundaria")
serie = A.carregar_serie_completa(finalidade="secundaria")
MACRO = spec_json["MACRO_FEATURES"]
sec_exog, _, _ = S.preparar_exogenas(exog, MACRO, C.SEC_EXOG_CAL, C.PANEL_START, C.TRAIN_TARGET_END, TEST_END)

janelas = S.JanelasLSTM(serie, C.LSTM_SEQ_LEN, sec_exog, MACRO)
static_multi = MACRO + spec_json["HOLIDAY_FEATURES"] + spec_json["COMMERCIAL_FEATURES"] + ["target_week_sin", "target_week_cos"]
VARIANTS = {"LSTM_global": None,                                   # v5: receita + pedidos + seno/cosseno
            "LSTM_uni": {"seq": ["revenue"], "static": []},
            "LSTM_multi": {"seq": ["revenue", "orders"], "static": static_multi}}'''),
    ("code", '''preds, info, y_scale = S.rodar_lstms(janelas, train, test, VARIANTS, C)
out = test[["seller_id", "week_start", "target_week"]].copy()
for m, p in preds.items():
    out[f"pred_{m}"] = p
A.salvar_df(out, "secundaria_pred_lstm")
A.tabela(info, "secundaria_lstm_treino.csv"); display(info)'''),
   ])

# ============================================================================ 08c
nb("08c_secundaria_avaliacao", "08c — Análise secundária: avaliação e univariada × multivariada",
   "**Lê:** `previsoes_teste` (05), `secundaria_pred_*` (08a, 08b), `selecao_modelo.json`, histórico do treino e a série completa (trava, *secundaria*).  \n"
   "**Grava:** tabelas `secundaria_*` e figuras.\n\n"
   "Compare cada par **dentro do mesmo nível**. Só afirme que as exógenas ajudam quando o IC95 de `d` ficar todo abaixo de zero.\n\n"
   "Inclui o **seasonal naive** (`SNaive_52`, mesma semana do ano anterior) como referência complementar: não entra na "
   "seleção porque as origens de validação de 2017 não têm 52 semanas de histórico; onde falta t−52, usa o Naive.",
   [
    ("code", '''from tcc_fluxo import secundaria as S, metricas as M, graficos

sel = A.carregar_json("selecao_modelo")
CHOSEN = sel["modelo_escolhido"]
comparison = A.carregar_df("previsoes_teste")
hist = A.carregar_historico_treino()
ts_pred = A.carregar_df("secundaria_pred_vendedor_estat")
lstm = A.carregar_df("secundaria_pred_lstm")
agg_pred = A.carregar_df("secundaria_pred_agregado")
scale_final = M.seller_scales(hist, C.TRAIN_TARGET_END)
agg_scale = M.aggregate_scale(hist, C.TRAIN_TARGET_END)
_, aggregate = M.evaluate_aggregate(comparison, ["Naive", CHOSEN], agg_scale)

# Seasonal naive (mesma semana do ano anterior): referência complementar, só no teste.
serie = A.carregar_serie_completa(finalidade="secundaria")
pred_snaive, disp_snaive = S.seasonal_naive(comparison, serie)
snaive_tab, snaive_cob = S.avaliar_snaive(comparison, CHOSEN, pred_snaive, disp_snaive, scale_final, agg_scale)
A.tabela(snaive_tab, "secundaria_seasonal_naive.csv")
A.salvar_json(snaive_cob, "secundaria_seasonal_naive_cobertura")
print("Seasonal naive — cobertura:", snaive_cob); display(snaive_tab)

res = S.avaliar(comparison, CHOSEN, ts_pred, lstm, agg_pred, aggregate, scale_final, agg_scale,
                extras={"SNaive_52": pred_snaive})
A.tabela(res["vendedor"], "secundaria_metricas_vendedor.csv"); A.tabela(res["agregado"], "secundaria_metricas_agregado.csv")
A.tabela(res["per_seller"], "secundaria_metricas_por_vendedor.csv")
print("Cortes em zero dos modelos lineares:"); display(res["clip"])
display(res["vendedor"]); display(res["agregado"])'''),
    ("code", '''um = S.univariada_vs_multivariada(res, C.N_BOOT, C.RANDOM_SEED, C.MIN_WEEKS_DM)
A.tabela(um, "secundaria_univariada_vs_multivariada.csv")
display(um[["nivel", "familia", "univariada", "multivariada", "n", "univ", "multi", "d_media", "ic95_inf", "ic95_sup",
            "prop_vendedores_multi_melhor", "p_valor_hln", "leitura"]])'''),
    ("code", '''C_UNI, C_MULTI, C_BASE = "#2a78d6", "#1baf7a", "#b4b2a9"
col = lambda t: {"univariada": C_UNI, "multivariada": C_MULTI}.get(t, C_BASE)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
d_s = res["vendedor"].sort_values("MASE_media", ascending=False)
axes[0].barh(d_s["modelo"], d_s["MASE_media"], color=[col(t) for t in d_s["tipo"]])
axes[0].axvline(float(res["vendedor"].set_index("modelo").loc["Naive", "MASE_media"]), color=graficos.INK2, ls="--", lw=1)
axes[0].set_title("Nível vendedor — MASE médio por vendedor (teste)", loc="left")
d_a = res["agregado"].sort_values("MASE", ascending=False)
axes[1].barh(d_a["modelo"], d_a["MASE"], color=[col(t) for t in d_a["tipo"]])
axes[1].axvline(float(res["agregado"].set_index("modelo").loc["Naive (soma por vendedor)", "MASE"]), color=graficos.INK2, ls="--", lw=1)
axes[1].set_title("Nível agregado — MASE do total semanal (teste)", loc="left")
for ax in axes:
    graficos.clean(ax); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
fig.tight_layout(); A.figura(fig, "secundaria_comparacao.png", bbox_inches="tight"); plt.show()'''),
   ])

# ============================================================================ 09
nb("09_resumo_e_pacote", "09 — Resumo metodológico, ambiente e pacote",
   "**Lê:** todos os JSON de decisão e o registro de acessos ao teste.  \n"
   "**Grava:** `ambiente_execucao.json`, `requirements_execucao.txt` e o `.zip` da execução.\n\n"
   "Os números do texto do TCC devem vir dos CSVs desta execução, nunca de outra versão.",
   [
    ("code", '''import json, shutil
from tcc_fluxo.io import sha256_arquivo, _ler_texto

sel = A.carregar_json("selecao_modelo")
exp1 = A.carregar_json("experimento1_modelo_escolhido") if A.existe("experimento1_modelo_escolhido") else None
div = A.carregar_json("divisao")
print("Modelo escolhido (seção 16):", sel["modelo_escolhido"], "| empate prático:", sel["modelos_em_empate_pratico"])
print("Experimento 1:", None if exp1 is None else exp1["decisao"])
print(pd.Series(div))

reg = A.dir / "registro_acesso_teste.jsonl"
acessos = pd.read_json(reg, lines=True) if reg.is_file() else pd.DataFrame()
print("Acessos ao teste registrados:"); display(acessos)'''),
    ("code", '''inputs = {str(p.relative_to(C.RAIZ)): sha256_arquivo(p)
          for pasta in (C.caminho("dados_olist"), C.caminho("macro_cache")) for p in sorted(pasta.glob("*")) if p.is_file()}
amb = {"versoes": A.versions, "config": C.como_dict(), "input_sha256": inputs,
       "artefatos": {p.name: json.loads(_ler_texto(p))["sha256"] for p in sorted(A.dir.glob("*.meta.json"))}}
(A.saidas / "ambiente_execucao.json").write_text(json.dumps(amb, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
(A.saidas / "requirements_execucao.txt").write_text("\\n".join(f"{k}=={v}" for k, v in A.versions.items() if k != "python" and v) + "\\n", encoding="utf-8")
zip_base = C.caminho("artefatos").parent
arq = shutil.make_archive(str(zip_base), "zip", str(zip_base))
print("Pacote:", arq)'''),
   ])

# ============================================================================ 10
nb("10_sensibilidade_status", "10 — Sensibilidade do alvo ao status do pedido",
   "**Pergunta:** vale a pena trocar o alvo de *pedidos com pagamento aprovado, incluindo cancelados* (atual) "
   "para *só pedidos entregues*?  \n"
   "**Lê:** `order_seller_revenue` e `corte` (01), `coorte` (03). **Não ajusta modelos e não altera nenhum artefato.**  \n"
   "**Grava:** `sensibilidade_status_*.csv` e `sensibilidade_status_semanal.png`.\n\n"
   "O status usado é o **final**, como estava na extração da base. Na data da previsão ele ainda não é conhecido "
   "para pedidos recentes: veja a seção de leitura no fim.",
   [
    ("code", '''from tcc_fluxo import alvo, graficos

corte = A.carregar_json("corte")
TEST_END = pd.Timestamp(corte["TEST_END"])
osr = A.carregar_df("order_seller_revenue")
cohort_ids = A.carregar_df("coorte")["seller_id"]
R = alvo.comparar_status(osr, cohort_ids, C.PANEL_START, TEST_END, C.TRAIN_TARGET_END)
print(pd.Series(R["resumo"]).round(4))'''),
    ("md", "### Composição do valor por status e perda por período"),
    ("code", '''display(R["por_status"]); display(R["por_periodo"])
A.tabela(R["por_status"], "sensibilidade_status_por_status.csv")
A.tabela(R["por_periodo"], "sensibilidade_status_por_periodo.csv")'''),
    ("md", "### Semana a semana e vendedor a vendedor"),
    ("code", '''ps = R["por_semana"]
A.tabela(ps, "sensibilidade_status_por_semana.csv")
A.tabela(R["por_vendedor"], "sensibilidade_status_por_vendedor.csv")
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(ps["week_start"], ps["pct_removido"], color=graficos.C_CHOSEN, lw=2)
ax.axvline(C.TRAIN_TARGET_END, color=graficos.INK2, ls="--", lw=1)
ax.text(C.TRAIN_TARGET_END, ax.get_ylim()[1] * 0.95, " início do teste", color=graficos.INK2, va="top", fontsize=9)
ax.set_ylabel("% do valor removido")
ax.set_title("Valor da coorte que sai do alvo se ficarem só pedidos entregues, por semana de aprovação", loc="left")
graficos.clean(ax); fig.tight_layout()
A.figura(fig, "sensibilidade_status_semanal.png"); plt.show()
print("Vendedores mais afetados:"); display(R["por_vendedor"].head(10))'''),
    ("md", """### Como ler
- **Se o valor removido for pequeno e estável** (total semanal quase igual, correlação perto de 1), a troca não muda as conclusões no agregado; manter o alvo atual e citar o percentual como limitação é suficiente.
- **Se a perda se concentrar em poucos vendedores**, a troca muda a série justamente deles: vale reportar quantos.
- **Defasagem de informação:** o status "delivered" só é conhecido dias ou semanas depois da aprovação. Para usar *só entregues* sem vazamento, os atributos da semana *t* teriam de usar o status conhecido em *t*, não o final. Trocar o alvo exige refazer o 01 com essa regra e rodar tudo de novo numa nova `EXECUCAO`; os números não seriam comparáveis um a um com os atuais, porque o alvo muda."""),
   ])

# ============================================================================ 11
nb("11_acumulado_4_semanas", "11 — Receita acumulada das próximas 4 semanas (análise complementar)",
   "**Pergunta:** quanto o vendedor deve receber nas próximas 4 semanas (o prazo de um boleto de fornecedor)?  \n"
   "**Lê:** treino, série completa e teste pela trava (*secundaria*), `exogenas_semanais`, `selecao_modelo.json`.  \n"
   "**Grava:** `acumulado_4_semanas` (artefato), `acumulado_4_semanas.csv` (usado pelo protótipo) e tabelas `acumulado_*`.\n\n"
   "O resultado principal do TCC continua sendo **h = 1**. Aqui o modelo escolhido prevê as 4 semanas de forma "
   "**recursiva**: a previsão de cada semana entra no lugar do realizado para prever a seguinte. Variáveis macro "
   "ficam congeladas na origem; o calendário segue a semana-alvo. Não participa da seleção. "
   "Somar as previsões h = 1 do notebook 05 seria vazamento: cada uma usa o realizado da semana anterior.",
   [
    ("code", '''from tcc_fluxo import acumulado as AC, Especificacao, graficos

H = 4
spec_json = A.carregar_json("painel_spec")
spec = Especificacao.de_json(spec_json, C)
MODELO = A.carregar_json("selecao_modelo")["modelo_escolhido"]
TEST_END = pd.Timestamp(A.carregar_json("corte")["TEST_END"])
train = A.carregar_treino()
serie = A.carregar_serie_completa(finalidade="secundaria")
exog = A.carregar_df("exogenas_semanais")
print(f"Modelo: {MODELO} | origens: {len(AC.origens(C.TRAIN_TARGET_END, TEST_END, H))}")'''),
    ("code", '''longo = AC.rodar(train, serie, exog, MODELO, spec, C, TEST_END, H, macro_cols=spec_json["MACRO_FEATURES"])

# conferência: em h = 1 a recursão tem de reproduzir as previsões do notebook 05
prev05 = A.carregar_df("previsoes_teste")
h1 = longo[longo["h"] == 1].merge(prev05[["seller_id", "target_week", f"pred_{MODELO}"]],
                                  left_on=["seller_id", "semana_alvo"], right_on=["seller_id", "target_week"])
assert len(h1) == (longo["h"] == 1).sum() and np.allclose(h1["previsto"], h1[f"pred_{MODELO}"], rtol=0, atol=1e-9)
print("h = 1 idêntico ao notebook 05: OK")

larga = AC.tabela_larga(longo, serie, H)
A.salvar_df(larga, "acumulado_4_semanas")
A.tabela(larga, "acumulado_4_semanas.csv")'''),
    ("md", "### Erro do acumulado de 4 semanas e erro por horizonte"),
    ("code", '''res, freq, por_h = AC.avaliar(larga, longo, MODELO)
A.tabela(res, "acumulado_metricas.csv"); A.tabela(por_h, "acumulado_erro_por_horizonte.csv")
A.salvar_json({"modelo": MODELO, "horizonte": H, **freq}, "acumulado_resumo")
display(res); display(por_h)
print(f"O acumulado do modelo ficou mais perto do realizado que o Naive × 4 em {freq['pct_mais_perto_que_naive4']:.0f}% "
      f"das {freq['n_comparacoes_sem_empate']:,} comparações vendedor-origem sem empate.")'''),
    ("code", '''tot = larga.groupby("semana_fim")[["real_acum", "previsto_acum", "naive_acum"]].sum()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(tot.index, tot["real_acum"], color=graficos.INK, lw=2, label="Realizado")
axes[0].plot(tot.index, tot["previsto_acum"], color=graficos.C_CHOSEN, lw=2, label=f"{MODELO} (recursivo)")
axes[0].plot(tot.index, tot["naive_acum"], color=graficos.C_NAIVE, lw=2, ls="--", label="Naive × 4")
axes[0].set_title("Total da coorte nas 4 semanas seguintes (por semana final)", loc="left", fontsize=10)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f} mil"))
axes[0].legend(frameon=False, fontsize=8)
axes[1].bar(por_h["h"].astype(str), por_h["WAPE"], color=graficos.C_CHOSEN)
for i, v in enumerate(por_h["WAPE"]):
    axes[1].text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=graficos.INK2)
axes[1].set_title("WAPE por horizonte (vendedor-semana): a incerteza cresce", loc="left", fontsize=10)
axes[1].set_xlabel("semanas à frente")
for ax in axes:
    graficos.clean(ax)
fig.tight_layout(); A.figura(fig, "acumulado_4_semanas.png"); plt.show()'''),
   ])

# ============================================================================ 12
nb("12_oe3_sazonalidade", "12 — OE3: padrões sazonais e calendário comercial em 2017",
   "**Objetivo (OE3, Parte 1):** analisar de forma exploratória os padrões sazonais da receita, com ênfase na Black Friday "
   "de 2017 e nos feriados nacionais, e identificar os eventos do calendário com maior associação com a receita.  \n"
   "**Lê:** histórico do treino (`panel_base` até `TRAIN_TARGET_END`), `calendario_eventos`, `perfis_treino`, treino e "
   "`selecao_modelo.json`. **Não lê o teste.**  \n"
   "**Grava:** `oe3_*.csv` e `oe3_receita_semanal_2017.png`.\n\n"
   "**Lift** = receita da semana ÷ média das 4 semanas anteriores à janela do evento, pulando semanas de outros "
   "eventos comerciais e a semana seguinte à Black Friday. Como a coorte cresce em 2017, "
   "cada lift também é comparado com a mediana dos lifts das semanas comuns. Cada evento ocorre uma vez: "
   "os resultados são **associações observadas**, não efeitos causais.",
   [
    ("code", '''from tcc_fluxo import sazonalidade as SZ, graficos

ANO = 2017
# Janelas dos eventos: (rótulo, chave no calendário, semanas ANTES do evento na janela). Ajuste à Tabela 3 da Parte 1.
EVENTOS = SZ.EVENTOS_PADRAO
hist = A.carregar_historico_treino()
cal = A.carregar_df("calendario_eventos")
perfis = A.carregar_df("perfis_treino")
tot, ativos = SZ.receita_semanal(hist, C.PANEL_START, C.TRAIN_TARGET_END)
print(f"{len(tot)} semanas: {tot.index.min().date()} a {(tot.index.max() + pd.Timedelta(days=6)).date()}")'''),
    ("md", "## 1. Padrão sazonal geral"),
    ("code", '''jan = SZ.janelas_eventos(cal, ANO, EVENTOS)
fer = SZ.semanas_feriado(cal, ANO)
fig, ax = plt.subplots(figsize=(12, 4.6))
for _, r in jan.drop_duplicates("evento").iterrows():
    w0 = r["primeira_semana_janela"]; w1 = SZ.segunda(pd.Timestamp(r["data"]))
    ax.axvspan(w0 - pd.Timedelta(days=3), w1 + pd.Timedelta(days=4), color="#eb6834", alpha=0.13, lw=0)
    ax.text(w1, tot.max() * 1.55, r["evento"], rotation=90, va="top", ha="center", fontsize=8, color=graficos.INK2)
ax.plot(tot.index, tot.values, color=graficos.C_CHOSEN, lw=2, marker="o", ms=3, label="Receita semanal da coorte")
yf = tot.reindex(sorted(fer["semana"].unique())).dropna()
ax.scatter(yf.index, [tot.max() * 0.03] * len(yf), marker="|", s=120, color=graficos.INK, label="Semana com feriado nacional")
ax.set_ylim(0, tot.max() * 1.6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f} mil"))
ax.set_ylabel("R$ por semana"); graficos.clean(ax)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Receita semanal da coorte em 2017 (treino), com eventos comerciais (faixas) e feriados (marcas)", loc="left", fontsize=10)
fig.tight_layout(); A.figura(fig, "oe3_receita_semanal_2017.png"); plt.show()

mensal = pd.DataFrame({"receita_R$": tot.groupby(tot.index.to_period("M")).sum(),
                       "vendedores_no_painel": ativos.groupby(ativos.index.to_period("M")).max()})
mensal["receita_por_vendedor_R$"] = mensal["receita_R$"] / mensal["vendedores_no_painel"]
mensal.index = mensal.index.astype(str)
A.tabela(mensal.reset_index(names="mes"), "oe3_receita_mensal_2017.csv"); display(mensal)
print("Meses mais fortes (receita por vendedor):", list(mensal["receita_por_vendedor_R$"].nlargest(3).index))
print("Meses mais fracos (receita por vendedor):", list(mensal["receita_por_vendedor_R$"].nsmallest(3).index))
print("A receita total cresce também porque vendedores entram na coorte ao longo do ano: veja vendedores_no_painel.")'''),
    ("md", "## 2. Lift por evento comercial (resposta do OE3: ordem de associação)"),
    ("code", '''lift, ref = SZ.tabela_lift(tot, cal, ANO, EVENTOS)
A.tabela(lift, "oe3_lift_eventos.csv"); display(lift)
print(f"Referência: lift mediano das semanas comuns = {ref['lift_mediano_semanas_comuns']:.3f} ({ref['n_semanas_comuns']} semanas). "
      "lift_vs_semana_comum > 1 indica receita acima do padrão de uma semana sem evento.")'''),
    ("md", "## 3. Black Friday 2017 (24/11; semana de 20 a 26/11)"),
    ("code", '''data_bf = cal.loc[(cal["ano"] == ANO) & (cal["evento"] == "commercial_black_friday"), "date"].min()
bf, base_bf = SZ.janela_black_friday(tot, data_bf, antes=2, depois=1, excluir=SZ.semanas_excluidas(cal, ANO, EVENTOS))
A.tabela(bf, "oe3_black_friday_janela.csv"); display(bf)
print(f"Base: média das 4 semanas antes da janela = R$ {base_bf:,.2f}")
bfv = SZ.black_friday_vendedores(hist, perfis, data_bf)
A.tabela(bfv, "oe3_black_friday_vendedores.csv"); display(bfv)'''),
    ("md", "## 4. Feriados nacionais (agrupados)"),
    ("code", '''tab_fer, res_fer = SZ.tabela_feriados(tot, cal, ANO, EVENTOS)
A.tabela(tab_fer, "oe3_feriados.csv"); display(tab_fer)
print(pd.Series(res_fer).round(3))
_ = A.salvar_json({"lift_referencia": ref, "feriados": res_fer, "base_black_friday": base_bf}, "oe3_resumo")'''),
    ("md", "## 5. Ligação com a modelagem: peso do calendário no modelo escolhido"),
    ("code", '''from tcc_fluxo import Especificacao
from tcc_fluxo.modelos import ajustar
spec_json = A.carregar_json("painel_spec"); spec = Especificacao.de_json(spec_json, C)
MODELO = A.carregar_json("selecao_modelo")["modelo_escolhido"]
prever = ajustar(MODELO, A.carregar_treino(), spec)          # mesmo ajuste do notebook 05 (só treino)
grupos, detalhe = SZ.importancia_grupos(prever, spec_json)
if grupos is None:
    print(f"Importância não disponível para {MODELO}.")
else:
    A.tabela(grupos, "oe3_importancia_grupos.csv"); A.tabela(detalhe, "oe3_importancia_atributos.csv")
    print(f"Importância por grupo — {MODELO}:"); display(grupos)
    print("Atributos de calendário mais usados:"); display(detalhe[detalhe["grupo"].str.startswith("calend")].head(8))'''),
    ("md", """### Para o texto (seção 5.6)
- **Figura:** `oe3_receita_semanal_2017.png`. **Tabela:** `oe3_lift_eventos.csv` (já ordenada pelo lift).
- Escrever "associação observada", não "efeito": cada evento ocorre uma única vez em 2017.
- Em 2017, Natal (25/12, segunda) e Réveillon (31/12, domingo) caem na **mesma** semana; a semana anterior é a de compras de Natal.
- Dia dos Namorados (12/06) e Corpus Christi (15/06) também dividem a semana: a coluna `feriados_na_mesma_semana` mostra essas sobreposições."""),
   ])

# ============================================================================ 13
nb("13_complementos_texto", "13 — Complementos para o texto do TCC",
   "Números e figuras pedidos pelo texto que não saíam de nenhum notebook. **Não altera nenhuma escolha**: "
   "só lê artefatos gravados pelos notebooks 01–05. Usa a série completa e as previsões do teste com "
   "finalidade *exploratória* (acesso registrado), depois da seleção congelada.  \n"
   "**Grava:** `fig_funil_coorte.png`, `fig_receita_cortes.png`, `funil_coorte.csv`, `tabela_w_perfis.csv`, "
   "`mase_validacao_vs_teste.csv`, `ablacao_exogenas.csv`, `diagnostico_arimax_extremos.csv`, `diagnostico_arimax_piores_vendedores.csv`, `vendedores_inativos_teste.csv`, `metricas_teste_so_ativos.csv`, `metricas_teste_so_ativos_por_perfil.csv`, `faixas_macro_treino_teste.csv`.",
   [
    ("code", '''from tcc_fluxo import complementos as CP, painel as P, graficos
from tcc_fluxo.cv import origens_cv

sel = A.carregar_json("selecao_modelo")
CHOSEN = sel["modelo_escolhido"]
weekly_observed = A.carregar_df("weekly_observed")
perfis = A.carregar_df("perfis_treino")
hist = A.carregar_historico_treino()
serie = A.carregar_serie_completa(finalidade="exploratoria")
comparison = A.carregar_df("previsoes_teste")
cv_det = A.carregar_df("cv_detalhe")
resumo_val = A.carregar_df("resumo_validacao")
TEST_END = pd.Timestamp(A.carregar_json("corte")["TEST_END"])
ORIGENS, _ = origens_cv(C)
print("Modelo escolhido:", CHOSEN)'''),
    ("md", "## 1. Funil da coorte (seção 4.2)"),
    ("code", '''elig, cohort, _, _ = P.coorte(weekly_observed, C.PANEL_START, C.TRAIN_END, C.MIN_CALENDAR_WEEKS,
                              C.MIN_SALES_WEEKS, C.USE_MEI_PROXY, C.MEI_MONTHLY_PROXY_LIMIT)
etapas, extra = CP.funil(elig, weekly_observed, C.TRAIN_END, C.MIN_CALENDAR_WEEKS, C.MIN_SALES_WEEKS)
assert etapas["vendedores"].iloc[-1] == len(cohort) == perfis["seller_id"].nunique()
A.tabela(etapas.assign(**{k: v for k, v in extra.items()}), "funil_coorte.csv")
display(etapas); print(pd.Series(extra))

fig, ax = plt.subplots(figsize=(9, 3.4))
rot = list(etapas["etapa"]) + ["Com venda só em 2018 (fora da coorte)"]
val = list(etapas["vendedores"]) + [extra["vendedores_so_em_2018 (fora por construção)"]]
cores = [graficos.C_OTHER] * (len(rot) - 2) + [graficos.C_CHOSEN, "#dedcd5"]
y = np.arange(len(rot))[::-1]
ax.barh(y, val, color=cores, height=0.6)
for yi, v, (_, r) in zip(y, val, pd.concat([etapas, pd.DataFrame([{"excluidos_nesta_etapa": 0}])]).iterrows()):
    txt = f"{v:,}".replace(",", ".")
    if r["excluidos_nesta_etapa"]:
        txt += f"  (−{int(r['excluidos_nesta_etapa']):,})".replace(",", ".")
    ax.text(v + max(val) * 0.01, yi, txt, va="center", fontsize=9)
ax.set_yticks(y, rot, fontsize=9); ax.set_xlim(0, max(val) * 1.25); ax.set_xticks([])
for sp in ["top", "right", "bottom"]: ax.spines[sp].set_visible(False)
ax.set_title("Seleção da coorte de vendedores", loc="left", fontsize=10)
fig.tight_layout(); A.figura(fig, "fig_funil_coorte.png"); plt.show()'''),
    ("md", "## 2. Receita semanal com os cortes de treino, validação e teste (seção 4.5)"),
    ("code", '''tot = serie.groupby("week_start")["revenue"].sum()
# A receita da semana w é o alvo da linha da semana w-1: os cortes abaixo são em semana-ALVO.
primeiro_alvo = A.carregar_treino()["target_week"].min()
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.axvspan(tot.index.min() - pd.Timedelta(days=3), primeiro_alvo - pd.Timedelta(days=4), color="#e5e7eb", lw=0,
           label="Aquecimento (só atributos: 8 semanas de defasagem)")
ax.axvspan(primeiro_alvo - pd.Timedelta(days=3), C.TRAIN_TARGET_END + pd.Timedelta(days=3), color="#dbeafe", lw=0,
           label="Treino (alvos 06/03 a 25/12/2017)")
for i, o in enumerate(ORIGENS):
    ax.axvspan(o + pd.Timedelta(days=4), o + pd.Timedelta(weeks=C.CV_VAL_WEEKS, days=3), ymin=0, ymax=0.06,
               color="#2563eb", lw=0, label="Janelas de validação (6 origens × 4 semanas)" if i == 0 else None)
ax.axvspan(C.TRAIN_TARGET_END + pd.Timedelta(days=4), TEST_END + pd.Timedelta(days=3), color="#fde8dc", lw=0,
           label=f"Teste (alvos 01/01 a {TEST_END:%d/%m/%Y})")
ax.plot(tot.index, tot.values, color=graficos.INK, lw=1.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f} mil"))
ax.set_ylabel("R$ por semana"); ax.set_ylim(0, tot.max() * 1.12); graficos.clean(ax)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Receita semanal da coorte (870 vendedores) e divisão temporal", loc="left", fontsize=10)
fig.tight_layout(); A.figura(fig, "fig_receita_cortes.png"); plt.show()'''),
    ("md", "## 3. Tabela W — perfis de vendedor (seção 5.1)"),
    ("code", '''tab_w = CP.tabela_perfis(serie, perfis, C.TRAIN_TARGET_END)
A.tabela(tab_w, "tabela_w_perfis.csv"); display(tab_w.round(1))'''),
    ("md", "## 4. Por que o MASE de validação é maior que o de teste (seções 4.7 e 5.2)\\n"
           "MASE = MAE ÷ escala, com escala = média de |Δ semanal| da receita do próprio vendedor até o corte. "
           "Duas coisas mudam entre validação e teste: (i) o **numerador** — as semanas de validação (jul–dez/2017, "
           "com a Black Friday) são mais difíceis que as de 2018 (veja `MAE_Naive_R$`); (ii) o **denominador** — nas "
           "origens iniciais há vendedores com pouco histórico e escala pequena, que inflam a média de MASE por "
           "vendedor (veja `media_1_sobre_escala`). Como as duas coisas afetam o Naive e o modelo igualmente, a "
           "`razao_escolhido_sobre_Naive` fica estável: a queda não indica sobreajuste nem vazamento."),
    ("code", '''vt = CP.validacao_vs_teste(cv_det, resumo_val, comparison, hist, ORIGENS, C.TRAIN_TARGET_END, ["Naive", CHOSEN])
A.tabela(vt, "mase_validacao_vs_teste.csv"); display(vt.round(3))'''),
    ("md", "## 5. Ablação: modelo escolhido sem × com exógenas (seção 5.6)"),
    ("code", '''SEM = CHOSEN.removesuffix("_Exog")
assert f"pred_{SEM}" in comparison.columns, f"{SEM} sem previsões no teste"
abl = CP.ablacao(comparison, hist, C.TRAIN_TARGET_END, SEM, CHOSEN, C.N_BOOT, C.RANDOM_SEED)
A.tabela(abl, "ablacao_exogenas.csv"); display(abl.round(4))
print("p > 0,05 em todas as linhas = sem diferença significativa entre os modelos com e sem exógenas.")'''),
    ("md", "## 6. ARIMAX/SARIMAX por vendedor: previsões extremas (seção 5.7)\\n"
           "Por que as versões com exógenas ajustadas por vendedor e somadas têm erro agregado tão alto: "
           "(i) vendedores cuja previsão máxima no teste supera 10× (ou 5×) a maior receita semanal do próprio "
           "vendedor no treino, e quanto do erro vem deles; (ii) efeito do corte das previsões negativas em zero."),
    ("code", '''pred_sec = A.carregar_df("secundaria_pred_vendedor_estat")     # previsões brutas da 08a
arq = A.saidas / "secundaria_metricas_agregado.csv"
mase_ref = None
if arq.exists():
    t = pd.read_csv(arq)
    t = t.loc[t["modelo"].str.endswith("(soma por vendedor)")]
    mase_ref = dict(zip(t["modelo"].str.replace(" (soma por vendedor)", "", regex=False), t["MASE"]))
diag_ext, casos_ext = CP.diagnostico_estat_extremos(pred_sec, comparison, hist, perfis, mase_ref)
A.tabela(diag_ext, "diagnostico_arimax_extremos.csv"); A.tabela(casos_ext, "diagnostico_arimax_piores_vendedores.csv")
display(diag_ext.round(3)); display(casos_ext.round(1))'''),
    ("md", "## 7. Vendedores sem venda no teste (seções 5.1, 5.3, 5.4 e 6.4)\\n"
           "Quantos vendedores da amostra não tiveram nenhuma venda aprovada nas 34 semanas de 2018, e como ficam as "
           "métricas do teste quando eles são retirados. A seleção não muda: é só uma leitura complementar."),
    ("code", '''MODELOS_ATIVOS = ["Naive", CHOSEN, "MediaMovel4", "HistGB_Poisson", "CatBoost_MAE"]
cont_inat, met_ativos, perfil_ativos = CP.vendedores_inativos_teste(serie, perfis, comparison, hist, C.TRAIN_TARGET_END,
                                                     [m for m in MODELOS_ATIVOS if f"pred_{m}" in comparison])
A.tabela(cont_inat, "vendedores_inativos_teste.csv"); A.tabela(met_ativos, "metricas_teste_so_ativos.csv")
A.tabela(perfil_ativos, "metricas_teste_so_ativos_por_perfil.csv")
display(cont_inat); display(met_ativos.round(3)); display(perfil_ativos.round(3))'''),
    ("md", "## 8. Faixas das variáveis macroeconômicas: treino × teste (seções 5.6 e 5.7)"),
    ("code", '''exog_sem = A.carregar_df("exogenas_semanais")
macro_cols = A.carregar_json("painel_spec")["MACRO_FEATURES"]
fx = CP.faixas_macro(exog_sem, macro_cols, C.PANEL_START, C.TRAIN_TARGET_END, TEST_END)
A.tabela(fx, "faixas_macro_treino_teste.csv"); display(fx.round(2))'''),
   ])
