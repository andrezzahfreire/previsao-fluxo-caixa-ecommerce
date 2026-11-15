# TCC — previsão semanal da entrada bruta de caixa por vendedor (Olist), versão modular v6

Esta é a divisão do notebook monolítico `olist_painel_semanal_macro_calendario_v5_modelos.ipynb` em
**11 notebooks curtos + um pacote Python (`src/tcc_fluxo`)**. A lógica é a mesma do v5: em dados de teste
no formato do Olist, todos os resultados foram reproduzidos **número por número** (ver `VERIFICACAO.md`).

## Estrutura

```
tcc_fluxo/
├── config.yaml              ← TODOS os parâmetros (datas, seeds, limiares). Única fonte.
├── requirements.txt
├── src/tcc_fluxo/           ← toda a lógica (funções); os notebooks só chamam
├── notebooks/               ← 01 a 09, rodar em ordem
├── dados_brutos/
│   ├── olist/               ← coloque aqui os 3 CSVs do Olist
│   └── macro_cache/         ← séries do SGS (baixadas uma vez e reutilizadas)
├── execucoes/<EXECUCAO>/    ← criado ao rodar
│   ├── artefatos/           ← contratos entre notebooks (.parquet/.json + .meta.json com sha256)
│   └── saidas/              ← tabelas .csv e figuras .png para o texto do TCC
├── rodar_tudo.py            ← executa os notebooks em ordem, sem abrir o VS Code
├── prototipo/app.py         ← protótipo Streamlit de apoio à decisão (lê as saídas da execução)
└── ferramentas/gerar_dados_sinteticos.py   ← dados falsos no formato Olist, para testar o pipeline
```

## Como rodar no Colab

1. Envie a pasta `tcc_fluxo/` inteira para `Meu Drive/` (fica `Meu Drive/tcc_fluxo/`).
2. Coloque `olist_orders_dataset.csv`, `olist_order_payments_dataset.csv` e `olist_order_items_dataset.csv`
   em `tcc_fluxo/dados_brutos/olist/`.
3. Se você tem o cache do SGS da execução v5 (`saida_.../macro_cache/*.csv`), copie os arquivos para
   `dados_brutos/macro_cache/`. Assim a v6 usa **exatamente** as mesmas séries da v5. Sem cache, o
   notebook 02 baixa da API do BCB (que pode ter revisado valores).
4. Abra os notebooks pela pasta `notebooks/` e rode **em ordem**. A primeira célula de cada um monta o Drive,
   instala os pacotes e chama `iniciar()`. Se a pasta não estiver em `MyDrive/tcc_fluxo`, ajuste `RAIZ` nessa célula.

Fora do Colab: defina a variável de ambiente `TCC_RAIZ` com o caminho da pasta, ou abra os notebooks de dentro dela.

## Rodar tudo de uma vez (local)

```powershell
.venv\Scripts\python rodar_tudo.py              # 01 ao 09, para no primeiro erro
.venv\Scripts\python rodar_tudo.py --de 05      # retoma a partir do 05
.venv\Scripts\python rodar_tudo.py --so 06      # só um notebook
```

Os notebooks em `notebooks/` não são alterados. As cópias executadas, com todas as saídas, ficam em
`execucoes/<EXECUCAO>/notebooks_executados/`, e os tempos em `execucoes/<EXECUCAO>/rodar_tudo.log`.

## Protótipo (Streamlit)

Depois de rodar até o notebook 05, na raiz do projeto:

```powershell
.venv\Scripts\python -m pip install -r prototipo\requirements.txt
.venv\Scripts\python -m streamlit run prototipo\app.py
```

O app lê a execução indicada em `EXECUCAO` no `config.yaml`: as previsões semanais do teste (`previsoes_teste.csv`, notebook 05),
as previsões das próximas 4 semanas (`acumulado_4_semanas.csv`, notebook 11; sem ele, só a opção "próxima semana"),
o modelo escolhido e os modelos elegíveis. Ele não altera nenhum arquivo.

Mostra **vendas aprovadas previstas**, com aviso de que não são dinheiro disponível, uma faixa provável baseada só em
erros de períodos anteriores, a frequência com que a previsão acertou mais que a referência ingênua e um campo único
de contas a pagar (semana ou 4 semanas). A comparação de modelos fica em "Detalhes técnicos".

## Ordem e contratos

| Notebook | Seções do v5 | Lê | Grava | Teste |
|---|---|---|---|---|
| `01_alvo` | 1–5 | CSVs Olist | `order_seller_revenue`, `weekly_observed`, `weeks_info`, `corte` | — |
| `02_exogenas` | 5.1 | `corte`, cache SGS | `exogenas_semanais`, `exogenas_listas` | — |
| `03_painel` | 6–11 | 01, 02 | `coorte`, `perfis_treino`, `panel_base`, `painel`, `painel_spec`, `divisao` | constrói, não avalia |
| `04_validacao_selecao` | 13–16 | só treino | `cv_detalhe`, `resumo_validacao`, **`selecao_modelo`** | **não** |
| `05_avaliacao_teste` | 17–22, 25 | seleção + teste | `previsoes_teste`, métricas, IC, DM | **confirmatório** |
| `06_diagnostico` | 23 | `previsoes_teste` | `diagnostico_*` | diagnóstico |
| `07_experimento1` | 24 | só treino | `limites_prototipo`, `experimento1_*` | só se escolher variante (exploratório) |
| `08a_secundaria_estatisticos` | secundária | série completa | ARIMA/SARIMA(X), Prophet | secundário |
| `08b_secundaria_lstm` | secundária | treino, teste | LSTM global/uni/multi | secundário |
| `08c_secundaria_avaliacao` | secundária | 05, 08a, 08b | métricas, univariada × multivariada | secundário |
| `09_resumo_e_pacote` | 26–27 | tudo | `ambiente_execucao.json`, `.zip` | — |
| `10_sensibilidade_status` | — | 01, 03 | `sensibilidade_status_*` | — (descritivo) |
| `11_acumulado_4_semanas` | — | treino, série completa, 05 | `acumulado_4_semanas`, `acumulado_*` (usados pelo protótipo) | complementar |

## Garantias embutidas

- **Hash em todo artefato.** `A.carregar_df()` confere o sha256; um arquivo alterado depois de gravado gera erro.
- **Trava do teste.** As linhas de 2018 só saem por `A.carregar_teste(finalidade)` ou
  `A.carregar_serie_completa(finalidade)`. Os dois exigem `selecao_modelo.json` e conferem se o painel é o
  mesmo da seleção. Cada acesso fica em `artefatos/registro_acesso_teste.jsonl` (notebook, finalidade, horário):
  é a prova, para a banca, de que o teste foi usado uma vez de forma confirmatória.
- **Sem alvos de 2018 no treino.** `A.carregar_painel(ate)` não passa de `TRAIN_TARGET_END` e mascara alvos do teste.
- **Registro de execução.** `registro_notebooks.jsonl` guarda o que cada notebook leu, gravou e as versões dos pacotes.

## O que mudou em relação ao v5

Os números do experimento principal, do Experimento 1 e da análise secundária são os mesmos. Mudanças:

1. **Diagnóstico (06)** usa o modelo de `selecao_modelo.json` (`DIAG_MODELO_PRINCIPAL: escolhido`).
   No v5 ele estava fixo em `CatBoost_MAE`, embora a seleção registrada tenha sido `LightGBM_MAE_Exog`.
   Os arquivos `diagnostico_catboost_*` passaram a se chamar `diagnostico_principal_*`.
2. **Acréscimos** (sem alterar colunas antigas): `vies_total_semanal.csv` (viés do total semanal) e a
   coluna `p_holm` (correção de Holm) em `teste_diebold_mariano.csv`.
3. **Cache do SGS** saiu da pasta da execução para `dados_brutos/macro_cache/`, com hash registrado.
4. **Experimento 1** calcula os atributos extras com o histórico até `TRAIN_TARGET_END`. Por causalidade, os
   valores nas linhas de treino são idênticos aos do v5 (verificado). A série completa só é lida se uma variante for ao teste.

## Regra de ouro para editar

Mudou um parâmetro? Edite o `config.yaml`, mude `EXECUCAO` para um nome novo e rode tudo de novo a partir do 01.
Mudou uma função? Edite o `.py` em `src/` (o `%autoreload` recarrega) e rode de novo os notebooks a partir do primeiro
que a usa. Nunca copie funções para dentro dos notebooks.
