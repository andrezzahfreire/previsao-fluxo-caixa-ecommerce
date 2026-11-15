"""Confere os feriados calculados pelo projeto contra a BrasilAPI (feriados nacionais).

Uso (da raiz do projeto):
    .venv\\Scripts\\python ferramentas\\conferir_feriados_brasilapi.py
    .venv\\Scripts\\python ferramentas\\conferir_feriados_brasilapi.py --anos 2016 2019
    .venv\\Scripts\\python ferramentas\\conferir_feriados_brasilapi.py --atualizar   # baixa de novo

- Só LÊ o calendário do projeto (exogenas.calendario_eventos); não altera nenhum artefato nem resultado.
- As respostas da API ficam em dados_brutos/brasilapi_feriados/<ano>.json (registro do que foi comparado).
- O relatório vai para execucoes/<EXECUCAO>/saidas/conferencia_feriados_brasilapi.csv.
- Compara só os feriados (holiday_*). Datas comerciais (Black Friday, Dia das Mães...) não são feriados.
- Além do dia, mostra se a SEMANA (seg–dom) é a mesma: os atributos do modelo são semanais.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from tcc_fluxo.config import carregar_config  # noqa: E402
from tcc_fluxo.exogenas import calendario_eventos  # noqa: E402

URL = "https://brasilapi.com.br/api/feriados/v1/{ano}"

NOMES = {
    "holiday_new_year": "Confraternização Universal (01/01)",
    "holiday_carnival": "Carnaval (terça)",
    "holiday_good_friday": "Sexta-feira Santa",
    "holiday_easter": "Páscoa",
    "holiday_corpus_christi": "Corpus Christi",
    "holiday_tiradentes": "Tiradentes (21/04)",
    "holiday_labor_day": "Dia do Trabalho (01/05)",
    "holiday_independence": "Independência (07/09)",
    "holiday_children_day": "Nossa Sra. Aparecida (12/10)",
    "holiday_all_souls": "Finados (02/11)",
    "holiday_republic": "Proclamação da República (15/11)",
    "holiday_christmas": "Natal (25/12)",
}

# Diferenças esperadas, que NÃO indicam erro de cálculo.
OBSERVACOES = {
    "holiday_carnival": "ponto facultativo federal, não feriado por lei; a API pode omitir",
    "holiday_corpus_christi": "ponto facultativo federal; a API pode omitir",
    "holiday_easter": "domingo; algumas fontes listam, outras não",
}


def baixar(ano, pasta, atualizar):
    arq = pasta / f"{ano}.json"
    if arq.is_file() and not atualizar:
        return json.loads(arq.read_text(encoding="utf-8")), "cache"
    import requests
    ultimo_erro = None
    for tentativa in range(3):
        try:
            r = requests.get(URL.format(ano=ano), timeout=15, headers={"User-Agent": "tcc-fluxo-conferencia"})
            r.raise_for_status()
            dados = r.json()
            if not isinstance(dados, list):
                raise ValueError(f"resposta inesperada: {str(dados)[:200]}")
            pasta.mkdir(parents=True, exist_ok=True)
            arq.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
            return dados, "API"
        except Exception as e:  # noqa: BLE001
            ultimo_erro = e
            time.sleep(2 * (tentativa + 1))
    raise SystemExit(f"Não consegui baixar {ano} da BrasilAPI: {ultimo_erro}")


def segunda(d):
    return d - pd.Timedelta(days=d.weekday())


def main():
    C = carregar_config(RAIZ)
    padrao = (C.PANEL_START.year - 1, C.EXOGENOUS_API_END.year + 1)   # mesmos anos do notebook 02
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", nargs=2, type=int, default=padrao, metavar=("INICIO", "FIM"))
    ap.add_argument("--atualizar", action="store_true", help="ignora o cache e baixa de novo")
    args = ap.parse_args()
    anos = list(range(args.anos[0], args.anos[1] + 1))

    proj = calendario_eventos(anos)
    proj = proj.loc[proj["evento"].str.startswith("holiday_")].copy()
    proj["date"] = pd.to_datetime(proj["date"]).dt.normalize()

    pasta = RAIZ / "dados_brutos" / "brasilapi_feriados"
    linhas_api = []
    for ano in anos:
        dados, origem = baixar(ano, pasta, args.atualizar)
        print(f"{ano}: {len(dados)} feriados ({origem})")
        for h in dados:
            linhas_api.append({"date": pd.Timestamp(h["date"]).normalize(), "nome_api": h.get("name"),
                               "tipo_api": h.get("type")})
    api = pd.DataFrame(linhas_api)

    m = proj.merge(api, on="date", how="outer", indicator=True)
    m["ano"] = m["date"].dt.year
    m["feriado_projeto"] = m["evento"].map(NOMES)
    m["resultado"] = m["_merge"].map({"both": "igual", "left_only": "só no projeto", "right_only": "só na BrasilAPI"})
    m["observacao"] = m["evento"].map(OBSERVACOES).fillna("")

    # para quem só aparece de um lado: existe algo do outro lado na MESMA semana?
    sem_proj = set(proj["date"].map(segunda))
    sem_api = set(api["date"].map(segunda))
    m["mesma_semana_no_outro_lado"] = [
        "" if r == "igual" else ("sim" if segunda(d) in (sem_api if r == "só no projeto" else sem_proj) else "não")
        for d, r in zip(m["date"], m["resultado"])
    ]
    m.loc[(m["resultado"] == "só na BrasilAPI") & (m["date"].dt.strftime("%m-%d") == "11-20"), "observacao"] = \
        "Consciência Negra: feriado nacional só a partir de 2024 (Lei 14.759/2023)"

    out = (m[["ano", "date", "feriado_projeto", "nome_api", "tipo_api", "resultado", "mesma_semana_no_outro_lado",
              "observacao"]].sort_values("date").reset_index(drop=True))
    out["date"] = out["date"].dt.date

    saidas = C.caminho("saidas")
    saidas.mkdir(parents=True, exist_ok=True)
    out.to_csv(saidas / "conferencia_feriados_brasilapi.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(out.to_string(index=False))
    resumo = out["resultado"].value_counts()
    print("\nResumo:", resumo.to_dict())
    div = out.loc[(out["resultado"] != "igual") & (out["observacao"] == "")]
    if div.empty:
        print("Conclusão: todas as datas batem, exceto diferenças esperadas (ver coluna observacao).")
    else:
        print(f"Conclusão: {len(div)} diferença(s) sem explicação prévia — conferir as linhas acima.")
    print(f"Relatório: {saidas / 'conferencia_feriados_brasilapi.csv'}")


if __name__ == "__main__":
    main()
