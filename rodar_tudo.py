"""Executa os notebooks do projeto em ordem, sem abrir o VS Code.

Uso (no terminal, dentro da pasta do projeto):
    .venv\\Scripts\\python rodar_tudo.py              # todos, do 01 ao 09
    .venv\\Scripts\\python rodar_tudo.py --de 05      # retoma a partir do 05 (os anteriores já rodaram)
    .venv\\Scripts\\python rodar_tudo.py --so 06      # só um notebook
    .venv\\Scripts\\python rodar_tudo.py --ate 05     # do 01 até o 05

- Para no primeiro erro e mostra a célula que falhou.
- Os notebooks em notebooks/ NÃO são alterados. A cópia executada, com todas as saídas,
  vai para execucoes/<EXECUCAO>/notebooks_executados/ (registro da execução para a banca).
- O log com tempos fica em execucoes/<EXECUCAO>/rodar_tudo.log.
"""
import argparse
import re
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ORDEM = ["01_alvo", "02_exogenas", "03_painel", "04_validacao_selecao", "05_avaliacao_teste",
         "06_diagnostico", "07_experimento1", "08a_secundaria_estatisticos", "08b_secundaria_lstm",
         "08c_secundaria_avaliacao", "09_resumo_e_pacote", "10_sensibilidade_status", "11_acumulado_4_semanas", "12_oe3_sazonalidade", "13_complementos_texto"]

if sys.platform.startswith("win"):   # evita avisos/travamentos do asyncio com o Jupyter no Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _indice(prefixo):
    for i, n in enumerate(ORDEM):
        if n.startswith(prefixo):
            return i
    sys.exit(f"Notebook '{prefixo}' não encontrado. Opções: {', '.join(n.split('_')[0] for n in ORDEM)}")


def main():
    ap = argparse.ArgumentParser(description="Roda os notebooks do TCC em ordem.")
    ap.add_argument("--de", help="primeiro notebook (ex.: 05)")
    ap.add_argument("--ate", help="último notebook (ex.: 07)")
    ap.add_argument("--so", help="roda só este notebook (ex.: 06)")
    args = ap.parse_args()

    raiz = Path(__file__).resolve().parent
    cfg = yaml.safe_load((raiz / "config.yaml").read_text(encoding="utf-8"))
    pasta_exec = raiz / cfg["CAMINHOS"]["artefatos"].format(EXECUCAO=cfg["EXECUCAO"])
    pasta_exec = pasta_exec.parent
    destino = pasta_exec / "notebooks_executados"
    destino.mkdir(parents=True, exist_ok=True)
    log = pasta_exec / "rodar_tudo.log"

    if args.so:
        lista = [ORDEM[_indice(args.so)]]
    else:
        i0 = _indice(args.de) if args.de else 0
        i1 = _indice(args.ate) if args.ate else len(ORDEM) - 1
        lista = ORDEM[i0:i1 + 1]

    os.environ["TCC_RAIZ"] = str(raiz)
    print(f"Execução '{cfg['EXECUCAO']}' | {len(lista)} notebook(s): {', '.join(lista)}\n", flush=True)
    t_total = time.time()
    for nome in lista:
        nb = nbformat.read(raiz / "notebooks" / f"{nome}.ipynb", as_version=4)
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(raiz / "notebooks")}})
        t0 = time.time()
        print(f"▶ {nome} rodando...", flush=True)
        try:
            client.execute()
        except CellExecutionError as e:
            nbformat.write(nb, destino / f"{nome}.ipynb")
            msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e))   # tira códigos de cor
            print(f"✗ {nome} FALHOU após {time.time() - t0:.0f}s\n")
            print(msg[-4000:])
            print(f"\nNotebook com o erro salvo em: {destino / (nome + '.ipynb')}")
            print(f"Depois de corrigir, retome com:  python rodar_tudo.py --de {nome.split('_')[0]}")
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  FALHOU  {nome}  {time.time() - t0:.0f}s\n")
            sys.exit(1)
        nbformat.write(nb, destino / f"{nome}.ipynb")
        dt = time.time() - t0
        print(f"✓ {nome} ok ({dt / 60:.1f} min)", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  ok      {nome}  {dt:.0f}s\n")

    print(f"\nConcluído em {(time.time() - t_total) / 60:.1f} min.")
    print(f"Tabelas e figuras: {pasta_exec / 'saidas'}")
    print(f"Notebooks executados: {destino}")


if __name__ == "__main__":
    main()
