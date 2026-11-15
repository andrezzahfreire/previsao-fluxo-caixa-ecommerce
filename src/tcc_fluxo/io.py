"""Contratos entre notebooks.

Regras:
1. Notebooks só trocam informação por arquivos em `artefatos/` (parquet ou json),
   sempre acompanhados de `<nome>.meta.json` com sha256, notebook de origem e entradas.
2. Ao carregar, o hash é conferido: um artefato alterado depois de gravado gera erro.
3. O conjunto de teste só sai por `carregar_teste()` / `carregar_serie_completa()`,
   que exigem a seleção congelada e registram cada acesso em
   `registro_acesso_teste.jsonl` (trilha de auditoria para a banca).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "target_next_week"


def sha256_arquivo(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (pd.Timestamp, _dt.date, _dt.datetime)):
        return str(o.date()) if isinstance(o, pd.Timestamp) else str(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"não serializável: {type(o)}")


def _ler_texto(path) -> str:
    """Lê em UTF-8; aceita arquivos antigos gravados na codificação do Windows (cp1252)."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _gravar_texto(path, texto: str):
    Path(path).write_text(texto, encoding="utf-8")


def _agora():
    return pd.Timestamp.now().isoformat(timespec="seconds")


class Artefatos:
    def __init__(self, C, notebook: str, versions: dict):
        self.C = C
        self.notebook = notebook
        self.versions = versions
        self.dir = C.caminho("artefatos")
        self.saidas = C.caminho("saidas")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.saidas.mkdir(parents=True, exist_ok=True)
        self._lidos: dict[str, str] = {}
        self._gravados: list[str] = []
        self._inicio = _agora()

    # ------------------------------------------------------------------ meta
    def _meta_path(self, nome):
        return self.dir / f"{nome}.meta.json"

    def _gravar_meta(self, nome, arquivo: Path, extra: dict):
        meta = {
            "arquivo": arquivo.name, "sha256": sha256_arquivo(arquivo),
            "notebook": self.notebook, "criado_em": _agora(),
            "entradas": dict(self._lidos), **extra,
        }
        _gravar_texto(self._meta_path(nome), json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default))
        self._gravados.append(arquivo.name)
        return meta

    def _conferir(self, nome) -> tuple[Path, dict]:
        mp = self._meta_path(nome)
        if not mp.is_file():
            raise FileNotFoundError(
                f"Artefato '{nome}' não existe em {self.dir}. Rode antes o notebook que o produz.")
        meta = json.loads(_ler_texto(mp))
        p = self.dir / meta["arquivo"]
        sha = sha256_arquivo(p)
        if sha != meta["sha256"]:
            raise RuntimeError(f"'{nome}': hash não confere — o arquivo foi alterado depois de gravado "
                               f"por {meta['notebook']}. Rode novamente o notebook de origem.")
        self._lidos[nome] = sha
        return p, meta

    def existe(self, nome) -> bool:
        return self._meta_path(nome).is_file()

    def sha(self, nome) -> str:
        return json.loads(_ler_texto(self._meta_path(nome)))["sha256"]

    # ------------------------------------------------------------ dataframes
    def salvar_df(self, df: pd.DataFrame, nome: str, **extra):
        p = self.dir / f"{nome}.parquet"
        df.to_parquet(p, index=False)
        return self._gravar_meta(nome, p, {"linhas": len(df), "colunas": list(map(str, df.columns)), **extra})

    def carregar_df(self, nome: str) -> pd.DataFrame:
        p, _ = self._conferir(nome)
        return pd.read_parquet(p)

    # ------------------------------------------------------------------ json
    def salvar_json(self, obj, nome: str):
        p = self.dir / f"{nome}.json"
        _gravar_texto(p, json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))
        return self._gravar_meta(nome, p, {})

    def carregar_json(self, nome: str):
        p, _ = self._conferir(nome)
        return json.loads(_ler_texto(p))

    # -------------------------------------------------- saídas para o texto
    def tabela(self, df: pd.DataFrame, nome_csv: str, index=False):
        df.to_csv(self.saidas / nome_csv, index=index, encoding="utf-8")
        return self.saidas / nome_csv

    def figura(self, fig, nome_png: str, **kw):
        fig.savefig(self.saidas / nome_png, dpi=150, **kw)
        return self.saidas / nome_png

    # ------------------------------------------------- acesso ao treino/teste
    def _painel(self):
        return self.carregar_df("painel")

    def carregar_treino(self) -> pd.DataFrame:
        """Linhas utilizáveis com target_week <= TRAIN_TARGET_END (idêntico a `train` do v5)."""
        p = self._painel()
        return p.loc[p["conjunto"] == "treino"].drop(columns="conjunto").reset_index(drop=True)

    def carregar_painel(self, ate_semana) -> pd.DataFrame:
        """Painel (com atributos) até `ate_semana`, sem expor alvos do teste.

        Linhas cuja semana-alvo cai depois de TRAIN_TARGET_END ficam com alvo NaN.
        """
        ate_semana = pd.Timestamp(ate_semana)
        if ate_semana > self.C.TRAIN_TARGET_END:
            raise PermissionError("carregar_painel só vai até TRAIN_TARGET_END; use carregar_teste().")
        p = self._painel()
        p = p.loc[p["week_start"] <= ate_semana].drop(columns="conjunto").copy()
        p.loc[p["target_week"] > self.C.TRAIN_TARGET_END, TARGET] = np.nan
        return p.reset_index(drop=True)

    def carregar_historico_treino(self) -> pd.DataFrame:
        """Série vendedor-semana (panel_base) até TRAIN_TARGET_END: receita, pedidos, had_record."""
        b = self.carregar_df("panel_base")
        return b.loc[b["week_start"] <= self.C.TRAIN_TARGET_END].reset_index(drop=True)

    def _exigir_selecao(self, finalidade):
        if not self.existe("selecao_modelo"):
            raise PermissionError("selecao_modelo.json ausente: o teste está bloqueado até a seleção "
                                  "ser congelada no notebook 04.")
        if finalidade not in {"confirmatoria", "diagnostico", "secundaria", "exploratoria"}:
            raise ValueError("finalidade deve ser confirmatoria, diagnostico, secundaria ou exploratoria")
        sel = self.carregar_json("selecao_modelo")
        painel_sha = self.sha("painel")
        if sel.get("painel_sha256") != painel_sha:
            raise RuntimeError("O painel mudou depois da seleção (hash diferente do registrado em "
                               "selecao_modelo.json). Rode de novo os notebooks 03 e 04.")
        return sel

    def _registrar_acesso(self, o_que, finalidade):
        reg = self.dir / "registro_acesso_teste.jsonl"
        with open(reg, "a", encoding="utf-8") as f:
            f.write(json.dumps({"notebook": self.notebook, "dado": o_que, "finalidade": finalidade,
                                "quando": _agora()}, ensure_ascii=False) + "\n")

    def carregar_teste(self, finalidade: str) -> pd.DataFrame:
        """Único caminho para as linhas de teste. Exige seleção congelada e registra o acesso."""
        self._exigir_selecao(finalidade)
        self._registrar_acesso("painel[teste]", finalidade)
        p = self._painel()
        return p.loc[p["conjunto"] == "teste"].drop(columns="conjunto").reset_index(drop=True)

    def carregar_serie_completa(self, finalidade: str) -> pd.DataFrame:
        """panel_base inteiro (inclui 2018). Mesma trava e registro de carregar_teste()."""
        self._exigir_selecao(finalidade)
        self._registrar_acesso("panel_base completo", finalidade)
        return self.carregar_df("panel_base")

    # ------------------------------------------------------------- registro
    def finalizar(self):
        reg = self.dir / "registro_notebooks.jsonl"
        with open(reg, "a", encoding="utf-8") as f:
            f.write(json.dumps({"notebook": self.notebook, "inicio": self._inicio, "fim": _agora(),
                                "leu": self._lidos, "gravou": self._gravados, "versoes": self.versions},
                               ensure_ascii=False) + "\n")
        print(f"{self.notebook}: leu {len(self._lidos)} artefato(s), gravou {len(self._gravados)}.")
