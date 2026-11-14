"""Leitura do config.yaml e preparação idêntica do ambiente em todos os notebooks."""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Chaves do config.yaml convertidas para pd.Timestamp na leitura.
_DATE_KEYS = {
    "EXOGENOUS_API_START", "EXOGENOUS_API_END", "TRAIN_END", "TRAIN_TARGET_END",
    "PANEL_START", "CV_FIRST_ORIGIN", "LEAK_TEST_CUT",
}
_TUPLE_LIST_KEYS = {"ARIMA_SELLER_ORDERS", "ARIMA_AGG_ORDERS", "SARIMA_SEASONAL_CANDIDATES"}


class Config:
    """Acesso por atributo (C.TRAIN_END) aos valores do config.yaml. Somente leitura."""

    def __init__(self, data: dict, raiz: Path):
        object.__setattr__(self, "_d", data)
        object.__setattr__(self, "RAIZ", raiz)

    def __getattr__(self, key):
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(f"'{key}' não existe no config.yaml") from None

    def __getitem__(self, key):
        return self._d[key]

    def __setattr__(self, key, value):
        raise AttributeError("Config é somente leitura: altere o config.yaml")

    def caminho(self, nome: str) -> Path:
        rel = self._d["CAMINHOS"][nome].format(EXECUCAO=self._d["EXECUCAO"])
        return (self.RAIZ / rel).resolve()

    def como_dict(self) -> dict:
        out = {}
        for k, v in self._d.items():
            out[k] = str(v.date()) if isinstance(v, pd.Timestamp) else v
        return out


def encontrar_raiz(raiz=None) -> Path:
    """Raiz do projeto = pasta que contém config.yaml."""
    if raiz is not None:
        p = Path(raiz).expanduser().resolve()
        if not (p / "config.yaml").is_file():
            raise FileNotFoundError(f"config.yaml não encontrado em {p}")
        return p
    if os.environ.get("TCC_RAIZ"):
        return encontrar_raiz(os.environ["TCC_RAIZ"])
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "config.yaml").is_file():
            return cand
    raise FileNotFoundError("config.yaml não encontrado; passe a raiz do projeto para iniciar().")


def carregar_config(raiz=None) -> Config:
    raiz = encontrar_raiz(raiz)
    data = yaml.safe_load((raiz / "config.yaml").read_text(encoding="utf-8"))
    for k in _DATE_KEYS:
        if k in data and data[k] is not None:
            data[k] = pd.Timestamp(data[k])
    for k in _TUPLE_LIST_KEYS:
        if k in data:
            data[k] = [tuple(x) for x in data[k]]
    for k in ("CV_FIRST_ORIGIN", "TRAIN_TARGET_END"):
        assert data[k].weekday() == 0, f"{k} precisa ser segunda-feira"
    return Config(data, raiz)


def versoes_atuais() -> dict:
    def v(pkg):
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return None
    pkgs = ["pandas", "numpy", "scikit-learn", "scipy", "catboost", "lightgbm", "statsmodels",
            "prophet", "torch", "pyarrow", "requests", "matplotlib"]
    return {"python": platform.python_version(), **{p: v(p) for p in pkgs}}


def fixar_sementes(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def iniciar(raiz=None, notebook: str = ""):
    """Primeira chamada de todo notebook. Devolve (C, A): configuração e gestor de artefatos."""
    from .io import Artefatos   # import tardio evita ciclo

    C = carregar_config(raiz)
    fixar_sementes(C.RANDOM_SEED)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 220)

    versions = versoes_atuais()
    for pkg, expected in C.RECOMMENDED_VERSIONS.items():
        actual = versions.get(pkg)
        if actual is not None and actual != expected:
            print(f"AVISO de reprodutibilidade: {pkg}={actual}; referência recomendada={expected}")

    A = Artefatos(C, notebook, versions)
    print(f"Notebook: {notebook} | execução: {C.EXECUCAO} | raiz: {C.RAIZ}")
    return C, A
