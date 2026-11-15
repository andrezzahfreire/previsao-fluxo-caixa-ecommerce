"""Seção 13 do v5: modelos do experimento principal (globais semanais e baselines)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .painel import Especificacao

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:   # pragma: no cover
    HAS_CATBOOST = False
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:   # pragma: no cover
    HAS_LIGHTGBM = False

if not HAS_CATBOOST:
    raise ImportError("CatBoost não foi carregado. Instale e reinicie o runtime do Colab.")


def lista_modelos(ridge_alphas):
    boost = (["HistGB_MAE", "HistGB_Poisson"]
             + (["LightGBM_MAE", "LightGBM_MAE_Exog"] if HAS_LIGHTGBM else [])
             + (["CatBoost_MAE", "CatBoost_MAE_raw", "CatBoost_Tweedie", "CatBoost_MAE_Exog"] if HAS_CATBOOST else []))
    ridge = [f"Ridge_a{a}" for a in ridge_alphas]
    return boost, ridge, ["Naive", "MediaMovel4", *ridge, *boost]


def _features_for_model(name, spec: Especificacao):
    return spec.numeric_exog if name.endswith("_Exog") else spec.numeric


def _xcat(df, spec: Especificacao, num_features=None):
    num_features = spec.numeric if num_features is None else num_features
    X = df[num_features + spec.cat].copy()
    X["seller_id"] = X["seller_id"].astype(str)
    return X


def _catboost(loss, spec):
    return CatBoostRegressor(loss_function=loss, iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                             random_seed=spec.seed, verbose=False, thread_count=spec.n_threads,
                             allow_writing_files=False)


def ajustar(name, fit_df, spec: Especificacao):
    """Treina o modelo `name` e devolve uma função prever(pred_df) -> previsões brutas.

    *_MAE usam MAE sobre log1p; CatBoost_MAE_raw usa MAE em reais. Treinar uma vez e prever várias vezes
    é o que permite a previsão recursiva de várias semanas (notebook 11).
    """
    T = spec.target
    y_raw = fit_df[T].clip(lower=0).to_numpy(float)
    y_log = np.log1p(y_raw)

    if name == "Naive":
        return lambda pred_df: pred_df["revenue_current"].to_numpy(float)
    if name == "MediaMovel4":
        return lambda pred_df: pred_df[["revenue_current", "rev_lag_1", "rev_lag_2", "rev_lag_3"]].mean(axis=1).to_numpy(float)
    if name.startswith("Ridge_a"):
        alpha = float(name.split("_a")[1])
        pre = ColumnTransformer([("num", StandardScaler(), spec.numeric),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), spec.cat)])
        model = Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])
        model.fit(_xcat(fit_df, spec), y_log)

        def prever(pred_df):
            with np.errstate(over="ignore"):
                return np.expm1(model.predict(_xcat(pred_df, spec)))
        return prever
    if name == "HistGB_MAE":
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.03, max_depth=6,
                                              l2_regularization=5.0, random_state=spec.seed)
        model.fit(fit_df[spec.numeric], y_log)
        return lambda pred_df: np.expm1(model.predict(pred_df[spec.numeric]))
    if name == "HistGB_Poisson":
        model = HistGradientBoostingRegressor(loss="poisson", max_iter=500, learning_rate=0.03, max_depth=6,
                                              l2_regularization=5.0, random_state=spec.seed)
        model.fit(fit_df[spec.numeric], y_raw)
        return lambda pred_df: model.predict(pred_df[spec.numeric])
    if name in {"LightGBM_MAE", "LightGBM_MAE_Exog"}:
        feats = _features_for_model(name, spec)
        model = LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.03, num_leaves=31,
                              max_depth=-1, min_child_samples=20, reg_lambda=5.0, random_state=spec.seed,
                              n_jobs=spec.n_threads, verbosity=-1)
        model.fit(fit_df[feats], y_log)

        def prever(pred_df):
            with np.errstate(over="ignore"):
                return np.expm1(model.predict(pred_df[feats]))
        prever.modelo, prever.atributos = model, list(feats)          # para importância dos atributos (notebook 12)
        return prever
    if name in {"CatBoost_MAE", "CatBoost_MAE_Exog"}:
        feats = _features_for_model(name, spec)
        model = _catboost("MAE", spec)
        model.fit(_xcat(fit_df, spec, feats), y_log, cat_features=spec.cat)

        def prever(pred_df):
            return np.expm1(model.predict(_xcat(pred_df, spec, feats)))
        prever.modelo, prever.atributos = model, list(feats) + list(spec.cat)
        return prever
    if name == "CatBoost_MAE_raw":
        model = _catboost("MAE", spec)
        model.fit(_xcat(fit_df, spec), y_raw, cat_features=spec.cat)
        return lambda pred_df: model.predict(_xcat(pred_df, spec))
    if name == "CatBoost_Tweedie":
        model = _catboost("Tweedie:variance_power=1.5", spec)
        model.fit(_xcat(fit_df, spec), y_raw, cat_features=spec.cat)
        return lambda pred_df: model.predict(_xcat(pred_df, spec), prediction_type="Exponent")
    raise ValueError(name)


def fit_predict(name, fit_df, pred_df, spec: Especificacao):
    """Treina e devolve previsões brutas (mesmo comportamento do v5)."""
    return ajustar(name, fit_df, spec)(pred_df)


def check_and_clip(raw):
    raw = np.asarray(raw, dtype=float)
    finite = np.isfinite(raw)
    info = {"n_nao_finitas": int((~finite).sum()), "n_negativas_brutas": int((raw[finite] < 0).sum())}
    return np.where(finite, np.clip(raw, 0, None), raw), info
