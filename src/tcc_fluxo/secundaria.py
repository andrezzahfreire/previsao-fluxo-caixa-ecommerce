"""Análise secundária do v5: ARIMA/SARIMA(X) por vendedor e agregado, Prophet(X) agregado e LSTM.

Não participa da seleção. Mesmos alvo, horizonte (h = 1), corte e métricas do experimento principal.
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd

from .inferencia import block_bootstrap_mean, dm_hln
from .metricas import evaluate_seller_level, summarize_sellers
from .modelos import check_and_clip
from .painel import TARGET


# ----------------------------------------------------------------------------- exógenas
def preparar_exogenas(exog_weekly, macro_features, cal_cols, panel_start, train_target_end, test_end):
    """Macro padronizada com média/desvio APENAS das semanas-alvo do treino."""
    exw = exog_weekly.set_index("target_week").sort_index()
    tr = (exw.index >= panel_start) & (exw.index <= train_target_end)
    te = (exw.index > train_target_end) & (exw.index <= test_end)
    cols = list(macro_features) + list(cal_cols)
    mean = exw.loc[tr, macro_features].mean()
    std = exw.loc[tr, macro_features].std(ddof=0).replace(0, 1.0)
    sec = exw[cols].astype(float).copy()
    sec[macro_features] = (sec[macro_features] - mean) / std
    rng = pd.DataFrame({"treino_min": exw.loc[tr, macro_features].min(), "treino_max": exw.loc[tr, macro_features].max(),
                        "teste_min": exw.loc[te, macro_features].min(), "teste_max": exw.loc[te, macro_features].max()})
    rng["teste_fora_da_faixa_treino"] = (rng["teste_min"] < rng["treino_min"]) | (rng["teste_max"] > rng["treino_max"])
    return sec, cols, rng


# ------------------------------------------------------------------ ARIMA por vendedor
def _fit(y, order, seasonal_order=(0, 0, 0, 0), exog=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend="c").fit(disp=False, maxiter=200)
        if np.isfinite(res.aic) and np.isfinite(res.params).all():
            return res
    except Exception:
        pass
    return None


def _one_step(res, y_full, n_tr, exog_full=None):
    """Previsões de 1 passo nas semanas de teste com os parâmetros do treino (filtro de Kalman, sem reestimar)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_full = res.apply(y_full, exog=exog_full) if exog_full is not None else res.apply(y_full)
            pred = np.asarray(res_full.get_prediction(start=n_tr, end=len(y_full) - 1).predicted_mean, float)
        return pred if np.isfinite(pred).all() else None
    except Exception:
        return None


def _n_params(order, seasonal, n_exog):
    return order[0] + order[2] + seasonal[0] + seasonal[2] + n_exog + 2


def _fit_with_exog(y_tr, y_full, X_tr, X_full, order, seasonal, par):
    n_tr = len(y_tr)
    for label, cols in [("macro+calendario", par["exog_cols"]), ("so_calendario", par["exog_cal"])]:
        cols = [c for c in cols if X_tr[c].std(ddof=0) > 0]
        if not cols or n_tr < par["min_obs_per_param"] * _n_params(order, seasonal, len(cols)):
            continue
        res = _fit(y_tr, order, seasonal, X_tr[cols].to_numpy(float))
        if res is None:
            continue
        pred = _one_step(res, y_full, n_tr, X_full[cols].to_numpy(float))
        if pred is not None:
            return f"{label} ({len(cols)} exog)", pred
    return None


def secondary_seller(sid, y_tr, y_full, X_tr, X_full, par):
    n_tr, n_te = len(y_tr), len(y_full) - len(y_tr)
    out = {}
    best = None
    for order in par["arima_orders"]:
        res = _fit(y_tr, order)
        if res is not None and (best is None or res.aic < best[1].aic):
            best = (order, res)
    pred_arima = _one_step(best[1], y_full, n_tr) if best is not None else None
    if pred_arima is None:
        order = (1, 0, 0)
        out["ARIMA_vendedor"] = ("falhou->media", np.full(n_te, float(np.mean(y_tr))))
    else:
        order = best[0]
        out["ARIMA_vendedor"] = (str(order), pred_arima)

    s = par["seasonal_period"]
    best_s = None
    for P, D, Q in par["seasonal_candidates"]:
        res = _fit(y_tr, order, (P, D, Q, s))
        if res is not None and (best_s is None or res.aic < best_s[1].aic):
            best_s = ((P, D, Q, s), res)
    pred_sarima = _one_step(best_s[1], y_full, n_tr) if best_s is not None else None
    if pred_sarima is None:
        seasonal = (0, 0, 0, 0)
        out["SARIMA_vendedor"] = ("falhou->ARIMA", out["ARIMA_vendedor"][1])
    else:
        seasonal = best_s[0]
        out["SARIMA_vendedor"] = (f"{order}x{seasonal}", pred_sarima)

    r = _fit_with_exog(y_tr, y_full, X_tr, X_full, order, (0, 0, 0, 0), par)
    out["ARIMAX_vendedor"] = (f"{order} {r[0]}", r[1]) if r else ("univariado (fallback)", out["ARIMA_vendedor"][1])
    if seasonal[3] > 0:
        r = _fit_with_exog(y_tr, y_full, X_tr, X_full, order, seasonal, par)
        out["SARIMAX_vendedor"] = ((f"{order}x{seasonal} {r[0]}", r[1]) if r
                                   else ("univariado (fallback)", out["SARIMA_vendedor"][1]))
    else:
        out["SARIMAX_vendedor"] = ("SARIMA falhou->ARIMAX", out["ARIMAX_vendedor"][1])
    return sid, out


SEC_TS_SELLER_MODELS = ["ARIMA_vendedor", "SARIMA_vendedor", "ARIMAX_vendedor", "SARIMAX_vendedor"]


def parametros(C, exog_cols):
    return {"arima_orders": C.ARIMA_SELLER_ORDERS, "seasonal_period": C.SARIMA_SEASONAL_PERIOD,
            "seasonal_candidates": C.SARIMA_SEASONAL_CANDIDATES, "exog_cols": exog_cols,
            "exog_cal": list(C.SEC_EXOG_CAL), "min_obs_per_param": C.SEC_MIN_OBS_PER_PARAM}


def rodar_arima_vendedores(serie_completa, sec_exog, test_weeks_sorted, train_target_end, par, n_jobs):
    from joblib import Parallel, delayed
    t0 = time.time()
    jobs, sec_weeks = [], {}
    for sid, g in serie_completa.sort_values(["seller_id", "week_start"]).groupby("seller_id", sort=False):
        s_full = g.set_index("week_start")["revenue"]
        s_train = s_full.loc[:train_target_end]
        X_full = sec_exog.loc[s_full.index]
        sec_weeks[sid] = s_full.index[len(s_train):]
        jobs.append(delayed(secondary_seller)(sid, s_train.to_numpy(float), s_full.to_numpy(float),
                                              X_full.iloc[:len(s_train)], X_full, par))
    results = Parallel(n_jobs=n_jobs)(jobs)
    print(f"ARIMA/SARIMA/ARIMAX/SARIMAX por vendedor: {len(results)} séries em {time.time() - t0:.0f}s")
    frames, specs = [], []
    for sid, out in results:
        weeks = sec_weeks[sid]
        assert np.array_equal(weeks.to_numpy(), test_weeks_sorted), sid
        fr = pd.DataFrame({"seller_id": sid, "target_week": weeks})
        for m in SEC_TS_SELLER_MODELS:
            fr[f"pred_{m}"] = out[m][1]
            specs.append({"seller_id": sid, "modelo": m, "especificacao": out[m][0]})
        frames.append(fr)
    pred = pd.concat(frames, ignore_index=True)
    spec_df = pd.DataFrame(specs)
    tipo = (spec_df["especificacao"]
            .str.replace(r"\(\d+, \d+, \d+\)x\(\d+, \d+, \d+, \d+\)|\(\d+, \d+, \d+\)", "", regex=True)
            .str.strip().replace("", "estimado (univariado)"))
    resumo = spec_df.assign(tipo=tipo).groupby(["modelo", "tipo"]).size().rename("vendedores").reset_index()
    return pred, spec_df, resumo


# ------------------------------------------------------------------------- agregado
def rodar_agregado(serie_completa, sec_exog, exog_cols, test_weeks, train_target_end, C):
    """Reestimado a cada semana (janela expansiva); previsão de um passo."""
    from prophet import Prophet
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore", message="IProgress not found")
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).setLevel(logging.ERROR)
        logging.getLogger(lg).disabled = True

    agg_series = serie_completa.groupby("week_start")["revenue"].sum()
    agg_train = agg_series.loc[:train_target_end]
    X_agg = sec_exog.loc[agg_series.index]
    agg_exog = [c for c in exog_cols if X_agg.loc[agg_train.index, c].std(ddof=0) > 0]

    def _aic(order, seasonal=(0, 0, 0, 0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return SARIMAX(agg_train.to_numpy(float), order=order, seasonal_order=seasonal, trend="c").fit(disp=False).aic
            except Exception:
                return np.inf

    aic = {order: _aic(order) for order in C.ARIMA_AGG_ORDERS}
    agg_order = min(aic, key=aic.get)
    sp = C.SARIMA_SEASONAL_PERIOD
    aic_s = {(P, D, Q, sp): _aic(agg_order, (P, D, Q, sp)) for P, D, Q in C.SARIMA_SEASONAL_CANDIDATES}
    agg_seasonal = min(aic_s, key=aic_s.get)
    specs = {"ARIMA_agregado": (agg_order, (0, 0, 0, 0), False), "SARIMA_agregado": (agg_order, agg_seasonal, False),
             "ARIMAX_agregado": (agg_order, (0, 0, 0, 0), True), "SARIMAX_agregado": (agg_order, agg_seasonal, True)}
    preds = {m: [] for m in [*specs, "Prophet_agregado", "ProphetX_agregado"]}
    fails = {m: 0 for m in preds}
    t0 = time.time()
    for w in test_weeks:
        hist = agg_series.loc[:w - pd.Timedelta(weeks=1)]
        Xh = X_agg.loc[hist.index, agg_exog].to_numpy(float)
        Xw = X_agg.loc[[w], agg_exog].to_numpy(float)
        for name, (order, seasonal, use_x) in specs.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = SARIMAX(hist.to_numpy(float), exog=Xh if use_x else None, order=order,
                                seasonal_order=seasonal, trend="c").fit(disp=False)
                    p = float(r.forecast(1, exog=Xw if use_x else None)[0])
                if not np.isfinite(p):
                    raise ValueError
            except Exception:
                p = float(hist.iloc[-1])
                fails[name] += 1
            preds[name].append(p)
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(pd.DataFrame({"ds": hist.index, "y": hist.to_numpy(float)}))
        preds["Prophet_agregado"].append(float(m.predict(pd.DataFrame({"ds": [w]}))["yhat"].iloc[0]))
        mx = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        for c in agg_exog:
            mx.add_regressor(c)
        df_fit = pd.DataFrame({"ds": hist.index, "y": hist.to_numpy(float)})
        df_fit[agg_exog] = Xh
        mx.fit(df_fit)
        df_new = pd.DataFrame({"ds": [w]})
        df_new[agg_exog] = Xw
        preds["ProphetX_agregado"].append(float(mx.predict(df_new)["yhat"].iloc[0]))
    print(f"{len(preds)} modelos agregados reestimados em {len(test_weeks)} semanas: {time.time() - t0:.0f}s")
    info = {"exogenas_agregado": agg_exog, "aic_arima": {str(k): float(v) for k, v in aic.items()},
            "ordem": list(agg_order), "aic_sarima": {str(k): float(v) for k, v in aic_s.items()},
            "ordem_sazonal": list(agg_seasonal), "falhas": fails}
    out = pd.DataFrame(preds, index=pd.DatetimeIndex(test_weeks, name="target_week")).reset_index()
    return out, agg_series, info


# ------------------------------------------------------------------------------ LSTM
class JanelasLSTM:
    """Janelas de 8 semanas (t-7..t) de log1p(receita) e log1p(pedidos) por vendedor."""

    def __init__(self, serie_completa, seq_len, exog_sec, macro_features):
        p = serie_completa.sort_values(["seller_id", "week_start"]).reset_index(drop=True)
        g = p.groupby("seller_id", sort=False)
        self.rev_seq = np.stack([g["revenue"].shift(k).to_numpy(float) for k in range(seq_len - 1, -1, -1)], axis=1)
        self.ord_seq = np.stack([g["orders"].shift(k).to_numpy(float) for k in range(seq_len - 1, -1, -1)], axis=1)
        p["_row"] = np.arange(len(p))
        self.key = p.set_index(["seller_id", "week_start"])["_row"]
        self.exog_sec = exog_sec
        self.macro = list(macro_features)

    def _rows(self, df):
        return self.key.loc[list(zip(df["seller_id"], df["week_start"]))].to_numpy()

    def make_xy(self, df):
        rows = self._rows(df)
        X = np.stack([np.log1p(self.rev_seq[rows]), np.log1p(self.ord_seq[rows])], axis=2)
        stat = df[["target_week_sin", "target_week_cos"]].to_numpy(float)
        assert np.isfinite(X).all()
        return X.astype(np.float32), stat.astype(np.float32), df[TARGET].to_numpy(float)

    def make_xy_cfg(self, df, cfg):
        rows = self._rows(df)
        arrays = {"revenue": self.rev_seq, "orders": self.ord_seq}
        X = np.stack([np.log1p(arrays[c][rows]) for c in cfg["seq"]], axis=2)
        assert np.isfinite(X).all()
        if not cfg["static"]:
            S = np.zeros((len(df), 0), dtype=np.float32)
        else:
            parts = []
            for c in cfg["static"]:
                if c in self.macro:
                    parts.append(df["target_week"].map(self.exog_sec[c]).to_numpy(float))
                else:
                    parts.append(df[c].to_numpy(float))
            S = np.column_stack(parts)
            assert np.isfinite(S).all()
            S = S.astype(np.float32)
        return X.astype(np.float32), S, df[TARGET].to_numpy(float)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _net(n_seq, n_static, hidden=32):
    torch, nn = _torch()

    class LSTMNetCfg(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, s):
            _, (h, _) = self.lstm(x)
            return self.head(torch.cat([h[-1], s], dim=1)).squeeze(1)

    return LSTMNetCfg()


def treinar_lstm(janelas, cfg, df_fit, n_epochs, y_scale, seed, df_hold=None):
    """cfg=None reproduz a LSTM_global do v5 (receita+pedidos e seno/cosseno da semana-alvo)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    mk = janelas.make_xy if cfg is None else (lambda d: janelas.make_xy_cfg(d, cfg))
    Xf, Sf, yf = mk(df_fit)
    net = _net(Xf.shape[2], Sf.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)
    Xt, St, yt = torch.tensor(Xf), torch.tensor(Sf), torch.tensor(yf / y_scale, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    if df_hold is not None:
        Xh, Sh, yh = mk(df_hold)
        Xh, Sh, yh = torch.tensor(Xh), torch.tensor(Sh), torch.tensor(yh / y_scale, dtype=torch.float32)
    for _ in range(n_epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=gen)
        for i in range(0, len(yt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b], St[b]), yt[b])
            loss.backward()
            opt.step()
        if df_hold is not None:
            net.eval()
            with torch.no_grad():
                curve.append(float(loss_fn(net(Xh, Sh), yh)))
    return net, curve


def prever_lstm(janelas, cfg, net, df, y_scale):
    torch, _ = _torch()
    mk = janelas.make_xy if cfg is None else (lambda d: janelas.make_xy_cfg(d, cfg))
    X, S, _ = mk(df)
    net.eval()
    with torch.no_grad():
        return np.exp(net(torch.tensor(X), torch.tensor(S)).numpy()) * y_scale


def rodar_lstms(janelas, train, test, variants, C):
    """Épocas escolhidas num holdout interno do TREINO (últimas semanas), depois treino completo."""
    torch, _ = _torch()
    torch.set_num_threads(C.N_THREADS)
    y_scale = float(train[TARGET].mean())
    hold_start = C.TRAIN_TARGET_END - pd.Timedelta(weeks=C.LSTM_HOLDOUT_WEEKS - 1)
    fit_l = train.loc[train["target_week"] < hold_start]
    hold_l = train.loc[train["target_week"] >= hold_start]
    preds, info = {}, []
    for name, cfg in variants.items():
        t0 = time.time()
        _, curve = treinar_lstm(janelas, cfg, fit_l, C.LSTM_MAX_EPOCHS, y_scale, C.RANDOM_SEED, hold_l)
        n_ep = int(np.argmin(curve)) + 1
        net, _ = treinar_lstm(janelas, cfg, train, n_ep, y_scale, C.RANDOM_SEED)
        pred, ck = check_and_clip(prever_lstm(janelas, cfg, net, test, y_scale))
        preds[name] = pred
        info.append({"modelo": name, "n_seq": 2 if cfg is None else len(cfg["seq"]),
                     "n_estaticas": 2 if cfg is None else len(cfg["static"]), "epocas": n_ep,
                     "perda_holdout": min(curve), **ck, "segundos": round(time.time() - t0, 1)})
        print(f"{name}: {n_ep} épocas | {time.time() - t0:.0f}s")
    return preds, pd.DataFrame(info), y_scale


# --------------------------------------------------------------------- seasonal naive
def seasonal_naive(comparison, serie_completa, lag_semanas=52):
    """Prevê a semana-alvo com a receita da mesma semana do ano anterior (target_week − 52 semanas).

    Referência complementar, só no teste: as origens de validação de 2017 não têm 52 semanas de histórico.
    Sem a semana t−52 (vendedor ainda não estava no painel), usa o Naive. Devolve (previsões, máscara de disponibilidade).
    """
    receita = serie_completa.set_index(["seller_id", "week_start"])["revenue"]
    chaves = pd.MultiIndex.from_arrays([comparison["seller_id"].to_numpy(),
                                        (comparison["target_week"] - pd.Timedelta(weeks=lag_semanas)).to_numpy()])
    sazonal = receita.reindex(chaves).to_numpy(float)
    disponivel = np.isfinite(sazonal)
    return np.where(disponivel, sazonal, comparison["pred_Naive"].to_numpy(float)), disponivel


def avaliar_snaive(comparison, chosen, pred_snaive, disponivel, scale_final, agg_scale_final):
    """Tabela do seasonal naive: teste completo (com Naive onde falta t−52) e só linhas com t−52."""
    df = comparison[["seller_id", "target_week", TARGET, "pred_Naive", "pred_MediaMovel4", f"pred_{chosen}"]].copy()
    df["pred_SNaive_52"] = pred_snaive
    models = list(dict.fromkeys(["Naive", "SNaive_52", "MediaMovel4", chosen]))
    from .metricas import evaluate_aggregate
    linhas = []
    for recorte, d in [("teste completo (Naive onde falta t-52)", df), ("só linhas com t-52 disponível", df.loc[disponivel])]:
        pooled, per = evaluate_seller_level(d, models, scale_final)
        summ = summarize_sellers(per).set_index("modelo")
        agg, _ = evaluate_aggregate(d, models, agg_scale_final)
        agg = agg.set_index("modelo")
        for m in models:
            linhas.append({"recorte": recorte, "modelo": m, "linhas": len(d), "vendedores": d["seller_id"].nunique(),
                           "MAE": float(pooled.set_index("modelo").loc[m, "MAE"]),
                           "skill_vs_naive": float(pooled.set_index("modelo").loc[m, "skill_vs_naive"]),
                           "MASE_medio_vendedor": float(summ.loc[m, "MASE_media"]),
                           "prop_vendedores_melhora": float(summ.loc[m, "prop_vendedores_melhora"]),
                           "WAPE_agregado": float(agg.loc[m, "WAPE"]), "MASE_agregado": float(agg.loc[m, "MASE"])})
    cobertura = {"pct_linhas_com_t52": float(disponivel.mean()), "linhas_com_t52": int(disponivel.sum()),
                 "linhas_teste": int(len(disponivel)),
                 "vendedores_com_t52_em_todas_as_semanas": int(pd.Series(disponivel).groupby(
                     comparison["seller_id"].to_numpy()).all().sum()),
                 "vendedores": int(comparison["seller_id"].nunique())}
    return pd.DataFrame(linhas), cobertura


# ------------------------------------------------------------------------- avaliação
SEC_FAMILY_BASE = {
    "Naive": ("baseline", "-"),
    "SNaive_52": ("baseline sazonal", "-"),
    "ARIMA_vendedor": ("ARIMA", "univariada"), "ARIMAX_vendedor": ("ARIMA", "multivariada"),
    "SARIMA_vendedor": ("SARIMA", "univariada"), "SARIMAX_vendedor": ("SARIMA", "multivariada"),
    "LSTM_uni": ("LSTM", "univariada"), "LSTM_global": ("LSTM", "intermediária"),
    "LSTM_multi": ("LSTM", "multivariada"),
    "ARIMA_agregado": ("ARIMA", "univariada"), "ARIMAX_agregado": ("ARIMA", "multivariada"),
    "SARIMA_agregado": ("SARIMA", "univariada"), "SARIMAX_agregado": ("SARIMA", "multivariada"),
    "Prophet_agregado": ("Prophet", "univariada"), "ProphetX_agregado": ("Prophet", "multivariada"),
}


def familias(chosen):
    fam = dict(SEC_FAMILY_BASE)
    fam[chosen] = ("principal (escolhido)", "multivariada" if chosen.endswith("_Exog") else "-")
    return fam


def avaliar(comparison, chosen, ts_seller_pred, lstm_preds, agg_preds, aggregate, scale_final, agg_scale_final,
            extras=None):
    """extras: {nome: previsões alinhadas com comparison} — referências complementares, como o SNaive_52."""
    extras = extras or {}
    fam = familias(chosen)
    models = list(dict.fromkeys(["Naive", *extras, chosen, *SEC_TS_SELLER_MODELS, "LSTM_uni", "LSTM_global",
                                 "LSTM_multi"]))
    sec = comparison[["seller_id", "week_start", "target_week", TARGET, "pred_Naive", f"pred_{chosen}"]].copy()
    for nome, pred in extras.items():
        sec[f"pred_{nome}"] = np.asarray(pred, float)
    sec = sec.merge(ts_seller_pred, on=["seller_id", "target_week"], how="left", validate="one_to_one")
    clip_rows = []
    for m in SEC_TS_SELLER_MODELS:
        sec[f"pred_{m}"], ck = check_and_clip(sec[f"pred_{m}"])
        clip_rows.append({"modelo": m, **ck})
    assert (lstm_preds["seller_id"].to_numpy() == sec["seller_id"].to_numpy()).all()
    assert (lstm_preds["target_week"].to_numpy() == sec["target_week"].to_numpy()).all()
    for m in ["LSTM_global", "LSTM_uni", "LSTM_multi"]:
        sec[f"pred_{m}"] = lstm_preds[f"pred_{m}"].to_numpy()
    assert sec[[f"pred_{m}" for m in models]].notna().all().all()

    pooled, per_seller = evaluate_seller_level(sec, models, scale_final)
    summ = summarize_sellers(per_seller)
    sec_seller = pooled.merge(summ[["modelo", "MASE_media", "MASE_mediana", "skill_vs_naive_mediana",
                                    "prop_vendedores_melhora"]], on="modelo")
    sec_seller.insert(1, "familia", [fam[m][0] for m in sec_seller["modelo"]])
    sec_seller.insert(2, "tipo", [fam[m][1] for m in sec_seller["modelo"]])
    sec_seller.insert(1, "experimento", ["baseline" if m in ("Naive", *extras) else
                                         ("principal (escolhido)" if m == chosen else "secundário")
                                         for m in sec_seller["modelo"]])

    agg_cols = {}
    for m in models:
        agg_cols[f"{m} (soma por vendedor)"] = sec.groupby("target_week")[f"pred_{m}"].sum()
    ap = agg_preds.set_index("target_week")
    for m in ap.columns:
        agg_cols[m] = pd.Series(np.clip(ap[m].to_numpy(float), 0, None), index=aggregate.index)
    sec_agg = pd.DataFrame(agg_cols)
    sec_agg[TARGET] = aggregate[TARGET]
    y_a = sec_agg[TARGET].to_numpy(float)
    base_a = float(np.mean(np.abs(y_a - sec_agg["Naive (soma por vendedor)"].to_numpy(float))))
    rows = []
    for c in agg_cols:
        p = sec_agg[c].to_numpy(float)
        err = float(np.mean(np.abs(y_a - p)))
        num_s, den_s = 2 * np.abs(y_a - p), np.abs(y_a) + np.abs(p)
        base_name = c.replace(" (soma por vendedor)", "")
        rows.append({"modelo": c, "familia": fam[base_name][0], "tipo": fam[base_name][1],
                     "origem": "soma por vendedor" if c.endswith("(soma por vendedor)") else "série agregada",
                     "MAE": err, "RMSE": float(np.sqrt(np.mean((y_a - p) ** 2))),
                     "WAPE": float(np.abs(y_a - p).sum() / y_a.sum()) if y_a.sum() > 0 else np.nan,
                     "sMAPE": float(np.divide(num_s, den_s, out=np.zeros_like(num_s), where=den_s > 0).mean() * 100),
                     "MASE": err / agg_scale_final, "skill_vs_naive": float(1 - err / base_a) if base_a > 0 else np.nan})
    return {"sec": sec, "models": models, "pooled": pooled, "per_seller": per_seller, "vendedor": sec_seller,
            "sec_agg": sec_agg, "agregado": pd.DataFrame(rows), "clip": pd.DataFrame(clip_rows)}


SELLER_PAIRS = [("ARIMA", "ARIMA_vendedor", "ARIMAX_vendedor"), ("SARIMA", "SARIMA_vendedor", "SARIMAX_vendedor"),
                ("LSTM", "LSTM_uni", "LSTM_multi")]
AGG_PAIRS = ([("ARIMA", "ARIMA_agregado", "ARIMAX_agregado"), ("SARIMA", "SARIMA_agregado", "SARIMAX_agregado"),
              ("Prophet", "Prophet_agregado", "ProphetX_agregado")]
             + [(f, f"{u} (soma por vendedor)", f"{m} (soma por vendedor)") for f, u, m in SELLER_PAIRS])


def univariada_vs_multivariada(res, n_boot, seed, min_weeks):
    """d = erro(multivariada) − erro(univariada); d < 0 favorece a multivariada."""
    sec, sec_agg, sec_pooled, per_seller = res["sec"], res["sec_agg"], res["pooled"], res["per_seller"]
    rng_um = np.random.default_rng(seed)
    rng_dm = np.random.default_rng(seed)
    mase_wide = per_seller.pivot(index="seller_id", columns="modelo", values="MASE").replace([np.inf, -np.inf], np.nan)
    L = (pd.DataFrame({m: np.abs(sec[TARGET] - sec[f"pred_{m}"]) for m in res["models"]})
         .groupby(sec["target_week"].to_numpy()).mean())

    def _dm_row(d):
        d = np.asarray(d, float)
        if not np.isfinite(d).all() or len(d) < min_weeks:
            return {"dm_hln": np.nan, "p_valor_hln": np.nan, "ic95_media_d_inf": np.nan,
                    "ic95_media_d_sup": np.nan, "p_valor_bootstrap_blocos": np.nan}
        stat, p = dm_hln(d)
        _, lo, hi, p_b = block_bootstrap_mean(d, n_boot, rng_dm)
        return {"dm_hln": stat, "p_valor_hln": p, "ic95_media_d_inf": lo, "ic95_media_d_sup": hi,
                "p_valor_bootstrap_blocos": p_b}

    def _leitura(lo, hi):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "não avaliado"
        if hi < 0:
            return "multivariada erra menos (IC95 < 0)"
        if lo > 0:
            return "multivariada erra mais (IC95 > 0)"
        return "sem diferença detectável (IC95 contém 0)"

    rows = []
    for fam, u, m in SELLER_PAIRS:
        d_s = (mase_wide[m] - mase_wide[u]).dropna()
        boot = np.array([d_s.to_numpy()[rng_um.integers(0, len(d_s), len(d_s))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mae_u = float(sec_pooled.set_index("modelo").loc[u, "MAE"])
        mae_m = float(sec_pooled.set_index("modelo").loc[m, "MAE"])
        row = {"nivel": "vendedor", "familia": fam, "univariada": u, "multivariada": m, "n": len(d_s),
               "metrica": "MASE médio por vendedor", "univ": mase_wide[u].mean(), "multi": mase_wide[m].mean(),
               "d_media": d_s.mean(), "d_mediana": d_s.median(), "ic95_inf": lo, "ic95_sup": hi,
               "prop_vendedores_multi_melhor": float((d_s < 0).mean()),
               "MAE_rel_multi_vs_univ": mae_m / mae_u - 1 if mae_u > 0 else np.nan, "leitura": _leitura(lo, hi)}
        row.update(_dm_row((L[m] - L[u]).to_numpy()))
        rows.append(row)
    for fam, u, m in AGG_PAIRS:
        e_u = np.abs(sec_agg[TARGET] - sec_agg[u]).to_numpy(float)
        e_m = np.abs(sec_agg[TARGET] - sec_agg[m]).to_numpy(float)
        d = e_m - e_u
        dm_part = _dm_row(d)
        row = {"nivel": "agregado", "familia": fam, "univariada": u, "multivariada": m, "n": len(d),
               "metrica": "erro absoluto do total semanal (R$)", "univ": e_u.mean(), "multi": e_m.mean(),
               "d_media": d.mean(), "d_mediana": float(np.median(d)),
               "ic95_inf": dm_part["ic95_media_d_inf"], "ic95_sup": dm_part["ic95_media_d_sup"],
               "prop_vendedores_multi_melhor": np.nan,
               "MAE_rel_multi_vs_univ": e_m.mean() / e_u.mean() - 1 if e_u.mean() > 0 else np.nan,
               "leitura": _leitura(dm_part["ic95_media_d_inf"], dm_part["ic95_media_d_sup"])}
        row.update(dm_part)
        rows.append(row)
    return pd.DataFrame(rows)
