"""Seção 24 do v5: Experimento 1 — atributos de ocorrência e intensidade (ablação no CatBoost_MAE).

Somente atributos mudam; perda, hiperparâmetros, alvo, corte e origens são os do CatBoost_MAE principal.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .metricas import seller_scales
from .modelos import check_and_clip
from .painel import TARGET

PROFILES = ["intermitente", "medio", "regular"]


# ----------------------------------------------------------------------------- atributos
def _win(a, t, w):
    return a[max(0, t - w + 1): t + 1]


def _prev_win(a, t, w):
    start, end = t - 2 * w + 1, t - w
    return None if start < 0 else a[start: end + 1]


def _slope(y):
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float) - (len(y) - 1) / 2
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _pmean(x):
    p = x[x > 0]
    return p.mean() if len(p) else np.nan


def seller_extra_features(r, o):
    """r, o: receitas e pedidos semanais em ordem temporal. A posição t usa só r[:t+1], o[:t+1]."""
    n = len(r)
    f = {}

    def put(name, t, v):
        if name not in f:
            f[name] = np.full(n, np.nan)
        f[name][t] = v

    last_sale = -1
    for t in range(n):
        h = r[: t + 1]
        if r[t] > 0:
            last_sale = t
        hp = h[h > 0]
        thr_t = np.percentile(hp, 90) if len(hp) >= 3 else np.nan

        # A. ocorrência
        put("weeks_since_last_sale", t, t - last_sale if last_sale >= 0 else t + 1)
        z = 0
        while z <= t and r[t - z] == 0:
            z += 1
        put("zero_streak", t, z)
        put("sold_current", t, float(r[t] > 0))
        for w in (4, 8, 13):
            x = _win(r, t, w)
            npos = int((x > 0).sum())
            put(f"n_pos_{w}", t, npos)
            put(f"freq_{w}", t, npos / len(x))
            put(f"sold_any_{w}", t, float(npos > 0))
        p4 = _prev_win(r, t, 4)
        if p4 is not None:
            put("freq_change_4", t, (_win(r, t, 4) > 0).mean() - (p4 > 0).mean())

        # B. intensidade
        for w in (4, 8, 13):
            p = _win(r, t, w)
            p = p[p > 0]
            put(f"pos_sum_{w}", t, p.sum())
            if len(p):
                put(f"pos_mean_{w}", t, p.mean())
                put(f"pos_median_{w}", t, np.median(p))
            if w in (8, 13) and len(p) >= 2:
                put(f"pos_std_{w}", t, p.std(ddof=1))
        if last_sale >= 0:
            put("last_pos_revenue", t, r[last_sale])
        put("has_orders_current", t, float(o[t] > 0))
        if o[t] > 0:
            put("rev_per_order_current", t, r[t] / o[t])
        r13, o13 = _win(r, t, 13), _win(o, t, 13)
        mo = o13 > 0
        if mo.any():
            rpo = r13[mo] / o13[mo]
            put("rpo_mean_13", t, rpo.mean())
            put("rpo_median_13", t, np.median(rpo))

        # C. picos
        for w in (4, 8, 13):
            x = _win(r, t, w)
            put(f"max_{w}", t, x.max())
            if np.isfinite(thr_t):
                put(f"peak_flag_{w}", t, float((x > thr_t).any()))
        for w in (8, 13):
            p = _win(r, t, w)
            p = p[p > 0]
            if len(p) >= 3:
                put(f"pos_p90_{w}", t, np.percentile(p, 90))
        if np.isfinite(thr_t):
            put("pos_p90_hist", t, thr_t)
            idx = np.flatnonzero(h > thr_t)
            if len(idx):
                put("weeks_since_peak", t, t - idx[-1])
        put("max_hist", t, h.max())
        pm13 = _pmean(r13)
        if np.isfinite(pm13):
            put("max_to_posmean_13", t, r13.max() / pm13)

        # D. tendência
        w4 = _win(r, t, 4)
        if p4 is not None:
            put("mean_diff_4", t, w4.mean() - p4.mean())
            put("prev4_zero", t, float(p4.sum() == 0))
            if p4.mean() > 0:
                put("mean_ratio_4", t, w4.mean() / p4.mean())
            o4p = _prev_win(o, t, 4)
            put("orders_change_4", t, _win(o, t, 4).sum() - o4p.sum())
        for w in (4, 8, 13):
            put(f"slope_{w}", t, _slope(_win(r, t, w)))
        p8 = _prev_win(r, t, 8)
        if p8 is not None:
            w8 = _win(r, t, 8)
            put("freq_change_8", t, (w8 > 0).mean() - (p8 > 0).mean())
            a, b = _pmean(w8), _pmean(p8)
            if np.isfinite(a) and np.isfinite(b):
                put("pos_mean_change_8", t, a - b)

        # pedidos
        for w in (4, 8, 13):
            x = _win(o, t, w)
            put(f"orders_sum_{w}", t, x.sum())
            put(f"orders_mean_{w}", t, x.mean())
            put(f"max_orders_{w}", t, x.max())

        # E. escala
        put("rev_mean_hist", t, h.mean())
        put("freq_hist", t, (h > 0).mean())
        if len(hp):
            put("pos_mean_hist", t, hp.mean())
            put("pos_median_hist", t, np.median(hp))
        if t >= 1:
            put("mase_scale_hist", t, np.abs(np.diff(h)).mean())
    return f


def build_extra_features(base):
    parts = []
    for sid, g in base.sort_values(["seller_id", "week_start"]).groupby("seller_id", sort=False):
        df = pd.DataFrame(seller_extra_features(g["revenue"].to_numpy(float), g["orders"].to_numpy(float)))
        df.insert(0, "week_start", g["week_start"].to_numpy())
        df.insert(0, "seller_id", sid)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


EXP1_BLOCKS = {
    "ocorrencia": ["weeks_since_last_sale", "sold_current", "sold_any_4", "sold_any_8", "sold_any_13",
                   "n_pos_4", "n_pos_8", "n_pos_13", "freq_4", "freq_8", "freq_13", "freq_change_4"],
    "intensidade": ["pos_mean_4", "pos_mean_8", "pos_mean_13", "pos_median_4", "pos_median_8", "pos_median_13",
                    "pos_std_8", "pos_std_13", "pos_sum_4", "pos_sum_8", "pos_sum_13", "last_pos_revenue",
                    "rev_per_order_current", "has_orders_current", "rpo_mean_13", "rpo_median_13"],
    "picos": ["max_4", "max_8", "max_13", "pos_p90_8", "pos_p90_13", "pos_p90_hist", "max_hist",
              "max_to_posmean_13", "peak_flag_4", "peak_flag_8", "peak_flag_13", "weeks_since_peak"],
    "tendencia": ["mean_diff_4", "mean_ratio_4", "prev4_zero", "slope_4", "slope_8", "slope_13",
                  "freq_change_8", "pos_mean_change_8", "orders_change_4"],
    "pedidos": ["orders_sum_4", "orders_sum_8", "orders_sum_13", "orders_mean_4", "orders_mean_8",
                "orders_mean_13", "max_orders_4", "max_orders_8", "max_orders_13"],
    "escala": ["rev_mean_hist", "pos_mean_hist", "pos_median_hist", "mase_scale_hist", "freq_hist"],
}
EXP1_ALL = [c for cols in EXP1_BLOCKS.values() for c in cols]
assert len(EXP1_ALL) == len(set(EXP1_ALL))

EXACT_REDUNDANT = {
    "has_orders_current": "idêntico a (orders_current > 0) e a (revenue_current > 0)",
    "pos_sum_4": "= revenue_current + rev_lag_1 + rev_lag_2 + rev_lag_3",
    "max_4": "= max(revenue_current, rev_lag_1, rev_lag_2, rev_lag_3)",
}


def verificar_atributos(extra, extra_pert, cut, hist):
    """Sem vazamento + redundâncias declaradas + exemplo conceitual do enunciado."""
    past = extra["week_start"] <= cut
    cols = ["seller_id", "week_start"] + EXP1_ALL
    pd.testing.assert_frame_equal(extra.loc[past, cols].reset_index(drop=True),
                                  extra_pert.loc[past, cols].reset_index(drop=True))
    changed = (extra.loc[~past, EXP1_ALL].fillna(-1) != extra_pert.loc[~past, EXP1_ALL].fillna(-1)).any().mean()
    assert np.array_equal(extra["weeks_since_last_sale"].to_numpy(), extra["zero_streak"].to_numpy())
    assert ((hist["orders"] > 0) == (hist["revenue"] > 0)).all()
    ex = seller_extra_features(np.array([0, 0, 120, 0, 80, 0, 0, 50.0]), np.array([0, 0, 1, 0, 1, 0, 0, 1.0]))
    check = {"n_pos_8": ex["n_pos_8"][-1], "freq_8": ex["freq_8"][-1], "pos_mean_8": ex["pos_mean_8"][-1],
             "pos_median_8": ex["pos_median_8"][-1], "max_8": ex["max_8"][-1],
             "weeks_since_last_sale": ex["weeks_since_last_sale"][-1]}
    assert check == {"n_pos_8": 3, "freq_8": 0.375, "pos_mean_8": 250 / 3, "pos_median_8": 80, "max_8": 120,
                     "weeks_since_last_sale": 0}
    return float(changed), check


def reduce_block(candidates, label, corr_df, numeric_features, corr_max):
    kept, log = [], []
    for c in candidates:
        if c in EXACT_REDUNDANT:
            log.append({"conjunto": label, "atributo": c, "decisao": "removido", "motivo": EXACT_REDUNDANT[c]})
            continue
        ref = numeric_features + kept
        rho = corr_df[ref + [c]].corr(method="spearman")[c].drop(c).abs()
        if rho.max() >= corr_max:
            log.append({"conjunto": label, "atributo": c, "decisao": "removido",
                        "motivo": f"|Spearman| = {rho.max():.3f} com {rho.idxmax()}"})
        else:
            kept.append(c)
            log.append({"conjunto": label, "atributo": c, "decisao": "mantido",
                        "motivo": f"maior |Spearman| = {rho.max():.3f} ({rho.idxmax()})"})
    return kept, log


def profile_labels_at(hist, origin):
    """Perfil estritamente histórico até a origem (tercis da frequência de semanas com venda)."""
    h = hist.loc[hist["week_start"] <= origin]
    g = h.groupby("seller_id")["revenue"].agg(semanas="size", positivas=lambda s: int((s > 0).sum()))
    g["freq"] = g["positivas"] / g["semanas"]
    g["perfil"] = pd.qcut(g["freq"].rank(method="first"), 3, labels=PROFILES).astype(str)
    return g


def calibrar_limites(train, hist, C):
    """Guardrails monetários: só o erro do Naive com target_week <= CV_FIRST_ORIGIN."""
    profile_cal = profile_labels_at(hist, C.CV_FIRST_ORIGIN)
    cal = train.loc[train["target_week"] <= C.CV_FIRST_ORIGIN].copy()
    cal["perfil"] = cal["seller_id"].map(profile_cal["perfil"])
    assert cal["perfil"].notna().all()
    rows, limits = [], {}
    for perfil in PROFILES:
        g = cal.loc[cal["perfil"] == perfil]
        y = g[TARGET].to_numpy(float)
        p = g["revenue_current"].to_numpy(float)
        e = p - y
        ae = np.abs(e)
        pos = y > 0
        nm = {"MAE": float(ae.mean()), "MAE_positiva": float(ae[pos].mean()) if pos.any() else 0.0,
              "P90": float(np.quantile(ae, 0.90)), "P95": float(np.quantile(ae, 0.95)), "vies": float(e.mean()),
              "real_medio": float(y.mean()), "n_linhas": int(len(g)), "n_vendedores": int(g["seller_id"].nunique())}
        k = C.CALIBRATION_ERROR_MULT
        lim = {"MAE": k * nm["MAE"], "MAE_positiva": k * nm["MAE_positiva"], "P90": k * nm["P90"],
               "P95": k * nm["P95"],
               "vies_abs_max": k * max(abs(nm["vies"]), C.CALIBRATION_BIAS_FLOOR_SHARE * nm["real_medio"])}
        limits[perfil] = {kk: float(v) for kk, v in lim.items()}
        rows.append({"perfil": perfil, **nm, **{f"limite_{kk}": v for kk, v in lim.items()}})
    return limits, pd.DataFrame(rows)


def rodadas(int_red, pic_red, intpic_red, exog_features):
    return {
        "base": [], "+intensidade": EXP1_BLOCKS["intensidade"], "+picos": EXP1_BLOCKS["picos"],
        "+todos": EXP1_ALL, "+pedidos": EXP1_BLOCKS["pedidos"], "+intensidade_red": int_red,
        "+picos_red": pic_red, "+intensidade_picos_red": intpic_red,
        "+macro_calendario": list(exog_features), "+todos_macro_calendario": EXP1_ALL + list(exog_features),
    }


def congelar_decisoes(limits, rounds, cv_origins, C):
    frozen = {
        "registrado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "momento": "fórmula congelada antes da validação; números calculados somente com target_week <= CV_FIRST_ORIGIN",
        "origem_dos_limites": "erro histórico do Naive no período de calibração pré-CV; nenhum dado do teste de 2018",
        "calibracao": {"modelo": "Naive", "cutoff": str(C.CV_FIRST_ORIGIN.date()),
                       "multiplicador_erros": C.CALIBRATION_ERROR_MULT,
                       "piso_bias_como_fracao_da_receita_media": C.CALIBRATION_BIAS_FLOOR_SHARE},
        "unidade": "R$ por vendedor-semana (proxy de entrada bruta de caixa)",
        "limites": limits,
        "viés": "bilateral: abs(vies) <= vies_abs_max",
        "conformidade_individual": {"min_percentual_vendedores": C.MIN_SELLER_COMPLIANCE,
                                    "min_origens": C.STAB_MIN_ORIGINS, "metricas": ["MAE", "MAE_positiva", "abs(vies)"]},
        "perfis_validacao": "tercis da frequência de semanas com venda até a origem",
        "estabilidade_min_origens": C.STAB_MIN_ORIGINS,
        "criterios_globais": {"MASE_tol": C.G_MASE_TOL, "falso_sinal_pp": C.G_FALSE_SIGNAL_PP,
                              "WAPE_agregado_tol": C.G_WAPE_TOL, "min_origens": C.G_MIN_ORIGINS,
                              "min_share_semanas_vendedores": C.G_MIN_SHARE},
        "criterios_picos": {"delta_min_previsto_real": C.P_MIN_DELTA, "min_origens": C.P_MIN_ORIGINS,
                            "min_share_regulares": C.P_MIN_REG_SHARE, "ic95_inferior_maior_que_zero": True},
        "desempate": f"menor MASE médio por vendedor; empate prático <= {C.PRACTICAL_TIE_TOL:.0%} -> menos atributos",
        "sem_elegivel": "mantém o modelo principal da seção 16; Experimento 1 não escolhe modelo",
        "origens": [str(o.date()) for o in cv_origins],
        "candidatos": {k: v for k, v in rounds.items()},
    }
    return frozen, hash_decisoes(frozen)


def hash_decisoes(frozen):
    return hashlib.sha256(json.dumps({k: frozen[k] for k in frozen if k != "registrado_em"},
                                     sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- validação
def fit_predict_catboost_mae(fit_df, pred_df, num_features, spec, return_model=False):
    X_fit = fit_df[num_features + spec.cat].copy()
    X_pred = pred_df[num_features + spec.cat].copy()
    X_fit["seller_id"] = X_fit["seller_id"].astype(str)
    X_pred["seller_id"] = X_pred["seller_id"].astype(str)
    model = CatBoostRegressor(loss_function="MAE", iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5,
                              random_seed=spec.seed, verbose=False, thread_count=spec.n_threads,
                              allow_writing_files=False)
    model.fit(X_fit, np.log1p(fit_df[TARGET].clip(lower=0).to_numpy(float)), cat_features=spec.cat)
    pred, _ = check_and_clip(np.expm1(model.predict(X_pred)))
    return (pred, model) if return_model else pred


def block_metrics(y, p, sellers, scale_o, peak_thr, signal_thr):
    e = p - y
    ae = np.abs(e)
    zero, pos, pk = y == 0, y > 0, y > peak_thr
    s = pd.DataFrame({"s": sellers, "ae": ae}).groupby("s")["ae"].mean()
    mase = (s / s.index.map(scale_o)).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n_linhas": len(y), "n_vendedores": len(s),
        "MAE": ae.mean(), "MAE_positiva": ae[pos].mean() if pos.any() else np.nan,
        "MAE_zero": ae[zero].mean() if zero.any() else np.nan,
        "P90": np.quantile(ae, 0.90), "P95": np.quantile(ae, 0.95),
        "vies": e.mean(), "vies_pct": 100 * e.sum() / y.sum() if y.sum() > 0 else np.nan,
        "pct_falso_sinal": 100 * (p[zero] >= signal_thr).mean() if zero.any() else np.nan,
        "n_picos": int(pk.sum()),
        "cobertura_picos_pct": 100 * (p[pk] > peak_thr).mean() if pk.any() else np.nan,
        "picos_previsto_real": p[pk].sum() / y[pk].sum() if pk.any() else np.nan,
        "MASE_vend_media": mase.mean(), "MASE_vend_mediana": mase.median(),
        "previsto_medio": p.mean(), "real_medio": y.mean(),
        "pct_previsao_quase_zero": 100 * (p < signal_thr).mean(),
    }


def validar(train_x, hist, cv_origins, rounds, spec, C, verbose=True):
    """Em cada origem: escala MASE, limiares de pico/sinal e perfis só com dados até ela."""
    models, rows, preds = {}, [], []
    t0 = time.time()
    for origin in cv_origins:
        val_end = origin + pd.Timedelta(weeks=C.CV_VAL_WEEKS)
        fit = train_x.loc[train_x["target_week"] <= origin]
        val = train_x.loc[(train_x["target_week"] > origin) & (train_x["target_week"] <= val_end)].copy()
        assert val["target_week"].nunique() == C.CV_VAL_WEEKS
        scale_o = seller_scales(hist, origin)
        hist_pos = hist.loc[(hist["week_start"] <= origin) & (hist["revenue"] > 0), "revenue"]
        peak_o, signal_o = float(hist_pos.quantile(0.90)), float(hist_pos.quantile(0.10))
        prof_o = profile_labels_at(hist, origin)
        val["perfil"] = val["seller_id"].map(prof_o["perfil"])
        assert val["perfil"].notna().all(), "vendedor na validação sem histórico até a origem"
        y = val[TARGET].to_numpy(float)
        naive_ae = np.abs(val["revenue_current"].to_numpy(float) - y)
        for name, feats in rounds.items():
            keep = name == "+todos" and origin == cv_origins[-1]
            ret = fit_predict_catboost_mae(fit, val, spec.numeric + feats, spec, return_model=keep)
            if keep:
                pred, models[name] = ret
            else:
                pred = ret
            wk = pd.DataFrame({"w": val["target_week"].to_numpy(), "y": y, "p": pred}).groupby("w").sum()
            for perfil in ["todos"] + PROFILES:
                msk = np.ones(len(val), bool) if perfil == "todos" else (val["perfil"] == perfil).to_numpy()
                m = block_metrics(y[msk], pred[msk], val["seller_id"].to_numpy()[msk], scale_o, peak_o, signal_o)
                m["skill_vs_naive"] = 1 - m["MAE"] / naive_ae[msk].mean()
                if perfil == "todos":
                    m["WAPE_agregado"] = np.abs(wk["p"] - wk["y"]).sum() / wk["y"].sum()
                rows.append({"origem": origin.date(), "rodada": name, "perfil": perfil,
                             "n_atributos": len(spec.numeric) + len(feats),
                             "limiar_pico": peak_o, "limiar_sinal": signal_o, **m})
            preds.append(pd.DataFrame({
                "origem": origin.date(), "rodada": name, "seller_id": val["seller_id"].to_numpy(),
                "target_week": val["target_week"].to_numpy(), "perfil": val["perfil"].to_numpy(),
                "y": y, "pred": pred, "scale": val["seller_id"].map(scale_o).to_numpy(float), "pico": y > peak_o}))
        if verbose:
            print(f"origem {origin.date()} | treino {len(fit):>6,} | val {len(val):>5,} | "
                  f"perfis {val.groupby('perfil')['seller_id'].nunique().to_dict()} | {time.time() - t0:5.0f}s")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), models


MCOLS = ["MAE", "MAE_positiva", "MAE_zero", "P90", "P95", "vies", "vies_pct", "pct_falso_sinal",
         "cobertura_picos_pct", "picos_previsto_real", "MASE_vend_media", "MASE_vend_mediana",
         "previsto_medio", "pct_previsao_quase_zero", "skill_vs_naive"]


def ganho_por_perfil(perfil_val, rounds):
    prof_mean = perfil_val.groupby(["rodada", "perfil"], sort=False)[MCOLS + ["n_atributos"]].mean()
    prof_mean["n_vendedores_medio"] = perfil_val.groupby(["rodada", "perfil"], sort=False)["n_vendedores"].mean()
    rows = []
    for r in rounds:
        if r == "base":
            continue
        for perfil in ["todos"] + PROFILES:
            x, b = prof_mean.loc[(r, perfil)], prof_mean.loc[("base", perfil)]
            d = {f"Δ {c}": x[c] - b[c] for c in ["MAE", "MAE_positiva", "MAE_zero", "P95", "vies",
                                                "picos_previsto_real", "previsto_medio", "pct_previsao_quase_zero"]}
            d["Δ MASE %"] = 100 * (x["MASE_vend_media"] / b["MASE_vend_media"] - 1)
            d["alerta_quase_zero"] = bool(d["Δ MASE %"] < 0 and d["Δ pct_previsao_quase_zero"] > 2
                                          and d["Δ previsto_medio"] < 0 and d["Δ MAE_positiva"] > 0)
            rows.append({"rodada": r, "perfil": perfil, **d})
    return prof_mean, pd.DataFrame(rows)


def perguntas(prof_delta):
    def answer(r, perfil, col, better="neg"):
        v = prof_delta.set_index(["rodada", "perfil"]).loc[(r, perfil), col]
        return v, (v < 0 if better == "neg" else v > 0)

    qa = []
    for r in ["+intensidade", "+intensidade_red"]:
        for perfil in PROFILES:
            v, ok = answer(r, perfil, "Δ MAE")
            vp, okp = answer(r, perfil, "Δ MAE_positiva")
            plural = {"intermitente": "intermitentes", "medio": "médios", "regular": "regulares"}[perfil]
            qa.append({"pergunta": f"{r} melhorou os {plural}?", "Δ MAE": v, "Δ MAE positiva": vp,
                       "resposta": "sim" if (ok and okp) else ("parcial" if (ok or okp) else "não")})
    for r in ["+picos", "+picos_red"]:
        v, ok = answer(r, "regular", "Δ picos_previsto_real", better="pos")
        vp, okp = answer(r, "regular", "Δ MAE_positiva")
        qa.append({"pergunta": f"{r} melhora os picos dos regulares?", "Δ previsto/real": v, "Δ MAE positiva": vp,
                   "resposta": "sim" if (ok and okp) else ("parcial" if (ok or okp) else "não")})
    flags = prof_delta.loc[prof_delta["alerta_quase_zero"], ["rodada", "perfil"]]
    qa.append({"pergunta": "Algum bloco reduz o MASE só por prever perto de zero?",
               "resposta": "não" if flags.empty else "sim: " + "; ".join(f"{a} ({b})" for a, b in flags.values)})
    return pd.DataFrame(qa)


def conformidade(exp1_long, perfil_val, prof_mean, rounds, cv_origins, limits, C):
    """Seção 24.5: limites na média do perfil, estabilidade e conformidade individual."""
    rows = []
    for r in rounds:
        for origin in cv_origins.date:
            for perfil in PROFILES:
                d = exp1_long.loc[(exp1_long["rodada"] == r) & (exp1_long["origem"] == origin)
                                  & (exp1_long["perfil"] == perfil)]
                lim = limits[perfil]
                for sid, g in d.groupby("seller_id", sort=False):
                    y = g["y"].to_numpy(float)
                    p = g["pred"].to_numpy(float)
                    e = p - y
                    ae = np.abs(e)
                    pos = y > 0
                    mae_pos = float(ae[pos].mean()) if pos.any() else np.nan
                    bias = float(e.mean())
                    ok = bool(ae.mean() <= lim["MAE"])
                    ok = ok and (np.isnan(mae_pos) or mae_pos <= lim["MAE_positiva"])
                    ok = ok and abs(bias) <= lim["vies_abs_max"]
                    rows.append({"rodada": r, "origem": origin, "perfil": perfil, "seller_id": sid,
                                 "MAE": float(ae.mean()), "MAE_positiva": mae_pos, "vies": bias,
                                 "dentro_dos_limites": ok})
    seller_compliance = pd.DataFrame(rows)
    comp_origin = (seller_compliance.groupby(["rodada", "origem", "perfil"], sort=False)
                   .agg(n_vendedores=("seller_id", "nunique"),
                        pct_vendedores_dentro_limites=("dentro_dos_limites", "mean")).reset_index())
    comp_summary = (comp_origin.groupby(["rodada", "perfil"], sort=False)
                    .agg(pct_vendedores_dentro_limites_media=("pct_vendedores_dentro_limites", "mean"),
                         menor_pct_origem=("pct_vendedores_dentro_limites", "min"),
                         origens_compliance_80=("pct_vendedores_dentro_limites",
                                                lambda s: int((s >= C.MIN_SELLER_COMPLIANCE).sum())))
                    .reset_index())
    comp_summary["conformidade_vendedores_ok"] = (
        (comp_summary["pct_vendedores_dentro_limites_media"] >= C.MIN_SELLER_COMPLIANCE)
        & (comp_summary["origens_compliance_80"] >= C.STAB_MIN_ORIGINS))

    crit = []
    for r in rounds:
        for perfil in PROFILES:
            lim = limits[perfil]
            per_o = perfil_val.query("rodada == @r and perfil == @perfil")
            ok_o = ((per_o["MAE"] <= lim["MAE"]) & (per_o["MAE_positiva"] <= lim["MAE_positiva"])
                    & (per_o["P90"] <= lim["P90"]) & (per_o["P95"] <= lim["P95"])
                    & (per_o["vies"].abs() <= lim["vies_abs_max"]))
            m = prof_mean.loc[(r, perfil)]
            checks = {"MAE_ok": m["MAE"] <= lim["MAE"], "MAE_positiva_ok": m["MAE_positiva"] <= lim["MAE_positiva"],
                      "P90_ok": m["P90"] <= lim["P90"], "P95_ok": m["P95"] <= lim["P95"],
                      "vies_abs_ok": abs(m["vies"]) <= lim["vies_abs_max"]}
            cs = comp_summary.loc[(comp_summary["rodada"] == r) & (comp_summary["perfil"] == perfil)].iloc[0]
            profile_ok = all(checks.values())
            stability_ok = int(ok_o.sum()) >= C.STAB_MIN_ORIGINS
            compliance_ok = bool(cs["conformidade_vendedores_ok"])
            crit.append({"rodada": r, "perfil": perfil, "MAE": m["MAE"], "MAE_positiva": m["MAE_positiva"],
                         "P90": m["P90"], "P95": m["P95"], "vies": m["vies"], **checks,
                         "limites_na_media": profile_ok, "origens_dentro_dos_limites": int(ok_o.sum()),
                         "estavel": stability_ok,
                         "pct_vendedores_dentro_limites_media": cs["pct_vendedores_dentro_limites_media"],
                         "menor_pct_vendedores_origem": cs["menor_pct_origem"],
                         "origens_compliance_80": int(cs["origens_compliance_80"]),
                         "conformidade_vendedores_ok": compliance_ok,
                         "passou": profile_ok and stability_ok and compliance_ok})
    crit_perfil = pd.DataFrame(crit)
    limits_all = crit_perfil.groupby("rodada", sort=False)["passou"].all()
    return seller_compliance, comp_origin, comp_summary, crit_perfil, limits_all


def criterios_globais(exp1_long, perfil_val, prof_mean, rounds, C):
    """Seção 24.6: bootstrap por vendedor com o mesmo estimando da seleção (média simples entre origens)."""
    P = exp1_long.assign(ae=lambda d: np.abs(d["pred"] - d["y"]))
    key = ["origem", "seller_id", "target_week"]
    base_P = P.loc[P["rodada"] == "base"].set_index(key)
    glob = prof_mean.xs("todos", level="perfil")
    glob_wape = perfil_val.query("perfil == 'todos'").groupby("rodada", sort=False)["WAPE_agregado"].mean()
    piv_mase = perfil_val.query("perfil == 'todos'").pivot(index="origem", columns="rodada", values="MASE_vend_media")
    piv_peak = perfil_val.query("perfil == 'todos'").pivot(index="origem", columns="rodada", values="picos_previsto_real")
    sellers_all = np.sort(P["seller_id"].unique())
    S_ = len(sellers_all)
    rng_b = np.random.default_rng(C.RANDOM_SEED)
    Wb = np.vstack([np.bincount(rng_b.integers(0, S_, S_), minlength=S_) for _ in range(C.N_BOOT)]).astype(float)
    ones_s = np.ones(S_)
    origins_ = sorted(P["origem"].unique())

    def origin_seller_arrays(df):
        d = df.copy()
        d["pos"] = d["y"] > 0
        d["ae_pos"] = np.where(d["pos"], d["ae"], 0.0)
        d["n_pos"] = d["pos"].astype(float)
        d["ae_zero"] = np.where(~d["pos"], d["ae"], 0.0)
        d["n_zero"] = (~d["pos"]).astype(float)
        d["pk_pred"] = np.where(d["pico"], d["pred"], 0.0)
        d["pk_y"] = np.where(d["pico"], d["y"], 0.0)
        g = d.groupby(["origem", "seller_id"]).agg(
            sae=("ae", "sum"), n=("ae", "size"), sae_pos=("ae_pos", "sum"), n_pos=("n_pos", "sum"),
            sae_zero=("ae_zero", "sum"), n_zero=("n_zero", "sum"), pk_pred=("pk_pred", "sum"), pk_y=("pk_y", "sum"),
            scale=("scale", "first"))
        g["mase"] = (g["sae"] / g["n"]) / g["scale"]
        out = {}
        for o in origins_:
            q = g.xs(o, level="origem", drop_level=True).reindex(sellers_all)
            out[o] = {c: q[c].fillna(0.0).to_numpy(float) for c in
                      ["sae", "n", "sae_pos", "n_pos", "sae_zero", "n_zero", "pk_pred", "pk_y"]}
            out[o]["mase"] = q["mase"].to_numpy(float)
            out[o]["mase_ok"] = np.isfinite(out[o]["mase"]).astype(float)
            out[o]["mase"] = np.nan_to_num(out[o]["mase"], nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _ratio(w, num, den):
        num_w, den_w = w @ num, w @ den
        if np.ndim(den_w) == 0:
            return float(num_w / den_w) if den_w > 0 else np.nan
        return np.divide(num_w, den_w, out=np.full_like(num_w, np.nan, dtype=float), where=den_w > 0)

    def _origin_mean(values):
        out = np.nanmean(np.vstack(values), axis=0)
        return float(out[0]) if out.ndim == 1 and out.size == 1 else out

    def stat_vec(arrays, w):
        per = {m: [] for m in ["MAE", "MAE_positiva", "MAE_zero", "picos_previsto_real", "MASE_vend_media"]}
        for o in origins_:
            a = arrays[o]
            per["MAE"].append((w @ a["sae"]) / (w @ a["n"]))
            per["MAE_positiva"].append((w @ a["sae_pos"]) / (w @ a["n_pos"]))
            per["MAE_zero"].append((w @ a["sae_zero"]) / (w @ a["n_zero"]))
            per["picos_previsto_real"].append(_ratio(w, a["pk_pred"], a["pk_y"]))
            per["MASE_vend_media"].append((w @ a["mase"]) / (w @ a["mase_ok"]))
        return {m: _origin_mean(v) for m, v in per.items()}

    gb = origin_seller_arrays(P.loc[P["rodada"] == "base"])
    boot_rows, glob_rows = [], []
    for r in rounds:
        if r == "base":
            continue
        gr = origin_seller_arrays(P.loc[P["rodada"] == r])
        est_r, est_b = stat_vec(gr, ones_s), stat_vec(gb, ones_s)
        bt_r, bt_b = stat_vec(gr, Wb), stat_vec(gb, Wb)
        for met in est_r:
            assert np.isclose(est_r[met], float(glob.loc[r, met]), rtol=1e-8, atol=1e-8), (r, met)
            assert np.isclose(est_b[met], float(glob.loc["base", met]), rtol=1e-8, atol=1e-8), ("base", met)
            diff = np.asarray(bt_r[met] - bt_b[met], float)
            boot_rows.append({"rodada": r, "metrica": met, "base": est_b[met], "rodada_valor": est_r[met],
                              "diferenca": est_r[met] - est_b[met], "ic95_inf": np.percentile(diff, 2.5),
                              "ic95_sup": np.percentile(diff, 97.5), "n_reamostragens": C.N_BOOT,
                              "unidade_reamostrada": "vendedor",
                              "estimando": "média simples da métrica entre as 6 origens"})
        both = pd.DataFrame({"r": P.loc[P["rodada"] == r].set_index(key)["ae"], "b": base_P["ae"]})
        wk = both.groupby(level=["origem", "target_week"]).mean()
        sl = both.groupby(level="seller_id").mean()
        sl = sl.loc[~np.isclose(sl["r"], sl["b"])]
        reg = (P.loc[(P["rodada"] == r) & P["pico"] & (P["perfil"] == "regular")]
               .groupby("seller_id")[["pred", "y"]].sum()
               .join(P.loc[(P["rodada"] == "base") & P["pico"] & (P["perfil"] == "regular")]
                     .groupby("seller_id")[["pred"]].sum().rename(columns={"pred": "pred_b"})))
        gap_r, gap_b = np.abs(1 - reg["pred"] / reg["y"]), np.abs(1 - reg["pred_b"] / reg["y"])
        untied = ~np.isclose(gap_r, gap_b)
        x, b = glob.loc[r], glob.loc["base"]
        ci_pk = [row for row in boot_rows if row["rodada"] == r and row["metrica"] == "picos_previsto_real"][0]
        c = {
            "G1_MASE": x["MASE_vend_media"] <= b["MASE_vend_media"] * (1 + C.G_MASE_TOL),
            "G2_MAE_positiva": x["MAE_positiva"] < b["MAE_positiva"],
            "G3_falso_sinal": x["pct_falso_sinal"] <= b["pct_falso_sinal"] + C.G_FALSE_SIGNAL_PP,
            "G4_WAPE_agregado": glob_wape[r] <= glob_wape["base"] * (1 + C.G_WAPE_TOL),
            "G5_origens": int((piv_mase[r] < piv_mase["base"]).sum()) >= C.G_MIN_ORIGINS,
            "G6_robustez": bool((wk["r"] < wk["b"]).mean() >= C.G_MIN_SHARE and (sl["r"] < sl["b"]).mean() >= C.G_MIN_SHARE),
            "P1_delta_min": (x["picos_previsto_real"] - b["picos_previsto_real"]) >= C.P_MIN_DELTA,
            "P2_origens": int((piv_peak[r] > piv_peak["base"]).sum()) >= C.P_MIN_ORIGINS,
            "P3_regulares": bool((gap_r[untied] < gap_b[untied]).mean() >= C.P_MIN_REG_SHARE),
            "P4_ic95": ci_pk["ic95_inf"] > 0,
        }
        glob_rows.append({"rodada": r, **c,
                          "globais_ok": all(v for k, v in c.items() if k.startswith("G")),
                          "picos_ok": all(v for k, v in c.items() if k.startswith("P")),
                          "delta_MASE_pct": 100 * (x["MASE_vend_media"] / b["MASE_vend_media"] - 1),
                          "delta_MAE_positiva": x["MAE_positiva"] - b["MAE_positiva"],
                          "delta_falso_sinal_pp": x["pct_falso_sinal"] - b["pct_falso_sinal"],
                          "delta_WAPE_agregado_pct": 100 * (glob_wape[r] / glob_wape["base"] - 1),
                          "delta_picos_previsto_real": x["picos_previsto_real"] - b["picos_previsto_real"],
                          "ic95_delta_picos": f"[{ci_pk['ic95_inf']:+.3f}; {ci_pk['ic95_sup']:+.3f}]",
                          "origens_MASE_melhor": int((piv_mase[r] < piv_mase["base"]).sum()),
                          "origens_picos_melhor": int((piv_peak[r] > piv_peak["base"]).sum()),
                          "pct_regulares_picos_melhor": 100 * (gap_r[untied] < gap_b[untied]).mean(),
                          "n_regulares_com_pico": int(untied.sum()),
                          "pct_semanas_melhor": 100 * (wk["r"] < wk["b"]).mean(),
                          "pct_vendedores_melhor": 100 * (sl["r"] < sl["b"]).mean(),
                          "bootstrap_estimando": "média simples entre origens"})
    return pd.DataFrame(boot_rows), pd.DataFrame(glob_rows).set_index("rodada"), glob, glob_wape


def selecionar(rounds, glob, glob_crit, limits_all, crit_perfil, limits_hash, cv_origins, C):
    rows = []
    for r in rounds:
        lim_ok = bool(limits_all[r])
        if r == "base":
            g_ok = p_ok = True
        else:
            g_ok, p_ok = bool(glob_crit.loc[r, "globais_ok"]), bool(glob_crit.loc[r, "picos_ok"])
        rows.append({"rodada": r, "n_atributos": int(glob.loc[r, "n_atributos"]),
                     "MASE_vend_media": glob.loc[r, "MASE_vend_media"], "limites_3_perfis": lim_ok,
                     "globais": g_ok if r != "base" else "n/a", "picos": p_ok if r != "base" else "n/a",
                     "elegivel": lim_ok and g_ok and p_ok})
    elig = pd.DataFrame(rows).set_index("rodada")
    eligible = elig.loc[elig["elegivel"]]
    if eligible.empty:
        chosen = None
        decision = ("nenhum candidato passou nos limites dos três perfis e nos demais critérios; "
                    "o Experimento 1 não escolhe modelo e o modelo principal de selecao_modelo.json é mantido")
    else:
        best = eligible["MASE_vend_media"].min()
        tied = eligible.loc[eligible["MASE_vend_media"] <= best * (1 + C.PRACTICAL_TIE_TOL)]
        chosen = tied.sort_values(["n_atributos", "MASE_vend_media"]).index[0]
        decision = (f"escolhido {chosen} entre os elegíveis {list(eligible.index)}; "
                    f"empate prático entre {list(tied.index)}" if len(tied) > 1 else f"escolhido {chosen}")
    failed = {r: crit_perfil.query("rodada == @r and not passou")["perfil"].tolist() for r in rounds}
    choice = {"modelo_escolhido": chosen, "decisao": decision, "elegiveis": list(eligible.index),
              "perfis_reprovados_por_rodada": failed,
              "MASE_validacao": {r: float(glob.loc[r, "MASE_vend_media"]) for r in rounds},
              "limites_sha256": limits_hash, "origens": [str(o.date()) for o in cv_origins],
              "registrado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
              "teste_usado_na_selecao": False}
    return elig, choice


def sensibilidade_sem_limites(rounds, glob, glob_crit, C):
    rows = []
    for r in rounds:
        ok = True if r == "base" else bool(glob_crit.loc[r, "globais_ok"] and glob_crit.loc[r, "picos_ok"])
        rows.append({"rodada": r, "n_atributos": int(glob.loc[r, "n_atributos"]),
                     "MASE_vend_media": glob.loc[r, "MASE_vend_media"], "elegivel_sem_limites": ok})
    sens = pd.DataFrame(rows).set_index("rodada")
    el = sens.loc[sens["elegivel_sem_limites"]]
    best = el["MASE_vend_media"].min()
    tied = el.loc[el["MASE_vend_media"] <= best * (1 + C.PRACTICAL_TIE_TOL)]
    choice = tied.sort_values(["n_atributos", "MASE_vend_media"]).index[0]
    sens["delta_MASE_vs_base_pct"] = 100 * (sens["MASE_vend_media"] / sens.loc["base", "MASE_vend_media"] - 1)
    return sens, list(el.index), list(tied.index), choice


def testar_escolhido(chosen, rounds, train_x, test_x, comparison, profile, scale_final, thr, limits, spec, C):
    """Seção 24.8: teste SOMENTE da variante escolhida pela validação (chamar só se chosen não for None/base)."""
    feats_new = spec.numeric + rounds[chosen]
    pred_new = fit_predict_catboost_mae(train_x, test_x, feats_new, spec)
    assert np.isfinite(pred_new).all() and (pred_new >= 0).all()
    name_new = f"CatBoost_MAE{chosen}"
    diag = comparison.merge(profile[["perfil"]], left_on="seller_id", right_index=True, how="left")
    assert (diag["seller_id"].to_numpy() == test_x["seller_id"].to_numpy()).all()
    diag[f"pred_{name_new}"] = pred_new
    y_t = diag[TARGET].to_numpy(float)
    naive_t = np.abs(diag["pred_Naive"].to_numpy(float) - y_t)
    rows = []
    for m in ["Naive", "CatBoost_MAE", name_new]:
        p_t = diag[f"pred_{m}"].to_numpy(float)
        wk = pd.DataFrame({"w": diag["target_week"], "y": y_t, "p": p_t}).groupby("w").sum()
        for perfil in ["todos"] + PROFILES:
            msk = np.ones(len(diag), bool) if perfil == "todos" else (diag["perfil"] == perfil).to_numpy()
            mm = block_metrics(y_t[msk], p_t[msk], diag["seller_id"].to_numpy()[msk], scale_final,
                               thr["PEAK_THRESHOLD"], thr["SIGNAL_THRESHOLD"])
            mm["skill_vs_naive"] = 1 - mm["MAE"] / naive_t[msk].mean()
            if perfil == "todos":
                mm["WAPE_agregado"] = np.abs(wk["p"] - wk["y"]).sum() / wk["y"].sum()
            else:
                lim = limits[perfil]
                mm["dentro_dos_limites"] = bool(mm["MAE"] <= lim["MAE"] and mm["MAE_positiva"] <= lim["MAE_positiva"]
                                                and mm["P90"] <= lim["P90"] and mm["P95"] <= lim["P95"]
                                                and abs(mm["vies"]) <= lim["vies_abs_max"])
            rows.append({"modelo": m, "perfil": perfil, **mm})
    abs_base = np.abs(diag["pred_CatBoost_MAE"].to_numpy(float) - y_t)
    d_s = (pd.DataFrame({"s": diag["seller_id"], "d": np.abs(pred_new - y_t) - abs_base})
           .groupby("s")["d"].agg(["sum", "size"]))
    rng_e = np.random.default_rng(C.RANDOM_SEED)
    idx_e = rng_e.integers(0, len(d_s), size=(C.N_BOOT, len(d_s)))
    boot = d_s["sum"].to_numpy()[idx_e].sum(1) / d_s["size"].to_numpy()[idx_e].sum(1)
    resumo = {"modelo": name_new, "diferenca_MAE": float(d_s["sum"].sum() / d_s["size"].sum()),
              "ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return pd.DataFrame(rows), resumo
