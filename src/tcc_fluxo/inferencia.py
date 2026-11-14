"""Seções 18–19 do v5: intervalos bootstrap e Diebold-Mariano com correção HLN."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .painel import TARGET


def _ci(est, boot):
    boot = boot[np.isfinite(boot)]
    return est, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def intervalos_bootstrap(per_seller_test, aggregate, final_models, n_boot, seed):
    """Nível vendedor: reamostra vendedores (preserva as semanas de cada um).
    Nível agregado: reamostra semanas do total da coorte. Mesma sequência de sorteios do v5."""
    rng = np.random.default_rng(seed)
    rows = []
    sellers = np.sort(per_seller_test["seller_id"].unique())
    S = len(sellers)
    idx = rng.integers(0, S, size=(n_boot, S))
    W = np.vstack([np.bincount(r, minlength=S) for r in idx]).astype(float)
    ones = np.ones(S)
    ref = per_seller_test.loc[per_seller_test["modelo"] == "Naive"].set_index("seller_id").loc[sellers]
    for m in final_models:
        s = per_seller_test.loc[per_seller_test["modelo"] == m].set_index("seller_id").loc[sellers]
        sae, n, sy, sae_ref = s["sae"].to_numpy(), s["n"].to_numpy(), s["sy"].to_numpy(), ref["sae"].to_numpy()
        mase_v = s["MASE"].replace([np.inf, -np.inf], np.nan).to_numpy()
        mase_ok = np.isfinite(mase_v)
        sk_v = s["skill_vs_naive"].to_numpy()
        sk_ok = np.isfinite(sk_v)
        stats_ = {
            "MAE (agrupado)": (lambda w: (w @ sae) / (w @ n)),
            "WAPE (agrupado)": (lambda w: (w @ sae) / (w @ sy)),
            "skill vs Naive (agrupado)": (lambda w: 1 - (w @ sae) / (w @ sae_ref)),
            "MASE médio por vendedor": (lambda w: (w @ np.where(mase_ok, mase_v, 0)) / (w @ mase_ok)),
            "proporção de vendedores com ganho": (lambda w: (w @ (sk_ok & (np.nan_to_num(sk_v) > 0))) / (w @ sk_ok)),
        }
        for met, f in stats_.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                est, lo, hi = _ci(float(f(ones)), np.asarray(f(W), dtype=float))
            rows.append({"modelo": m, "metrica": met, "estimativa": est, "ic95_inf": lo, "ic95_sup": hi,
                         "nivel": "vendedor (bootstrap por vendedor)", "n_reamostragens": n_boot})

    yw = aggregate[TARGET].to_numpy(float)
    T_w = len(yw)
    idx_w = rng.integers(0, T_w, size=(n_boot, T_w))
    e_ref_w = np.abs(yw - aggregate["pred_Naive"].to_numpy(float))
    for m in final_models:
        e_w = np.abs(yw - aggregate[f"pred_{m}"].to_numpy(float))
        for met, f in {
            "MAE": lambda ii: e_w[ii].mean(axis=-1),
            "WAPE": lambda ii: e_w[ii].sum(axis=-1) / yw[ii].sum(axis=-1),
            "skill vs Naive": lambda ii: 1 - e_w[ii].sum(axis=-1) / e_ref_w[ii].sum(axis=-1),
        }.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                est, lo, hi = _ci(float(f(np.arange(T_w))), f(idx_w))
            rows.append({"modelo": m, "metrica": met, "estimativa": est, "ic95_inf": lo, "ic95_sup": hi,
                         "nivel": "agregado semanal (bootstrap por semana)", "n_reamostragens": n_boot})
    return pd.DataFrame(rows)


def dm_hln(d, h=1):
    """Diebold & Mariano (1995) com a correção de Harvey, Leybourne & Newbold (1997); t com n-1 g.l."""
    d = np.asarray(d, float)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    dm = dbar / np.sqrt(gamma0 / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = k * dm
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def block_bootstrap_mean(d, B, rng):
    """Bootstrap de blocos móveis da média de d, bloco = ceil(n^(1/3))."""
    d = np.asarray(d, float)
    n = len(d)
    L = max(2, int(np.ceil(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, n_blocks))
    sample_idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    means = d[sample_idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(np.mean(np.abs(means - d.mean()) >= abs(d.mean())))
    return L, lo, hi, p


def tabela_dm(comparison, aggregate, final_models, chosen, n_boot, seed, min_weeks):
    """Perda semanal (média entre vendedores e erro do total) contra o Naive."""
    rng_dm = np.random.default_rng(seed)
    loss = pd.DataFrame({"target_week": comparison["target_week"]})
    for m in final_models:
        loss[m] = np.abs(comparison[TARGET] - comparison[f"pred_{m}"])
    L_seller = loss.groupby("target_week").mean()
    L_agg = pd.DataFrame({m: np.abs(aggregate[TARGET] - aggregate[f"pred_{m}"]) for m in final_models})
    rows = []
    for nivel, L in [("vendedor: perda absoluta média por semana", L_seller),
                     ("agregado: erro absoluto do total semanal", L_agg)]:
        n_weeks = len(L)
        for m in [x for x in final_models if x != "Naive"]:
            d = (L[m] - L["Naive"]).to_numpy()
            row = {"nivel": nivel, "modelo": m, "referencia": "Naive", "n_semanas_efetivas": n_weeks,
                   "media_d (modelo - Naive)": d.mean(), "escolhido_validacao": m == chosen}
            if not np.isfinite(d).all():
                row.update({"aplicado": False, "observacao": "perdas não finitas"})
            elif n_weeks < min_weeks:
                row.update({"aplicado": False, "observacao": f"menos de {min_weeks} semanas"})
            else:
                stat, p = dm_hln(d)
                blk, lo, hi, p_b = block_bootstrap_mean(d, n_boot, rng_dm)
                if p < 0.05:
                    txt = "modelo erra menos que o Naive" if d.mean() < 0 else "modelo erra mais que o Naive"
                else:
                    txt = "sem diferença significativa a 5%"
                row.update({"aplicado": True, "dm_hln": stat, "p_valor_hln": p, "bloco": blk,
                            "ic95_media_d_inf": lo, "ic95_media_d_sup": hi, "p_valor_bootstrap_blocos": p_b,
                            "observacao": txt})
            rows.append(row)
    return pd.DataFrame(rows)


def holm(pvals):
    """Correção de Holm-Bonferroni (acréscimo da v6; não altera as colunas do v5)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out
