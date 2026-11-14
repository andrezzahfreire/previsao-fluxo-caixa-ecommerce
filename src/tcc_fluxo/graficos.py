"""Figuras do v5 como funções. Cada função devolve a figura; o notebook decide se mostra e grava."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .painel import TARGET

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C_CHOSEN, C_NAIVE, C_OTHER = "#2a78d6", "#eb6834", "#b4b2a9"
PROFILE_COLORS = {"intermitente": "#2a78d6", "medio": "#eb6834", "regular": "#1baf7a"}
MODEL_COLORS = {"LightGBM_MAE": "#16a085", "LightGBM_MAE_Exog": "#0e7490", "CatBoost_MAE": "#2a78d6",
                "CatBoost_MAE_Exog": "#8b5cf6", "CatBoost_MAE_raw": "#7b4ab5", "Naive": "#eb6834",
                "CatBoost_Tweedie": "#1baf7a", "MediaMovel4": "#eda100", "HistGB_MAE": "#9a9a9a",
                "HistGB_Poisson": "#6b6b6b"}
MODEL_STYLE = {"LightGBM_MAE": "--", "LightGBM_MAE_Exog": ":", "CatBoost_MAE": "-", "CatBoost_MAE_Exog": "-.",
               "CatBoost_MAE_raw": "-", "Naive": "--", "CatBoost_Tweedie": "-", "MediaMovel4": ":",
               "HistGB_MAE": "-", "HistGB_Poisson": "-"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def total_semanal_observado(weeks_info, test_end, desde="2017-06-01"):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ws = weeks_info.loc[desde:, "valor"]
    ax.plot(ws.index, ws.values, color="#2a78d6", lw=2, marker="o", ms=3)
    ax.axvline(test_end, color=INK2, ls="--", lw=1)
    ax.text(test_end, ax.get_ylim()[1] * 0.95, " TEST_END", color=INK2, va="top", fontsize=9)
    ax.set_title("Proxy de entrada bruta de caixa — total semanal (todas as vendas aprovadas)")
    ax.set_ylabel("R$")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def agregado_teste(aggregate, chosen):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(aggregate.index, aggregate[TARGET], color=INK, lw=2, label="Real")
    ax.plot(aggregate.index, aggregate["pred_Naive"], color=C_NAIVE, lw=2, ls="--", label="Naive")
    if chosen != "Naive":
        ax.plot(aggregate.index, aggregate[f"pred_{chosen}"], color=C_CHOSEN, lw=2,
                label=f"{chosen} (escolhido na validação)")
    ax.set_title("Total semanal da coorte no teste — proxy de entrada bruta de caixa (h = 1 semana)",
                 color=INK, loc="left")
    ax.set_ylabel("R$ por semana", color=INK2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f} mil"))
    clean(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def comparacao_modelos(resumo_val, ci, final_models, eligible_final, chosen, ridge_name, ridge_eligible):
    val_plot = resumo_val.loc[resumo_val["modelo"].isin(final_models)].set_index("modelo")
    order = val_plot.sort_values("MASE_vend_media", ascending=False).index.tolist()
    sk_ci = ci.loc[(ci["metrica"] == "skill vs Naive (agrupado)")].set_index("modelo")

    def _label(m):
        return m + ("  (removido: instável)" if m == ridge_name and not ridge_eligible else "")

    colors = [C_CHOSEN if m == chosen else C_OTHER for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(order) + 1.6), sharey=True)
    ax = axes[0]
    vals = val_plot.loc[order, "MASE_vend_media"]
    err = val_plot.loc[order, "MASE_vend_media_dp"]
    elig_mask = np.array([m in eligible_final for m in order])
    x_max = float((vals[elig_mask] + err[elig_mask]).max() * 1.25)
    shown = np.minimum(np.where(np.isfinite(vals), vals, x_max), x_max)
    ax.barh(range(len(order)), shown, xerr=np.where(elig_mask, err, 0), color=colors, height=0.6,
            error_kw={"ecolor": INK2, "lw": 1})
    for i, v in enumerate(vals):
        txt = f"  {v:,.3f}" if v <= x_max else f"  {v:,.0f} (fora da escala)"
        ax.text(min(v + err.iloc[i], x_max) if v <= x_max else 0.02 * x_max, i, txt, va="center",
                color=INK if v > x_max else INK2, fontsize=9)
    ax.set_xlim(0, x_max)
    ax.axvline(1, color=INK2, lw=1, ls=":")
    ax.set_yticks(range(len(order)), [_label(m) for m in order])
    ax.set_title("Validação: MASE médio por vendedor\n(média ± dp entre origens; menor é melhor)",
                 color=INK, loc="left", fontsize=10)
    clean(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    est = sk_ci.loc[order, "estimativa"].clip(lower=-1.0)
    lo = sk_ci.loc[order, "ic95_inf"].clip(lower=-1.0)
    hi = sk_ci.loc[order, "ic95_sup"].clip(lower=-1.0)
    ax.barh(range(len(order)), est, color=colors, height=0.6)
    ax.errorbar(est, range(len(order)), xerr=[est - lo, hi - est], fmt="none", ecolor=INK2, lw=1)
    ax.axvline(0, color=INK2, lw=1)
    for i, m in enumerate(order):
        ax.text(max(hi.iloc[i], 0), i, f"  {sk_ci.loc[m, 'estimativa']:+.1%}", va="center", color=INK2, fontsize=9)
    ax.set_title("Teste: skill vs Naive, nível vendedor-semana\n(IC 95% bootstrap por vendedor; eixo truncado em −100%)",
                 color=INK, loc="left", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    clean(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    fig.suptitle(f"Azul = modelo escolhido pela validação ({chosen})", x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def erro_por_vendedor(diag_sellers, main):
    sk_def = diag_sellers["skill_vs_naive"].dropna()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lo_ax = 1.0
    for perfil, color in PROFILE_COLORS.items():
        g = diag_sellers.loc[diag_sellers["perfil"] == perfil]
        ax.scatter(g["MAE_Naive"].clip(lower=lo_ax), g[f"MAE_{main}"].clip(lower=lo_ax), s=14, alpha=0.65,
                   color=color, edgecolor="white", linewidth=0.4, label=f"{perfil} ({len(g)})")
    hi_ax = float(max(diag_sellers["MAE_Naive"].max(), diag_sellers[f"MAE_{main}"].max())) * 1.3
    ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], color=INK2, lw=1, ls="--")
    ax.text(hi_ax / 1.6, hi_ax / 1.25, "mesmo erro", color=INK2, fontsize=8, rotation=45, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_ax, hi_ax)
    ax.set_ylim(lo_ax, hi_ax)
    ax.set_xlabel("MAE do Naive por vendedor (R$, escala log)", color=INK2)
    ax.set_ylabel(f"MAE do {main} por vendedor (R$, escala log)", color=INK2)
    ax.set_title(f"Erro por vendedor no teste: abaixo da diagonal = {main} melhor\n"
                 f"({(sk_def > 0).mean():.1%} dos {len(sk_def)} vendedores com skill definido; valores < R$ 1 exibidos em 1)",
                 color=INK, loc="left", fontsize=10)
    clean(ax)
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.legend(frameon=False, title="perfil (nº de vendedores)", fontsize=8, title_fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def erro_por_semana(diag_week, models):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for m in models:
        axes[0].plot(diag_week["target_week"], diag_week[f"MAE_{m}"], color=MODEL_COLORS.get(m, C_OTHER),
                     ls=MODEL_STYLE.get(m, "-"), lw=2, label=m)
        axes[1].plot(diag_week["target_week"], diag_week[f"vies_{m}"], color=MODEL_COLORS.get(m, C_OTHER),
                     ls=MODEL_STYLE.get(m, "-"), lw=2, label=m)
    axes[0].set_title("Erro absoluto médio por vendedor em cada semana do teste (R$)", color=INK, loc="left", fontsize=10)
    axes[1].set_title("Viés médio por vendedor em cada semana (previsto − real, R$): abaixo de 0 = subestimação",
                      color=INK, loc="left", fontsize=10)
    axes[1].axhline(0, color=INK2, lw=1)
    for ax in axes:
        clean(ax)
    fig.autofmt_xdate()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def picos(picos_df, models, peak_thr, n_picos):
    order_pk = [m for m in models][::-1]
    pk = picos_df.set_index("modelo").loc[order_pk]
    bar_colors = [MODEL_COLORS.get(m, C_OTHER) for m in order_pk]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    axes[0].barh(order_pk, pk["Previsto/real"], color=bar_colors, height=0.6)
    axes[0].axvline(1, color=INK2, lw=1, ls="--")
    for i, v in enumerate(pk["Previsto/real"]):
        axes[0].text(v, i, f"  {v:.2f}", va="center", color=INK2, fontsize=9)
    axes[0].set_title("Previsão média ÷ valor real médio nos picos\n(1 = sem viés; < 1 = subestima)", color=INK,
                      loc="left", fontsize=10)
    axes[1].barh(order_pk, pk["% picos subestimados"], color=bar_colors, height=0.6)
    axes[1].axvline(50, color=INK2, lw=1, ls="--")
    for i, v in enumerate(pk["% picos subestimados"]):
        axes[1].text(v, i, f"  {v:.0f}%", va="center", color=INK2, fontsize=9)
    axes[1].set_xlim(0, 110)
    axes[1].set_title("% dos picos subestimados", color=INK, loc="left", fontsize=10)
    for ax in axes:
        clean(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Picos no teste: real > R$ {peak_thr:,.0f} (P90 da receita positiva no treino), n = {n_picos:,}",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig
