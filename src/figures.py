"""Figures for the report. One chart, one question.

Palette: validated categorical slots 1 (blue) and 2 (orange) on the light surface.
Small multiples use ONE series colour against grey context, so no multi-hue
separation question arises.
"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import features as F, validate as V
from components import loadings

THEME = os.environ.get("FIG_THEME", "light")
PFX = "" if THEME == "light" else "dark-"
if THEME == "dark":
    BLUE, ORANGE = "#3987e5", "#d95926"
    SURFACE, INK, INK2, GREY = "#181c24", "#f2f4f7", "#a7b1c0", "#39414e"
    GRID, EDGE, RULE = "#252b35", "#3a4350", "#5a6472"
else:
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    SURFACE, INK, INK2, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
    GRID, EDGE, RULE = "#ececea", "#c9c8c3", "#9a9994"
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": EDGE,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fig1_space(D):
    """One distinction, not four.

    A density strip carries the distributional claim (duelists shifted right, everyone
    else spread) because 726 overlapping dots cannot. Labels use LEADER LINES -- a
    floating label next to a dense scatter is ambiguous about which point it names.
    """
    from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(9.4, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.9], hspace=.06)
    top, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    XLIM = (-7.0, 8.2)

    nd, du = D[D.role != "duelist"], D[D.role == "duelist"]
    xs = np.linspace(*XLIM, 400)
    for g, c, lab in [(nd, BLUE, "not a duelist"), (du, ORANGE, "duelist")]:
        y = gaussian_kde(g.PC1)(xs)
        top.fill_between(xs, y, color=c, alpha=.22, lw=0)
        top.plot(xs, y, color=c, lw=2)
        top.annotate(f"{lab}  (n={len(g)})", (g.PC1.median(), gaussian_kde(g.PC1)(g.PC1.median())[0]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=9, color=c, weight="bold")
    top.set_xlim(*XLIM); top.set_ylim(0, None)
    top.set_yticks([]); top.set_xticks([]); top.grid(False)
    for sp in ("left", "bottom"): top.spines[sp].set_visible(False)

    ax.scatter(nd.PC1, nd.PC2, s=13, c=BLUE, lw=0, alpha=.38, zorder=1)
    ax.scatter(du.PC1, du.PC2, s=13, c=ORANGE, lw=0, alpha=.52, zorder=2)
    ax.axhline(0, color=EDGE, lw=.8, zorder=0)
    ax.axvline(0, color=EDGE, lw=.8, zorder=0)

    # Names only on the plot; the key below carries the numbers.
    # Note text is GENERATED FROM THE ROW, never hardcoded -- hand-written labels
    # repeatedly drifted to the player's career mean when the dot is one season.
    NOTE = [("leo",       2023, (-14,  -6), "right", "the fewest of any player measured"),
            ("something", 2026, (-12,  12), "right", "Paper Rex"),
            ("jinggg",    2026, (-12, -12), "right", "Paper Rex \u00b7 same roster, same choice"),
            ("aspas",     2024, ( 12,  11), "left",  ""),
            ("oxy",       2024, ( 13,   0), "left",  "")]
    key = []
    for h, y, off, ha, tag in NOTE:
        q = D[(D.handle == h) & (D.year == y)]
        if not len(q) or not bool(q.label_ok.iat[0]):
            continue
        r = q.iloc[0]
        x0, y0 = float(r.PC1), float(r.PC2)
        ax.scatter([x0], [y0], s=62, facecolor="none", edgecolor=INK, lw=1.5, zorder=6)
        ax.annotate(h, (x0, y0), textcoords="offset points", xytext=off, fontsize=9,
                    color=INK, weight="bold", ha=ha, va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=.92))
        key.append((h, f"{r.first_engagement:.1f}", f"{r.deaths:.1f}", tag))

    ax.set_xlabel("aggression  \u2192"); ax.set_ylabel("isolation / gunplay  \u2192")
    ax.set_xlim(*XLIM); ax.set_ylim(-3.9, 4.4)

    # key below the chart: one row per player, fields aligned
    fig.text(.02, -.012, "C I R C L E D   P L A Y E R S", fontsize=7.4, color=INK2,
             weight="bold", ha="left")
    fig.text(.135, -.048, "opening duels", fontsize=7.4, color=INK2, ha="right", va="top")
    fig.text(.215, -.048, "deaths", fontsize=7.4, color=INK2, ha="right", va="top")
    for i, (h, fe, dth, tag) in enumerate(key):
        yy = -.078 - i * .036
        fig.text(.02,  yy, h,   fontsize=8.8, color=INK,  weight="bold", ha="left",  va="top")
        fig.text(.135, yy, fe,  fontsize=8.8, color=INK,  ha="right", va="top")
        fig.text(.215, yy, dth, fontsize=8.8, color=INK,  ha="right", va="top")
        if tag:
            fig.text(.245, yy, tag, fontsize=8.4, color=INK2, ha="left", va="top")
    yb = -.078 - len(key) * .036 - .022
    gdu, gnd = du[du.label_ok], nd[nd.label_ok]
    fig.text(.02, yb, f"Per map, averaged over players whose role label is solid "
             f"(\u226570% of maps in that role). The average duelist takes "
             f"{gdu.first_engagement.mean():.1f} opening duels and dies "
             f"{gdu.deaths.mean():.1f} times;\nthe average non-duelist takes "
             f"{gnd.first_engagement.mean():.1f} and dies {gnd.deaths.mean():.1f}. "
             f"Part-time duelists are excluded from that average \u2014 they take 5.4.",
             fontsize=8, color=INK2, ha="left", va="top", style="italic",
             linespacing=1.5)

    fig.suptitle("One distinction, not four: duelists and everyone else",
                 fontsize=13, color=INK, y=1.075, x=.02, ha="left")
    fig.text(.02, .995, f"{len(D)} player-seasons, 2023\u20132026. A duelist/not flag "
             "explains 53% of aggression; the full four-way role label adds 3 points "
             "more.\nThe curves show the claim: duelists sit right, everyone else spans "
             "the axis, and they overlap heavily.",
             fontsize=8.5, color=INK2, ha="left", linespacing=1.5)
    fig.savefig(f"figures/{PFX}1-style-space.png"); plt.close(fig)


def fig1b_roles(D):
    """Supporting: initiator, controller and sentinel are visually interchangeable."""
    roles = ["duelist", "initiator", "controller", "sentinel"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
    for ax, r in zip(axes, roles):
        s_ = D[D.role == r]
        ax.scatter(D.PC1, D.PC2, s=8, c=GREY, lw=0, zorder=1)
        ax.scatter(s_.PC1, s_.PC2, s=10, c=BLUE, lw=.4, edgecolor=SURFACE, zorder=2)
        ax.set_title(f"{r}  (n={len(s_)})", fontsize=9.5, color=INK, pad=6)
        ax.axvline(0, color=EDGE, lw=.8, zorder=0)
    axes[0].set_ylabel("isolation  \u2192")
    for ax in axes: ax.set_xlabel("aggression  \u2192")
    fig.suptitle("The other three roles are interchangeable",
                 fontsize=12, color=INK, y=1.12, x=.085, ha="left")
    fig.text(.085, 1.03, "Only the duelist panel is shifted. Initiator, controller and "
             "sentinel sit on top of each other.", fontsize=8.5, color=INK2, ha="left")
    fig.savefig(f"figures/{PFX}1b-roles.png"); plt.close(fig)


def fig2_validation(H):
    """Held-out: 2025 position vs 2026 position, model never saw 2026."""
    a = H[H.year == 2025].set_index("player_id"); b = H[H.year == 2026].set_index("player_id")
    j = a.index.intersection(b.index)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for ax, col, name in zip(axes, ["PC1", "PC2"], ["aggression", "isolation / gunplay"]):
        x, y = a.loc[j, col], b.loc[j, col]
        lim = [min(x.min(), y.min()) - .4, max(x.max(), y.max()) + .4]
        ax.plot(lim, lim, color=EDGE, lw=1.2, zorder=1)
        ax.scatter(x, y, s=16, c=BLUE, lw=.5, edgecolor=SURFACE, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("2025"); ax.set_title(f"{name}    r = {x.corr(y):.2f}",
                                            fontsize=10, color=INK, pad=7)
    axes[0].set_ylabel("2026  (never seen by the model)")
    fig.suptitle("Style position holds in a year the model never saw",
                 fontsize=12.5, color=INK, y=1.10, x=.02, ha="left")
    fig.text(.02, 1.01, f"n={len(j)} players. Quality adjustment, scaling and rotation "
             "all fitted on 2023–2025 only.", fontsize=8.5, color=INK2, ha="left")
    fig.savefig(f"figures/{PFX}2-held-out.png"); plt.close(fig)


def fig3_axes(D, raw):
    """What the axes mean, in the game's own units."""
    UN = {"hs": "headshot %", "assists": "assists / map", "plants": "plants / map",
          "clutch_att": "clutches / map", "first_engagement": "opening duels / map",
          "deaths": "deaths / map"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    fig.subplots_adjust(wspace=.62)
    for ax, pc, name in zip(axes, ["PC1", "PC2"], ["aggression", "isolation / gunplay"]):
        q = pd.qcut(D[pc].values, 5, labels=False)
        lo = raw.loc[q == 0, list(UN)].mean(); hi = raw.loc[q == 4, list(UN)].mean()
        rel = ((hi - lo) / raw[list(UN)].std()).sort_values()
        ypos = np.arange(len(rel))
        ax.barh(ypos, rel.values, height=.55,
                color=[ORANGE if v < 0 else BLUE for v in rel.values], lw=0)
        ax.set_yticks(ypos); ax.set_yticklabels([UN[i] for i in rel.index])
        for y, v, k in zip(ypos, rel.values, rel.index):
            ax.text(v + (.07 if v > 0 else -.07), y, f"{lo[k]:.1f} → {hi[k]:.1f}",
                    va="center", ha="left" if v > 0 else "right", fontsize=8, color=INK2)
        ax.axvline(0, color=RULE, lw=1)
        ax.set_xlim(rel.min() - 1.9, rel.max() + 1.9)
        ax.set_xlabel("change from bottom fifth to top fifth (sd)")
        ax.set_title(name, fontsize=10.5, color=INK, pad=7)
        ax.grid(axis="y", visible=False)
    fig.suptitle("What each axis actually measures", fontsize=12.5, color=INK,
                 y=1.11, x=.06, ha="left")
    fig.text(.06, 1.02, "Bar length is standardised so the two panels compare; the label gives the "
             "real per-map value, bottom fifth → top fifth.",
             fontsize=8.5, color=INK2, ha="left")
    fig.savefig(f"figures/{PFX}3-axes.png"); plt.close(fig)


def fig4_kd(D):
    """Entry fragging costs deaths AND earns damage. In K/D they cancel.

    Only aspas is highlighted, in both panels, with a leader line -- the earlier
    version scattered unexplained blue and orange dots across two panels and labelled
    them in one, which told the reader nothing about which dot was whom.
    """
    du = D[(D.role == "duelist") & D.label_ok & (D.maps >= 30)]
    a_ = du[du.handle == "aspas"]
    panels = [("deaths", "deaths per map", "THE COST", "more entries, more deaths"),
              ("adr", "damage per round (ADR)", "THE GAIN", "more entries, more damage")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.subplots_adjust(wspace=.3)
    for ax, (col, ylab, tag, sub) in zip(axes, panels):
        ax.scatter(du.first_engagement, du[col], s=20, c=GREY, lw=0, zorder=1)
        b0, a0 = np.polyfit(du.first_engagement, du[col], 1)
        xs = np.linspace(du.first_engagement.min(), du.first_engagement.max(), 2)
        ax.plot(xs, a0 + b0 * xs, color=RULE, lw=2, zorder=2)
        ax.scatter(a_.first_engagement, a_[col], s=52, c=ORANGE, lw=1.2,
                   edgecolor=SURFACE, zorder=4)
        r = du.first_engagement.corr(du[col])
        ax.set_title(f"{tag}   \u00b7   {sub}   \u00b7   r = {r:+.2f}",
                     fontsize=10, color=INK, pad=9, loc="left")
        ax.set_xlabel("opening duels taken per map  \u2192")
        ax.set_ylabel(ylab)

    # one highlighted player needs a key, not a label with a leader line
    fig.text(.02, -.02, "\u25cf", fontsize=10, color=ORANGE, ha="left", va="top")
    fig.text(.042, -.018, "aspas", fontsize=9, color=INK, weight="bold", ha="left", va="top")
    fig.text(.085, -.018, f"all four seasons, 2023\u20132026 \u00b7 "
             f"{a_.first_engagement.mean():.1f} opening duels and "
             f"{a_.deaths.mean():.1f} deaths per map, against a duelist average of "
             f"{du.first_engagement.mean():.1f} and {du.deaths.mean():.1f}",
             fontsize=8.6, color=INK2, ha="left", va="top")

    fig.suptitle("Entry fragging is K/D-neutral: it costs deaths and earns damage",
                 fontsize=12.5, color=INK, y=1.19, x=.02, ha="left")
    fig.text(.02, 1.08, f"Each grey dot is one duelist-season with 30+ maps "
             f"(n={len(du)}). The two effects pull K/D in opposite directions and "
             f"roughly cancel:\nr(opening duels, K/D) = \u22120.02, so K/D cannot tell "
             f"you whether a duelist takes the first fight.",
             fontsize=8.5, color=INK2, ha="left", linespacing=1.5)
    fig.savefig(f"figures/{PFX}4-kd-hides.png"); plt.close(fig)


if __name__ == "__main__":
    M, L, sc = loadings(F.STYLE)
    D = pd.concat([M[["player_id","handle","year","team","role","main_agent"]], sc], axis=1)
    raw = (M[["player_id","year"]]
             .merge(F.seasons(F.build()), on=["player_id","year"], how="left")
             .reset_index(drop=True))
    H, _, _ = V.fit_and_project(F.STYLE)
    fig2_validation(H); fig3_axes(D, raw)
    from examples import guarded
    G = guarded().merge(F.seasons(F.build()), on=["player_id","year"], how="left",
                        suffixes=("","_r"))  # carries raw per-map stats for the key
    fig1_space(G); fig1b_roles(G); fig4_kd(G)
    print(f"wrote 5 figures, theme={THEME}")
