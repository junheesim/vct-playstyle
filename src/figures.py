"""Figures for the report. One chart, one question.

The static style-space scatter that used to live here was for REPORT.md, which was a
second copy of the report kept as markdown. That copy is gone -- the site's explorer
draws the same 775 points with filtering and hover -- so the static duplicate went
with it.

Palette: validated categorical slots 1 (blue) and 2 (orange) on the light surface.
Small multiples use ONE series color against gray context, so no multi-hue
separation question arises.
"""
import os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import paths
import features as F
from components import loadings

THEME = os.environ.get("FIG_THEME", "light")
PFX = "" if THEME == "light" else "dark-"
if THEME == "dark":
    BLUE, ORANGE = "#3987e5", "#d95926"
    SURFACE, INK, INK2, GRAY = "#181c24", "#f2f4f7", "#a7b1c0", "#39414e"
    GRID, EDGE, RULE = "#252b35", "#3a4350", "#5a6472"
    # text that sits ON a blue/orange fill. The fills are mid-tone in BOTH themes, so
    # this cannot follow the surface: one dark ink clears 4.1:1 on all four fills.
    ONFILL = "#12151a"
else:
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    SURFACE, INK, INK2, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
    GRID, EDGE, RULE = "#ececea", "#c9c8c3", "#9a9994"
    ONFILL = "#12151a"
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": EDGE,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fig1b_roles(D):
    """Supporting: initiator, controller and sentinel are visually interchangeable."""
    roles = ["duelist", "initiator", "controller", "sentinel"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
    for ax, r in zip(axes, roles):
        s_ = D[D.role == r]
        ax.scatter(D.PC1, D.PC2, s=8, c=GRAY, lw=0, zorder=1)
        ax.scatter(s_.PC1, s_.PC2, s=10, c=BLUE, lw=.4, edgecolor=SURFACE, zorder=2)
        ax.set_title(f"{r}  (n={len(s_)})", fontsize=9.5, color=INK, pad=6)
        ax.axvline(0, color=EDGE, lw=.8, zorder=0)
    axes[0].set_ylabel("isolation  \u2192")
    for ax in axes: ax.set_xlabel("aggression  \u2192")
    fig.suptitle("The other three roles are interchangeable",
                 fontsize=12, color=INK, y=1.12, x=.085, ha="left")
    fig.text(.085, 1.03, "Only the duelist panel is shifted. Initiator, controller and "
             "sentinel sit on top of each other.", fontsize=8.5, color=INK2, ha="left")
    fig.savefig(paths.FIGURES / f"{PFX}1b-roles.png"); plt.close(fig)


def fig2_validation(H):
    """Does a player keep their position from one season to the next?

    Was a held-out plot: fitted on 2023-25, 2026 projected cold. That framing went,
    because the test could not fail -- dropping one year of four moves a rotation by
    about .01 against a criterion of .15, and the 2025 half of every pair sat inside
    the training data regardless. This is the same scatter under the model the rest of
    the report uses, and it makes the claim the section actually supports: a player's
    position is a stable property of the player.
    """
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
    axes[0].set_ylabel("2026")
    fig.suptitle("A player lands in much the same place a season later",
                 fontsize=12.5, color=INK, y=1.10, x=.02, ha="left")
    fig.text(.02, 1.01, f"One dot is one of the {len(j)} players who appear in both seasons. "
             "The line is where a player who did not move at all would sit.",
             fontsize=8.5, color=INK2, ha="left")
    fig.savefig(paths.FIGURES / f"{PFX}2-held-out.png"); plt.close(fig)


def fig3_axes(D, raw):
    """What each axis is a collective measure OF -- its recipe, not its consequences.

    One bar per axis, split into the six behaviors by how much each contributes to
    it. Everything to the right of the center line pushes the axis up; everything to
    the left pushes it down; the segments of one bar sum to 100%.

    Earlier versions plotted what high-scoring players DO differently, in standard
    deviations and then in percent. Both answered "what follows from being high on
    this axis" when the question is "what is this axis made of". A part-to-whole
    question wants a part-to-whole chart.
    """
    _, L, _ = loadings(F.STYLE)
    UN = F.UNITS
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    deep = 0                                     # alternate the depth of outside labels

    for i, (pc, name) in enumerate([("PC1", "aggression"), ("PC2", "isolation / gunplay")]):
        w = L[pc]
        # SQUARED weights, which is the standard contribution of a variable to a
        # component and the only version for which "the blocks add up to the whole
        # axis" is literally true: a component's weight vector has unit length, so
        # the squares sum to 1 by construction. Dividing |w| by the sum of |w| also
        # yields something that sums to 100, but only because it was made to.
        share = w**2 / (w**2).sum() * 100
        y = 1 - i
        for sign, color in ((+1, BLUE), (-1, ORANGE)):
            part = share[np.sign(w) == sign].sort_values(ascending=False)
            edge = 0.0                           # distance from the center line so far
            for k, v in part.items():
                # both sides stack OUTWARD from 0: left = -(edge+v) going left,
                # left = edge going right. Always a positive width.
                ax.barh(y, v, left=(edge if sign > 0 else -(edge + v)), height=.52,
                        color=color, lw=1.4, edgecolor=SURFACE, zorder=2)
                mid = sign * (edge + v / 2)
                if v >= 8:                       # fits inside the segment
                    ax.text(mid, y + .085, UN[k].split(" /")[0], ha="center", va="center",
                            fontsize=8.6, color=ONFILL, zorder=3)
                    ax.text(mid, y - .095, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8.6, color=ONFILL, zorder=3, fontweight="bold")
                else:                            # too thin -- label it outside
                    d = .50 + .20 * (deep % 2)   # stagger, or two thin blocks collide
                    deep += 1
                    ax.annotate(f"{UN[k].split(' /')[0]} {v:.0f}%",
                                xy=(mid, y - .27), xytext=(mid, y - d),
                                ha="center", va="top", fontsize=8, color=INK2,
                                arrowprops=dict(arrowstyle="-", color=EDGE, lw=.8,
                                                shrinkA=0, shrinkB=2))
                edge += v

    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["aggression", "isolation /\ngunplay"], fontsize=11, color=INK)
    ax.set_xlim(-62, 78); ax.set_ylim(-.95, 1.55)
    ax.set_xticks([]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(31, 1.48, "these push the axis UP", ha="center", fontsize=8.6, color=BLUE)
    ax.text(-31, 1.48, "these push it DOWN", ha="center", fontsize=8.6, color=ORANGE)
    fig.suptitle("What each axis is made of", fontsize=12.5, color=INK, y=1.12, x=.01, ha="left")
    fig.text(.01, 1.04, "Each axis is one number combining all six behaviors. The width of a "
             "block is how much of that axis the behavior\naccounts for; the six blocks of a bar "
             "add up to the whole axis.", fontsize=8.5, color=INK2, ha="left", va="top",
             linespacing=1.5)
    fig.savefig(paths.FIGURES / f"{PFX}3-axes.png", bbox_inches="tight"); plt.close(fig)


def fig4_kd(D):
    """Does entering less come with a better kill-death ratio? One panel, not three.

    It used to be three scatters -- deaths, damage and K/D against opening duels --
    which illustrated the cancelling mechanism but never tested the accusation. The
    accusation is about one player, so the figure now shows the comparison that
    answers it: every duelist, the ones who enter even less than aspas picked out, and
    aspas himself. If avoiding the first fight were what produced his ratio, the cloud
    would slope down and he would sit on that slope.

    aspas is ONE point, his four-season average, because the comparison group is
    defined against that average. Plotting his four seasons separately put three of
    them on the wrong side of the line that was drawn to separate them.
    """
    du = D[(D.role == "duelist") & D.label_ok]
    a = D[D.handle == "aspas"]
    fe_a, kd_a = a.first_engagement.mean(), a.kd_ratio.mean()
    low = du[du.first_engagement < fe_a]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvspan(du.first_engagement.min() - .45, fe_a, color=ORANGE, alpha=.06, zorder=0)
    ax.scatter(du.first_engagement, du.kd_ratio, s=26, c=GRAY, lw=.5,
               edgecolor=SURFACE, zorder=2, label=f"{len(du)} duelist-seasons")
    ax.scatter(low.first_engagement, low.kd_ratio, s=26, c=ORANGE, lw=.5,
               edgecolor=SURFACE, zorder=3, label=f"the {len(low)} who enter less than aspas")
    ax.scatter([fe_a], [kd_a], s=78, c=BLUE, lw=1.3, edgecolor=SURFACE, zorder=5,
               label="aspas, four seasons averaged")

    ax.axhline(du.kd_ratio.mean(), color=RULE, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(du.first_engagement.max() + .1, du.kd_ratio.mean() - .012,
            f"all duelists  {du.kd_ratio.mean():.2f}", ha="right", va="top",
            fontsize=8.2, color=INK2, zorder=6)
    # the orange line is identified in the legend rather than by floating text, and
    # aspas needs no annotation -- the legend already says what the blue point is.
    ax.plot([du.first_engagement.min() - .45, fe_a], [low.kd_ratio.mean()] * 2,
            color=ORANGE, lw=1.4, ls=(0, (4, 3)), zorder=4,
            label=f"their average K/D, {low.kd_ratio.mean():.2f}")

    ax.set_xlabel("opening duels taken per map")
    ax.set_ylabel("kills \u00f7 deaths")
    ax.set_xlim(du.first_engagement.min() - .45, du.first_engagement.max() + .35)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.6, handletextpad=.4,
              borderaxespad=.4, labelcolor=INK2)
    fig.suptitle("Entering less does not come with a better kill-death ratio",
                 fontsize=12.5, color=INK, y=1.10, x=.015, ha="left")
    fig.text(.015, 1.035, "If avoiding the first fight were what produced a high ratio, this "
             "cloud would slope downwards and aspas would sit\non that slope. It does not, and "
             "he does not.", fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.savefig(paths.FIGURES / f"{PFX}4-kd-hides.png", bbox_inches="tight"); plt.close(fig)


def fig5_archetypes(D):
    """What three real types would look like, next to what the data looks like.

    The section used to argue this with a table of separation values -- 0.37 for the
    real players against 0.32 for a cloud with nothing in it and 0.53 for three
    genuine types. All correct, and unreadable: nobody has intuitions about a
    silhouette score. The shape of the data settles it without a number, so the
    number becomes a caption on a picture instead of the argument itself.

    Same axes, same n, same spread in all three panels. Only the grouping differs.
    """
    from clusters import null_like, separation
    X = D[["PC1", "PC2"]].values
    rng = np.random.default_rng(0)
    sd = float(X.std(0).mean())
    d = 2.80                                  # the duelist-to-initiator gap, decision 12
    cen = np.array([[-d*sd, 0.0], [0.0, d*sd*.87], [d*sd, 0.0]])
    lab = rng.integers(0, 3, len(X))
    types = cen[lab] + rng.normal(scale=sd, size=(len(X), 2))
    cloud = null_like(X, seed=0)

    panels = [(cloud, "simulated: no grouping", "one draw from the Monte Carlo null"),
              (X,     "the real players",       "775 player-seasons"),
              (types, "simulated: three types",  "as far apart as duelists are from initiators")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    fig.subplots_adjust(top=.80, left=.02, right=.99, wspace=.10)
    for ax, (P, name, sub) in zip(axes, panels):
        # measure on the data as it is; standardize only for DRAWING, so the three
        # panels share a frame and the numbers still match the ones in the prose.
        sep = separation(P, 3)
        Q = (P - P.mean(0)) / P.std(0)
        ax.scatter(Q[:, 0], Q[:, 1], s=9, c=BLUE if name.startswith("the real") else GRAY,
                   lw=0, alpha=.75, zorder=2)
        ax.set_title(name, fontsize=10.5, color=INK, pad=24)
        ax.text(.5, 1.028, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.2, color=INK2)
        ax.text(.5, -.09, f"separation {sep:.2f}", transform=ax.transAxes,
                ha="center", fontsize=9.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(EDGE)
    fig.suptitle("What three real types would look like", fontsize=12.5, color=INK,
                 y=1.22, x=.012, ha="left")
    fig.text(.012, 1.14, "The data sits between two simulations. The number under each panel is "
             "the same measurement, and it is the evidence.", fontsize=8.5, color=INK2,
             ha="left", va="top")
    fig.savefig(paths.FIGURES / f"{PFX}5-archetypes.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    M, L, sc = loadings(F.STYLE)
    D = pd.concat([M[["player_id","handle","year","team","role","main_agent"]], sc], axis=1)
    raw = (M[["player_id","year"]]
             .merge(F.seasons(F.build()), on=["player_id","year"], how="left")
             .reset_index(drop=True))
    fig2_validation(D); fig3_axes(D, raw)
    from examples import guarded
    G = guarded().merge(F.seasons(F.build()), on=["player_id","year"], how="left",
                        suffixes=("","_r"))  # carries raw per-map stats for the key
    fig1b_roles(G); fig4_kd(G); fig5_archetypes(D)
    print(f"wrote 4 figures, theme={THEME}")
