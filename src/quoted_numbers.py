"""Every number the prose quotes, printed from the data.

Hand-copied figures drifted repeatedly while this project was being built -- a career
mean pasted into a single-season label, a percentile computed on a different
population than the claim implied. This is the single source of truth: run it, and
update the prose from its output rather than from memory.

    python src/quoted_numbers.py report     what the site's report quotes
    python src/quoted_numbers.py decisions  what decisions/ 07, 10, 11, 12 and 18 quote

(Named quoted_numbers, not numbers: a module called `numbers` shadows the stdlib one
that numpy imports, and the whole package fails to load.)
"""
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import data as D
import features as F
import pca
import style as St
import validate as V
from components import N_DRAWS, eigenvalues, loadings, null_cutoffs, replication
from examples import guarded
from roles import ROLES, agent_retention, shares


def section(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


def sub(t):
    print(f"\n-- {t}")


def fit(cols=None):
    cols = cols or F.STYLE
    M = St.build(cols)
    Z = pca.standardize(M, cols)
    model, flip = pca.fit(Z.values, cols)
    return M, pca.loadings(model, flip, cols), pca.scores(model, flip, Z, M.index)


def yoy(D_, y0, y1, col):
    a = D_[D_.year == y0].set_index("player_id")[col]
    b = D_[D_.year == y1].set_index("player_id")[col]
    j = a.index.intersection(b.index)
    return a.loc[j].corr(b.loc[j]), len(j)


@lru_cache(maxsize=None)          # 200 null clouds twice over is ~30s; both
@lru_cache(maxsize=None)
def archetype_scale(n_draws=200) -> dict:
    """What 0.369 means, by putting a floor AND a ceiling either side of it.

    The section used to compare the real separation only against a structureless
    cloud. Beating that establishes "not a featureless blob", which is not the claim
    being made -- so the comparison could not prove there are no types. The missing
    half is an upper reference: what the same measurement returns on data that really
    IS three types.

    The ceiling is anchored, not invented. Its groups are placed as far apart as
    duelists are from initiators on aggression (Cohen's d = 2.80, from `role_matrix`),
    so it asks: if there were three archetypes as distinct as the one distinction this
    report has already established, what would this number look like?
    """
    from clusters import null_like, separation
    M, _, sc = loadings(F.STYLE)
    X = sc[["PC1", "PC2"]].values
    d_ref = role_matrix()["d"][("duelist", "initiator")]
    rng = np.random.default_rng(0)
    sd = float(X.std(0).mean())

    def three_types(d, seed):
        r = np.random.default_rng(seed)
        cen = np.array([[-d*sd, 0.0], [0.0, d*sd*0.87], [d*sd, 0.0]])
        lab = r.integers(0, 3, len(X))
        return separation(cen[lab] + r.normal(scale=sd, size=(len(X), 2)), 3)

    floor = [separation(null_like(X, seed=i), 3) for i in range(n_draws)]
    ceil  = [three_types(d_ref, i) for i in range(30)]
    return dict(real=separation(X, 3), d_ref=d_ref,
                floor=float(np.mean(floor)), floor_sd=float(np.std(floor, ddof=1)),
                ceil=float(np.mean(ceil)),  ceil_sd=float(np.std(ceil, ddof=1)),
                mid={d: float(np.mean([three_types(d, i) for i in range(30)]))
                     for d in (1.0, 2.0)})


@lru_cache(maxsize=None)
def role_specialists(n_draws=100) -> dict:
    """Does isolation survive among players who stay in one role -- and is the sample
    size what decides it?

    Restricting to role specialists is the friendliest test available: a player who
    splits a season between two jobs has a season average describing neither, so a
    real dimension should be CLEAREST where the measurement is cleanest. Isolation
    instead stops clearing its chance cutoff there.

    The obvious rebuttal is that a third of the rows went with the restriction, and a
    marginal result weakens in any smaller sample. So the same test runs on random
    subsets of the same size, with no role restriction, as the control.
    """
    from roles import ROLES, shares, with_role
    d = F.build()
    ps = with_role(F.seasons(d), d)
    R = St.residualize(ps, F.STYLE)
    M = pd.concat([ps[["player_id", "year"]].reset_index(drop=True),
                   R.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
    P = shares(); P["conc"] = P[ROLES].max(axis=1)
    M = M.reset_index(drop=True).merge(P[["player_id", "year", "conc"]],
                                       on=["player_id", "year"], how="left")
    spec = M[M.conc >= .70]

    def check(X, n):
        ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle", n=n)
        return int((ev > cut).sum()), float(ev[1] - cut[1])

    k_all, m_all = check(pca.standardize(M, F.STYLE).values, N_DRAWS)
    k_sp,  m_sp  = check(pca.standardize(spec, F.STYLE).values, N_DRAWS)
    rng = np.random.default_rng(0)
    ctrl = [check(pca.standardize(M.iloc[rng.choice(len(M), len(spec), replace=False)],
                                  F.STYLE).values, 200) for _ in range(n_draws)]
    return dict(n_all=len(M), n_spec=len(spec), kept_all=k_all, kept_spec=k_sp,
                margin_all=m_all, margin_spec=m_sp, n_draws=n_draws,
                ctrl_kept2=sum(k >= 2 for k, _ in ctrl),
                ctrl_positive=sum(m > 0 for _, m in ctrl),
                ctrl_min=min(m for _, m in ctrl))


def archetypes(n_sil=200, n_ari=40, reps=8) -> dict:   # `report` and `site_claims` want it
    """Are there discrete player types, or one continuous cloud?

    Two questions, each against clouds with the same shape and spread as the real
    data and no groups in them -- and each against MANY such clouds, not one.

      separation   is there empty space between the groups?  (silhouette)
      replication  do two halves of the players cut in the same place?  (adjusted
                   Rand index)

    The two answer differently here, and that difference is the finding: a cloud
    with a consistent shape can be cut in the same place twice without there being
    any gap to cut along.
    """
    from clusters import null_band, replication, separation
    M, _, sc = loadings(F.STYLE)
    XY = sc[["PC1", "PC2"]].values
    pid = M.player_id.values
    out = {"k": {}}
    for k in (2, 3, 4):
        sil = separation(XY, k)
        ar = replication(XY, k, pid, "kmeans", n=20)
        out["k"][k] = dict(sil=sil, sil_null=null_band(XY, k, separation, n=n_sil),
                           ari=ar, ari_null=null_band(XY, k, replication, n=n_ari,
                                                      pid=pid, model="kmeans", n_=reps))
    km = KMeans(3, n_init=20, random_state=0).fit(XY)
    near = np.sort(km.transform(XY), axis=1)
    ratio = near[:, 0] / near[:, 1]
    out["boundary_n"] = int((ratio >= .75).sum())
    out["boundary_share"] = float((ratio >= .75).mean())
    return out


def entry_comparison() -> dict:
    """Does entering less come with a better kill-death ratio? The like-for-like test.

    The section used to answer this with a correlation across all duelists, which
    invited a fair objection: a slope fitted over everyone says little about one
    player at the low end of it. So instead, find the duelists who enter LESS than
    aspas does and look at what their ratio actually is. He is at the 15th percentile
    of a populated range, not off the end of it, so that comparison group exists.
    """
    G = guarded().merge(F.seasons(F.build()), on=["player_id", "year"],
                        how="left", suffixes=("", "_r"))
    gdu = G[(G.role == "duelist") & G.label_ok]
    a = G[G.handle == "aspas"]
    fe_a = float(a.first_engagement.mean())
    low = gdu[gdu.first_engagement < fe_a]
    m = sm.OLS(gdu.kd_ratio, sm.add_constant(gdu.first_engagement)).fit()
    pred = m.params["const"] + m.params["first_engagement"] * fe_a
    resid = gdu.kd_ratio - m.predict(sm.add_constant(gdu.first_engagement))
    return dict(n_gdu=len(gdu), n_low=len(low), fe_aspas=fe_a,
                fe_min=float(gdu.first_engagement.min()),
                fe_max=float(gdu.first_engagement.max()),
                kd_low=float(low.kd_ratio.mean()), kd_all=float(gdu.kd_ratio.mean()),
                kd_aspas=float(a.kd_ratio.mean()), kd_pred=float(pred),
                resid=float(a.kd_ratio.mean() - pred),
                resid_pct=float((resid < a.kd_ratio.mean() - pred).mean()))


def case_study() -> dict:
    """Every number the aspas case study quotes, on the 149 solidly-labeled duelists.

    Includes both style axes, which the section did not use until 2026-09-21. The
    accusation is that he avoids the round's first fight to protect his own numbers,
    and the report had only tested the second half of that (whether avoiding entries
    raises K/D). Aggression tests the first half. Isolation and clutch rate test the
    part about playing away from teammates, which is what "baiting" actually claims.
    """
    G = guarded().merge(F.seasons(F.build()), on=["player_id", "year"],
                        how="left", suffixes=("", "_r"))
    gdu = G[(G.role == "duelist") & G.label_ok]
    a = G[G.handle == "aspas"]
    pct = lambda col: (gdu[col] < a[col].mean()).mean()

    # Being last alive requires being alive. A duelist who dies rarely gets more
    # clutch situations without hanging back, so the raw rate overstates the case.
    X = sm.add_constant(gdu[["deaths"]])
    m = sm.OLS(gdu.clutch_att, X).fit()
    resid = gdu.clutch_att - m.predict(X)
    a_res = a.clutch_att.mean() - (m.params["const"] + m.params["deaths"]*a.deaths.mean())

    prx = gdu[gdu.team == "Paper Rex"]
    return dict(
        n_gdu=len(gdu), seasons=len(a), teams=a.team.nunique(),
        rows=[("aggression (PC1)", "PC1"), ("isolation (PC2)", "PC2"),
              ("opening duels / map", "first_engagement"),
              ("clutch situations / map", "clutch_att"), ("assists / map", "assists"),
              ("deaths / map", "deaths"), ("K/D", "kd_ratio"),
              ("damage / round", "adr")],
        val={c: a[c].mean() for _, c in [("", "PC1"), ("", "PC2"), ("", "first_engagement"),
             ("", "clutch_att"), ("", "assists"), ("", "deaths"), ("", "kd_ratio"), ("", "adr")]},
        avg={c: gdu[c].mean() for c in ["PC1", "PC2", "first_engagement", "clutch_att",
             "assists", "deaths", "kd_ratio", "adr"]},
        pct={c: pct(c) for c in ["PC1", "PC2", "first_engagement", "clutch_att",
             "assists", "deaths", "kd_ratio", "adr"]},
        r_deaths=gdu.first_engagement.corr(gdu.deaths),
        r_adr=gdu.first_engagement.corr(gdu.adr),
        r_kd=gdu.first_engagement.corr(gdu.kd_ratio),
        r_clutch_deaths=gdu.clutch_att.corr(gdu.deaths),
        clutch_deaths_r2=m.rsquared,
        clutch_adj_pct=(resid < a_res).mean(),
        prx_n=len(prx), prx_fe=prx.first_engagement.mean(), prx_pc2=prx.PC2.mean(),
        gdu_fe=gdu.first_engagement.mean(), gdu_pc2=gdu.PC2.mean())


@lru_cache(maxsize=None)
def refit_2026() -> dict:
    """Refit the whole thing on 2026 alone and ask whether the same axes come back.

    This replaced the two-row "held-out 2026" check, which compared a model fitted on
    2023-25 against one fitted on all four years. That test could not fail: dropping
    one year of four moves a PCA rotation by about .01, and the criterion set for it
    was .15. It also correlated a 2025 score -- inside the training data either way --
    against a 2026 one, on the same 130 people, so nothing new was being predicted.

    Fitting on 2026 by itself CAN fail, and isolation does. The cost is a small
    sample, so the size control is reported beside it.
    """
    from components import N_DRAWS
    from roles import with_role
    d = F.build()
    ps = with_role(F.seasons(d), d)

    def fit(years):
        sub = ps[ps.year.isin(years)]
        R = St.residualize(sub, F.STYLE).dropna()
        Z = pca.standardize(R, F.STYLE)
        m, flip = pca.fit(Z.values, F.STYLE)
        ev, cut = eigenvalues(Z.values), null_cutoffs(Z.values, "shuffle", n=N_DRAWS)
        return pca.loadings(m, flip, F.STYLE), int((ev > cut).sum()), len(Z)

    Ltr, k_tr, n_tr = fit([2023, 2024, 2025])
    L26, k_26, n_26 = fit([2026])
    cong = {c: abs(float(np.dot(Ltr[c], L26[c]) /
            (np.linalg.norm(Ltr[c]) * np.linalg.norm(L26[c])))) for c in ("PC1", "PC2")}

    # could the smaller sample alone explain it? same size, drawn from the other years.
    R = St.residualize(ps[ps.year.isin([2023, 2024, 2025])], F.STYLE).dropna()
    rng = np.random.default_rng(0)
    keeps = []
    for _ in range(100):
        Z = pca.standardize(R.iloc[rng.choice(len(R), n_26, replace=False)], F.STYLE).values
        keeps.append(int((eigenvalues(Z) > null_cutoffs(Z, "shuffle", n=200)).sum()))
    return dict(n_train=n_tr, n_2026=n_26, kept_train=k_tr, kept_2026=k_26,
                cong_pc1=cong["PC1"], cong_pc2=cong["PC2"],
                ctrl_kept2=sum(k >= 2 for k in keeps), ctrl_n=len(keeps))


def persistence_anchors() -> list:
    """How much each measure of a player persists from 2025 to 2026, same players.

    The report quotes 0.72 for aggression. On its own that number means nothing --
    a reader has no scale to judge it against. These are the scale: the same
    correlation, over the same two seasons and the same 130 players, for things a
    reader already has intuitions about.
    """
    M, _, sc = loadings(F.STYLE)
    P = pd.concat([M[["player_id", "year"]], sc], axis=1).merge(
        F.seasons(F.build()), on=["player_id", "year"])

    def yoy_(c):
        a = P[P.year == 2025].set_index("player_id")[c]
        b = P[P.year == 2026].set_index("player_id")[c]
        j = a.index.intersection(b.index)
        return a.loc[j].corr(b.loc[j])

    rows = [("headshot rate", "hs"), ("combat score (ACS)", "acs"),
            ("aggression", "PC1"), ("isolation", "PC2"),
            ("kills \u00f7 deaths", "kd_ratio"), ("their team\u2019s win rate", "team_win")]
    out = [(lab, yoy_(c), c in ("PC1", "PC2")) for lab, c in rows]
    return sorted(out, key=lambda r: -r[1])


def role_matrix() -> dict:
    """Cohen's d between EVERY pair of roles on aggression, not just adjacent ones.

    A chain of "gap to the role below" gives three of the six comparisons and hides
    the largest -- duelist against initiator -- so it could not show how far apart
    the ends of the taxonomy are.
    """
    M, _, sc = loadings(F.STYLE)
    P = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    order = P.groupby("role").PC1.mean().sort_values(ascending=False).index.tolist()

    def d(a, b):
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

    return dict(order=order,
                n={r: int((P.role == r).sum()) for r in order},
                mean={r: float(P[P.role == r].PC1.mean()) for r in order},
                d={(a, b): float(d(P[P.role == a].PC1, P[P.role == b].PC1))
                   for i, a in enumerate(order) for b in order[i+1:]})


def role_ladder() -> dict:
    """Where each role sits on aggression, and how far apart consecutive roles are.

    Replaces "share of each role above the overall median", which could not carry the
    claim it was used for. Duelists are 26% of the population and 97% of them are
    above the median, so they fill 198 of the 387 above-median places -- which FORCES
    189 of them onto non-duelists, 33% of that group, before any behavior is
    measured. The observed figure was 33%. The statistic was arithmetic, not evidence.

    Cohen's d has no such floor: it is the gap between two groups measured in their
    own standard deviations, so it says how much the groups actually overlap.
    """
    M, _, sc = loadings(F.STYLE)
    P = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    order = P.groupby("role").PC1.mean().sort_values(ascending=False).index.tolist()

    def d(a, b):
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

    out = {}
    for i, r in enumerate(order):
        s_ = P[P.role == r].PC1
        below = P[P.role == order[i + 1]].PC1 if i + 1 < len(order) else None
        out[r] = dict(n=len(s_), mean=s_.mean(),
                      d_below=None if below is None else d(s_, below))
    du, nd = P[P.role == "duelist"].PC1, P[P.role != "duelist"].PC1
    out["_sep"] = dict(d_sen_con=d(P[P.role == "sentinel"].PC1, P[P.role == "controller"].PC1),
                       d_sen_ini=d(P[P.role == "sentinel"].PC1, P[P.role == "initiator"].PC1),
                       du_below_nd=(du < nd.median()).mean())
    return out


SCREEN = ["first_engagement", "hs", "assists", "plants", "creds_per_round",
          "clutch_att", "deaths", "kast", "defuses", "first_engagement_gap"]


def screen_table() -> dict:
    """The feature screen the report's "What was measured, and why" table prints.

    Reliability and the quality correlation, per candidate, on the 887 player-seasons
    that clear the map floor -- the population the screen was run on (decision 04),
    not the 775 with every feature complete. The dropped candidates are included:
    `deaths` is the one that passes both tests and was dropped anyway, on the
    principle, and that is the row worth showing.
    """
    d = F.build()
    ps = F.seasons(d)
    out = {}
    for c in SCREEN:
        rel, _, _ = F.reliability(d, c, ps)
        out[c] = (rel, ps[c].corr(ps[F.QUALITY]))
    return out


def _ordinal(n: int) -> str:
    """15 -> '15th'. 11-13 are 'th' whatever their last digit says."""
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def site_claims() -> dict:
    """Every pipeline-derived number the site states, formatted exactly as the page
    prints it.

    `tests/test_site.py` asserts each of these strings appears in `site/`. That is the
    guard this project was missing: the report shipped with an agent-retention figure
    whose two halves were reversed, a clutch median that was really the defuse median,
    and an explorer landing line quoting a superseded R2 -- three hand-copied numbers,
    in the one project whose whole claim is that every number traces to a record.

    A number goes in here if the pipeline produces it. Numbers the prose reasons WITH
    rather than reports (a threshold, a count of decision files) do not belong.
    """
    M, L, sc = loadings(F.STYLE)
    raw_ov = D.overview()
    P = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    P["is_du"] = (P.role == "duelist").astype(int)
    nd = P[P.role != "duelist"]
    G = guarded().merge(F.seasons(F.build()), on=["player_id", "year"],
                        how="left", suffixes=("", "_r"))
    X = pca.standardize(M, F.STYLE).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    med_rep, _ = replication(X, M.player_id.values)
    H, _, _ = V.fit_and_project(F.STYLE)
    IN = pd.concat([M[["player_id", "year"]], sc], axis=1)
    med_all = P.PC1.median()
    zero = G[G.duelist == 0]
    r4 = smf.ols("PC1 ~ C(role)", P).fit().rsquared
    rb = smf.ols("PC1 ~ is_du", P).fit().rsquared
    rn = smf.ols("PC1 ~ C(role)", nd).fit().rsquared
    rz = smf.ols("PC1 ~ C(role)", zero).fit().rsquared
    gdu = G[(G.role == "duelist") & G.label_ok]
    claims = c = {}   # two names, one dict: `c` reads better inline, `claims` in the loop

    # -- scope
    c["n_seasons"]   = f"{len(M)}"
    c["n_players"]   = f"{M.player_id.nunique()}"
    c["n_matches"]   = f"{raw_ov.groupby(D.MATCH).ngroups:,}"
    # the phrase, not the bare word: "seven" alone would match a stray sentence and
    # pass while the report described a different number of behaviors.
    c["n_features"]  = {5: "five", 6: "six", 7: "seven", 8: "eight"}[len(F.STYLE)] + " behaviors"
    c["n_guarded"]   = f"{int(G.label_ok.sum())}"

    # -- components
    for i, pc in enumerate(["pc1", "pc2", "pc3"]):
        c[f"{pc}_eigen"]  = f"{ev[i]:.3f}"
        c[f"{pc}_cutoff"] = f"{cut[i]:.3f}"
        c[f"{pc}_margin"] = f"{ev[i]-cut[i]:+.3f}".replace("+", "").replace("-", "−")
        c[f"{pc}_repl"]   = f"{med_rep[i]:.3f}"
    c["pc1_var"] = f"{ev[0]/len(ev):.1%}"
    c["pc2_var"] = f"{ev[1]/len(ev):.1%}"
    c["pc3_var"] = f"{ev[2]/len(ev):.1%}"

    # -- role recovery
    c["r2_role"]      = f"{r4:.3f}"
    c["r2_flag"]      = f"{rb:.3f}"
    c["r2_nondu"]     = f"{rn:.3f}"
    c["r2_nondu_pct"] = f"{rn:.0%}"
    c["n_nondu"]      = f"{len(nd)}"
    c["r2_zerodu"]    = f"{rz:.3f}"
    c["n_zerodu"]     = f"{len(zero)}"
    # `taxonomy_adds`, `n_zerodu_above` and the four `above_median_*` shares were
    # claims until 2026-09-21. The report stopped making them: the share of a role
    # above the overall median cannot carry a claim about overlap, because duelists
    # are a minority and so a fixed number of above-median places FALL to the other
    # roles whatever they do. See `role_ladder`.
    # Every pair of roles, not just adjacent ones: a chain of "gap to the role below"
    # gave three of the six comparisons and hid the largest, duelist vs initiator.
    R = role_matrix()
    for r in R["order"]:
        c[f"role_n_{r}"]    = f">{R['n'][r]}<"
        c[f"role_mean_{r}"] = f"{R['mean'][r]:+.2f}".replace("-", "\u2212")
    for (a, b), v in R["d"].items():
        c[f"d_{a}_{b}"] = f">{v:.2f}<"
    sep = role_ladder()["_sep"]
    c["duelists_below_nd_median"] = f">{sep['du_below_nd']:.1%}<"

    # -- persistence. The two-row "held out 2026" comparison was a claim until
    # 2026-09-21; it could not fail (see `refit_2026`) and the report dropped it.
    for col in ("PC1", "PC2"):
        i_, n = V.yoy(IN, 2025, 2026, col)
        c[f"insample_{col.lower()}"] = f"{i_:.3f}"
        c["n_heldout_pairs"] = f"{n}"
    rf = refit_2026()
    c["refit_n"]       = f"{rf['n_2026']} player-seasons"
    c["refit_cong"]    = f"{rf['cong_pc1']:.3f}"
    c["refit_ctrl"]    = f"two dimensions {rf['ctrl_kept2']} times"
    # `cost_pc1` / `cost_pc2` were claims until 2026-09-21. The report stopped
    # printing the difference between the two rows, because a third number that is
    # only the subtraction of the two above it earns nothing -- and "cost" named a
    # quantity without saying what it was the cost of.
    for lab, r, _own in persistence_anchors():
        key = lab.split()[0].strip("\u2019s").lower()
        c[f"persist_{key}"] = f">{r:.3f}<"

    # -- case study
    c["n_gdu"] = f"{len(gdu)}"
    c["n_du"]  = f"{len(G[G.role=='duelist'])}"
    c["gdu_fe"] = f"{gdu.first_engagement.mean():.2f}"
    c["ungdu_fe"] = f"{G[(G.role=='duelist') & ~G.label_ok].first_engagement.mean():.2f}"
    for lab, col in [("deaths", "deaths"), ("adr", "adr"), ("kd", "kd_ratio")]:
        c[f"du_r_{lab}"] = f"{gdu.first_engagement.corr(gdu[col]):+.2f}".replace("+", "+")
    a_ = G[G.handle == "aspas"]
    c["aspas_seasons"] = f"{len(a_)}"
    c["aspas_fe"] = f"{a_.first_engagement.mean():.2f}"
    for lab, col in [("fe","first_engagement"), ("deaths","deaths"),
                     ("kd","kd_ratio"), ("adr","adr")]:
        # as an ORDINAL, which is how the page writes it. "15%" would also match the
        # page by accident, somewhere else entirely; "15th" can only match the cell
        # that means it.
        c[f"aspas_pct_{lab}"] = _ordinal(round((gdu[col] < a_[col].mean()).mean() * 100))
    c["aspas_kd"]  = f"{a_.kd_ratio.mean():.2f}"
    c["aspas_adr"] = f"{a_.adr.mean():.1f}"
    c["aspas_deaths"] = f"{a_.deaths.mean():.2f}"

    # -- the feature screen
    #
    # Matched with the surrounding cell delimiters. These values are two or three
    # characters long, so a bare ".60" would match inside "0.605" elsewhere on the
    # page and pass while saying nothing. ">.60<" can only match the cell that means it.
    # Only the behaviors that were KEPT. The report's screen table used to carry the
    # four dropped candidates as extra rows; they made it the widest table on the page
    # and Method explains what happened to each anyway, so the rows went and the
    # claims went with them. `screen_table()` still computes all ten.
    for feat, (rel, r) in ((f, screen_table()[f]) for f in F.STYLE):
        c_ = feat.replace("first_engagement_gap", "fegap").replace("first_engagement", "fe")
        c_ = c_.replace("creds_per_round", "creds").replace("clutch_att", "clutch")
        claims[f"screen_rel_{c_}"] = f">{rel:.2f}<".replace(">0.", ">.")
        claims[f"screen_q_{c_}"]   = f">{r:+.2f}<".replace("+0.", "+.").replace("-0.", "\u2212.")

    # -- the role-specialist stress test
    rs = role_specialists()
    c["spec_n"]        = f">{rs['n_spec']}<"
    c["spec_margin"]   = f"\u2212{abs(rs['margin_spec']):.3f}"
    c["spec_ctrl_n"]   = f">{rs['n_draws']}</b> random"
    c["spec_ctrl_kept"] = f"in {rs['ctrl_kept2']} of them"

    # -- archetypes
    A = archetypes()
    # The section reports three groups only: the k=2 and k=4 rows were dropped when it
    # was restructured around a floor-and-ceiling scale rather than a table per k.
    v = A["k"][3]
    c["sil_3"]      = f"{v['sil']:.2f}"
    c["ari_3"]      = f"({v['ari']:.3f} against"
    c["ari_null_3"] = f"{v['ari_null'][3]:.3f} &plusmn;"
    c["ari_sd_3"]   = f"{v['ari_null'][4]:.3f})"

    S = archetype_scale()
    c["scale_real"]  = f">{S['real']:.3f}</b>"
    c["scale_floor"] = f">{S['floor']:.3f} &plusmn; {S['floor_sd']:.3f}</b>"
    c["scale_ceil"]  = f">{S['ceil']:.3f} &plusmn; {S['ceil_sd']:.3f}</b>"
    c["boundary_n"]     = f">{A['boundary_n']}<"
    c["boundary_share"] = f"{A['boundary_share']:.0%} of them"

    # -- the like-for-like entry comparison, which now leads the case study
    E = entry_comparison()
    # spelled out in the prose, so the claim matches the words, not the digits
    c["entry_n_low"]  = "Twenty-two duelist-seasons"
    c["entry_kd_low"] = f"average {E['kd_low']:.2f}"
    c["entry_kd_all"] = f"{E['kd_all']:.2f} for duelists as a whole"
    c["entry_resid"]  = f"{E['resid']:.2f} above"
    c["entry_fe"]     = f">{E['fe_aspas']:.2f}<"

    # -- method cards
    stay, move = agent_retention()
    c["agent_stay"] = f"{stay:.0%}"
    c["agent_move"] = f"{move:.0%}"
    tot = F.build().groupby(["player_id", "year"]).clutch_att.sum()
    tot = tot[tot.index.isin(pd.MultiIndex.from_frame(M[["player_id", "year"]]))]
    c["clutch_median"] = f"{tot.median():.0f}"

    return c


def report():
    raw = D.overview(); ps_all = D.player_seasons(raw)
    M, L, sc = loadings(F.STYLE)
    G = guarded().merge(F.seasons(F.build()), on=["player_id", "year"],
                        how="left", suffixes=("", "_r"))
    P = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)

    section("SCOPE")
    print(f"  min maps per season                {D.MIN_MAPS}")
    print(f"  player-seasons at that floor       {len(ps_all)}")
    print(f"  ...with all {len(F.STYLE)} features complete    {len(M)}")
    print(f"  distinct players                   {M.player_id.nunique()}")
    print(f"  matches                            {raw.groupby(D.MATCH).ngroups:,}")
    print(f"  years                              {raw.year.min()}-{raw.year.max()}")

    section("COMPONENTS")
    X = pca.standardize(M, F.STYLE).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    med, lo = replication(X, M.player_id.values)
    for i in range(4):
        print(f"  PC{i+1}  eigenvalue {ev[i]:.2f}  variance {ev[i]/len(ev):>6.1%}  "
              f"null cutoff {cut[i]:.2f}  replicates {med[i]:.3f}  "
              f"{'KEEP' if ev[i] > cut[i] else ''}")
    print(f"  retained: {(ev > cut).sum()}")

    section("ROLE RECOVERY")
    P["is_du"] = (P.role == "duelist").astype(int)
    nd = P[P.role != "duelist"]
    print(f"  R2 of PC1 from the 4-way role label   {smf.ols('PC1 ~ C(role)', P).fit().rsquared:.3f}")
    print(f"  R2 from a duelist/not flag            {smf.ols('PC1 ~ is_du', P).fit().rsquared:.3f}")
    print(f"  R2 among the {len(nd)} non-duelists       "
          f"{smf.ols('PC1 ~ C(role)', nd).fit().rsquared:.3f}")
    med_all = P.PC1.median()
    zero = G[(G.duelist == 0)]
    print(f"  never played a duelist agent          {len(zero)}")
    print(f"  ...above the overall aggression median {(zero.PC1 > med_all).sum()} "
          f"({(zero.PC1 > med_all).mean():.0%})")

    section("HELD-OUT VALIDATION")
    H, model, _ = V.fit_and_project(F.STYLE)
    IN = pd.concat([M[["player_id", "year"]], sc], axis=1)
    for col in ("PC1", "PC2"):
        h, n = V.yoy(H, 2025, 2026, col); i, _ = V.yoy(IN, 2025, 2026, col)
        print(f"  {col}  held-out {h:.3f}   in-sample {i:.3f}   n={n}")
    print("  train variance explained: " +
          ", ".join(f"{v:.1%}" for v in model.explained_variance_ratio_))

    section("AXES, IN GAME UNITS (bottom fifth -> top fifth)")
    UN = F.UNITS
    rawv = M[["player_id", "year"]].merge(F.seasons(F.build()), on=["player_id", "year"])
    for pc in ("PC1", "PC2"):
        q = pd.qcut(M[pc].values if pc in M else sc[pc].values, 5, labels=False)
        print(f"  {pc}:")
        for f, lab in UN.items():
            print(f"    {lab:<20}{rawv.loc[q==0, f].mean():>8.2f} -> {rawv.loc[q==4, f].mean():>8.2f}")

    section("DUELIST POPULATION (label guard applied: >=70% of maps in role)")
    du, gdu = G[G.role == "duelist"], G[(G.role == "duelist") & G.label_ok]
    print(f"  duelist-labeled seasons        {len(du)}")
    print(f"  ...passing the label guard      {len(gdu)}")
    print(f"  guarded mean opening duels      {gdu.first_engagement.mean():.2f}")
    print(f"  guarded mean deaths             {gdu.deaths.mean():.2f}")
    print(f"  unguarded mean opening duels    {du[~du.label_ok].first_engagement.mean():.2f}")
    a = G[G.handle == "aspas"]
    print(f"\n  aspas, {len(a)} seasons, mean opening duels {a.first_engagement.mean():.2f}")
    print(f"    percentile among guarded duelists  "
          f"{(gdu.first_engagement < a.first_engagement.mean()).mean():.0%}")
    print(f"    percentile among ALL duelists      "
          f"{(du.first_engagement < a.first_engagement.mean()).mean():.0%}")
    for f in ("first_engagement", "deaths", "kd_ratio", "adr"):
        print(f"    {f:<18}{a[f].mean():>8.2f}   pctile "
              f"{(gdu[f] < a[f].mean()).mean():>4.0%}")

    section("ARCHETYPES")
    A = archetypes()
    print(f"  {'groups':<8}{'separation':>12}{'null band':>22}"
          f"{'repeatability':>16}{'null band':>22}")
    for k, v in A["k"].items():
        # null_band returns (median, p5, p95, mean, sd); the report quotes mean +/- sd.
        _, _, _, sm, ssd = v["sil_null"]
        _, _, _, am, asd = v["ari_null"]
        print(f"  {k:<8}{v['sil']:>12.3f}{f'{sm:.3f} +/- {ssd:.3f}':>22}"
              f"{v['ari']:>16.3f}{f'{am:.3f} +/- {asd:.3f}':>22}")
    print(f"\n  at 3 groups, player-seasons within 75% of a boundary: "
          f"{A['boundary_n']} ({A['boundary_share']:.0%})")
    print("  null bands are 200 structureless clouds: mean and sd across them.")

    S = archetype_scale()
    print(f"\n  the scale the report puts 0.37 on:")
    print(f"    no grouping at all            {S['floor']:.3f} +/- {S['floor_sd']:.3f}")
    print(f"    THE REAL PLAYERS              {S['real']:.3f}")
    print(f"    three types {S['d_ref']:.1f} sd apart      {S['ceil']:.3f} +/- {S['ceil_sd']:.3f}")


def decisions():
    M, L, sc = fit()
    # M's index is gapped (dropna); raw/`P` are rebuilt with a RangeIndex. Pandas
    # aligns on LABELS, so mixing them silently correlates the wrong rows. One index.
    M = M.reset_index(drop=True)
    sc = sc.reset_index(drop=True)
    P = pd.concat([M[["player_id", "year", "role", "handle", "main_agent"]], sc], axis=1)
    raw = M[["player_id", "year"]].merge(F.seasons(F.build()), on=["player_id", "year"])

    section("DECISION 07 -- scope of each feature set")
    print(f"  {len(F.STYLE)} behaviors: complete player-seasons {len(St.build(F.STYLE)):>5}")
    print(f"  floor in use: {D.MIN_MAPS} maps  (decision 13)")

    section("DECISION 11 -- how many components")
    X = pca.standardize(M, F.STYLE).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    nrm = null_cutoffs(X, "normal")
    med, lo = replication(X, M.player_id.values)
    print(f"  {'':<6}{'eigenvalue':>12}{'var':>8}{'null(shuf)':>12}{'null(norm)':>12}"
          f"{'replicates':>12}{'5th pct':>10}")
    for i in range(4):
        print(f"  PC{i+1:<4}{ev[i]:>12.2f}{ev[i]/len(ev):>8.1%}{cut[i]:>12.2f}{nrm[i]:>12.2f}"
              f"{med[i]:>12.3f}{lo[i]:>10.2f}{'   KEEP' if ev[i]>cut[i] else ''}")
    print(f"  retained (shuffle null): {(ev>cut).sum()}   normal null: {(ev>nrm).sum()}"
          f"   Kaiser: {(ev>1).sum()}")
    for t in (.70, .80, .90):
        print(f"  cumulative >= {t:.0%} would keep {int(np.argmax(np.cumsum(ev)/len(ev)>=t)+1)}")

    sub("loadings")
    print("   " + L.round(2).to_string().replace("\n", "\n   "))

    sub("year-over-year (in-sample)")
    for c in ("PC1", "PC2"):
        rs = [yoy(P, y0, y1, c) for y0, y1 in [(2023,2024),(2024,2025),(2025,2026)]]
        print(f"  {c}  " + "   ".join(f"{y}:{r:.3f}(n={n})"
              for (r,n),y in zip(rs,["23-24","24-25","25-26"]))
              + f"   mean {np.mean([r for r,_ in rs]):.3f}")

    sub("role recovery")
    P["is_du"] = (P.role == "duelist").astype(int)
    nd = P[P.role != "duelist"]
    r4 = smf.ols("PC1 ~ C(role)", P).fit().rsquared
    rb = smf.ols("PC1 ~ is_du", P).fit().rsquared
    rn = smf.ols("PC1 ~ C(role)", nd).fit().rsquared
    print(f"  R2 from the full 4-way role label        {r4:.3f}")
    print(f"  R2 from a duelist / not-duelist flag     {rb:.3f}")
    print(f"  R2 from role, EXCLUDING duelists         {rn:.3f}   (n={len(nd)})")
    print(f"  the 4-way taxonomy adds                  {r4-rb:+.3f} over the binary flag")
    span = (nd.PC1.max()-nd.PC1.min())/(P.PC1.max()-P.PC1.min())
    print(f"  non-duelists span                        {span:.0%} of the population range")
    med_all = P.PC1.median()
    for r in ROLES:
        s = P[P.role == r]
        print(f"    {r:<12}{(s.PC1>med_all).mean():>6.0%} sit above the overall median  (n={len(s)})")

    sub("role SHARES instead of the hard label (amendment 1)")
    Sh = shares()
    A = P.merge(Sh[["player_id","year"]+ROLES], on=["player_id","year"], how="left")
    print(f"  R2 of PC1 from the hard 4-way label      {r4:.3f}")
    print("  R2 of PC1 from the four role SHARES      "
          f"{smf.ols('PC1 ~ duelist + initiator + controller', A).fit().rsquared:.3f}")
    print("  R2 of PC1 from duelist SHARE alone       "
          f"{smf.ols('PC1 ~ duelist', A).fit().rsquared:.3f}")

    sub("the clean cross-role test (amendment 3)")
    zero = A[A.duelist == 0]
    print(f"  never played a duelist agent             {len(zero)} of {len(A)}")
    print(f"  ...above the overall aggression median   {(zero.PC1>med_all).sum()} "
          f"({(zero.PC1>med_all).mean():.0%})")
    print("  R2 of PC1 from role among them           "
          f"{smf.ols('PC1 ~ C(role)', zero).fit().rsquared:.3f}   (vs {r4:.3f} overall)")
    print("\n  most aggressive with ZERO duelist maps:")
    print("   " + zero.nlargest(10,"PC1")[["handle","year","role","main_agent","PC1"]]
            .round(2).to_string(index=False).replace("\n","\n   "))

    sub("duelist map share of the five originally cited cross-role cases")
    for h, y in [("reduxx",2026),("victor",2024),("meteor",2025),("havoc",2024),("ange1",2023)]:
        row = A[(A.handle == h) & (A.year == y)]
        if row.empty:
            print(f"  {h} {y:<6}not in the sample at this floor"); continue
        x = row.iloc[0]
        print(f"  {h} {y}  labeled {x.role:<11}{x.duelist:>5.0%} duelist maps   PC1 {x.PC1:+.2f}")

    section("DECISION 11 -- PC2")
    q = pd.qcut(sc.PC2.values, 5, labels=False)
    UN = F.UNITS
    print("  bottom fifth -> top fifth, in game units:")
    for f, lab in UN.items():
        a, b = raw.loc[q==0, f].mean(), raw.loc[q==4, f].mean()
        print(f"    {lab:<20}{a:>8.2f} -> {b:>8.2f}   ({b-a:+.2f})")
    # Correlations with the FEATURES are against the residualized, standardized
    # columns the component was fitted on -- not the raw season means, which still
    # carry the quality variance the model removed. The two differ a lot: PC2 vs
    # residual assists is -.50, vs raw assists -.12.
    print(f"\n  r of PC2 with residual assists          {sc.PC2.corr(M.assists):+.3f}")
    print(f"  r of PC2 with residual plants           {sc.PC2.corr(M.plants):+.3f}")
    print(f"  r of PC2 with residual clutch_att       {sc.PC2.corr(M.clutch_att):+.3f}")
    print(f"  r of PC2 with residual hs               {sc.PC2.corr(M.hs):+.3f}")
    kres = St.residualize(raw, ["kast"])["kast"].reset_index(drop=True)
    print("  r of PC2 with residual KAST             "
          f"{kres.corr(sc.PC2.reset_index(drop=True)):+.3f}   (KAST is not a feature; "
          f"residualized on {F.QUALITY} the same way for comparability)")
    print(f"  r of PC2 with RAW KAST                  {P.PC2.corr(raw.kast):+.3f}")
    print(f"\n  r of PC2 with K/D (quality)             {P.PC2.corr(raw.kd_ratio):+.3f}")
    print(f"  r of PC1 with K/D (quality)             {P.PC1.corr(raw.kd_ratio):+.3f}")
    print(f"  r of PC2 with year                      {P.PC2.corr(P.year):+.3f}")
    ag = P.assign(a=P.main_agent.str.lower()).groupby("a").PC2.agg(["mean","size"])
    ag = ag[ag["size"] >= 15].sort_values("mean")
    print("\n  by modal agent (n>=15), lowest and highest:")
    print("    top:    " + ", ".join(f"{i} {v:+.2f}" for i,v in ag["mean"].tail(3)[::-1].items()))
    print("    bottom: " + ", ".join(f"{i} {v:+.2f}" for i,v in ag["mean"].head(3).items()))

    section("DECISION 10 -- is each adjustment necessary?")
    d = F.build()
    b = d[d.Side == "both"].copy()
    keep = pd.MultiIndex.from_frame(M[["player_id", "year"]])
    b = b[pd.MultiIndex.from_frame(b[["player_id", "year"]]).isin(keep)]

    sub("does adjusting change the player-season number?")
    print(f"  {'feature':<22}{'r(raw, map-adj)':>18}{'r(raw, quality-adj)':>22}")
    for f in F.STYLE:
        g = b.dropna(subset=[f])
        dm = pd.get_dummies(g.Map, drop_first=True).astype(float)
        res = sm.OLS(g[f].values, sm.add_constant(dm.values)).fit().resid
        madj = (pd.Series(res, index=g.index).groupby([g.player_id, g.year]).mean()
                  .rename("m").reset_index())
        j = M[["player_id", "year"]].merge(madj, on=["player_id", "year"], how="left")
        r_map = raw[f].corr(j.m)
        r_q = raw[f].corr(M[f])          # both RangeIndex now (see above)
        print(f"  {f:<22}{r_map:>18.4f}{r_q:>22.4f}")

    sub("does opponent strength vary?")
    ms = F.map_results()
    tw = ms.groupby(["Team", "year"]).won.mean().rename("team_str").reset_index()
    pair = ms.merge(ms, on=D.MATCH + ["Map"], suffixes=("", "_o"))
    pair = pair[pair.Team != pair.Team_o]
    pair = pair.merge(tw.rename(columns={"Team": "Team_o", "team_str": "opp_str"}),
                      on=["Team_o", "year"], how="left")
    po = (b.merge(pair[D.MATCH + ["Map", "Team", "opp_str"]],
                  on=D.MATCH + ["Map", "Team"], how="left")
            .groupby(["player_id", "year"]).opp_str.mean().rename("opp").reset_index())
    O = M[["player_id", "year"]].merge(po, on=["player_id", "year"], how="left")
    print(f"  mean opponent strength across {len(O)} player-seasons: "
          f"{O.opp.mean():.3f},  sd {O.opp.std():.3f},  range {O.opp.min():.3f}-{O.opp.max():.3f}")
    r2 = {f: smf.ols("y ~ opp", pd.DataFrame({"y": raw[f].values, "opp": O.opp.values})
                     .dropna()).fit().rsquared for f in F.STYLE}
    print("  R2 of each feature on opponent strength: "
          f"{min(r2.values()):.4f}-{max(r2.values()):.4f}  (max: {max(r2, key=r2.get)})")
    sh = b.groupby(["player_id", "year"]).Map.value_counts(normalize=True).rename("s")
    print(f"  sd of a player's share of any one map: {sh.groupby('Map', level=2).std().mean():.3f}")

    sub("K/D-only spec vs controlling for all four quality measures")
    QUAD = ["kd_ratio", "acs", "adr", "Rating"]
    rq = raw.copy()
    rq["Rating"] = (b.groupby(["player_id", "year"]).Rating.mean()
                     .reindex(pd.MultiIndex.from_frame(raw[["player_id", "year"]])).values)
    out = {}
    for f in F.STYLE:
        ok = rq[f].notna() & rq[QUAD].notna().all(axis=1)
        r4_ = sm.OLS(rq.loc[ok, f], sm.add_constant(rq.loc[ok, QUAD])).fit().resid
        out[f] = M.loc[ok.values, f].reset_index(drop=True).corr(r4_.reset_index(drop=True))
    print("  r between the K/D-adjusted estimate and the all-four-adjusted estimate:")
    for f, v in sorted(out.items(), key=lambda x: x[1]):
        print(f"    {f:<24}{v:.3f}")

    sub("residual quality correlation of the style estimate")
    for lab, frame in [("unadjusted", raw), ("K/D-adjusted", M)]:
        vals = [abs(frame[f].corr(rq[qq])) for f in F.STYLE for qq in QUAD]
        print(f"  mean |r| with the four quality measures, {lab:<14}{np.mean(vals):.3f}")

    section("DECISION 18 -- per-map counts, not per-round rates")
    # Computed by `robustness.rounds_summary`, not here, so that
    # `src/robustness.py rounds` and this section cannot disagree.
    from robustness import rounds_summary
    rs = rounds_summary()
    print(f"  map length      mean {rs['map_mean']:.2f}  sd {rs['map_sd']:.2f}   "
          f"p5 {rs['map_p5']:.0f}  p95 {rs['map_p95']:.0f}  max {rs['map_max']:.0f}"
          f"   ({rs['n_maps']:,} maps)")
    print(f"  season mean     {rs['season_mean']:.2f}  sd {rs['observed_sd']:.2f}   "
          f"p10 {rs['season_p10']:.2f}  p90 {rs['season_p90']:.2f}")
    print(f"  of that sd, the part that is persistent rather than map-level noise: "
          f"{rs['persistent_sd']:.2f}")
    print(f"  largest share of any feature's between-player variance explained: "
          f"{max(rs['r2'].values()):.2%} ({max(rs['r2'], key=rs['r2'].get)})")
    print(f"  first_engagement specifically           {rs['r2']['first_engagement']:.2%}")
    print(f"  r of season-mean rounds with PC1        {rs['r_pc1']:+.3f}")
    sub("rebuilt end to end on per-round rates")
    for x in rs["specs"]:
        print(f"  {x['name']:<22}n={x['n']:<5}kept {x['kept']}   "
              f"PC1 var {x['var1']:.1%}   PC2 margin {x['pc2_margin']:+.3f}   "
              f"R2 role {x['r_role']:.3f}   yoy {x['yoy']:.3f}")
    print(f"  same players, same places: PC1 r = {rs['pc1_r']:.4f}   "
          f"PC2 r = {rs['pc2_r']:.4f}   (n={rs['n_shared']})")

    section("DECISION 12 -- archetypes or a continuum?")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from clusters import replication as crep, null_like
    XY = sc[["PC1", "PC2"]].values
    pid = M.player_id.values
    Z = null_like(XY)
    print(f"  {'k':<4}{'kmeans real':>13}{'kmeans null':>13}{'gmm real':>11}{'gmm null':>11}"
          f"{'silhouette':>12}{'sil null':>10}")
    for k in range(2, 9):
        kr, kn = crep(XY, k, pid, "kmeans"), crep(Z, k, pid, "kmeans")
        gr, gn = crep(XY, k, pid, "gmm"), crep(Z, k, pid, "gmm")
        s = silhouette_score(XY, KMeans(k, n_init=20, random_state=0).fit(XY).labels_)
        sn = silhouette_score(Z, KMeans(k, n_init=20, random_state=0).fit(Z).labels_)
        print(f"  {k:<4}{kr:>13.3f}{kn:>13.3f}{gr:>11.3f}{gn:>11.3f}{s:>12.3f}{sn:>10.3f}")
    km = KMeans(3, n_init=20, random_state=0).fit(XY)
    near = np.sort(km.transform(XY), axis=1)
    ratio = near[:, 0] / near[:, 1]
    print(f"\n  at k=3, within 75% of a boundary: {(ratio>=.75).sum()} of {len(XY)} "
          f"({(ratio>=.75).mean():.0%})")

    section("DECISION 11 -- the lurker mechanism test (PC2)")
    # "Lurker-shaped" roles are the ones that hold ground alone: sentinel and
    # controller. The test asks whether players who MOVE into one already shot
    # straighter beforehand (selection) or start shooting straighter after (mechanism).
    LURK, OTHER = {"sentinel", "controller"}, {"duelist", "initiator"}
    hs_ = pd.DataFrame({"player_id": raw.player_id, "year": raw.year,
                        "hs": raw.hs.values, "role": P.role.values})
    hs_["grp"] = np.where(hs_.role.isin(LURK), "lurk",
                          np.where(hs_.role.isin(OTHER), "other", None))
    w = hs_.pivot_table(index="player_id", columns="year", values="hs")
    grp = hs_.pivot_table(index="player_id", columns="year", values="grp", aggfunc="first")

    moved_in, moved_out, stayed = [], [], []
    for a, bnext in [(2023, 2024), (2024, 2025), (2025, 2026)]:
        if a not in grp or bnext not in grp:
            continue
        ok = grp[a].notna() & grp[bnext].notna() & w[a].notna() & w[bnext].notna()
        g0, g1 = grp.loc[ok, a], grp.loc[ok, bnext]
        before, after = w.loc[ok, a], w.loc[ok, bnext]
        for mask, bucket in [((g0 == "other") & (g1 == "lurk"), moved_in),
                             ((g0 == "lurk") & (g1 == "other"), moved_out),
                             ((g0 == g1), stayed)]:
            bucket.extend(zip(before[mask].tolist(), after[mask].tolist()))

    drift = float(np.mean([y - x for x, y in stayed]))
    base_in = float(np.mean([x for x, _ in moved_in]))
    chg_in = float(np.mean([y - x for x, y in moved_in])) - drift
    chg_out = float(np.mean([y - x for x, y in moved_out])) - drift
    print(f"  headshot % BEFORE moving in     {base_in:>7.2f}   vs "
          f"{hs_[hs_.grp=='other'].hs.mean():.2f} for duelists/initiators overall "
          f"({base_in - hs_[hs_.grp=='other'].hs.mean():+.2f})")
    print(f"  within-player change, moved IN  {chg_in:>+7.2f} points vs non-movers"
          f"   (n={len(moved_in)})")
    print(f"  within-player change, moved OUT {chg_out:>+7.2f} points vs non-movers"
          f"   (n={len(moved_out)})")
    print("  between-role gap in headshot %  "
          f"{hs_[hs_.grp=='lurk'].hs.mean() - hs_[hs_.grp=='other'].hs.mean():>+7.2f} points")
    print(f"  (non-movers: {len(stayed)} players, drifting {drift:+.2f} points year to year)")

    section("DECISION 11 -- is PC2 attack-specific? (the failed prediction)")
    ov = D.overview()
    for c in ["Assists", "First Kills", "First Deaths"]:
        ov[c] = pd.to_numeric(ov[c], errors="coerce")
    ov["fe"] = ov["First Kills"] + ov["First Deaths"]
    key = ["player_id", "year"]
    gaps = {}
    for col, name in [("Assists", "assist"), ("fe", "first_engagement")]:
        w2 = (ov[ov.Side.isin(["attack", "defend"])]
                .groupby(key + ["Side"])[col].mean().unstack("Side"))
        gaps[name] = (w2["attack"] - w2["defend"]).rename(name)
    G2 = M[key].merge(pd.concat(gaps.values(), axis=1).reset_index(), on=key, how="left")
    for name in gaps:
        print(f"  r of PC2 with the attack-minus-defend {name} gap: "
              f"{sc.PC2.corr(G2[name]):+.3f}")

    section("DECISION 10 -- the clutch claim behind the role-baseline argument")
    du = P[P.role == "duelist"]
    se = P[P.role == "sentinel"]
    cl = raw.clutch_att
    n = (cl[du.index] > cl[se.index].mean()).sum()
    print(f"  duelists clutching more than the average sentinel: {n} of {len(du)} "
          f"({n/len(du):.0%})")

    sub("how badly does a 3-label system describe players?")
    for lab, lo_, hi in [("clearly in one group", 0, .5), ("leaning one way", .5, .75),
                         ("borderline", .75, .9), ("essentially a coin flip", .9, 1.01)]:
        m = (ratio >= lo_) & (ratio < hi)
        print(f"    {lab:<26}{m.mean():>5.0%}   ({m.sum()})")
    print(f"    pairs compared by the Rand index: {len(XY)*(len(XY)-1)//2:,}")

    sub("the three groups at k=3")
    C = P.assign(g=km.labels_, PC1=sc.PC1.values, PC2=sc.PC2.values)
    for g in sorted(C.g.unique()):
        s = C[C.g == g]
        comp = (s.role.value_counts(normalize=True).head(2)
                 .map(lambda v: f"{v:.0%}").to_dict())
        dist = np.linalg.norm(XY[C.g.values == g] - km.cluster_centers_[g], axis=1)
        near = s.iloc[np.argsort(dist)[:4]]
        print(f"  group {g}  n={len(s):<5}PC1 {s.PC1.mean():+.2f}  PC2 {s.PC2.mean():+.2f}  "
              f"{', '.join(f'{k} {v}' for k, v in comp.items())}")
        print("     nearest the center: " +
              ", ".join(f"{r.handle} ({r.main_agent})" for _, r in near.iterrows()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="*", default=[], choices=["report", "decisions", []],
                    help="default: both")
    a = ap.parse_args()
    for name in (a.which or ["report", "decisions"]):
        {"report": report, "decisions": decisions}[name]()
