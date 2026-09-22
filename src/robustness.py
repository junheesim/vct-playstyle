"""Do the judgment calls change the answer?

Two sweeps, both asking the same question of a choice that was argued rather than
derived. Was sensitivity.py and floor_sweep.py.

FEATURE SET (decisions 15, 19)
  There is one feature set now. It was two -- STRICT and STYLE -- while two features
  were contested; decision 19 dropped `first_engagement_gap` and stopped treating
  `clutch_att` as a special case, so the comparison has nothing left to compare.

  What remains under `features` is the restriction that is NOT about feature choice:
  CONC, the player-seasons with >=70% of maps in one role. A flex player's season
  average describes a blend of two jobs rather than either of them (decision 14), and
  this is where Finding 1's "isolation fails among role specialists" comes from.

MAP FLOOR (decision 13)
  The floor was argued from reliability-by-season-length, but the bands in that
  argument were chosen by hand and "reliability .32 is acceptable" is itself an
  unstated threshold. The honest test is not to defend a value but to show the
  conclusions do not depend on it.

THE MEASURE OF QUALITY (decision 06)
  Every behavior is adjusted for quality before the rotation. No quality measure is
  clean -- each is partly caused by playstyle -- so the choice is swept rather than
  defended.

ROUND NORMALIZATION (decision 18)
  A map runs 13 to 48 rounds, so a per-map COUNT is a rate times an opportunity
  count. The published features are per map. This rebuilds them as per-round rates
  and shows what changes.

    python src/robustness.py features | floor | quality | rounds
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import data as D
import features as F
import pca
import style as S
import validate as V
from components import N_DRAWS, eigenvalues, loadings, null_cutoffs
from roles import ROLES as R, shares, with_role

FLOORS = [5, 10, 15, 20, 25, 30]

# The four features that are raw per-map counts. `hs` is already a percentage and
# `creds_per_round` already a rate, so neither is exposed.
COUNTS = ["first_engagement", "assists", "plants", "clutch_att"]






def spec(name, cols, conc_min=None):
    M, L, sc = loadings(cols)
    D = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    if conc_min is not None:
        P = shares(); P["conc"] = P[R].max(axis=1)
        D = D.merge(P[["player_id", "year", "conc"]], on=["player_id", "year"], how="left")
        keep = D.conc >= conc_min
        M, D = M[keep.values], D[keep.values]
        Z = pca.standardize(M, cols)
        model, flip = pca.fit(Z.values, cols)
        s2 = pca.scores(model, flip, Z, M.index)
        D = pd.concat([M[["player_id", "year", "role"]], s2], axis=1)
    X = pca.standardize(M, cols).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    D["is_du"] = (D.role == "duelist").astype(int)
    nd = D[D.role != "duelist"]
    return dict(name=name, n=len(D), kept=int((ev > cut).sum()),
                var1=ev[0]/len(ev), var2=ev[1]/len(ev),
                r_role=smf.ols("PC1 ~ C(role)", D).fit().rsquared,
                r_flag=smf.ols("PC1 ~ is_du", D).fit().rsquared,
                r_nd=smf.ols("PC1 ~ C(role)", nd).fit().rsquared,
                D=D[["player_id", "year", "PC1", "PC2"]])







def per_round_frame() -> pd.DataFrame:
    """`features.build()` with every per-map COUNT re-expressed as a per-round rate.

    Scaled back up by 24 (the rounds in a 13-11 map) and the side gap by 12, so the
    columns stay in the units the report quotes and NOTHING but the normalization
    changes. The columns keep their original NAMES, because `pca.ANCHOR` and every
    diagnostic key on them.

    The quality measure is untouched: `kd_ratio` is built from season TOTALS of
    kills and deaths, so it is already a rate and the adjustment is identical under
    both specifications. Only the features move.
    """
    d = F.build().merge(F.round_counts(), on=D.MATCH + ["Map", "Team"], how="left")

    for c in COUNTS:
        d[c] = 24 * d[c] / d.rounds.replace(0, np.nan)
    return d


def _fit(d):
    """Residualize, rotate, score. The order is `style.build`'s -- residualize FIRST
    and drop after -- because fitting the quality adjustment on the smaller sample
    shifts every score slightly (see the note in `run`)."""
    ps = with_role(F.seasons(d), d)
    Rz = S.residualize(ps, F.STYLE)
    M = pd.concat([ps[["player_id", "year", "role"]].reset_index(drop=True),
                   Rz.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
    M = M.reset_index(drop=True)
    Z = pca.standardize(M, F.STYLE)
    model, flip = pca.fit(Z.values, F.STYLE)
    M = pd.concat([M, pca.scores(model, flip, Z, M.index)], axis=1)
    M["is_du"] = (M.role == "duelist").astype(int)
    return M, Z


def _spec_row(name, d, n_null=None):
    # None -> components.N_DRAWS, the same null the published numbers use. PC2's
    # margin is ~.06, small enough that 250 draws and 500 draws disagree in the
    # third decimal, so a number the site states must not be computed at 250.
    M, Z = _fit(d)
    ev, cut = eigenvalues(Z.values), null_cutoffs(Z.values, "shuffle",
                                                  n=n_null or N_DRAWS)
    a = M[M.year == 2025].set_index("player_id").PC1
    b = M[M.year == 2026].set_index("player_id").PC1
    j = a.index.intersection(b.index)
    return dict(name=name, n=len(M), kept=int((ev > cut).sum()),
                var1=ev[0] / len(ev), var2=ev[1] / len(ev), pc2_margin=ev[1] - cut[1],
                r_role=smf.ols("PC1 ~ C(role)", M).fit().rsquared,
                r_flag=smf.ols("PC1 ~ is_du", M).fit().rsquared,
                r_nd=smf.ols("PC1 ~ C(role)", M[M.role != "duelist"]).fit().rsquared,
                yoy=a.loc[j].corr(b.loc[j]), M=M)


def rounds_summary(n_null=None) -> dict:
    """Every number decision 18 quotes. One implementation; `rounds` prints it and
    `quoted_numbers.decisions` prints it, so the two cannot disagree."""
    rc = F.round_counts()
    # rc carries one row per TEAM per map, so a map appears twice. Counting map
    # lengths off it unchecked doubles the n -- decision 01's mistake, one level down.
    r = rc.drop_duplicates(D.MATCH + ["Map"]).rounds.dropna()

    d = F.build().merge(rc, on=D.MATCH + ["Map", "Team"], how="left")
    ps = F.seasons(d)
    sm = (d.groupby(["player_id", "year"]).rounds.mean().rename("rmean").reset_index()
            .merge(ps[["player_id", "year", "maps"]], on=["player_id", "year"]))
    raw = (d.groupby(["player_id", "year"])[F.STYLE].mean().reset_index()
             .merge(sm, on=["player_id", "year"]))

    base = _spec_row("per MAP (published)", F.build(), n_null)
    pr   = _spec_row("per ROUND", per_round_frame(), n_null)
    A = base["M"].set_index(["player_id", "year"])
    B = pr["M"].set_index(["player_id", "year"])
    j = A.index.intersection(B.index)

    # If map length were an independent draw per map, averaging a season of them
    # would divide its sd by sqrt(n). What the observed sd exceeds that by is the
    # part that really is persistent -- the only part that can confound anything.
    predicted = r.std() / np.sqrt(ps.maps.mean())
    observed = sm.rmean.std()
    persistent = np.sqrt(max(observed**2 - predicted**2, 0.0))

    return dict(
        coverage=d.rounds.notna().mean(), n_maps=len(r),
        map_mean=r.mean(), map_sd=r.std(),
        map_p5=np.percentile(r, 5), map_p95=np.percentile(r, 95), map_max=r.max(),
        maps_median=ps.maps.median(), maps_mean=ps.maps.mean(),
        predicted_sd=predicted, observed_sd=observed, persistent_sd=persistent,
        season_mean=sm.rmean.mean(),
        season_p10=sm.rmean.quantile(.10), season_p90=sm.rmean.quantile(.90),
        r2={c: raw[c].corr(raw.rmean) ** 2 for c in F.STYLE},
        r_pc1=base["M"].merge(sm, on=["player_id", "year"]).PC1.corr(
              base["M"].merge(sm, on=["player_id", "year"]).rmean),
        specs=[base, pr], n_shared=len(j),
        pc1_r=A.loc[j, "PC1"].corr(B.loc[j, "PC1"]),
        pc2_r=A.loc[j, "PC2"].corr(B.loc[j, "PC2"]))


def rounds():
    """Decision 18. Is a per-map count a rate, or an opportunity count?"""
    s = rounds_summary()

    print(f"\n  MAP LENGTH  ({s['n_maps']:,} maps, joined to {s['coverage']:.1%} of player-maps)")
    print(f"    mean {s['map_mean']:.2f}  sd {s['map_sd']:.2f}   "
          f"p5 {s['map_p5']:.0f}  p95 {s['map_p95']:.0f}  max {s['map_max']:.0f}"
          f"   (p95/p5 = {s['map_p95']/s['map_p5']:.2f}x)")

    print(f"\n  DOES IT SURVIVE THE SEASON AVERAGE?   median {s['maps_median']:.0f} maps, "
          f"mean {s['maps_mean']:.1f}")
    print(f"    map-level sd                       {s['map_sd']:>6.2f}")
    print(f"    predicted season-mean sd, if independent   {s['predicted_sd']:>6.2f}")
    print(f"    observed season-mean sd            {s['observed_sd']:>6.2f}"
          f"   -> persistent part {s['persistent_sd']:.2f}"
          f" ({s['persistent_sd']/s['season_mean']:.1%} of a map)")
    print(f"    between players: p10 {s['season_p10']:.2f}  p90 {s['season_p90']:.2f}")

    print("\n  SHARE OF BETWEEN-PLAYER VARIANCE IN EACH RAW FEATURE EXPLAINED BY IT")
    for c, v in sorted(s["r2"].items(), key=lambda kv: -kv[1]):
        print(f"    {c:<24}{v:>8.2%}")
    print(f"    {'r with PC1 itself':<24}{s['r_pc1']:>+8.3f}")

    print(f"\n  REBUILT END TO END ON PER-ROUND RATES")
    print(f"    {'spec':<22}{'n':>5}{'kept':>6}{'PC1 var':>9}{'PC2 var':>9}"
          f"{'PC2 margin':>12}{'R2 role':>9}{'R2 flag':>9}{'R2 non-du':>11}{'yoy':>8}")
    for x in s["specs"]:
        print(f"    {x['name']:<22}{x['n']:>5}{x['kept']:>6}{x['var1']:>9.1%}"
              f"{x['var2']:>9.1%}{x['pc2_margin']:>+12.3f}{x['r_role']:>9.3f}"
              f"{x['r_flag']:>9.3f}{x['r_nd']:>11.3f}{x['yoy']:>8.3f}")
    print(f"\n    same players, same places?  n={s['n_shared']}   "
          f"PC1 r = {s['pc1_r']:.4f}   PC2 r = {s['pc2_r']:.4f}")
    print("\n  The opportunity count does not vary between players, so the published")
    print("  per-map features are kept. COST, recorded: PC2's margin over its null")
    print(f"  falls {s['specs'][0]['pc2_margin']:+.3f} -> {s['specs'][1]['pc2_margin']:+.3f}"
          " -- the marginal component gets MORE marginal.")


def run(floor):
    d = F.build()
    # Residualize FIRST, drop after -- the order `style.build` uses. Dropping first
    # fits the quality adjustment on a smaller sample and shifts every score very
    # slightly (r = .99994), which was enough to make this sweep report R2 role .599
    # where `quoted_numbers` reported .605 for the same 775 rows.
    ps = with_role(F.seasons(d, min_maps=floor), d)
    R = S.residualize(ps, F.STYLE)
    M = pd.concat([ps[["player_id", "year", "role"]].reset_index(drop=True),
                   R.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
    # dropna leaves a gapped index; pca.scores must be given it, or the concat below
    # aligns two different indexes and silently unions them into a longer, scrambled
    # frame. (That is exactly what happened: 847 rows and R2 role .015.)
    M = M.reset_index(drop=True)
    X = pca.standardize(M, F.STYLE).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle", n=250)
    model, flip = pca.fit(X, F.STYLE)
    sc = pca.scores(model, flip, X, M.index)
    M = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    M["is_du"] = (M.role == "duelist").astype(int)
    nd = M[M.role != "duelist"]
    a = M[M.year == 2025].set_index("player_id").PC1
    b = M[M.year == 2026].set_index("player_id").PC1
    j = a.index.intersection(b.index)
    return dict(floor=floor, n=len(M), kept=int((ev > cut).sum()),
                pc2_margin=ev[1] - cut[1], var1=ev[0] / len(ev),
                r_role=smf.ols("PC1 ~ C(role)", M).fit().rsquared,
                r_flag=smf.ols("PC1 ~ is_du", M).fit().rsquared,
                r_nd=smf.ols("PC1 ~ C(role)", nd).fit().rsquared,
                yoy=a.loc[j].corr(b.loc[j]), pairs=len(j))

def feature_sets():
    specs = [spec(f"all {len(F.STYLE)} behaviors", F.STYLE),
             spec("only >=70% one-role seasons", F.STYLE, conc_min=.70)]

    print(f"\n{'spec':<26}{'n':>6}{'kept':>6}{'PC1 var':>9}{'PC2 var':>9}"
          f"{'R2 role':>9}{'R2 flag':>9}{'R2 non-du':>11}")
    for s in specs:
        print(f"  {s['name']:<24}{s['n']:>6}{s['kept']:>6}{s['var1']:>9.1%}{s['var2']:>9.1%}"
              f"{s['r_role']:>9.3f}{s['r_flag']:>9.3f}{s['r_nd']:>11.3f}")

    print("\n  Do players land in the same place? (correlation of positions, shared rows)\n")
    base = specs[0]["D"].set_index(["player_id", "year"])
    for s in specs[1:]:
        o = s["D"].set_index(["player_id", "year"])
        j = base.index.intersection(o.index)
        print(f"  {'vs ' + s['name']:<30}n={len(j):>5}   "
              f"PC1 r={base.loc[j,'PC1'].corr(o.loc[j,'PC1']):.3f}   "
              f"PC2 r={base.loc[j,'PC2'].corr(o.loc[j,'PC2']):.3f}")

    print("\n  Held-out 2026 (whole pipeline refitted on 2023-2025):\n")
    H, _, _ = V.fit_and_project(F.STYLE)
    r1, n = V.yoy(H, 2025, 2026, "PC1"); r2, _ = V.yoy(H, 2025, 2026, "PC2")
    print(f"  {'all behaviors':<24}PC1 {r1:.3f}   PC2 {r2:.3f}   n={n}")


def floor_sweep():
    print(f"\n{'floor':>6}{'n':>7}{'kept':>6}{'PC2 margin':>12}{'PC1 var':>9}"
          f"{'R2 role':>9}{'R2 flag':>9}{'R2 non-du':>11}{'yoy PC1':>9}{'pairs':>7}")
    rows = [run(f) for f in FLOORS]
    for r in rows:
        print(f"{r['floor']:>6}{r['n']:>7}{r['kept']:>6}{r['pc2_margin']:>+12.3f}"
              f"{r['var1']:>9.1%}{r['r_role']:>9.3f}{r['r_flag']:>9.3f}"
              f"{r['r_nd']:>11.3f}{r['yoy']:>9.3f}{r['pairs']:>7}")
    t = pd.DataFrame(rows)
    print("\n  across floors 5 to 30:")
    for k, lab in [("r_role", "R2 from 4-way role"), ("r_flag", "R2 from duelist flag"),
                   ("r_nd", "R2 among non-duelists"), ("yoy", "year-over-year PC1")]:
        print(f"    {lab:<24}{t[k].min():.3f} to {t[k].max():.3f}   "
              f"(range {t[k].max()-t[k].min():.3f})")
    print(f"    components kept          {sorted(t.kept.unique())}")


def quality():
    """Swap the measure of quality and see what survives.

    `none` standardizes the raw behaviors instead of residualizing. The columns to
    watch are R2 role -- how much the adjustment removes along with the skill -- and
    r(PC1, ACS), which is how much skill is left IN the axis afterwards."""
    from roles import with_role
    d = F.build()
    ps = with_role(F.seasons(d), d)
    ref = ps.dropna(subset=F.STYLE + ["kd_ratio", "acs", "team_win"])[
        ["player_id", "year", "kd_ratio", "acs"]]
    print(f"\n{'adjusted for':<24}{'n':>6}{'kept':>6}{'PC1 var':>9}{'R2 role':>9}"
          f"{'R2 flag':>9}{'r(PC1,ACS)':>12}{'vs K/D':>9}")
    base = None
    for lab, cov in [("K/D  (chosen)", "kd_ratio"), ("team win rate", "team_win"),
                     ("ACS", "acs"), ("none - raw behavior", None)]:
        # residualize on everything available, then drop -- same order as style.build
        R = (S.residualize(ps, F.STYLE, covariate=cov) if cov else
             pd.DataFrame({c: (ps[c] - ps[c].mean()) / ps[c].std() for c in F.STYLE},
                          index=ps.index))
        M = pd.concat([ps[["player_id", "year", "role"]].reset_index(drop=True),
                       R.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
        M = M.reset_index(drop=True)
        Z = pca.standardize(M, F.STYLE)
        model, flip = pca.fit(Z.values, F.STYLE)
        D = pd.concat([M[["player_id", "year", "role"]],
                       pca.scores(model, flip, Z, M.index)], axis=1)
        D["is_du"] = (D.role == "duelist").astype(int)
        ev, cut = eigenvalues(Z.values), null_cutoffs(Z.values, "shuffle", n=250)
        j = D.merge(ref, on=["player_id", "year"])
        s = D.set_index(["player_id", "year"]).PC1
        vs = "" if base is None else f"{base.loc[base.index.intersection(s.index)].corr(s.loc[base.index.intersection(s.index)]):.3f}"
        if base is None:
            base = s
        print(f"  {lab:<22}{len(D):>6}{int((ev>cut).sum()):>6}{ev[0]/len(ev):>9.1%}"
              f"{smf.ols('PC1 ~ C(role)', D).fit().rsquared:>9.3f}"
              f"{smf.ols('PC1 ~ is_du', D).fit().rsquared:>9.3f}"
              f"{j.PC1.corr(j.acs):>+12.3f}{vs:>9}")
    print("\n  Two components whatever we adjust for, and players land in the same place.")
    print("  What moves is how much role APPEARS to explain, and how much skill is left in.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", nargs="*", default=[],
                    choices=["features", "floor", "quality", "rounds", []],
                    help="default: all")
    a = ap.parse_args()
    for name in (a.check or ["features", "floor", "quality", "rounds"]):
        {"features": feature_sets, "floor": floor_sweep,
         "quality": quality, "rounds": rounds}[name]()
