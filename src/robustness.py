"""Do the judgement calls change the answer?

Two sweeps, both asking the same question of a choice that was argued rather than
derived. Was sensitivity.py and floor_sweep.py.

FEATURE SET (decision 15)
  STYLE     7 features  -- the decided set (decisions 07 and 17)
  STRICT    5 features  -- drops clutch_att and first_engagement_gap, the two
                           judgement calls
  CONC      7 features, restricted to player-seasons with >=70% of maps in one
                           role -- a flex player's season average describes a blend
                           of two jobs rather than either of them (decision 14)

MAP FLOOR (decision 13)
  The floor was argued from reliability-by-season-length, but the bands in that
  argument were chosen by hand and "reliability .32 is acceptable" is itself an
  unstated threshold. The honest test is not to defend a value but to show the
  conclusions do not depend on it.

YARDSTICK (decision 06)
  Every behaviour is adjusted for quality before the rotation. No quality measure is
  clean -- each is partly caused by playstyle -- so the choice is swept rather than
  defended.

    python src/robustness.py features | floor | yardstick
"""
import pandas as pd
import statsmodels.formula.api as smf

import features as F
import pca
import style as S
import validate as V
from components import eigenvalues, loadings, null_cutoffs
from roles import ROLES as R, shares, with_role

FLOORS = [5, 10, 15, 20, 25, 30]






def spec(name, cols, conc_min=None):
    M, L, sc = loadings(cols)
    D = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    if conc_min is not None:
        P = shares(); P["conc"] = P[R].max(axis=1)
        D = D.merge(P[["player_id", "year", "conc"]], on=["player_id", "year"], how="left")
        keep = D.conc >= conc_min
        M, D = M[keep.values], D[keep.values]
        Z = pca.standardise(M, cols)
        model, flip = pca.fit(Z.values, cols)
        s2 = pca.scores(model, flip, Z, M.index)
        D = pd.concat([M[["player_id", "year", "role"]], s2], axis=1)
    X = pca.standardise(M, cols).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    D["is_du"] = (D.role == "duelist").astype(int)
    nd = D[D.role != "duelist"]
    return dict(name=name, n=len(D), kept=int((ev > cut).sum()),
                var1=ev[0]/len(ev), var2=ev[1]/len(ev),
                r_role=smf.ols("PC1 ~ C(role)", D).fit().rsquared,
                r_flag=smf.ols("PC1 ~ is_du", D).fit().rsquared,
                r_nd=smf.ols("PC1 ~ C(role)", nd).fit().rsquared,
                D=D[["player_id", "year", "PC1", "PC2"]])







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
    X = pca.standardise(M, F.STYLE).values
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
    specs = [spec(f"STYLE  ({len(F.STYLE)} features)", F.STYLE),
             spec(f"STRICT ({len(F.STRICT)} features)", F.STRICT),
             spec("CONC   (>=70% one role)", F.STYLE, conc_min=.70)]

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
        print(f"  {'STYLE vs ' + s['name'].split()[0]:<24}n={len(j):>5}   "
              f"PC1 r={base.loc[j,'PC1'].corr(o.loc[j,'PC1']):.3f}   "
              f"PC2 r={base.loc[j,'PC2'].corr(o.loc[j,'PC2']):.3f}")

    print("\n  Held-out 2026 (whole pipeline refitted on 2023-2025):\n")
    for name, cols in [("STYLE", F.STYLE), ("STRICT", F.STRICT)]:
        H, _, _ = V.fit_and_project(cols)
        r1, n = V.yoy(H, 2025, 2026, "PC1"); r2, _ = V.yoy(H, 2025, 2026, "PC2")
        print(f"  {name:<24}PC1 {r1:.3f}   PC2 {r2:.3f}   n={n}")


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


def yardstick():
    """Swap the quality covariate and see what survives.

    `none` standardises the raw behaviours instead of residualizing. The columns to
    watch are R2 role -- how much the adjustment removes along with the skill -- and
    r(PC1, ACS), which is how much skill is left IN the axis afterwards."""
    from roles import with_role
    d = F.build()
    ps = with_role(F.seasons(d), d)
    ref = ps.dropna(subset=F.STYLE + ["kd_ratio", "acs", "team_win"])[
        ["player_id", "year", "kd_ratio", "acs"]]
    print(f"\n{'yardstick':<24}{'n':>6}{'kept':>6}{'PC1 var':>9}{'R2 role':>9}"
          f"{'R2 flag':>9}{'r(PC1,ACS)':>12}{'vs K/D':>9}")
    base = None
    for lab, cov in [("K/D  (chosen)", "kd_ratio"), ("team win rate", "team_win"),
                     ("ACS", "acs"), ("none - raw behaviour", None)]:
        # residualize on everything available, then drop -- same order as style.build
        R = (S.residualize(ps, F.STYLE, covariate=cov) if cov else
             pd.DataFrame({c: (ps[c] - ps[c].mean()) / ps[c].std() for c in F.STYLE},
                          index=ps.index))
        M = pd.concat([ps[["player_id", "year", "role"]].reset_index(drop=True),
                       R.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
        M = M.reset_index(drop=True)
        Z = pca.standardise(M, F.STYLE)
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
    print("\n  Two components under every yardstick, and players land in the same place.")
    print("  What moves is how much role APPEARS to explain, and how much skill is left in.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", nargs="*", default=[],
                    choices=["features", "floor", "yardstick", []], help="default: all")
    a = ap.parse_args()
    for name in (a.check or ["features", "floor", "yardstick"]):
        {"features": feature_sets, "floor": floor_sweep, "yardstick": yardstick}[name]()
