"""Do the judgement calls change the answer?

Three specifications, each a choice that was argued rather than derived:

  STYLE     8 features  -- the decided set (decision 07)
  STRICT    6 features  -- drops clutch_att and first_engagement_gap, the two
                           judgement calls
  CONC      8 features, restricted to player-seasons with >=70% of maps in one
                           role -- a flex player's season average describes a blend
                           of two jobs rather than either of them (decision 14)

For each: how many components survive the null, what the axes are made of, whether
players land in the same place, how much role explains, and whether it still holds
on a year the model never saw.
"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, "src")
import features as F, style as S, validate as V
from components import eigenvalues, null_cutoffs, loadings
from step13_label_validity import shares

R = ["duelist", "initiator", "controller", "sentinel"]


def spec(name, cols, conc_min=None):
    M, L, sc = loadings(cols)
    D = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)
    if conc_min is not None:
        P = shares(); P["conc"] = P[R].max(axis=1)
        D = D.merge(P[["player_id", "year", "conc"]], on=["player_id", "year"], how="left")
        keep = D.conc >= conc_min
        M, D = M[keep.values], D[keep.values]
        Z = ((M[cols] - M[cols].mean()) / M[cols].std()).values
        from sklearn.decomposition import PCA
        p = PCA(n_components=2).fit(Z)
        flip = -1 if p.components_[0][cols.index("first_engagement")] < 0 else 1
        s2 = pd.DataFrame(p.transform(Z), index=M.index, columns=["PC1", "PC2"])
        s2["PC1"] *= flip
        D = pd.concat([M[["player_id", "year", "role"]], s2], axis=1)
    X = ((M[cols] - M[cols].mean()) / M[cols].std()).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle")
    D["is_du"] = (D.role == "duelist").astype(int)
    nd = D[D.role != "duelist"]
    return dict(name=name, n=len(D), kept=int((ev > cut).sum()),
                var1=ev[0]/len(ev), var2=ev[1]/len(ev),
                r_role=smf.ols("PC1 ~ C(role)", D).fit().rsquared,
                r_flag=smf.ols("PC1 ~ is_du", D).fit().rsquared,
                r_nd=smf.ols("PC1 ~ C(role)", nd).fit().rsquared,
                D=D[["player_id", "year", "PC1", "PC2"]])


if __name__ == "__main__":
    specs = [spec("STYLE  (8 features)", F.STYLE),
             spec("STRICT (6 features)", F.STRICT),
             spec("CONC   (>=70% one role)", F.STYLE, conc_min=.70)]

    print(f"\n{'spec':<26}{'n':>6}{'kept':>6}{'PC1 var':>9}{'PC2 var':>9}"
          f"{'R2 role':>9}{'R2 flag':>9}{'R2 non-du':>11}")
    for s in specs:
        print(f"  {s['name']:<24}{s['n']:>6}{s['kept']:>6}{s['var1']:>9.1%}{s['var2']:>9.1%}"
              f"{s['r_role']:>9.3f}{s['r_flag']:>9.3f}{s['r_nd']:>11.3f}")

    print(f"\n  Do players land in the same place? (correlation of positions, shared rows)\n")
    base = specs[0]["D"].set_index(["player_id", "year"])
    for s in specs[1:]:
        o = s["D"].set_index(["player_id", "year"])
        j = base.index.intersection(o.index)
        print(f"  {'STYLE vs ' + s['name'].split()[0]:<24}n={len(j):>5}   "
              f"PC1 r={base.loc[j,'PC1'].corr(o.loc[j,'PC1']):.3f}   "
              f"PC2 r={base.loc[j,'PC2'].corr(o.loc[j,'PC2']):.3f}")

    print(f"\n  Held-out 2026 (whole pipeline refitted on 2023-2025):\n")
    for name, cols in [("STYLE", F.STYLE), ("STRICT", F.STRICT)]:
        H, _, _ = V.fit_and_project(cols)
        r1, n = V.yoy(H, 2025, 2026, "PC1"); r2, _ = V.yoy(H, 2025, 2026, "PC2")
        print(f"  {name:<24}PC1 {r1:.3f}   PC2 {r2:.3f}   n={n}")
