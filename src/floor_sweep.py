"""Does the minimum-maps floor matter at all?

The floor was argued from reliability-by-season-length (decision 13), but the bands
in that argument were chosen by hand and "reliability .32 is acceptable" is itself an
unstated threshold. The honest test is not to defend a value but to show the
conclusions do not depend on it.
"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, "src")
import data as D, features as F, style as S, validate as V
from components import eigenvalues, null_cutoffs
from sklearn.decomposition import PCA
from step11_role_independence import with_role

FLOORS = [5, 10, 15, 20, 25, 30]


def run(floor):
    d = F.build()
    ps = with_role(F.seasons(d, min_maps=floor), d).dropna(subset=F.STYLE + [F.QUALITY])
    R = S.residualize(ps, F.STYLE)
    M = pd.concat([ps[["player_id", "year", "role"]].reset_index(drop=True),
                   R.reset_index(drop=True)], axis=1).dropna(subset=F.STYLE)
    X = ((M[F.STYLE] - M[F.STYLE].mean()) / M[F.STYLE].std()).values
    ev, cut = eigenvalues(X), null_cutoffs(X, "shuffle", n=250)
    p = PCA(n_components=2).fit(X)
    flip = -1 if p.components_[0][F.STYLE.index("first_engagement")] < 0 else 1
    sc = pd.DataFrame(p.transform(X), columns=["PC1", "PC2"]); sc["PC1"] *= flip
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


if __name__ == "__main__":
    print(f"\n{'floor':>6}{'n':>7}{'kept':>6}{'PC2 margin':>12}{'PC1 var':>9}"
          f"{'R2 role':>9}{'R2 flag':>9}{'R2 non-du':>11}{'yoy PC1':>9}{'pairs':>7}")
    rows = [run(f) for f in FLOORS]
    for r in rows:
        print(f"{r['floor']:>6}{r['n']:>7}{r['kept']:>6}{r['pc2_margin']:>+12.3f}"
              f"{r['var1']:>9.1%}{r['r_role']:>9.3f}{r['r_flag']:>9.3f}"
              f"{r['r_nd']:>11.3f}{r['yoy']:>9.3f}{r['pairs']:>7}")
    t = pd.DataFrame(rows)
    print(f"\n  across floors 5 to 30:")
    for k, lab in [("r_role", "R2 from 4-way role"), ("r_flag", "R2 from duelist flag"),
                   ("r_nd", "R2 among non-duelists"), ("yoy", "year-over-year PC1")]:
        print(f"    {lab:<24}{t[k].min():.3f} to {t[k].max():.3f}   "
              f"(range {t[k].max()-t[k].min():.3f})")
    print(f"    components kept          {sorted(t.kept.unique())}")
