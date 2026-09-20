"""Every number the report and the decision files quote, printed from the data.

(Named report_numbers, not numbers: a module called `numbers` shadows the stdlib one
that numpy imports, and the whole package fails to load.)

Hand-copied figures drifted repeatedly during this project -- a career mean pasted
into a single-season label, a percentile computed on a different population than the
claim implied. This is the single source of truth: run it, and update the prose from
its output rather than from memory.
"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, "src")
import data as D, features as F, style as S, validate as V
from components import loadings, eigenvalues, null_cutoffs, replication
from examples import guarded


def section(t): print(f"\n{'='*66}\n{t}\n{'='*66}")


if __name__ == "__main__":
    raw = D.overview(); ps_all = D.player_seasons(raw)
    M, L, sc = loadings(F.STYLE)
    G = guarded().merge(F.seasons(F.build()), on=["player_id", "year"],
                        how="left", suffixes=("", "_r"))
    P = pd.concat([M[["player_id", "year", "role"]], sc], axis=1)

    section("SCOPE")
    print(f"  min maps per season                {D.MIN_MAPS}")
    print(f"  player-seasons at that floor       {len(ps_all)}")
    print(f"  ...with all 8 features complete    {len(M)}")
    print(f"  distinct players                   {M.player_id.nunique()}")
    print(f"  matches                            {raw.groupby(D.MATCH).ngroups:,}")
    print(f"  years                              {raw.year.min()}-{raw.year.max()}")

    section("COMPONENTS")
    X = ((M[F.STYLE] - M[F.STYLE].mean()) / M[F.STYLE].std()).values
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
    H, pca, _ = V.fit_and_project(F.STYLE)
    IN = pd.concat([M[["player_id", "year"]], sc], axis=1)
    for col in ("PC1", "PC2"):
        h, n = V.yoy(H, 2025, 2026, col); i, _ = V.yoy(IN, 2025, 2026, col)
        print(f"  {col}  held-out {h:.3f}   in-sample {i:.3f}   n={n}")
    print(f"  train variance explained: " +
          ", ".join(f"{v:.1%}" for v in pca.explained_variance_ratio_))

    section("AXES, IN GAME UNITS (bottom fifth -> top fifth)")
    UN = {"hs": "headshot %", "assists": "assists/map", "plants": "plants/map",
          "clutch_att": "clutches/map", "first_engagement": "opening duels/map",
          "deaths": "deaths/map"}
    rawv = M[["player_id", "year"]].merge(F.seasons(F.build()), on=["player_id", "year"])
    for pc in ("PC1", "PC2"):
        q = pd.qcut(M[pc].values if pc in M else sc[pc].values, 5, labels=False)
        print(f"  {pc}:")
        for f, lab in UN.items():
            print(f"    {lab:<20}{rawv.loc[q==0, f].mean():>8.2f} -> {rawv.loc[q==4, f].mean():>8.2f}")

    section("DUELIST POPULATION (label guard applied: >=70% of maps in role)")
    du, gdu = G[G.role == "duelist"], G[(G.role == "duelist") & G.label_ok]
    print(f"  duelist-labelled seasons        {len(du)}")
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
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    XY = sc[["PC1", "PC2"]].values
    rng = np.random.default_rng(0)
    Z = rng.multivariate_normal(XY.mean(0), np.cov(XY, rowvar=False), len(XY))
    for k in (2, 3, 4):
        km = KMeans(k, n_init=20, random_state=0).fit(XY)
        kz = KMeans(k, n_init=20, random_state=0).fit(Z)
        print(f"  k={k}  separation: your data {silhouette_score(XY, km.labels_):.3f}   "
              f"structureless cloud {silhouette_score(Z, kz.labels_):.3f}")
    km = KMeans(3, n_init=20, random_state=0).fit(XY)
    d3 = km.transform(XY); near = np.sort(d3, axis=1)
    ratio = near[:, 0] / near[:, 1]
    print(f"  at k=3, player-seasons within 75% of a boundary: "
          f"{(ratio >= .75).sum()} ({(ratio >= .75).mean():.0%})")
