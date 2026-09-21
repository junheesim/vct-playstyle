"""Every number the prose quotes, printed from the data.

Hand-copied figures drifted repeatedly while this project was being built -- a career
mean pasted into a single-season label, a percentile computed on a different
population than the claim implied. This is the single source of truth: run it, and
update the prose from its output rather than from memory.

    python src/quoted_numbers.py report     what the site's report quotes
    python src/quoted_numbers.py decisions  what decisions/ 07, 10, 11 and 12 quote

(Named quoted_numbers, not numbers: a module called `numbers` shadows the stdlib one
that numpy imports, and the whole package fails to load.)
"""
import argparse

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
from components import eigenvalues, loadings, null_cutoffs, replication
from examples import guarded
from roles import ROLES, shares


def section(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


def sub(t):
    print(f"\n-- {t}")


def fit(cols=None):
    cols = cols or F.STYLE
    M = St.build(cols)
    Z = pca.standardise(M, cols)
    model, flip = pca.fit(Z.values, cols)
    return M, pca.loadings(model, flip, cols), pca.scores(model, flip, Z, M.index)


def yoy(D_, y0, y1, col):
    a = D_[D_.year == y0].set_index("player_id")[col]
    b = D_[D_.year == y1].set_index("player_id")[col]
    j = a.index.intersection(b.index)
    return a.loc[j].corr(b.loc[j]), len(j)


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
    X = pca.standardise(M, F.STYLE).values
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


def decisions():
    M, L, sc = fit()
    # M's index is gapped (dropna); raw/`P` are rebuilt with a RangeIndex. Pandas
    # aligns on LABELS, so mixing them silently correlates the wrong rows. One index.
    M = M.reset_index(drop=True)
    sc = sc.reset_index(drop=True)
    P = pd.concat([M[["player_id", "year", "role", "handle", "main_agent"]], sc], axis=1)
    raw = M[["player_id", "year"]].merge(F.seasons(F.build()), on=["player_id", "year"])

    section("DECISION 07 -- scope of each feature set")
    for name, cols in [(f"STRICT ({len(F.STRICT)})", F.STRICT), (f"STYLE ({len(F.STYLE)})", F.STYLE)]:
        print(f"  {name:<12}complete player-seasons {len(St.build(cols)):>5}")
    print(f"  floor in use: {D.MIN_MAPS} maps  (decision 13)")

    section("DECISION 11 -- how many components")
    X = pca.standardise(M, F.STYLE).values
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
    Xs = pca.standardise(St.build(F.STRICT), F.STRICT).values
    evs = eigenvalues(Xs)
    print(f"  STRICT (6 features) retains: {(evs > null_cutoffs(Xs,'shuffle')).sum()}")
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
        print(f"  {h} {y}  labelled {x.role:<11}{x.duelist:>5.0%} duelist maps   PC1 {x.PC1:+.2f}")

    section("DECISION 11 -- PC2")
    q = pd.qcut(sc.PC2.values, 5, labels=False)
    UN = {"hs":"headshot %","assists":"assists/map","plants":"plants/map",
          "clutch_att":"clutch situations","deaths":"deaths/map",
          "creds_per_round":"creds/round","first_engagement":"opening duels/map"}
    print("  bottom fifth -> top fifth, in game units:")
    for f, lab in UN.items():
        a, b = raw.loc[q==0, f].mean(), raw.loc[q==4, f].mean()
        print(f"    {lab:<20}{a:>8.2f} -> {b:>8.2f}   ({b-a:+.2f})")
    # Correlations with the FEATURES are against the residualized, standardised
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
        print("     nearest the centre: " +
              ", ".join(f"{r.handle} ({r.main_agent})" for _, r in near.iterrows()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="*", default=[], choices=["report", "decisions", []],
                    help="default: both")
    a = ap.parse_args()
    for name in (a.which or ["report", "decisions"]):
        {"report": report, "decisions": decisions}[name]()
