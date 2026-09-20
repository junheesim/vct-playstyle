"""Step 10 -- every test, every feature, one table.

No single number decides. A feature has to survive five separate ways of failing:
  1 STYLE?      r with the K/D yardstick -- is it really a quality measure?
  2 MEASURABLE? reliability -- can one season estimate it at all?
  3 PERSISTS?   true stability = observed / reliability
  4 THE PLAYER? stability among players who CHANGED TEAM
  5 DISTINCT?   VIF -- do the other features already say this?
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src"); import features as F
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

d  = F.build(); ps = F.seasons(d)
tm = (d[d.Side=="both"].groupby(["player_id","year"]).Team
        .agg(lambda s: s.mode().iat[0]).rename("team"))
q  = ps.merge(tm, on=["player_id","year"])
SET = F.STYLE + ["defuses", "kast"]

rows = []
for f in SET:
    rel, lo, hi = F.reliability(d, f, ps)
    pv = q.pivot_table(index="player_id", columns="year", values=f)
    tv = q.pivot_table(index="player_id", columns="year", values="team", aggfunc="first")
    pr = pd.concat([pd.DataFrame({"t":pv[y],"t1":pv[y+1],"same":tv[y]==tv[y+1]}).dropna()
                    for y in (2023,2024,2025) if y+1 in pv.columns])
    obs   = pr.t.corr(pr.t1)
    moved = pr[~pr.same]
    rows.append({"feature": f,
                 "1_r_kd": ps[f].corr(ps.kd_ratio),
                 "2_rel": rel,
                 "3_true": min(obs/rel, 1.0) if pd.notna(rel) else np.nan,
                 "4_newteam": moved.t.corr(moved.t1), "n_moved": len(moved),
                 "cover": ps[f].notna().mean()})
t = pd.DataFrame(rows).set_index("feature")

Z = ps[SET].dropna().apply(lambda s:(s-s.mean())/s.std()).assign(const=1.0)
t["5_vif"] = pd.Series({f: vif(Z.values,i) for i,f in enumerate(Z.columns) if f!="const"})

def flag(r):
    bad = []
    if abs(r["1_r_kd"]) > .40:  bad.append("quality")
    if r["2_rel"]      < .50:   bad.append("noisy")
    if r["4_newteam"]  < .30:   bad.append("team")
    if r["5_vif"]      > 5:     bad.append("redundant")
    return ",".join(bad) or "clean"
t["concerns"] = t.apply(flag, axis=1)
print(t.round(3).to_string())
print("""
  1_r_kd    >|.40| = looks like quality
  2_rel     <.50   = can barely be measured in one season; 3_true is then unstable
  3_true    capped at 1.0 (defuses exceeds it -- correction out of range)
  4_newteam <.30   = travels with the roster, not the player
  5_vif     >5     = the other features already say this""")
