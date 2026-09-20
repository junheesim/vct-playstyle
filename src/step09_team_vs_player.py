"""Step 9 -- how much of each feature's persistence is the PLAYER, and how much is
the TEAM staying the same?

Year-over-year stability was read as "this is a player trait". But most players stay
on the same roster, in the same system, with the same teammates. Splitting the pairs
by whether the player CHANGED TEAM separates the two.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src"); import features as F

d  = F.build(); ps = F.seasons(d)
tm = (d[d.Side=="both"].groupby(["player_id","year"]).Team
        .agg(lambda s: s.mode().iat[0]).rename("team"))
q  = ps.merge(tm, on=["player_id","year"])

rows = []
for f in F.STYLE + F.PENDING + ["kd_ratio"]:
    pv = q.pivot_table(index="player_id", columns="year", values=f)
    tv = q.pivot_table(index="player_id", columns="year", values="team", aggfunc="first")
    pr = pd.concat([pd.DataFrame({"t": pv[y], "t1": pv[y+1], "same": tv[y] == tv[y+1]}).dropna()
                    for y in (2023,2024,2025) if y+1 in pv.columns])
    same, moved = pr[pr.same], pr[~pr.same]
    rows.append({"feature": f, "n_stay": len(same), "n_moved": len(moved),
                 "r_same_team": same.t.corr(same.t1),
                 "r_new_team":  moved.t.corr(moved.t1),
                 "drop": same.t.corr(same.t1) - moved.t.corr(moved.t1)})

t = pd.DataFrame(rows).set_index("feature").sort_values("drop")
print(t.round(3).to_string())
print("\n  r_new_team is the honest PLAYER-trait figure: the player kept it after")
print("  changing roster, system and teammates.")
print("  A large 'drop' means the feature travels with the TEAM, not the player.")
