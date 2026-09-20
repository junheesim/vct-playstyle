"""Step 12 -- is a player's agent THEIRS, or the team's assignment?

Q27 asks whether to control for agent. That only makes sense if agent is a confound.
If players carry their agents across rosters it is a personal signature, and removing
it destroys real information.

Test: when a player changes team, do they keep their agent? Compared against players
who stayed -- agents drift anyway with patches and new releases, so the STAYER rate
is the baseline, not 100%.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src"); import features as F, data as D
from step11_role_independence import ROLE

d  = F.build(); b = d[d.Side=="both"]
ps = (b.groupby(["player_id","year"])
        .agg(main=("Agents", lambda s: s.mode().iat[0]),
             team=("Team",   lambda s: s.mode().iat[0]),
             maps=("Map","size")).reset_index())
ps = ps[ps.maps >= 20]
pool = b.groupby(["player_id","year"]).Agents.apply(lambda s: set(s.str.lower()))
ps = ps.merge(pool.rename("pool"), on=["player_id","year"])
ps["role"] = ps.main.str.lower().str.split(",").str[0].map(ROLE)

rows = []
for y in (2023, 2024, 2025):
    a = ps[ps.year == y].set_index("player_id")
    c = ps[ps.year == y+1].set_index("player_id")
    for pid in a.index.intersection(c.index):
        x, z = a.loc[pid], c.loc[pid]
        j = len(x["pool"] & z["pool"]) / len(x["pool"] | z["pool"])
        rows.append({"moved": x.team != z.team, "same_main": x["main"] == z["main"],
                     "pool_overlap": j, "role": x.role, "same_role": x.role == z.role})
t = pd.DataFrame(rows)

print("="*66, "\nA. Did they keep the agent?\n", "="*66)
g = t.groupby("moved").agg(n=("same_main","size"), same_main=("same_main","mean"),
                           same_role=("same_role","mean"), pool_overlap=("pool_overlap","mean"))
g.index = ["stayed on team", "CHANGED TEAM"]
print(g.round(3).to_string())

print("\n" + "="*66, "\nB. By role -- who carries their agent?\n", "="*66)
m = t[t.moved]
print(m.groupby("role").agg(n=("same_main","size"), same_main=("same_main","mean"),
                            same_role=("same_role","mean")).round(3).to_string())

print("\n" + "="*66, "\nC. What the numbers mean for Q27\n", "="*66)
sm = g.loc["CHANGED TEAM","same_main"]; sr = g.loc["CHANGED TEAM","same_role"]
print(f"  After a team change: {sm:.0%} keep the exact agent, {sr:.0%} keep the ROLE.")
print(f"  Stayers:             {g.loc['stayed on team','same_main']:.0%} exact, "
      f"{g.loc['stayed on team','same_role']:.0%} role.")
print(f"  Gap attributable to the team: "
      f"{g.loc['stayed on team','same_main']-sm:+.0%} exact, "
      f"{g.loc['stayed on team','same_role']-sr:+.0%} role.")
