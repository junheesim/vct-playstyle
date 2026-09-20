"""Step 11 -- does a feature tell you anything Riot's role label does not?

The five screening tests all asked "is this a real, measurable, individual trait?"
None asked whether the feature is REDUNDANT WITH THE ROLE TAXONOMY -- which is the
thing this project claims to go beyond.

Test: regress each feature on Riot role (from the player-season's modal agent) and
read R2. High R2 means the feature largely restates the role label.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src"); import features as F
import statsmodels.formula.api as smf

ROLE = {**{a:"duelist"    for a in ["jett","raze","reyna","phoenix","yoru","neon","iso","waylay"]},
        **{a:"initiator"  for a in ["sova","breach","skye","kayo","fade","gekko","tejo"]},
        **{a:"controller" for a in ["omen","brimstone","viper","astra","harbor","clove"]},
        **{a:"sentinel"   for a in ["killjoy","cypher","sage","chamber","deadlock","vyse"]}}

def with_role(ps, d):
    ag = (d[d.Side=="both"].groupby(["player_id","year"]).Agents
            .agg(lambda s: s.mode().iat[0]).rename("main_agent"))
    ps = ps.merge(ag, on=["player_id","year"])
    ps["role"] = ps.main_agent.str.lower().str.split(",").str[0].map(ROLE)
    return ps

if __name__ == "__main__":
    d = F.build(); p = with_role(F.seasons(d), d).dropna(subset=["role"])
    print(f"role assigned for {p.role.notna().mean():.0%} of player-seasons\n")
    print(f"  {'feature':<18}{'R2 from Riot role':>20}")
    for f, r2 in sorted(((f, smf.ols(f"{f} ~ C(role)", p.dropna(subset=[f])).fit().rsquared)
                         for f in F.STYLE), key=lambda x: -x[1]):
        tag = "IS largely role" if r2 > .30 else ("partly role" if r2 > .12 else "independent of role")
        print(f"  {f:<18}{r2:>20.3f}   {tag}")

    print("\n  clutch_att by role -- is there real variation INSIDE each role?")
    g = p.dropna(subset=["clutch_att"]).groupby("role").clutch_att.agg(["mean","std","size"])
    print("   " + g.round(3).to_string().replace("\n","\n   "))
    print(f"\n   between-role spread of means : {g['mean'].std():.3f}")
    print(f"   average within-role spread    : {g['std'].mean():.3f}  <- 2.5x larger")

    q  = p.dropna(subset=["clutch_att"])
    du, se = q[q.role=="duelist"], q[q.role=="sentinel"]
    n = (du.clutch_att > se.clutch_att.mean()).sum()
    print(f"\n   duelists with a HIGHER clutch rate than the average sentinel: "
          f"{n} of {len(du)} ({n/len(du):.0%})")
    print("   -- these are duelists being played as anchors, against role convention.")
