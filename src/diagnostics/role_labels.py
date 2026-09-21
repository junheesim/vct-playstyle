"""How good is the role label, and does anything depend on it being wrong?

Decisions 07 (amendment), 11 (amendment) and 14. Was step11_role_independence,
step12_agent_ownership and step13_label_validity.

    independence  does a feature tell you anything the role label does not?
    ownership     how much of a season does one agent actually cover?
    validity      how weak is `main_agent`, and does it confound the cross-role claims?
"""
from _run import main, rule     # first: also puts src/ on the path

import pandas as pd
import statsmodels.formula.api as smf

import features as F
from components import loadings
from roles import ROLE, ROLES, role_of, shares, with_role



def independence():
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
    print(f"   average within-role spread    : {g['std'].mean():.3f}"
          f"  <- {g['std'].mean()/g['mean'].std():.1f}x larger")

    q  = p.dropna(subset=["clutch_att"])
    du, se = q[q.role=="duelist"], q[q.role=="sentinel"]
    n = (du.clutch_att > se.clutch_att.mean()).sum()
    print(f"\n   duelists with a HIGHER clutch rate than the average sentinel: "
          f"{n} of {len(du)} ({n/len(du):.0%})")
    print("   -- these are duelists being played as anchors, against role convention.")


def ownership():
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

    rule("A. Did they keep the agent?", 66)
    g = t.groupby("moved").agg(n=("same_main","size"), same_main=("same_main","mean"),
                               same_role=("same_role","mean"), pool_overlap=("pool_overlap","mean"))
    g.index = ["stayed on team", "CHANGED TEAM"]
    print(g.round(3).to_string())

    rule("B. By role -- who carries their agent?", 66)
    m = t[t.moved]
    print(m.groupby("role").agg(n=("same_main","size"), same_main=("same_main","mean"),
                                same_role=("same_role","mean")).round(3).to_string())

    rule("C. What the numbers mean for Q27", 66)
    sm = g.loc["CHANGED TEAM","same_main"]; sr = g.loc["CHANGED TEAM","same_role"]
    print(f"  After a team change: {sm:.0%} keep the exact agent, {sr:.0%} keep the ROLE.")
    print(f"  Stayers:             {g.loc['stayed on team','same_main']:.0%} exact, "
          f"{g.loc['stayed on team','same_role']:.0%} role.")
    print(f"  Gap attributable to the team: "
          f"{g.loc['stayed on team','same_main']-sm:+.0%} exact, "
          f"{g.loc['stayed on team','same_role']-sr:+.0%} role.")


def validity():
    P = shares(); P["role"] = role_of(P.main)
    print("A. HOW WEAK IS THE LABEL?\n")
    print(f"  agents played per player-season      median {P.n_agents.median():.0f}")
    print(f"  share of maps on the modal AGENT     median {P.agent_share.median():.0%}")
    print(f"  share of maps on the modal ROLE      median "
          f"{P[ROLES].max(axis=1).median():.0%}")
    print("  -> the ROLE label is solid; the AGENT label is not.")

    M, L, sc = loadings(F.STYLE)
    D = pd.concat([M[["player_id", "year"]], sc], axis=1).merge(P, on=["player_id", "year"])
    print("\nB. THE ROLE NUMBER, THREE WAYS\n")
    for expr, lab in [("C(role)", "hard 4-way label (understates)"),
                      ("duelist + initiator + controller", "the four role SHARES"),
                      ("duelist", "duelist SHARE alone")]:
        print(f"  R2 of PC1 from {lab:<34}: {smf.ols(f'PC1 ~ {expr}', D).fit().rsquared:.3f}")

    print("\nC. THE CLEAN CROSS-ROLE TEST -- players who NEVER played a duelist agent\n")
    pure = D[D.duelist == 0]; med = D.PC1.median()
    print(f"  {len(pure)} of {len(D)} player-seasons, of which "
          f"{(pure.PC1>med).sum()} ({(pure.PC1>med).mean():.0%}) are above the aggression median")
    print(f"  R2 of PC1 from role among them: {smf.ols('PC1 ~ C(role)', pure).fit().rsquared:.3f}"
          f"   (vs {smf.ols('PC1 ~ C(role)', D).fit().rsquared:.3f} overall)")
    print("\n  " + pure.nlargest(8, "PC1")[["handle","year","role","main","PC1"]]
            .round(2).to_string(index=False).replace("\n", "\n  "))

    print("\nD. THE CLUTCH CLAIM, CLEANED\n")
    du, se = P[P.duelist >= .8], P[(P.sentinel + P.controller) >= .8]
    n = (du.clutch_att > se.clutch_att.mean()).sum()
    print(f"  near-pure duelists (>=80% duelist maps)          {len(du)}")
    print(f"  near-pure sentinel/controller                    {len(se)}")
    print(f"  duelists exceeding the sent/ctrl mean clutch rate {n} ({n/len(du):.0%})")
    print("\n  " + du.nlargest(6,"clutch_att")[["handle","year","main","duelist","clutch_att"]]
            .round(2).to_string(index=False).replace("\n", "\n  "))


CHECKS = {
    "independence": ("is a feature redundant with the role taxonomy?  (decision 07)", independence),
    "ownership":    ("how much of a season is one agent?  (decision 14)", ownership),
    "validity":     ("how weak is `main_agent`, and what does it cost?  (decision 11)", validity),
}

if __name__ == "__main__":
    main(CHECKS, __doc__)
