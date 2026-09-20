"""Step 13 -- how good is the `main_agent` label, and does it confound the
cross-role findings?

`main_agent` is the MODE of a player-season's agents. A player plays a median of 6
agents a season and spends only ~43% of maps on their modal one, so the label is much
weaker than it looks. Any claim of the form "an X-role player who behaves like a Y"
must therefore check what the player ACTUALLY played, not what they are labelled.
"""
import sys, pandas as pd, numpy as np, statsmodels.formula.api as smf
sys.path.insert(0, "src")
import features as F
from step11_role_independence import ROLE
from components import loadings


def shares():
    """Per player-season: fraction of maps in each Riot role, plus the modal agent."""
    d = F.build(); ps = F.seasons(d); b = d[d.Side == "both"].copy()
    b["r"] = b.Agents.str.lower().map(ROLE)
    keep = pd.MultiIndex.from_frame(ps[["player_id", "year"]])
    b = b[pd.MultiIndex.from_frame(b[["player_id", "year"]]).isin(keep)]
    sh = b.groupby(["player_id", "year"]).r.value_counts(normalize=True).unstack().fillna(0)
    lab = b.groupby(["player_id", "year"]).agg(
        handle=("handle", "last"),
        main=("Agents", lambda s: s.mode().iat[0]),
        agent_share=("Agents", lambda s: s.value_counts(normalize=True).iat[0]),
        n_agents=("Agents", "nunique"))
    return ps.merge(sh, on=["player_id", "year"]).merge(lab, on=["player_id", "year"])


if __name__ == "__main__":
    P = shares(); P["role"] = P.main.str.lower().map(ROLE)
    print("A. HOW WEAK IS THE LABEL?\n")
    print(f"  agents played per player-season      median {P.n_agents.median():.0f}")
    print(f"  share of maps on the modal AGENT     median {P.agent_share.median():.0%}")
    print(f"  share of maps on the modal ROLE      median "
          f"{P[['duelist','initiator','controller','sentinel']].max(axis=1).median():.0%}")
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
