"""Named players who are extreme FOR THEIR ROLE, on each axis.

Every candidate must pass the label guard first (decision 11 amendment): the role
label must not be WRONG (modal agent's role == modal role) and must not be WEAK
(>=70% of maps in that role). `main_agent` is the mode of a season and covers only
~43% of a player's maps, so an unguarded "an X who plays like a Y" claim can just be
a mislabelled Y.
"""

import pandas as pd

import features as F
from components import loadings
from roles import ROLES, shares

MIN_ROLE_SHARE = .70


def guarded():
    """Player-seasons whose role label is safe to cite, with within-role positions.

    The guard is one condition now: did the player actually spend most of the season
    in the role they are labeled with. It used to be two, because `role` was the role
    of the single most-played AGENT and could disagree with the role the player
    actually played most -- so the first condition tested those two against each
    other. `role` IS the modal role now, and that comparison no longer means anything.
    """
    M, L, sc = loadings(F.STYLE)
    D = pd.concat([M[["player_id", "year", "handle", "team", "region", "role",
                      "role_share", "main_agent"]], sc], axis=1)
    P = shares()
    D = D.merge(P[["player_id", "year"] + ROLES], on=["player_id", "year"], how="left")
    D["label_ok"] = D.role_share >= MIN_ROLE_SHARE
    for c in ("PC1", "PC2"):                       # position within the player's own role
        g = D.groupby("role")[c]
        D[c + "_inrole"] = (D[c] - g.transform("mean")) / g.transform("std")
    return D


def table(D, axis, n=3):
    rows = []
    ok = D[D.label_ok]
    for r in ROLES:
        s = ok[ok.role == r]
        for lab, part in [("most", s.nlargest(n, axis + "_inrole")),
                          ("least", s.nsmallest(n, axis + "_inrole"))]:
            for _, x in part.iterrows():
                rows.append({"role": r, "end": lab, "handle": x.handle, "year": int(x.year),
                             "team": x.team, "agent": x.main_agent,
                             "role_share": f"{x.role_share:.0%}",
                             "raw": round(x[axis], 2), "vs_role": round(x[axis+"_inrole"], 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = guarded()
    print(f"{len(D)} player-seasons | {D.label_ok.sum()} pass the label guard "
          f"({D.label_ok.mean():.0%})\n")
    for axis, name in [("PC1", "AGGRESSION"), ("PC2", "ISOLATION / GUNPLAY")]:
        t = table(D, axis)
        print("=" * 84)
        print(f"{name} — most and least, WITHIN each role")
        print("=" * 84)
        for r in ROLES:
            s = t[t.role == r]
            print(f"\n  {r.upper()}  (role mean {D[D.role==r][axis].mean():+.2f})")
            print("   " + s.drop(columns="role").to_string(index=False).replace("\n", "\n   "))
        print()
