"""Features, one row per player-season. Encodes decisions 04, 05 and 06.

Nothing here screens or models. It builds the columns the decisions settled on and
labels which is which, so no downstream script has to remember.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

import data as D

# ---------------------------------------------------------------- decision 04
# Clean on all five screening tests (see decisions/07).
#
# `deaths` was here until decision 17 and is not any more. It passed every screening
# test, but it is an OUTCOME -- the count of fights you lost -- and the rule the rest
# of the set was chosen by is "count the situations, don't score the outcomes". It was
# also near-redundant: adjusted for quality it correlates +.74 with first_engagement,
# and dropping it moves every player by r = .987.
STRICT = ["first_engagement", "hs", "assists", "plants", "creds_per_round"]

# clutch_att: new-team stability .14 (interval includes 0), but role-independent
# (R2 .110) and 80% unique. Junhee's construct -- last alive => baiting, or anchoring
# and rotating late. Confirmed by 32 of 203 duelists clutching more than the average
# sentinel (diagnostics/role_labels.py independence). Kept, but see decisions/07: its
# new-team stability is .009 on 104 movers, so it travels with the roster. STRICT drops
# it and lands players in the same place (PC1 r .978), so nothing rests on the call.
# decision 09: attack-minus-defend opening duels. Separates the entry duelist who
# opens on attack from the aggressive defender. Only first_engagement has a persistent
# side asymmetry (.458); the other splittable features' gaps are noise (.06-.16).
SIDED = ["first_engagement_gap"]

STYLE = STRICT + ["clutch_att"] + SIDED

# No thresholds are applied anywhere. Verdicts are hand-written in decisions/07, and
# Phase 4 is run under BOTH sets to show the choice is not load-bearing.
# Empty since decision 07 closed: every candidate is now either in STYLE or in DROPPED.
# Kept because steps 05, 08 and 09 print STYLE + PENDING, and it is the slot a new
# candidate feature goes into while it is being screened.
PENDING = []

# Read by nothing; kept as the record of what was considered and rejected, in the
# same file as what was kept, so the two cannot drift apart.
DROPPED = {"econ":   "r with K/D high; vlr Econ is damage per 1000 credits, an "
                     "EFFICIENCY ratio, not spending. Replaced by creds_per_round.",
           "kast":   "r=+.45 with the K/D yardstick -- it tracks how good the player "
                     "is. Separately, 62% of it is explained by Kills+Assists+Deaths, "
                     "two of which are already features. Trading, the part worth "
                     "having, is not separable and is not measurable in this dataset.",
           "acs":    "quality. Was the yardstick until decision 06; kept as a column "
                     "for comparison only.",
           "adr":    "quality, r=.97 with ACS.",
           "Rating": "quality; heavily death-penalizing (r=-.45 with deaths), which "
                     "points its bend straight at the style set.",
           "Kills":  "quality.",
           "2k-5k":  "quality (multikills)."}

# ---------------------------------------------------------------- decision 06
QUALITY = "kd_ratio"     # a RATE, so fight volume cancels. See decisions/06.

# Everything averaged per season. Beyond STYLE + SIDED these are carried as columns
# for reference and comparison only; nothing downstream models them. `deaths` moved
# into this group in decision 17 -- still reported, no longer a feature.
ALL = STRICT + ["clutch_att", "deaths", "defuses", "kast", "econ", "acs", "adr"]


def pct(s):  # vlr writes percentages as '68%'
    return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")


@lru_cache(maxsize=None)
def _build() -> pd.DataFrame:
    ov = D.overview()
    ov = ov[ov.Side == "both"].copy()
    for c in ["Kills","Deaths","Assists","First Kills","First Deaths","Rating",
              "Average Combat Score","Average Damage Per Round"]:
        ov[c] = pd.to_numeric(ov[c], errors="coerce")
    ov["hs"]   = pct(ov["Headshot %"])
    ov["kast"] = pct(ov["Kill, Assist, Trade, Survive %"])
    # decision 04 -- count the situations, don't score the outcomes.
    # FK+FD = opening duels TAKEN (style). FK/(FK+FD) = duels WON (quality).
    ov["first_engagement"] = ov["First Kills"] + ov["First Deaths"]
    ov = ov.rename(columns={"Deaths":"deaths","Assists":"assists",
                            "Average Combat Score":"acs","Average Damage Per Round":"adr"})

    ks = D._stack("kills_stats")
    ks = ks[ks.Map != "All Maps"].copy()
    for c in ["1v1","1v2","1v3","1v4","1v5","Econ","Spike Plants","Spike Defuses"]:
        ks[c] = pd.to_numeric(ks[c], errors="coerce")
    ks["clutch_att"] = ks[["1v1","1v2","1v3","1v4","1v5"]].sum(axis=1)   # attempts, not wins
    ks = ks.rename(columns={"Spike Plants":"plants","Spike Defuses":"defuses","Econ":"econ"})

    # kills_stats has no Side column, so it joins to the 'both' rows only.
    d = ov.merge(ks[D.MATCH + ["Map","Player","clutch_att","plants","defuses","econ"]],
                 on=D.MATCH + ["Map","Player"], how="left")

    # Inverting Econ recovers the spending half alone; ADR cancels.
    d["creds_per_round"] = 1000 * d.adr / d.econ.replace(0, np.nan)
    d.loc[~d.creds_per_round.between(500, 8000), "creds_per_round"] = np.nan

    d = d.merge(map_results(), on=D.MATCH + ["Map", "Team"], how="left")
    return d.merge(side_gaps(), on=D.MATCH + ["Map", "player_id"], how="left")


def build() -> pd.DataFrame:
    """Per player-map rows with every candidate feature and the map result attached.

    Cached, because several scripts rebuild it repeatedly in one run. The copy keeps
    callers that add columns from corrupting the shared frame."""
    return _build().copy()


def side_gaps() -> pd.DataFrame:
    """attack-minus-defend opening duels, per player-MAP.  decision 09.

    Computed per map rather than per season so that every downstream diagnostic
    (reliability, split-half, residualization) treats it like any other feature.
    """
    ov = D.overview()
    for c in ["First Kills", "First Deaths"]:
        ov[c] = pd.to_numeric(ov[c], errors="coerce")
    ov["fe"] = ov["First Kills"] + ov["First Deaths"]
    key = D.MATCH + ["Map", "player_id"]
    w = (ov[ov.Side.isin(["attack", "defend"])]
           .set_index(key + ["Side"]).fe.unstack("Side"))
    return (w["attack"] - w["defend"]).rename("first_engagement_gap").reset_index()


def map_results() -> pd.DataFrame:
    """Did the player's team win this map? Used for team strength, and as an
    independent quality check that cannot be padded by individual stats."""
    ms = D._stack("maps_scores")
    ms = ms[ms.Map != "All Maps"].copy()
    for c in ["Team A Score", "Team B Score"]:
        ms[c] = pd.to_numeric(ms[c], errors="coerce")
    return pd.concat([
        ms.assign(Team=ms["Team A"], won=(ms["Team A Score"] > ms["Team B Score"]).astype(float)),
        ms.assign(Team=ms["Team B"], won=(ms["Team B Score"] > ms["Team A Score"]).astype(float)),
    ])[D.MATCH + ["Map", "Team", "won"]]


def seasons(d: pd.DataFrame, min_maps: int = D.MIN_MAPS) -> pd.DataFrame:
    """One row per player-season: feature means, the quality yardstick, team win rate."""
    g = d.groupby(["player_id", "year"])
    out = g[ALL + SIDED].mean()
    out["maps"]     = g.size()
    out["team_win"] = g.won.mean()
    tot = g[["Kills", "deaths", "First Kills", "First Deaths"]].sum()
    out["kd_ratio"] = tot.Kills / tot.deaths                            # decision 06
    out["fk_win"]   = tot["First Kills"] / (tot["First Kills"] + tot["First Deaths"])
    return out[out.maps >= min_maps].reset_index()


def reliability(d: pd.DataFrame, col: str, ps: pd.DataFrame, n_splits: int = 200):
    """Split-half reliability, averaged over many RANDOM splits.

    decision 05. A single odd/even split is one arbitrary partition, and for a rare
    feature (defuses: 10 events a season) the resulting correlation swings widely --
    ordering maps differently moved defuses from .497 to .410, enough to push its
    corrected stability above 1.0, which is impossible. Averaging many random splits
    removes the dependence on that arbitrary choice and gives an interval.

    Splits are RANDOM, not first-half vs second-half: both halves must span the whole
    season, or genuine mid-season change is miscounted as measurement noise.
    """
    keep = pd.MultiIndex.from_frame(ps[["player_id", "year"]])
    d = d[["player_id", "year", col]].dropna()
    d = d[pd.MultiIndex.from_frame(d[["player_id", "year"]]).isin(keep)]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_splits):
        h = d.assign(half=rng.integers(0, 2, len(d)))
        w = h.groupby(["player_id", "year", "half"])[col].mean().unstack("half").dropna()
        if w.shape[1] < 2 or len(w) < 30:
            continue
        r = w[0].corr(w[1])
        if pd.notna(r) and r > 0:
            out.append(2*r / (1 + r))
    if not out:
        return np.nan, np.nan, np.nan
    q = np.percentile(out, [5, 50, 95])
    return q[1], q[0], q[2]


if __name__ == "__main__":
    d  = build()
    ps = seasons(d)
    print(f"{len(ps)} player-seasons | map result joined {d.won.notna().mean():.1%}")
    print(f"\nSTRICT ({len(STRICT)}): {', '.join(STRICT)}")
    print(f"STYLE  ({len(STYLE)}): + {', '.join(set(STYLE)-set(STRICT))}")
    print(f"QUALITY: {QUALITY}\n")
    print(ps[STYLE + [QUALITY, "team_win"]].describe().loc[
          ["count","mean","std"]].round(2).to_string())
