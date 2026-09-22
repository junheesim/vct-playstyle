"""Features, one row per player-season. Encodes decisions 04, 05 and 06.

Nothing here screens or models. It builds the columns the decisions settled on and
labels which is which, so no downstream script has to remember.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

import data as D

# ---------------------------------------------------------------- decision 04
# THE feature set. One list, not two.
#
# There were two -- STRICT (5) and STYLE (7) -- because two features were contested
# and decision 07 committed to running Phase 4 under both. Decision 19 collapsed them:
# `first_engagement_gap` is gone, and `clutch_att` is simply a feature now rather than
# a special case carried with an escort. Two named sets was overhead the reader paid
# for and the analysis no longer needs.
#
# Dropped and why:
#   deaths                (decision 17) an OUTCOME -- the count of fights lost -- where
#                         the rule is to measure the tendency, not whether it went well.
#   first_engagement_gap  (decision 19) attack-minus-defend opening duels. Only 4 of the
#                         7 behaviors CAN be split by side -- plants, creds and clutches
#                         come from kills_stats, which has no Side column -- so it gave
#                         one behavior a dimension the others could not have. Second
#                         weakest reliability in the set, and a difference of raw counts
#                         rather than a rate per side-round (decision 18). Removing it
#                         moves every player by r = .982 and costs no coverage at all:
#                         775 player-seasons either way.
STYLE = ["first_engagement", "hs", "assists", "plants", "creds_per_round", "clutch_att"]

# Readable name and unit per feature, for figures and prose.
#
# Keyed on STYLE and asserted against it, so a feature cannot enter or leave the set
# without this moving with it. Two hand-maintained copies of this mapping used to
# live in `figures.py` and `quoted_numbers.py`; decision 17 dropped `deaths` and
# neither copy noticed, so a figure captioned "what each axis actually measures"
# showed a column that is not a model input and hid two that are.
UNITS = {"first_engagement": "opening duels / map",
         "hs":               "headshot %",
         "assists":          "assists / map",
         "plants":           "plants / map",
         "creds_per_round":  "credits / round",
         "clutch_att":       "clutches / map"}

assert set(UNITS) == set(STYLE), \
    f"UNITS must name exactly STYLE: {set(UNITS) ^ set(STYLE)}"

# ---------------------------------------------------------------- decision 06
QUALITY = "kd_ratio"     # a RATE, so fight volume cancels. See decisions/06.

# Everything averaged per season. Beyond STYLE + SIDED these are carried as columns
# for reference and comparison only; nothing downstream models them. `deaths` moved
# into this group in decision 17 -- still reported, no longer a feature.
# `deaths` and `first_engagement_gap` are carried as columns and reported, but are
# not features -- see the note above. The rest are here for comparison only.
ALL = STYLE + ["deaths", "defuses", "kast", "econ", "acs", "adr"]
SIDED = ["first_engagement_gap"]


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


def round_counts() -> pd.DataFrame:
    """Rounds played, per team per map, split by side.  decision 18.

    A map is first to 13, so it runs 13 to 48 rounds. Every per-map COUNT feature is
    therefore a rate multiplied by an opportunity count, and this is the column that
    prices that.

    Deliberately NOT merged into `build()`. Decision 18 keeps the published features
    per map, having measured that the opportunity count does not vary between players;
    this exists so `robustness.py rounds` can rebuild the alternative and show it.

    `atk` and `dfd` are REGULATION rounds only -- overtime alternates sides and vlr
    records it in one combined column -- so atk + dfd <= rounds.
    """
    ms = D._stack("maps_scores")
    ms = ms[ms.Map != "All Maps"].copy()
    for c in ["Team A Score", "Team B Score", "Team A Attacker Score",
              "Team A Defender Score", "Team B Attacker Score", "Team B Defender Score"]:
        ms[c] = pd.to_numeric(ms[c], errors="coerce")
    ms["rounds"] = ms["Team A Score"] + ms["Team B Score"]

    def side(x, y):
        """One team's view: its attacking rounds are the ones IT won attacking plus
        the ones the opponent won defending."""
        return ms.assign(Team=ms[f"Team {x}"],
                         atk=ms[f"Team {x} Attacker Score"] + ms[f"Team {y} Defender Score"],
                         dfd=ms[f"Team {x} Defender Score"] + ms[f"Team {y} Attacker Score"])

    return pd.concat([side("A", "B"), side("B", "A")])[
        D.MATCH + ["Map", "Team", "rounds", "atk", "dfd"]]


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
    """One row per player-season: feature means, the quality measure, team win rate."""
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
        # No `r > 0` filter. Discarding the splits that come out negative truncates
        # the distribution from below, so the median and the 5th percentile both read
        # HIGH -- and reliability is the quantity every disattenuation divides by, so
        # an inflated estimate quietly shrinks every corrected stability. The filter
        # never fired on this data (0 of 200 splits for every feature in the set),
        # which is exactly why it could sit here unnoticed.
        if pd.notna(r):
            out.append(2*r / (1 + r))
    if not out:
        return np.nan, np.nan, np.nan
    q = np.percentile(out, [5, 50, 95])
    return q[1], q[0], q[2]


if __name__ == "__main__":
    d  = build()
    ps = seasons(d)
    print(f"{len(ps)} player-seasons | map result joined {d.won.notna().mean():.1%}")
    print(f"\nFEATURES ({len(STYLE)}): {', '.join(STYLE)}")
    print(f"QUALITY: {QUALITY}\n")
    print(ps[STYLE + [QUALITY, "team_win"]].describe().loc[
          ["count","mean","std"]].round(2).to_string())
