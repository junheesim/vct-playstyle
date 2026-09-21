"""Loading and scoping. Implements decisions 01-03; see `decisions/`.

Nothing in here makes an analytical choice. It produces the table every later step
starts from, with the grain, window and identity already settled.
"""

from functools import lru_cache

import pandas as pd

import paths

RAW      = paths.RAW
YEARS    = range(2023, 2027)                     # decision 02
MATCH    = ["Tournament", "Stage", "Match Type", "Match Name", "year"]   # decision 01
GRAIN    = MATCH + ["Map", "player_id", "Side"]
QUAL     = "challengers|qualifier"
MIN_MAPS = 15                                    # decision 13


def _require_raw() -> None:
    """Say what is wrong once, instead of five FileNotFoundErrors deep in a concat.

    `data/raw` is a symlink to the Kaggle dump and is not committed, so this is the
    first thing that breaks on a fresh clone or if the link target moves."""
    if RAW.is_dir():
        return
    hint = f" (broken symlink -> {RAW.readlink()})" if RAW.is_symlink() else ""
    raise FileNotFoundError(
        f"{RAW} not found{hint}.\n"
        "  Unpack the Kaggle dataset 'ryanluong1/valorant-champion-tour-2021-2023-data'\n"
        f"  so that {RAW}/vct_2023/matches/overview.csv exists, or symlink an existing copy:\n"
        f"    ln -s /path/to/data/raw {RAW}")


@lru_cache(maxsize=None)
def _stack(kind: str, sub: str = "matches") -> pd.DataFrame:
    """Same filename in every year folder, so this is a concat, not a join.

    Cached: `overview` is rebuilt by several scripts in one run, and re-reading the
    same CSVs each time is pure waste. Callers must not mutate the result in place.
    """
    _require_raw()
    return pd.concat(
        [pd.read_csv(RAW / f"vct_{y}/{sub}/{kind}.csv", low_memory=False).assign(year=y)
         for y in YEARS], ignore_index=True)


@lru_cache(maxsize=None)
def player_ids() -> pd.DataFrame:
    """handle + year -> stable player_id.  decision 03."""
    ids = _stack("players_ids", sub="ids").dropna(subset=["Player ID"])
    ids["player_id"] = ids["Player ID"].astype(int)
    return ids[["Player", "year", "player_id"]].drop_duplicates(["Player", "year"])


LEAGUES = ("Americas", "EMEA", "Pacific", "China")

# Renames, and one mislabel. vlr gives these the same franchise slot and a continuous
# roster across the change, so they are one organisation for any "did the player
# change team?" question -- 10 of 417 year-over-year pairs read as roster moves
# otherwise. `Mega Minors` is different in kind: it is not a rebrand but a wrong
# label on NRG's Americas slot for 2025-26, confirmed by the slot, the roster
# carry-over (ethan, finesse, s0m) and NRG appearing zero times in those years.
TEAM_ALIAS = {"Mega Minors": "NRG Esports",     # mislabel; corrected for display too
              "GIANTX":      "Giants Gaming",
              "TALON":       "Talon Esports",
              "KIWOOM DRX":  "DRX"}
MISLABELLED = {"Mega Minors"}


def league_of(tournament: str):
    """The regional league a tournament belongs to, or None for an international one.

    Masters and Champions are a TIER, not a region. Reading them as a fifth region
    put 58 player-seasons in an "International" bucket that no player belongs to --
    they have a home league, they just played enough international maps that quarter
    for the mode to tip.
    """
    for r in LEAGUES:
        if r in tournament:
            return r
    return None


def _regions(df: pd.DataFrame) -> pd.Series:
    """Region per row, from the TEAM's league rather than the tournament's name.

    A team's region is where it plays its league games, in any year. That covers the
    Chinese orgs in 2023, before China had a league in this data -- they appear only
    at international events, and the tournament name says nothing about where they
    are from. The 5 player-seasons whose team never plays a league game at all
    (Attacking Soul Esports) fall back to the player's own league elsewhere; all five
    of those players went on to play in China.
    """
    lg = df[df.league.notna()]
    by_team = lg.groupby("Team").league.agg(lambda s: s.mode().iat[0])
    by_player = lg.groupby("player_id").league.agg(lambda s: s.mode().iat[0])
    return df.Team.map(by_team).fillna(df.player_id.map(by_player))


@lru_cache(maxsize=None)
def _overview() -> tuple:
    df = _stack("overview").copy()
    n0 = len(df)

    # decision 01 -- 'All Maps' is the series total, i.e. the same stats summed.
    df = df[df.Map != "All Maps"]
    n1 = len(df)

    # decision 02 -- 2023 is a transition year; drop the amateur tier inside it.
    df = df[~(df.Tournament.str.contains(QUAL, case=False, na=False) & (df.year == 2023))]
    n2 = len(df)

    # decision 03 -- key on player_id, display a lowercased handle.
    df = df.merge(player_ids(), on=["Player", "year"], how="left")
    assert df.player_id.notna().all(), "unresolved handles: " + \
        str(sorted(df.loc[df.player_id.isna(), "Player"].unique())[:10])
    df["player_id"] = df.player_id.astype(int)
    df["handle"]    = df.Player.str.lower()

    df["league"] = df.Tournament.map(league_of)
    df["region"] = _regions(df)
    assert df.region.notna().all(), "unresolved region: " + \
        str(sorted(df.loc[df.region.isna(), "Team"].unique())[:10])

    # Canonical organisation, for continuity across a rename. `Team` stays as recorded
    # so a 2023 season is not shown under a name coined in 2025 -- except the one that
    # is simply wrong, which is corrected everywhere.
    df["org"] = df.Team.replace(TEAM_ALIAS)
    df["Team"] = df.Team.replace({k: v for k, v in TEAM_ALIAS.items() if k in MISLABELLED})

    assert not df.duplicated(GRAIN).any(), "grain is not unique"

    return df, (n0, n1, n2)


def overview(verbose: bool = False) -> pd.DataFrame:
    """One row per player x map x side, scoped and identified."""
    df, (n0, n1, n2) = _overview()
    df = df.copy()          # the cached frame is shared; hand out a private one
    if verbose:
        print(f"  raw {n0:>7,} -> drop All Maps {n1:>7,} -> drop 2023 qualifiers {n2:>7,}")
        print(f"  {df.player_id.nunique()} players, {df.groupby(MATCH).ngroups:,} matches, "
              f"{df.year.min()}-{df.year.max()}")
    return df


def player_seasons(df: pd.DataFrame, min_maps: int = MIN_MAPS) -> pd.DataFrame:
    """player_id x year, restricted to seasons with enough maps to estimate anything.

    15, not 20 (decision 13). Split-half reliability by season length puts the cliff at
    15: below it `clutch_att` falls to .10 and is unmeasurable. Between 15 and 25 maps
    it holds at .32, close to its .42 at 25-40. The headline features are reliable at
    any length (`first_engagement` .85 even at 5-15 maps), so the weakest feature sets
    the floor.
    """
    m = (df[df.Side == "both"].groupby(["player_id", "year"])
           .size().rename("maps").reset_index())
    return m[m.maps >= min_maps]


if __name__ == "__main__":
    ov = overview(verbose=True)
    ps = player_seasons(ov)
    print(f"\n  player-seasons at >={MIN_MAPS} maps: {len(ps)}")
    print("  " + ps.groupby("year").size().to_string().replace("\n", "\n  "))
