"""Loading and scoping. Implements decisions 01-03; see `decisions/`.

Nothing in here makes an analytical choice. It produces the table every later step
starts from, with the grain, window and identity already settled.
"""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")

RAW    = pathlib.Path("data/raw")
YEARS  = range(2023, 2027)                       # decision 02
MATCH  = ["Tournament", "Stage", "Match Type", "Match Name", "year"]   # decision 01
GRAIN  = MATCH + ["Map", "player_id", "Side"]
QUAL   = "challengers|qualifier"


def _stack(kind: str, sub: str = "matches") -> pd.DataFrame:
    """Same filename in every year folder, so this is a concat, not a join."""
    return pd.concat(
        [pd.read_csv(RAW / f"vct_{y}/{sub}/{kind}.csv", low_memory=False).assign(year=y)
         for y in YEARS], ignore_index=True)


def player_ids() -> pd.DataFrame:
    """handle + year -> stable player_id.  decision 03."""
    ids = _stack("players_ids", sub="ids").dropna(subset=["Player ID"])
    ids["player_id"] = ids["Player ID"].astype(int)
    return ids[["Player", "year", "player_id"]].drop_duplicates(["Player", "year"])


def region_of(tournament: str) -> str:
    for r in ("Americas", "EMEA", "Pacific", "China"):
        if r in tournament:
            return r
    return "International"


def overview(verbose: bool = False) -> pd.DataFrame:
    """One row per player x map x side, scoped and identified."""
    df = _stack("overview")
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

    df["region"] = df.Tournament.map(region_of)
    assert not df.duplicated(GRAIN).any(), "grain is not unique"

    if verbose:
        print(f"  raw {n0:>7,} -> drop All Maps {n1:>7,} -> drop 2023 qualifiers {n2:>7,}")
        print(f"  {df.player_id.nunique()} players, {df.groupby(MATCH).ngroups:,} matches, "
              f"{df.year.min()}-{df.year.max()}")
    return df


MIN_MAPS = 15                                    # decision 13


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
