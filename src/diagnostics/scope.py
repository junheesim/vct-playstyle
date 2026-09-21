"""What is one row, which years belong in the window, and what to key on.

Decisions 01 (grain), 02 (era) and 03 (identity). Was step01_audit, step01b_probe,
step02_era, step02b_cost and step03_identity.

    grain       what does one row describe?
    duplicates  the 93 duplicate rows, and whether `Match Name` is a usable id
    era         why the row count collapses after 2022
    cost        what dropping 2023 Challengers costs, and whether 2026 is thin
    identity    handle vs player_id: collisions, renames, and the cost of each
"""
from _run import main, rule     # first: also puts src/ on the path

import pandas as pd

import paths

# These deliberately reach back to 2021, BEFORE the analysis window. The point of
# decision 02 is to justify the window, which needs the era it excludes.
AUDIT_YEARS = range(2021, 2027)
SCOPED_YEARS = range(2023, 2027)
MATCH = ["Tournament", "Stage", "Match Type", "Match Name", "year"]
KEY = ["Tournament", "Stage", "Match Type", "Match Name", "Map", "year"]
FLOOR = 15                      # decision 13


def overview(years=AUDIT_YEARS, kind="overview", sub="matches") -> pd.DataFrame:
    """Raw, unscoped. `data.overview()` is the scoped table; these checks predate it."""
    frames = [pd.read_csv(paths.RAW / f"vct_{y}/{sub}/{kind}.csv", low_memory=False).assign(year=y)
              for y in years if (paths.RAW / f"vct_{y}/{sub}/{kind}.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def played(years=AUDIT_YEARS) -> pd.DataFrame:
    """Decision 01 applied: real maps, both sides combined."""
    return overview(years).query("Map != 'All Maps' and Side == 'both'").copy()


# ------------------------------------------------------------------ grain
def grain():
    ov = overview()
    rule("A. Size", 70)
    print(ov.groupby("year").size().rename("rows").to_frame())

    rule("B. What does one row describe?", 70)
    # A candidate key is a set of columns that should identify a row uniquely.
    # If it does not, either the key is wrong or the table has repeats.
    for name, key in [
        ("match + map + player",        ["Tournament","Stage","Match Type","Match Name","Map","Player","year"]),
        ("match + map + player + side", ["Tournament","Stage","Match Type","Match Name","Map","Player","Side","year"]),
    ]:
        dup = ov.duplicated(key).sum()
        print(f"  {name:<30} duplicate rows: {dup:>7,}   {'UNIQUE' if dup == 0 else 'NOT unique'}")

    rule("C. Column values that are not what you would guess", 70)
    print("  Side:", ov.Side.value_counts().to_dict())
    print("\n  Map -- 12 most common:")
    print(ov.Map.value_counts().head(12).to_string())

    rule("D. Is 'Match Name' a match id?", 70)
    print(f"  distinct 'Match Name' values           {ov['Match Name'].nunique():>7,}")
    print(f"  distinct tournament+stage+type+name    {ov.groupby(MATCH).ngroups:>7,}")

    rule("E. Pick one player-match and print every row for it", 70)
    g = (ov[ov.Side == "both"]
           .groupby(["Tournament","Stage","Match Type","Match Name","Player","year"]).size())
    probe = g[g == 3].index[0]                      # a player with 3 rows in one match
    cols = ["Map","Side","Kills","Deaths","Assists","First Kills","Average Combat Score"]
    sel = ov.set_index(["Tournament","Stage","Match Type","Match Name","Player","year"]).loc[[probe]]
    print(f"  {probe[4]} -- {probe[3]} ({probe[0]})\n")
    print(sel[cols].to_string(index=False))


# ------------------------------------------------------------ duplicates
def duplicates():
    ov = overview()
    rule("Q3 -- the 93 duplicates", 70)
    full = KEY + ["Player", "Side"]
    dups = ov[ov.duplicated(full, keep=False)].sort_values(full)
    print(f"  rows involved: {len(dups)}   distinct keys: {dups.groupby(full).ngroups}")
    print(f"  by year: {dups.year.value_counts().sort_index().to_dict()}")
    print(f"  by tournament:\n{dups.Tournament.value_counts().head(5).to_string()}")

    print("\n  -- are the duplicate rows identical, or do they disagree? --")
    stat_cols = ["Kills","Deaths","Assists","Average Combat Score","Agents","Team"]
    print(f"  rows that are byte-identical copies: "
          f"{dups.duplicated(full + stat_cols, keep=False).sum()} / {len(dups)}")

    print("\n  -- first two conflicting keys, printed in full --")
    shown = 0
    for k, g in dups.groupby(full):
        if g[stat_cols].drop_duplicates().shape[0] > 1:
            print(f"\n  {k[3]} | {k[4]} | {k[6]} | {k[7]}")
            print(g[stat_cols].to_string(index=False))
            shown += 1
            if shown == 2:
                break
    if shown == 0:
        print("  none conflict -- every duplicate is an exact copy")

    rule("Q4 -- is 'Match Name' a usable id?", 70)
    mn = (ov[["Match Name","Tournament","Stage","Match Type","year"]].drop_duplicates()
            .groupby("Match Name").size().sort_values(ascending=False))
    print(f"  Match Names used in more than one tournament/stage/type: {(mn>1).sum():,}")
    print(f"  worst offender is used {mn.iloc[0]} times\n")
    worst = mn.index[0]
    print(f"  '{worst}' appears in:")
    print(ov[ov["Match Name"] == worst][["year","Tournament","Stage","Match Type"]]
            .drop_duplicates().sort_values(["year","Tournament"]).head(12).to_string(index=False))

    print("\n  -- what it costs you --")
    sel = ov[(ov["Match Name"] == worst) & (ov.Side == "both") & (ov.Map != "All Maps")]
    print(f"  groupby('Match Name')                      -> {sel.groupby('Match Name').ngroups:>6,} group(s)")
    print(f"  groupby(tournament+stage+type+name)        -> {sel.groupby(KEY[:4]).ngroups:>6,} group(s)")
    print(f"  ...for the SAME {len(sel):,} rows.")


# --------------------------------------------------------------------- era
def era():
    ov = played()
    ov["Average Combat Score"] = pd.to_numeric(ov["Average Combat Score"], errors="coerce")
    per_year = ov.groupby("year")
    t = pd.DataFrame({
        "rows":     per_year.size(),
        "teams":    per_year.Team.nunique(),
        "players":  per_year.Player.nunique(),
        "matches":  per_year.apply(lambda g: g.groupby(MATCH[:-1]).ngroups),
    })
    t["matches_per_team"] = (2 * t.matches / t.teams).round(1)   # a match involves 2 teams
    t["rows_per_match"]   = (t.rows / t.matches).round(1)
    t["players_per_team"] = (t.players / t.teams).round(1)
    rule("WORLD 1 vs 2 vs 3", 78)
    print(t[["rows","teams","matches","matches_per_team",
             "rows_per_match","players_per_team"]].to_string())

    rule("What is IN the extra data? -- spread of player skill (season mean ACS)", 78)
    ps = (ov.groupby(["Player","year"])
            .agg(acs=("Average Combat Score","mean"), maps=("Map","size")).reset_index())
    q = ps[ps.maps >= 5].groupby("year").acs.describe(percentiles=[.05,.25,.5,.75,.95])
    q["p95_minus_p5"] = (q["95%"] - q["5%"]).round(1)
    print(q[["count","mean","std","5%","50%","95%","p95_minus_p5"]].round(1).to_string())

    rule("Team count by region-tier proxy (tournament name)", 78)
    # First match wins, most specific first. An overwriting loop mislabels
    # "Champions Tour NA Stage 1: Challengers 2" as a franchised league event.
    def tier(s):
        s = s.lower()
        if "challengers" in s or "qualifier" in s:          return "Challengers / qualifier"
        if "game changers" in s:                            return "Game Changers"
        if "champions" in s and "champions tour" not in s:  return "Champions (intl)"
        if "masters" in s:                                  return "Masters (intl)"
        return "League / other"
    print(pd.crosstab(ov.year, ov.Tournament.map(tier)).to_string())


# -------------------------------------------------------------------- cost
def cost():
    ov = played(SCOPED_YEARS)
    ov["chal"] = ov.Tournament.str.contains("challengers|qualifier", case=False, na=False)

    rule("Q6 -- cost of dropping Challengers rows inside 2023", 76)
    y23 = ov[ov.year == 2023]
    chal, franch = set(y23[y23.chal].Player), set(y23[~y23.chal].Player)
    print(f"  2023 rows            {len(y23):>6,}   of which Challengers "
          f"{y23.chal.sum():>6,} ({y23.chal.mean():.1%})")
    print(f"  2023 players         {y23.Player.nunique():>6,}")
    print(f"  ...only ever seen in Challengers: {len(chal - franch):>4,}")
    print(f"  ...in BOTH (they lose maps, not their season): {len(chal & franch):>4,}")

    for label, d in [("keep all 2023", ov),
                     ("drop 2023 Challengers", ov[~(ov.chal & (ov.year == 2023))])]:
        m = d.groupby(["Player","year"]).size().rename("maps").reset_index()
        surv = m[m.maps >= FLOOR]
        print(f"\n  {label:<24} player-seasons with >={FLOOR} maps, by year:")
        print("    " + "  ".join(f"{y}:{n:>4}" for y, n in surv.groupby("year").size().items())
              + f"   TOTAL {len(surv)}")

    rule("Q7 -- is 2026 actually thin?", 76)
    d = ov[~(ov.chal & (ov.year == 2023))]
    m = d.groupby(["Player","year"]).size().rename("maps").reset_index()
    print(m.groupby("year").maps.describe(percentiles=[.25,.5,.75])
           [["count","mean","25%","50%","75%","max"]].round(1).to_string())

    print("\n  event composition (rows):")
    def event(s):
        s = s.lower()
        if "challengers" in s or "qualifier" in s:          return "Challengers"
        if "champions" in s and "champions tour" not in s:  return "Champions"
        if "masters" in s:                                  return "Masters"
        return "League"
    print(pd.crosstab(d.year, d.Tournament.map(event)).to_string())

    print(f"\n  players surviving >={FLOOR} maps, and how many appear in the PRIOR year too:")
    s = m[m.maps >= FLOOR]
    for y in (2024, 2025, 2026):
        cur, prev = set(s[s.year == y].Player), set(s[s.year == y-1].Player)
        print(f"    {y}: {len(cur):>3} players, {len(cur & prev):>3} also qualified in {y-1}"
              f"  -> usable year-over-year pairs: {len(cur & prev)}")


# ---------------------------------------------------------------- identity
def identity():
    ids = (overview(SCOPED_YEARS, kind="players_ids", sub="ids")
             .dropna(subset=["Player ID"]))
    ids["Player ID"] = ids["Player ID"].astype(int)
    ids = ids.drop_duplicates(["Player", "Player ID", "year"])
    print(f"id file: {len(ids):,} handle-id-year rows | "
          f"{ids.Player.nunique():,} handles | {ids['Player ID'].nunique():,} ids")

    rule("A. COLLISION -- one handle, several player_ids", 72)
    a = ids.groupby("Player")["Player ID"].nunique()
    print(f"  handles mapping to >1 id: {(a>1).sum()} of {len(a):,}")
    for h in a[a > 1].index[:6]:
        print("   " + ids[ids.Player == h].sort_values("year")[["Player","Player ID","year"]]
                .to_string(index=False).replace("\n", "\n    "))

    rule("B. RENAME -- one player_id, several handles", 72)
    b = ids.groupby("Player ID")["Player"].nunique()
    print(f"  ids mapping to >1 handle: {(b>1).sum()} of {len(b):,}")
    for i in b[b > 1].index[:6]:
        r = ids[ids["Player ID"] == i].sort_values("year")
        print(f"    id {i}: " + "  ".join(f"{y}={p}" for p, y in zip(r.Player, r.year)))

    print("\n  -- COST: how many year-over-year pairs does keying on handle LOSE? --")
    ov = played(SCOPED_YEARS)
    ov = ov[~(ov.Tournament.str.contains("challengers|qualifier", case=False, na=False)
              & (ov.year == 2023))]
    ov = ov.merge(ids, on=["Player", "year"], how="left")
    print(f"    rows with no id resolved: {ov['Player ID'].isna().sum():,} of {len(ov):,}")
    o = ov.dropna(subset=["Player ID"])
    for key, name in [("Player", "handle"), ("Player ID", "player_id")]:
        m = o.groupby([key, "year"]).size().rename("maps").reset_index()
        s = m[m.maps >= FLOOR]
        pairs = sum(len(set(s[s.year == y][key]) & set(s[s.year == y-1][key]))
                    for y in (2024, 2025, 2026))
        print(f"    key = {name:<10} player-seasons {len(s):>4}   "
              f"year-over-year pairs {pairs:>4}")

    rule("C. GRAIN -- one id twice in the same match-map", 72)
    d = o.groupby(["Player ID"] + MATCH + ["Map"]).size()
    print(f"  (player_id, match, map) combos appearing >1 time: {(d>1).sum()}")


CHECKS = {
    "grain":      ("what does one row describe?  (decision 01)", grain),
    "duplicates": ("the 93 duplicate rows; is `Match Name` an id?", duplicates),
    "era":        ("why the row count collapses after 2022  (decision 02)", era),
    "cost":       ("what the 2023 filter costs; is 2026 thin?", cost),
    "identity":   ("handle vs player_id  (decision 03)", identity),
}

if __name__ == "__main__":
    main(CHECKS, __doc__)
