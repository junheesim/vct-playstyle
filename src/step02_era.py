"""
Step 2 -- why does the row count collapse after 2022?

Three worlds produce the same 17x row drop. This separates them by counting at
three different grains, per year:
    (1) distinct teams            (2) matches per team        (3) rows per match
"""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
RAW = pathlib.Path("data/raw"); YEARS = range(2021, 2027)
MATCH = ["Tournament","Stage","Match Type","Match Name","year"]

ov = pd.concat([pd.read_csv(RAW/f"vct_{y}/matches/overview.csv", low_memory=False).assign(year=y)
                for y in YEARS], ignore_index=True)
# Step 1 decisions, applied.
ov = ov[(ov.Map != "All Maps") & (ov.Side == "both")].copy()
ov["Average Combat Score"] = pd.to_numeric(ov["Average Combat Score"], errors="coerce")

per_year = ov.groupby("year")
t = pd.DataFrame({
    "rows":            per_year.size(),
    "teams":           per_year.Team.nunique(),
    "players":         per_year.Player.nunique(),
    "matches":         ov.groupby(MATCH).ngroups and per_year.apply(lambda g: g.groupby(MATCH[:-1]).ngroups),
})
t["matches_per_team"] = (2 * t.matches / t.teams).round(1)   # each match involves 2 teams
t["rows_per_match"]   = (t.rows / t.matches).round(1)
t["players_per_team"] = (t.players / t.teams).round(1)

print("="*78, "\nWORLD 1 vs 2 vs 3\n", "="*78)
print(t[["rows","teams","matches","matches_per_team","rows_per_match","players_per_team"]].to_string())

print("\n" + "="*78, "\nWhat is IN the extra data? -- spread of player skill (season mean ACS)\n", "="*78)
ps = (ov.groupby(["Player","year"])
        .agg(acs=("Average Combat Score","mean"), maps=("Map","size"))
        .reset_index())
ps = ps[ps.maps >= 5]
q = ps.groupby("year").acs.describe(percentiles=[.05,.25,.5,.75,.95])
q["p95_minus_p5"] = (q["95%"] - q["5%"]).round(1)
print(q[["count","mean","std","5%","50%","95%","p95_minus_p5"]].round(1).to_string())

print("\n" + "="*78, "\nTeam count by region-tier proxy (tournament name)\n", "="*78)
# First match wins, most specific first. An overwriting loop mislabels
# "Champions Tour NA Stage 1: Challengers 2" as a franchised league event.
def tier(t):
    t = t.lower()
    if "challengers" in t or "qualifier" in t: return "Challengers / qualifier"
    if "game changers" in t:                   return "Game Changers"
    if "champions" in t and "champions tour" not in t: return "Champions (intl)"
    if "masters" in t:                         return "Masters (intl)"
    return "League / other"
ov["tier"] = ov.Tournament.map(tier)
print(pd.crosstab(ov.year, ov.tier).to_string())
