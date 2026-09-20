"""Step 3 -- is the player handle safe to key on?

Three checks. A and B compare the two identity columns against each other.
C is the grain check, run again on the scoped window.
"""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
RAW = pathlib.Path("data/raw"); YEARS = range(2023, 2027)
MATCH = ["Tournament","Stage","Match Type","Match Name","year"]

ids = pd.concat([pd.read_csv(RAW/f"vct_{y}/ids/players_ids.csv").assign(year=y) for y in YEARS],
                ignore_index=True).dropna(subset=["Player ID"])
ids["Player ID"] = ids["Player ID"].astype(int)
ids = ids.drop_duplicates(["Player","Player ID","year"])
print(f"id file: {len(ids):,} handle-id-year rows | "
      f"{ids.Player.nunique():,} handles | {ids['Player ID'].nunique():,} ids\n")

print("="*72, "\nA. COLLISION -- one handle, several player_ids\n", "="*72)
a = ids.groupby("Player")["Player ID"].nunique()
print(f"  handles mapping to >1 id: {(a>1).sum()} of {len(a):,}")
for h in a[a>1].index[:6]:
    print("   ", ids[ids.Player==h].sort_values("year")[["Player","Player ID","year"]].to_string(index=False).replace("\n","\n    "))

print("\n" + "="*72, "\nB. RENAME -- one player_id, several handles\n", "="*72)
b = ids.groupby("Player ID")["Player"].nunique()
print(f"  ids mapping to >1 handle: {(b>1).sum()} of {len(b):,}")
for i in b[b>1].index[:6]:
    r = ids[ids["Player ID"]==i].sort_values("year")
    print(f"    id {i}: " + "  ".join(f"{y}={p}" for p, y in zip(r.Player, r.year)))

print("\n  -- COST: how many year-over-year pairs does keying on handle LOSE? --")
ov = pd.concat([pd.read_csv(RAW/f"vct_{y}/matches/overview.csv", low_memory=False).assign(year=y)
                for y in YEARS], ignore_index=True)
ov = ov[(ov.Map!="All Maps") & (ov.Side=="both")]
ov = ov[~(ov.Tournament.str.contains("challengers|qualifier", case=False, na=False) & (ov.year==2023))]
ov = ov.merge(ids, on=["Player","year"], how="left")
print(f"    rows with no id resolved: {ov['Player ID'].isna().sum():,} of {len(ov):,}")
o = ov.dropna(subset=["Player ID"])
for key, name in [("Player","handle"), ("Player ID","player_id")]:
    m = o.groupby([key,"year"]).size().rename("maps").reset_index()
    s = m[m.maps>=20]
    pairs = sum(len(set(s[s.year==y][key]) & set(s[s.year==y-1][key])) for y in (2024,2025,2026))
    print(f"    key = {name:<10} player-seasons {len(s):>4}   year-over-year pairs {pairs:>4}")

print("\n" + "="*72, "\nC. GRAIN -- your query: one id twice in the same match-map\n", "="*72)
d = o.groupby(["Player ID"]+MATCH+["Map"]).size()
print(f"  (player_id, match, map) combos appearing >1 time: {(d>1).sum()}")
