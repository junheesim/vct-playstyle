"""Step 2b -- price the two open decisions.
Q6: what does option (b) cost inside 2023?
Q7: how incomplete is 2026, and where does that land?
"""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
RAW = pathlib.Path("data/raw"); YEARS = range(2023, 2027)
ov = pd.concat([pd.read_csv(RAW/f"vct_{y}/matches/overview.csv", low_memory=False).assign(year=y)
                for y in YEARS], ignore_index=True)
ov = ov[(ov.Map != "All Maps") & (ov.Side == "both")].copy()
ov["chal"] = ov.Tournament.str.contains("challengers|qualifier", case=False, na=False)

print("="*76, "\nQ6 -- cost of dropping Challengers rows inside 2023\n", "="*76)
y23 = ov[ov.year == 2023]
print(f"  2023 rows            {len(y23):>6,}   of which Challengers {y23.chal.sum():>6,} ({y23.chal.mean():.1%})")
print(f"  2023 players         {y23.Player.nunique():>6,}")
print(f"  ...only ever seen in Challengers: {y23[y23.chal].Player.nunique() - y23[y23.chal].Player.isin(y23[~y23.chal].Player).sum() and len(set(y23[y23.chal].Player) - set(y23[~y23.chal].Player)):>4,}")
print(f"  ...in BOTH (they lose maps, not their season): "
      f"{len(set(y23[y23.chal].Player) & set(y23[~y23.chal].Player)):>4,}")

for label, d in [("keep all 2023", ov), ("drop 2023 Challengers", ov[~(ov.chal & (ov.year==2023))])]:
    m = d.groupby(["Player","year"]).size().rename("maps").reset_index()
    surv = m[m.maps >= 20]
    print(f"\n  {label:<24} player-seasons with >=20 maps, by year:")
    print("    " + "  ".join(f"{y}:{n:>4}" for y, n in surv.groupby('year').size().items())
          + f"   TOTAL {len(surv)}")

print("\n" + "="*76, "\nQ7 -- is 2026 actually thin?\n", "="*76)
d = ov[~(ov.chal & (ov.year==2023))]
m = d.groupby(["Player","year"]).size().rename("maps").reset_index()
print(m.groupby("year").maps.describe(percentiles=[.25,.5,.75])[["count","mean","25%","50%","75%","max"]].round(1).to_string())
print("\n  event composition (rows):")
def ev(t):
    t = t.lower()
    if "challengers" in t or "qualifier" in t: return "Challengers"
    if "champions" in t and "champions tour" not in t: return "Champions"
    if "masters" in t: return "Masters"
    return "League"
print(pd.crosstab(d.year, d.Tournament.map(ev)).to_string())

print("\n  players surviving >=20 maps, and how many appear in the PRIOR year too:")
s = m[m.maps >= 20]
for y in (2024, 2025, 2026):
    cur, prev = set(s[s.year==y].Player), set(s[s.year==y-1].Player)
    print(f"    {y}: {len(cur):>3} players, {len(cur & prev):>3} also qualified in {y-1}  "
          f"-> usable year-over-year pairs: {len(cur & prev)}")
