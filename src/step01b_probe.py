"""Step 1b -- follow up on the two loose ends from the audit."""
import pandas as pd, pathlib, warnings
warnings.filterwarnings("ignore")
RAW = pathlib.Path("data/raw"); YEARS = range(2021, 2027)
KEY = ["Tournament","Stage","Match Type","Match Name","Map","year"]

ov = pd.concat([pd.read_csv(RAW/f"vct_{y}/matches/overview.csv", low_memory=False).assign(year=y)
                for y in YEARS], ignore_index=True)

print("="*70, "\nQ3 -- the 93 duplicates\n", "="*70)
full = KEY + ["Player","Side"]
dups = ov[ov.duplicated(full, keep=False)].sort_values(full)
print(f"  rows involved: {len(dups)}   distinct keys: {dups.groupby(full).ngroups}")
print(f"  by year: {dups.year.value_counts().sort_index().to_dict()}")
print(f"  by tournament:\n{dups.Tournament.value_counts().head(5).to_string()}")

print("\n  -- are the duplicate rows identical, or do they disagree? --")
stat_cols = ["Kills","Deaths","Assists","Average Combat Score","Agents","Team"]
n_ident = dups.duplicated(full + stat_cols, keep=False).sum()
print(f"  rows that are byte-identical copies: {n_ident} / {len(dups)}")

print("\n  -- first two conflicting keys, printed in full --")
shown = 0
for k, g in dups.groupby(full):
    if g[stat_cols].drop_duplicates().shape[0] > 1:
        print(f"\n  {k[3]} | {k[4]} | {k[6]} | {k[7]}")
        print(g[stat_cols].to_string(index=False))
        shown += 1
        if shown == 2: break
if shown == 0:
    print("  none conflict -- every duplicate is an exact copy")

print("\n" + "="*70, "\nQ4 -- is 'Match Name' a usable id?\n", "="*70)
mn = (ov[["Match Name","Tournament","Stage","Match Type","year"]].drop_duplicates()
        .groupby("Match Name").size().sort_values(ascending=False))
print(f"  Match Names used in more than one tournament/stage/type: {(mn>1).sum():,}")
print(f"  worst offender is used {mn.iloc[0]} times\n")
worst = mn.index[0]
print(f"  '{worst}' appears in:")
print(ov[ov["Match Name"]==worst][["year","Tournament","Stage","Match Type"]]
        .drop_duplicates().sort_values(["year","Tournament"]).head(12).to_string(index=False))

print("\n  -- what it costs you --")
sel = ov[(ov["Match Name"]==worst) & (ov.Side=="both") & (ov.Map!="All Maps")]
print(f"  groupby('Match Name')                      -> {sel.groupby('Match Name').ngroups:>6,} group(s)")
print(f"  groupby(tournament+stage+type+name)        -> {sel.groupby(KEY[:4]).ngroups:>6,} group(s)")
print(f"  ...for the SAME {len(sel):,} rows.")
