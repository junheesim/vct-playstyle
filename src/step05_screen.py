"""Step 5 -- check the style/quality classification against the data.

Three numbers per feature (decisions 04, 05, 06):
  reliability -- can it be MEASURED in one season?   (split-half, Spearman-Brown)
  observed    -- raw year-over-year correlation
  true        -- observed / reliability
  r_quality   -- how much it moves with the K/D yardstick (ACS shown for comparison)
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "src"); import features as F

d  = F.build()
ps = F.seasons(d)
print(f"{len(ps)} player-seasons | yardstick = {F.QUALITY}\n")

FEATS = F.STYLE + ["defuses", "kast", "econ"]
rows = []
for f in FEATS:
    rel, lo, hi = F.reliability(d, f, ps)

    a  = ps.pivot_table(index="player_id", columns="year", values=f)
    pr = pd.concat([pd.DataFrame({"t": a[y], "t1": a[y+1]}).dropna()
                    for y in (2023, 2024, 2025) if y+1 in a.columns])
    obs = pr.t.corr(pr.t1)
    rows.append({"feature": f, "verdict": "STYLE" if f in F.STYLE else
                 ("PENDING" if f in F.PENDING else "dropped"),
                 "rel": rel, "rel_lo": lo, "rel_hi": hi, "observed": obs,
                 "true": obs/rel if pd.notna(rel) else np.nan,
                 "true_lo": obs/hi if pd.notna(hi) else np.nan,
                 "true_hi": min(obs/lo, 1.0) if pd.notna(lo) else np.nan,
                 "r_kd": ps[f].corr(ps[F.QUALITY]), "r_acs": ps[f].corr(ps.acs)})

t = pd.DataFrame(rows).set_index("feature").sort_values("true", ascending=False)
print(t.round(3).to_string())
print(f"\n  mean |r| with K/D across STYLE: "
      f"{t.loc[t.verdict=='STYLE','r_kd'].abs().mean():.3f}"
      f"   (with ACS: {t.loc[t.verdict=='STYLE','r_acs'].abs().mean():.3f})")
