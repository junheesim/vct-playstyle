"""Step 4 -- Rating or ACS as the quality yardstick?

If Rating already contains a first-blood term, then screening the First Kill family
against Rating is circular. Test: does FK predict Rating AFTER the ordinary quality
components (ACS, ADR, KAST, K-D) are accounted for? If Rating were a pure summary of
those, FK should add nothing.
"""
import sys, pandas as pd, numpy as np, statsmodels.api as sm
sys.path.insert(0, "src"); import data as D

ov = D.overview()
ov = ov[ov.Side == "both"].copy()
for c in ["Rating","Average Combat Score","Average Damage Per Round","Kills","Deaths",
          "First Kills","First Deaths","Assists"]:
    ov[c] = pd.to_numeric(ov[c], errors="coerce")
ov["kast"] = pd.to_numeric(ov["Kill, Assist, Trade, Survive %"].astype(str).str.rstrip("%"), errors="coerce")
ov["kd"]   = ov.Kills - ov.Deaths
ov["involvement"] = ov["First Kills"] + ov["First Deaths"]
d = ov.dropna(subset=["Rating","Average Combat Score","Average Damage Per Round","kast",
                      "kd","First Kills","First Deaths","involvement"])
print(f"n = {len(d):,} player-maps\n")

print("="*70, "\nRaw correlations with each candidate yardstick\n", "="*70)
print(f"  {'feature':<14}{'r with Rating':>15}{'r with ACS':>14}")
for f in ["First Kills","First Deaths","involvement","Assists","Deaths","kast"]:
    print(f"  {f:<14}{d[f].corr(d.Rating):>15.3f}{d[f].corr(d['Average Combat Score']):>14.3f}")

print("\n" + "="*70, "\nDoes FK add to Rating beyond ordinary quality?\n", "="*70)
base = ["Average Combat Score","Average Damage Per Round","kast","kd"]
for target in ["Rating", "Average Combat Score"]:
    X0 = sm.add_constant(d[[c for c in base if c != target]])
    m0 = sm.OLS(d[target], X0).fit()
    X1 = sm.add_constant(d[[c for c in base if c != target] + ["First Kills","First Deaths"]])
    m1 = sm.OLS(d[target], X1).fit()
    print(f"\n  target = {target}")
    print(f"    R2 without FK/FD : {m0.rsquared:.4f}")
    print(f"    R2 with    FK/FD : {m1.rsquared:.4f}   (+{m1.rsquared-m0.rsquared:.4f})")
    print(f"    FK coef {m1.params['First Kills']:+.4f} (t={m1.tvalues['First Kills']:.1f})   "
          f"FD coef {m1.params['First Deaths']:+.4f} (t={m1.tvalues['First Deaths']:.1f})")
