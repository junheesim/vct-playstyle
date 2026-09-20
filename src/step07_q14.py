"""Step 7 -- is first_engagement's +0.681 with ACS real quality, or a bent yardstick?

first_engagement = FK + FD.
  Reading (i)  better players take more opening duels -> BOTH halves should load
               positively on ACS.
  Reading (ii) mechanical: a first kill IS a kill, and kills raise ACS. Then the
               loading lives entirely in FK, and FD (a death, worth nothing to ACS)
               should be flat. Controlling for total kills should also remove it.
"""
import sys, pandas as pd, numpy as np, statsmodels.api as sm
sys.path.insert(0,"src"); import features as F

d = F.build(); ps = F.seasons(d)
g = d.groupby(["player_id","year"])
fk = g[["First Kills","First Deaths","Kills"]].mean()
ps = ps.merge(fk, on=["player_id","year"])

print("="*64, "\nA. Split first_engagement into its halves\n", "="*64)
for c in ["First Kills","First Deaths","first_engagement"]:
    print(f"  r({c:<17}, ACS) = {ps[c].corr(ps.acs):+.3f}")

print("\n" + "="*64, "\nB. Control for total kills\n", "="*64)
X = sm.add_constant(ps[["Kills"]])
r_fe  = sm.OLS(ps.first_engagement, X).fit().resid
r_acs = sm.OLS(ps.acs, X).fit().resid
print(f"  r(first_engagement, ACS)              = {ps.first_engagement.corr(ps.acs):+.3f}")
print(f"  r(first_engagement, ACS | total kills)= {np.corrcoef(r_fe, r_acs)[0,1]:+.3f}")
print(f"  r(First Deaths,     ACS | total kills)= "
      f"{np.corrcoef(sm.OLS(ps['First Deaths'],X).fit().resid, r_acs)[0,1]:+.3f}")

print("\n" + "="*64, "\nC. Your 9 features together -- correlation\n", "="*64)
SET = ["first_engagement","hs","assists","plants","defuses","clutch_att",
       "deaths","creds_per_round","kast"]
C = ps[SET].corr()
print(C.round(2).to_string())
print("\n  |r| > 0.5 pairs:")
for i,a in enumerate(SET):
    for b in SET[i+1:]:
        if abs(C.loc[a,b]) > 0.5: print(f"    {a} ~ {b}: {C.loc[a,b]:+.2f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
Z = ps[SET].dropna().apply(lambda s:(s-s.mean())/s.std()).assign(const=1.0)
v = {f: vif(Z.values,i) for i,f in enumerate(Z.columns) if f!="const"}
print("\n  VIF (>5 = redundant): " + ", ".join(f"{k}={x:.1f}" for k,x in sorted(v.items(), key=lambda kv:-kv[1])))
