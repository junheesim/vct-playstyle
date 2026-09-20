"""Step 8 -- is there a better quality yardstick than ACS?  (decision 06)

ACS is a VOLUME measure: combat score accumulated per round. It rises with how much
you fight, so it is bent in exactly the direction that matters -- fighting a lot is
what `first_engagement` measures. A RATE cannot be inflated that way: volume appears
in numerator and denominator and cancels.
"""
import sys, pandas as pd, numpy as np
sys.path.insert(0,"src"); import features as F

ps = F.seasons(F.build())
ps["Rating"] = ps.get("Rating", np.nan)
CAND = ["acs","adr","kd_ratio","fk_win","team_win"]
KIND = {"acs":"volume","adr":"volume","kd_ratio":"rate","fk_win":"rate","team_win":"outcome"}

print("="*70, "\nA. Do the candidates agree on who is good?\n", "="*70)
print(ps[CAND].corr().round(2).to_string())

print("\n" + "="*70, "\nB. Volume inflation: how much does each track FIGHT VOLUME?\n", "="*70)
print(f"  {'candidate':<12}{'kind':<10}{'r with first_engagement':>25}")
for c in CAND:
    print(f"  {c:<12}{KIND[c]:<10}{ps[c].corr(ps.first_engagement):>25.3f}")

print("\n" + "="*70, "\nC. Is it a stable player property? (a quality measure should be)\n", "="*70)
for c in CAND:
    a  = ps.pivot_table(index="player_id", columns="year", values=c)
    pr = pd.concat([pd.DataFrame({"t":a[y],"t1":a[y+1]}).dropna()
                    for y in (2023,2024,2025) if y+1 in a.columns])
    print(f"  {c:<12}{pr.t.corr(pr.t1):>8.3f}")

print("\n" + "="*70, "\nD. Contamination of the STYLE set, by yardstick\n", "="*70)
print(f"  {'feature':<18}{'r with ACS':>12}{'r with K/D':>12}")
for f in F.STYLE + F.PENDING:
    print(f"  {f:<18}{ps[f].corr(ps.acs):>+12.3f}{ps[f].corr(ps.kd_ratio):>+12.3f}")
print(f"\n  mean |r| over STYLE -- ACS: {ps[F.STYLE].corrwith(ps.acs).abs().mean():.3f}"
      f"   K/D: {ps[F.STYLE].corrwith(ps.kd_ratio).abs().mean():.3f}")
