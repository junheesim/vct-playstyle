"""A demonstration on FAKE data where the true answer is known in advance."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

N, MAPS = 400, 40

def season(true_rate):
    """Each player plays 40 maps. Defuses per map are random around their true rate."""
    return rng.poisson(np.outer(true_rate, np.ones(MAPS)) )

def split_half(maps):
    odd, even = maps[:, 0::2].mean(1), maps[:, 1::2].mean(1)
    r = np.corrcoef(odd, even)[0,1]
    return r, 2*r/(1+r)

print("="*70)
print("SETUP: 400 players. Each has a TRUE defuse tendency that we choose.")
print("       We make it IDENTICAL in 2024 and 2025 -- nobody changes at all.")
print("       So the TRUE year-over-year correlation is 1.000, by construction.")
print("="*70)

true_rate = rng.gamma(2.0, 0.12, N)          # true tendency, ~0.24 defuses/map
m24, m25 = season(true_rate), season(true_rate)
obs24, obs25 = m24.mean(1), m25.mean(1)

obs = np.corrcoef(obs24, obs25)[0,1]
rh, rel = split_half(m24)

print(f"\n  TRUE year-over-year correlation (we set it)      {1.000:.3f}")
print(f"  what you actually MEASURE                       {obs:.3f}   <- too low!")
print(f"\n  split-half: odd maps vs even maps, same season   {rh:.3f}")
print(f"  scaled to full season  2r/(1+r)                  {rel:.3f}   <- reliability")
print(f"\n  corrected:  {obs:.3f} / {rel:.3f} = {obs/rel:.3f}   <- recovers the truth")

print("\n" + "="*70)
print("SECOND CASE: now let players genuinely drift between seasons.")
print("="*70)
drift = true_rate * np.exp(rng.normal(0, 0.25, N))
m24, m25 = season(true_rate), season(drift)
true_corr = np.corrcoef(true_rate, drift)[0,1]
obs = np.corrcoef(m24.mean(1), m25.mean(1))[0,1]
rh, rel = split_half(m24)
print(f"\n  TRUE correlation between the two tendencies      {true_corr:.3f}")
print(f"  what you MEASURE                                 {obs:.3f}")
print(f"  reliability                                      {rel:.3f}")
print(f"  corrected                                        {obs/rel:.3f}   <- recovers it again")

print("\n" + "="*70)
print("WHY 2r/(1+r): more maps = less luck. Reliability RISES with season length.")
print("="*70)
print(f"\n  {'maps/season':>12}{'measured reliability':>24}{'predicted from 20 maps':>26}")
base = None
for mp in (10, 20, 40, 80):
    MAPS = mp
    m = season(true_rate)
    r = np.corrcoef(m[:,0::2].mean(1), m[:,1::2].mean(1))[0,1]
    full = 2*r/(1+r)
    if mp == 20: base = full
    k = mp/20
    pred = k*base/(1+(k-1)*base) if base else np.nan
    print(f"  {mp:>12}{full:>24.3f}{pred:>26.3f}")
