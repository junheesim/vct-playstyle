# Methods — what each diagnostic computes

Reference card. Every one of these is Phase 2 justification: run once to defend the
feature list, never recomputed downstream. See `LOGIC.md` for where they fit.

---

## 1. Reliability — can it be measured in one season?

    split the season into two random halves, A and B
    both measure the SAME player in the SAME year, so any disagreement is luck

    r_half = Corr(A, B)                  reliability of a HALF-season estimate
    rel    = 2*r_half / (1 + r_half)     scaled up to a FULL season

Output 0-1: the share of between-player differences that is real, not luck.
The scaling exists because averaging twice as many maps halves the noise variance.
Averaged over 200 random splits; report the 5-95% interval.
defuses: r_half = .247 -> rel = .396.

## 2. True stability — does the trait persist?

    obs  = Corr(player's value in year t, same player in year t+1)
    true = obs / rel

Even with NO change, `obs` comes out equal to `rel`, not 1 -- noise in both years
drags the correlation down. Dividing removes it.
Breaks when `rel` is small: you are dividing by a noisy number.
defuses: .420/.396 -> above 1.0, out of range.

## 3. Yardstick — is the feature really quality?

    r_kd = Corr(feature, kd_ratio)
    where kd_ratio = total kills / total deaths, per player-season

Output -1 to +1. Near zero = style. Large either way = it tracks how GOOD the player is.
K/D beats ACS because it is a RATIO -- fight volume cancels between numerator and
denominator -- whereas ACS ACCUMULATES with volume and is therefore inflated by
aggression, the very thing being measured.

## 4. Team vs player — the individual or the roster?

    split the year-over-year pairs by whether the player changed team
    r_same  = Corr(t, t+1)  among stayers
    r_moved = Corr(t, t+1)  among movers

`r_moved` is the honest player-trait figure: kept through a new roster, system and
teammates. The gap (r_same - r_moved) is what belongs to the team.
clutch_att: .355 -> .142. Almost all of it was the team.

## 5. VIF — does another feature already say this?

    for feature j:  regress j on all the other features, take R2_j
    VIF_j = 1 / (1 - R2_j)

    R2 = .00 -> VIF  1.0   its own thing
    R2 = .50 -> VIF  2.0   fine
    R2 = .80 -> VIF  5.0   worry
    R2 = .90 -> VIF 10.0   redundant

Worst in this set: first_engagement = 3.5 (others explain ~71% of it). Under the line.

---

## What carries into Phase 3
Only the feature LIST and the table. None of the five above is computed again.
Reliability returns at the very END, to judge whether two players' positions on the
final style axis are distinguishable.
