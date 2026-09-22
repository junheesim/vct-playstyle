# The project logic — how the pieces fit

## The question
Do VCT pros have distinguishable playstyles — and is playstyle something other than
"how good they are" or "which agent they play"?

## Why it is hard
Every observable number about a player is a blend, and only one part is style:

    observed  =  STYLE
              +  how good they are        (quality)
              +  which agent they played
              +  which map, which side
              +  who they played against
              +  which team they are on
              +  random luck

**The project is a sequence of subtractions.** Every decision is either one
subtraction, or a check that a subtraction can be trusted.

See `METHODS.md` for what each Phase 2 diagnostic actually computes.

## The chain

    PHASE 1 — make the data mean what you think it means
      01 grain        drop series totals; fix the match key          DONE
      02 scope        franchised era only (removes skill tier)       DONE
      03 identity     one player = one person                        DONE

    PHASE 2 — choose what to measure, and know how well you measured it
      04 features     candidates that plausibly measure style        DONE
      06 quality      a quality detector that is not itself bent     DONE
      05 reliability  can each measurement be trusted?               DONE
      09 team test    is it the player, or the roster?               DONE
      11 role test    does it say anything the role label does not?  DONE
      12 agent test   is the agent the player's, or the team's?      DONE
      07 THE LIST     the 8 features; kast and defuses out           DONE
      08 no agent/role control — role is a held-out validation label DONE
      09 side split   one gap feature only                           DONE

    PHASE 3 — subtract the contaminants
      10 covariates  quality only; map/opponent do not vary       DONE
      18 round count does not vary either; swept anyway           DONE
      residualize     remove quality (kd_ratio) only

    PHASE 4 — find the structure
      11 components  2 axes: aggression, isolation/gunplay       DONE
      12 archetypes  CONTINUUM, with 3 landmark regions          DONE

    PHASE 5 — is it real?                                      <-- CURRENT
      does it hold on a year the model never saw?
      is it different from Riot's role labels?
      does it predict anything about winning?

## Where reliability sits — it is NOT a step in the chain
It is the calibration of the instrument. Weighing something small on a kitchen scale,
on a wobbly table:

  - **reliability** — how precise is the scale? If it reads +/-50g, do not report a
    30g difference.
  - **the quality measure (06)** — is the scale zeroed, or does it read heavy?
  - **residualization (next)** — subtract the container so you get the contents.
  - **team test (09)** — is the object on the scale, or partly resting on the table?

Reliability removes no contaminant. It tells you how much precision you have to spend.
Every downstream number — is a feature stable, does a player's style persist, is this
player different from that one — is uninterpretable without it. That is why it recurs
rather than being finished.

## What each phase buys

| phase | removes / establishes |
|-------|------------------------|
| 1 | 2021-22 amateurs, double-counted rows, merged identities |
| 2 | features that are really quality; features that cannot be measured; features that belong to the team |
| 3 | quality (K/D). Map, opponent, side and round count were MEASURED not to vary between players, and agent/role are held out by decision 08 |
| 4 | the actual finding |
| 5 | whether to believe it |

At the end of Phase 3 each player-feature is a single number meaning: *how much does
this player do this, compared to what you would expect of a player of equal SKILL?*
Map, opponent, side and round count are not in that sentence because decisions 10 and
18 measured them and they do not vary between players; agent and role are not in it
because decision 08 holds them out as validation labels. That number is style. Phase 4
finds its shape.
