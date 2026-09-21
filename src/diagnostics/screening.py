"""Is each candidate feature a real, measurable, individual STYLE trait?

Decisions 04, 05, 06 and 07. These were seven files named after the order they were
run in -- step04_proxy, step05_screen, step06b_demo, step07_q14, step08_yardstick,
step09_team_vs_player, step10_scorecard -- which recorded WHEN a check happened
rather than what it asks.

    rating-proxy    does vlr `Rating` hide a first-blood term? (it does not)
    quality-metric  ACS vs Rating vs K/D vs team win rate -- which yardstick?
    screen          reliability, stability and quality loading, per feature
    attenuation     a simulation proving the reliability correction recovers an
                    answer that is known in advance
    entry-vs-acs    is first_engagement's +.681 with ACS real, or a bent yardstick?
    team-vs-player  how much of a feature's persistence is the roster, not the player?
    scorecard       all five screening tests, side by side
"""
from _run import main, rule     # first: also puts src/ on the path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

import data as D
import features as F



def rating_proxy():
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

    rule("Raw correlations with each candidate yardstick", 70)
    print(f"  {'feature':<14}{'r with Rating':>15}{'r with ACS':>14}")
    for f in ["First Kills","First Deaths","involvement","Assists","Deaths","kast"]:
        print(f"  {f:<14}{d[f].corr(d.Rating):>15.3f}{d[f].corr(d['Average Combat Score']):>14.3f}")

    rule("Does FK add to Rating beyond ordinary quality?", 70)
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


def screen():
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


def attenuation():
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


def entry_vs_acs():
    d = F.build(); ps = F.seasons(d)
    g = d.groupby(["player_id","year"])
    fk = g[["First Kills","First Deaths","Kills"]].mean()
    ps = ps.merge(fk, on=["player_id","year"])

    rule("A. Split first_engagement into its halves", 64)
    for c in ["First Kills","First Deaths","first_engagement"]:
        print(f"  r({c:<17}, ACS) = {ps[c].corr(ps.acs):+.3f}")

    rule("B. Control for total kills", 64)
    X = sm.add_constant(ps[["Kills"]])
    r_fe  = sm.OLS(ps.first_engagement, X).fit().resid
    r_acs = sm.OLS(ps.acs, X).fit().resid
    print(f"  r(first_engagement, ACS)              = {ps.first_engagement.corr(ps.acs):+.3f}")
    print(f"  r(first_engagement, ACS | total kills)= {np.corrcoef(r_fe, r_acs)[0,1]:+.3f}")
    print(f"  r(First Deaths,     ACS | total kills)= "
          f"{np.corrcoef(sm.OLS(ps['First Deaths'],X).fit().resid, r_acs)[0,1]:+.3f}")

    rule("C. Your 9 features together -- correlation", 64)
    SET = ["first_engagement","hs","assists","plants","defuses","clutch_att",
           "deaths","creds_per_round","kast"]
    C = ps[SET].corr()
    print(C.round(2).to_string())
    print("\n  |r| > 0.5 pairs:")
    for i,a in enumerate(SET):
        for b in SET[i+1:]:
            if abs(C.loc[a,b]) > 0.5: print(f"    {a} ~ {b}: {C.loc[a,b]:+.2f}")

    Z = ps[SET].dropna().apply(lambda s:(s-s.mean())/s.std()).assign(const=1.0)
    v = {f: vif(Z.values,i) for i,f in enumerate(Z.columns) if f!="const"}
    print("\n  VIF (>5 = redundant): " + ", ".join(f"{k}={x:.1f}" for k,x in sorted(v.items(), key=lambda kv:-kv[1])))


def quality_metric():
    ps = F.seasons(F.build())
    ps["Rating"] = ps.get("Rating", np.nan)
    CAND = ["acs","adr","kd_ratio","fk_win","team_win"]
    KIND = {"acs":"volume","adr":"volume","kd_ratio":"rate","fk_win":"rate","team_win":"outcome"}

    rule("A. Do the candidates agree on who is good?", 70)
    print(ps[CAND].corr().round(2).to_string())

    rule("B. Volume inflation: how much does each track FIGHT VOLUME?", 70)
    print(f"  {'candidate':<12}{'kind':<10}{'r with first_engagement':>25}")
    for c in CAND:
        print(f"  {c:<12}{KIND[c]:<10}{ps[c].corr(ps.first_engagement):>25.3f}")

    rule("C. Is it a stable player property? (a quality measure should be)", 70)
    for c in CAND:
        a  = ps.pivot_table(index="player_id", columns="year", values=c)
        pr = pd.concat([pd.DataFrame({"t":a[y],"t1":a[y+1]}).dropna()
                        for y in (2023,2024,2025) if y+1 in a.columns])
        print(f"  {c:<12}{pr.t.corr(pr.t1):>8.3f}")

    rule("D. Contamination of the STYLE set, by yardstick", 70)
    print(f"  {'feature':<18}{'r with ACS':>12}{'r with K/D':>12}")
    for f in F.STYLE + F.PENDING:
        print(f"  {f:<18}{ps[f].corr(ps.acs):>+12.3f}{ps[f].corr(ps.kd_ratio):>+12.3f}")
    print(f"\n  mean |r| over STYLE -- ACS: {ps[F.STYLE].corrwith(ps.acs).abs().mean():.3f}"
          f"   K/D: {ps[F.STYLE].corrwith(ps.kd_ratio).abs().mean():.3f}")


def team_vs_player():
    d  = F.build(); ps = F.seasons(d)
    # `org`, not `Team`: four franchises were renamed mid-window (Giants Gaming ->
    # GIANTX, Talon -> TALON, DRX -> KIWOOM DRX, and NRG mislabelled as Mega Minors).
    # Keying on the displayed name counts 10 of 417 pairs as roster moves that never
    # happened, which is exactly the group this test treats as the honest one.
    tm = (d[d.Side=="both"].groupby(["player_id","year"]).org
            .agg(lambda s: s.mode().iat[0]).rename("team"))
    q  = ps.merge(tm, on=["player_id","year"])

    rows = []
    for f in F.STYLE + F.PENDING + ["kd_ratio"]:
        pv = q.pivot_table(index="player_id", columns="year", values=f)
        tv = q.pivot_table(index="player_id", columns="year", values="team", aggfunc="first")
        pr = pd.concat([pd.DataFrame({"t": pv[y], "t1": pv[y+1], "same": tv[y] == tv[y+1]}).dropna()
                        for y in (2023,2024,2025) if y+1 in pv.columns])
        same, moved = pr[pr.same], pr[~pr.same]
        rows.append({"feature": f, "n_stay": len(same), "n_moved": len(moved),
                     "r_same_team": same.t.corr(same.t1),
                     "r_new_team":  moved.t.corr(moved.t1),
                     "drop": same.t.corr(same.t1) - moved.t.corr(moved.t1)})

    t = pd.DataFrame(rows).set_index("feature").sort_values("drop")
    print(t.round(3).to_string())
    print("\n  r_new_team is the honest PLAYER-trait figure: the player kept it after")
    print("  changing roster, system and teammates.")
    print("  A large 'drop' means the feature travels with the TEAM, not the player.")


def scorecard():
    d  = F.build(); ps = F.seasons(d)
    # `org`, not `Team`: four franchises were renamed mid-window (Giants Gaming ->
    # GIANTX, Talon -> TALON, DRX -> KIWOOM DRX, and NRG mislabelled as Mega Minors).
    # Keying on the displayed name counts 10 of 417 pairs as roster moves that never
    # happened, which is exactly the group this test treats as the honest one.
    tm = (d[d.Side=="both"].groupby(["player_id","year"]).org
            .agg(lambda s: s.mode().iat[0]).rename("team"))
    q  = ps.merge(tm, on=["player_id","year"])
    SET = F.STYLE + ["defuses", "kast"]

    rows = []
    for f in SET:
        rel, lo, hi = F.reliability(d, f, ps)
        pv = q.pivot_table(index="player_id", columns="year", values=f)
        tv = q.pivot_table(index="player_id", columns="year", values="team", aggfunc="first")
        pr = pd.concat([pd.DataFrame({"t":pv[y],"t1":pv[y+1],"same":tv[y]==tv[y+1]}).dropna()
                        for y in (2023,2024,2025) if y+1 in pv.columns])
        obs   = pr.t.corr(pr.t1)
        moved = pr[~pr.same]
        rows.append({"feature": f,
                     "1_r_kd": ps[f].corr(ps.kd_ratio),
                     "2_rel": rel,
                     "3_true": min(obs/rel, 1.0) if pd.notna(rel) else np.nan,
                     "4_newteam": moved.t.corr(moved.t1), "n_moved": len(moved),
                     "cover": ps[f].notna().mean()})
    t = pd.DataFrame(rows).set_index("feature")

    Z = ps[SET].dropna().apply(lambda s:(s-s.mean())/s.std()).assign(const=1.0)
    t["5_vif"] = pd.Series({f: vif(Z.values,i) for i,f in enumerate(Z.columns) if f!="const"})

    def flag(r):
        bad = []
        if abs(r["1_r_kd"]) > .40:  bad.append("quality")
        if r["2_rel"]      < .50:   bad.append("noisy")
        if r["4_newteam"]  < .30:   bad.append("team")
        if r["5_vif"]      > 5:     bad.append("redundant")
        return ",".join(bad) or "clean"
    t["concerns"] = t.apply(flag, axis=1)
    print(t.round(3).to_string())
    print("""
  1_r_kd    >|.40| = looks like quality
  2_rel     <.50   = can barely be measured in one season; 3_true is then unstable
  3_true    capped at 1.0 (defuses exceeds it -- correction out of range)
  4_newteam <.30   = travels with the roster, not the player
  5_vif     >5     = the other features already say this""")


CHECKS = {
    "rating-proxy":   ("does vlr Rating hide a first-blood term?  (decision 04)", rating_proxy),
    "quality-metric": ("ACS vs Rating vs K/D vs team win rate  (decision 06)", quality_metric),
    "screen":         ("reliability, stability and quality loading  (decision 05)", screen),
    "attenuation":    ("simulation: does the reliability correction work?", attenuation),
    "entry-vs-acs":   ("is first_engagement's loading on ACS real?  (decision 07)", entry_vs_acs),
    "team-vs-player": ("is the persistence the player or the roster?", team_vs_player),
    "scorecard":      ("all five screening tests, side by side  (decision 07)", scorecard),
}

if __name__ == "__main__":
    main(CHECKS, __doc__)
