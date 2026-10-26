"""

Design (because MTTR is only defined for repositories that had a failure AND a recovery):
  Sample A  all repositories
  Sample B  valid pipelines (>= MIN_RUNS completed runs in the window)  -> success rate, failure incidence
  Sample C  valid pipelines with >= 1 failure episode                  -> recovery incidence
  Sample D  valid pipelines with a recovered failure (MTTR defined)    -> MTTR comparisons + regressions
"""
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_FILE = r"C:\Users\User\Desktop\Minig github\DataSet.csv"
OUT_DIR = "results_v2"
COL = {                                   # name used here : column in your CSV
    "group": "classified_posture",
    "mttr": "mttr_hours_mean",
    "status": "mttr_status",
    "success": "success_rate_pct",
    "lifespan": "project_lifespan_days",
    "contributors": "contributors_count",
    "runs": "n_completed_runs",
    "episodes": "n_failure_episodes",
    "n_recovered": "n_recovered",
    "valid": "valid_pipeline",
    "stars": "stars",
    "forks": "forks",
    "commits": "total_commits",
    "repo": "repository",
    "age": "repo_age_days",               # age at collection date (collected_at - created_at)
    "cluster": None,                      # set to "owner" for cluster-robust SEs
}
GROUP_MAP = {
    "Basic / No Automated Pipeline Security": "Basic",
    "DevSecOps (Shift-Left Integrated)": "DevSecOps",
    "SecDevOps (Strict Security Gating)": "SecDevOps",
}
GROUP_ORDER = ["Basic", "DevSecOps", "SecDevOps"]   # Basic = reference
# Optional eligibility rule: keep only repositories at least this old at collection time.
# 365 guarantees every repository could have produced runs during the whole 365-day window. None = no filter.
MIN_REPO_AGE_DAYS = 365
N_BOOT = 2000
SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 0. Load data and build the samples
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.rename(columns={v: k for k, v in COL.items() if v and v in df.columns})
    print("\n=== classified_posture values in your file ===")
    print(df["group"].value_counts(dropna=False))

    def _map(v):
        v = str(v).strip()
        if v in GROUP_MAP:
            return GROUP_MAP[v]
        low = v.lower()
        for key, lab in (("secdevops", "SecDevOps"), ("devsecops", "DevSecOps"), ("basic", "Basic")):
            if low.startswith(key):
                return lab
        return np.nan
    df["group"] = df["group"].map(_map)
    if df["group"].isna().any():
        print(f"WARNING: dropping {int(df['group'].isna().sum())} rows with unmapped posture labels.")
        df = df[df["group"].notna()].copy()
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    if "repo" in df.columns:
        n0 = len(df)
        df = df.drop_duplicates(subset="repo")
        if len(df) < n0:
            print(f"Dropped {n0 - len(df)} duplicate repository rows.")
        df["owner"] = df["repo"].astype(str).str.split("/").str[0]
    if COL["cluster"] == "owner":
        df["cluster"] = df["owner"]
    df["valid"] = df["valid"].astype(str).str.strip().str.lower().eq("true")
    for c in ("mttr", "success", "lifespan", "contributors", "runs", "episodes", "n_recovered", "age"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["success"] = df["success"] / 100.0                   # percent -> proportion
    return df


def build_samples(df):
    if MIN_REPO_AGE_DAYS is not None:
        n0 = len(df)
        df = df[df["age"] >= MIN_REPO_AGE_DAYS].copy()
        print(f"\nAge filter (>= {MIN_REPO_AGE_DAYS} days): kept {len(df)} of {n0} repositories.")
    valid = df["valid"]
    failed = valid & (df["episodes"] > 0)
    recovered = (valid & (df["status"] == "recovered") & (df["mttr"] > 0)
                 & (df["lifespan"] > 0) & (df["contributors"] > 0))
    flow = pd.DataFrame({
        "A. All repositories": df.groupby("group", observed=True).size(),
        "B. Valid pipelines": df[valid].groupby("group", observed=True).size(),
        "C. With >=1 failure episode": df[failed].groupby("group", observed=True).size(),
        "D. With recovered failure (regression sample)": df[recovered].groupby("group", observed=True).size(),
    }).T.fillna(0).astype(int)
    flow["Total"] = flow.sum(axis=1)
    print("\n=== SAMPLE FLOW ===\n", flow)
    flow.to_csv(f"{OUT_DIR}/00_sample_flow.csv")
    return df[valid].copy(), df[failed].copy(), df[recovered].copy()


def descriptives(valid, failed, recovered):
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)
    g = valid.groupby("group", observed=True)
    t = pd.DataFrame({
        "n_valid": g.size(),
        "success_median": g["success"].median(), "success_IQR": g["success"].apply(iqr),
        "share_with_failure": g["episodes"].apply(lambda s: (s > 0).mean()),
    })
    t["share_recovered_among_failed"] = failed.groupby("group", observed=True)["status"].apply(
        lambda s: (s == "recovered").mean())
    r = recovered.groupby("group", observed=True)["mttr"]
    t["n_recovered_sample"] = r.size()
    t["mttr_median_h"], t["mttr_IQR_h"], t["mttr_mean_h"] = r.median(), r.apply(iqr), r.mean()
    print("\n=== DESCRIPTIVES ===\n", t.round(3).to_string())
    t.to_csv(f"{OUT_DIR}/01_descriptives.csv")


# ----------------------------------------------------------------------------
# 2. Kruskal-Wallis + Dunn (Holm)
# ----------------------------------------------------------------------------
def dunn_test(d, value, group="group", p_adjust="holm"):
    """Dunn's post hoc test with tie correction, rank-based."""
    x = d[[value, group]].dropna()
    x["rank"] = stats.rankdata(x[value])
    N = len(x)
    _, counts = np.unique(x[value], return_counts=True)
    tie_corr = (counts ** 3 - counts).sum() / (12.0 * (N - 1))
    sigma2 = N * (N + 1) / 12.0 - tie_corr
    g = x.groupby(group, observed=True)["rank"].agg(["mean", "count"])
    rows = []
    for a, b in itertools.combinations(g.index, 2):
        se = np.sqrt(sigma2 * (1 / g.loc[a, "count"] + 1 / g.loc[b, "count"]))
        z = (g.loc[a, "mean"] - g.loc[b, "mean"]) / se
        rows.append({"Group_1": a, "Group_2": b, "z": z, "p_raw": 2 * stats.norm.sf(abs(z))})
    res = pd.DataFrame(rows)
    res["p_adj"] = multipletests(res["p_raw"], method=p_adjust)[1]
    res["significant"] = res["p_adj"] < 0.05
    return res


def kruskal_and_dunn(d, value):
    groups = [g[value].values for _, g in d.groupby("group", observed=True)]
    H, p = stats.kruskal(*groups)
    k, N = len(groups), sum(len(g) for g in groups)
    eps2 = H * (N + 1) / (N ** 2 - 1)               # epsilon-squared effect size
    print(f"\n=== KRUSKAL-WALLIS: {value} ===  H({k-1}) = {H:.3f}, p = {p:.4g}, epsilon^2 = {eps2:.4f}")
    dunn = dunn_test(d, value)
    print(dunn.round(4))
    dunn.to_csv(f"{OUT_DIR}/02_dunn_{value}.csv", index=False)
    pd.DataFrame([{"metric": value, "H": H, "df": k - 1, "p": p, "epsilon_sq": eps2, "N": N}]).to_csv(
        f"{OUT_DIR}/02_kruskal_{value}.csv", index=False)


# ----------------------------------------------------------------------------
# 3. Pairwise Mann-Whitney with effect sizes, bootstrap CIs, Brunner-Munzel
# ----------------------------------------------------------------------------
def pairwise_mwu(d, value, method="holm"):
    rows = []
    for a, b in itertools.combinations(GROUP_ORDER, 2):
        x = d.loc[d["group"] == a, value].dropna().values
        y = d.loc[d["group"] == b, value].dropna().values
        n1, n2 = len(x), len(y)
        U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        ps = U / (n1 * n2)                      # probability of superiority P(X>Y)+0.5P(X=Y)
        rbc = 1 - 2 * U / (n1 * n2)             # same sign convention as your earlier tables
        boot = []
        for _ in range(N_BOOT):
            xb, yb = rng.choice(x, n1), rng.choice(y, n2)
            boot.append(1 - 2 * stats.mannwhitneyu(xb, yb)[0] / (n1 * n2))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        bm = stats.brunnermunzel(x, y)
        rows.append({
            "Group_1": a, "Group_2": b, "N1": n1, "N2": n2,
            "Median_1": np.median(x), "Median_2": np.median(y),
            "U": U, "p_raw": p, "r_rb": rbc, "r_rb_CI_low": lo, "r_rb_CI_high": hi,
            "P_superiority_G1>G2": ps,
            "BrunnerMunzel_p": bm.pvalue,
        })
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res["p_raw"], method="holm")[1]
    res["p_bonferroni"] = multipletests(res["p_raw"], method="bonferroni")[1]
    res["BM_p_holm"] = multipletests(res["BrunnerMunzel_p"], method="holm")[1]
    print(f"\n=== PAIRWISE MANN-WHITNEY: {value} ===\n", res.round(4).to_string())
    res.to_csv(f"{OUT_DIR}/03_pairwise_mwu_{value}.csv", index=False)


# ----------------------------------------------------------------------------
# 4. Regression helpers
# ----------------------------------------------------------------------------
def prep_regression_vars(d, log_fn=np.log):
    d = d.copy()
    d["log_mttr"] = log_fn(d["mttr"])
    d["log_lifespan"] = log_fn(d["lifespan"])
    d["log_contributors"] = log_fn(d["contributors"])
    d["is_devsecops"] = (d["group"] == "DevSecOps").astype(int)
    d["is_secdevops"] = (d["group"] == "SecDevOps").astype(int)
    d["is_any_security"] = (d["group"] != "Basic").astype(int)
    return d


def fit(formula, d, family=None, cov="HC3"):
    """OLS (family=None) or GLM with robust (HC3 or cluster) covariance."""
    kw = {"cov_type": cov}
    if "cluster" in d.columns and COL["cluster"]:
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d["cluster"])[0]}}
    model = smf.ols(formula, d) if family is None else smf.glm(formula, d, family=family)
    return model.fit(**kw)


def tidy(res, label, pct=True):
    ci = res.conf_int()
    t = pd.DataFrame({"Beta": res.params, "Robust_SE": res.bse, "p": res.pvalues,
                      "CI_low": ci[0], "CI_high": ci[1]})
    if pct:
        t["Pct_change"] = (np.exp(t["Beta"]) - 1) * 100
        t["Pct_CI_low"] = (np.exp(t["CI_low"]) - 1) * 100
        t["Pct_CI_high"] = (np.exp(t["CI_high"]) - 1) * 100
        # per-doubling interpretation for ln-predictors: 2**beta - 1
        for v in ("log_lifespan", "log_contributors"):
            if v in t.index:
                t.loc[v, ["Pct_change", "Pct_CI_low", "Pct_CI_high"]] = (
                    2 ** t.loc[v, ["Beta", "CI_low", "CI_high"]].values - 1) * 100
    t.insert(0, "Model", label)
    return t


def diagnostics(d, formula, label):
    ols = smf.ols(formula, d).fit()
    bp = het_breuschpagan(ols.resid, ols.model.exog)
    sw = stats.shapiro(ols.resid)
    vifs = {n: variance_inflation_factor(ols.model.exog, i)
            for i, n in enumerate(ols.model.exog_names) if n != "Intercept"}
    row = {"Model": label, "N": int(ols.nobs), "R2": ols.rsquared, "Adj_R2": ols.rsquared_adj,
           "F": ols.fvalue, "F_p": ols.f_pvalue, "BP_LM": bp[0], "BP_p": bp[1],
           "Shapiro_W": sw[0], "Shapiro_p": sw[1], "Max_VIF": max(vifs.values())}
    return row


# ----------------------------------------------------------------------------
# 5. Regressions (recovered sample), robustness, hurdle parts, success rate
# ----------------------------------------------------------------------------
def prep_regression_vars(d, log_fn=np.log):
    d = d.copy()
    d["log_mttr"] = log_fn(d["mttr"])
    d["log_lifespan"] = log_fn(d["lifespan"])
    d["log_contributors"] = log_fn(d["contributors"])
    d["is_devsecops"] = (d["group"] == "DevSecOps").astype(int)
    d["is_secdevops"] = (d["group"] == "SecDevOps").astype(int)
    d["is_any_security"] = (d["group"] != "Basic").astype(int)
    return d


def add_common_vars(d):
    d = d.copy()
    d["log_lifespan"] = np.log1p(d["lifespan"].clip(lower=0))
    d["log_contributors"] = np.log1p(d["contributors"].clip(lower=0))
    d["log_runs"] = np.log1p(d["runs"])
    d["is_devsecops"] = (d["group"] == "DevSecOps").astype(int)
    d["is_secdevops"] = (d["group"] == "SecDevOps").astype(int)
    return d


COVS = "log_lifespan + log_contributors + log_runs"


def main_regressions(rec):
    d = prep_regression_vars(rec)
    d["log_runs"] = np.log1p(d["runs"])
    f1 = "log_mttr ~ is_any_security + log_lifespan + log_contributors"
    f2 = "log_mttr ~ is_devsecops + is_secdevops + log_lifespan + log_contributors"
    res = pd.concat([tidy(fit(f1, d), "M1 any security (HC3)"), tidy(fit(f2, d), "M2 separate (HC3)")])
    diag = pd.DataFrame([diagnostics(d, f1, "M1"), diagnostics(d, f2, "M2")])
    print("\n=== MAIN REGRESSIONS (recovered sample, HC3) ===\n", res.round(4).to_string())
    print("\n=== DIAGNOSTICS ===\n", diag.round(4).to_string())
    res.to_csv(f"{OUT_DIR}/04_regression_main.csv")
    diag.to_csv(f"{OUT_DIR}/04_regression_diagnostics.csv", index=False)

    # formal test of DevSecOps vs SecDevOps
    r2 = fit(f2, d)
    tt = r2.t_test("is_devsecops = is_secdevops")
    print("\nDevSecOps vs SecDevOps (M2):", tt.summary_frame().round(4).to_string())

    extra = [c for c in ("stars", "forks", "commits") if c in rec.columns and rec[c].notna().all()]
    for c in extra:
        d[f"log1p_{c}"] = np.log1p(rec[c])
    f3 = f2 + " + log_runs" + "".join(f" + log1p_{c}" for c in extra)
    r3 = tidy(fit(f3, d), "M2 + runs, stars, forks, commits (HC3)")
    print("\n=== M2 WITH EXTRA COVARIATES ===\n", r3.round(4).to_string())
    r3.to_csv(f"{OUT_DIR}/04_regression_extra_covariates.csv")


def sensitivity(rec):
    f = "{y} ~ is_devsecops + is_secdevops + log_lifespan + log_contributors"
    out = []
    d = prep_regression_vars(rec)
    w = d.copy()
    lo, hi = w["mttr"].quantile([0.01, 0.99])
    w["mttr"] = w["mttr"].clip(lo, hi)
    w = prep_regression_vars(w)
    out.append(tidy(fit(f.format(y="log_mttr"), w), f"(a) winsorised 1/99, n={len(w)}", pct=False))
    q = smf.quantreg(f.format(y="log_mttr"), d).fit(q=0.5)
    out.append(tidy(q, f"(b) median regression, n={int(q.nobs)}", pct=False))
    g = fit("mttr ~ is_devsecops + is_secdevops + log_lifespan + log_contributors", d,
            family=sm.families.Gamma(link=sm.families.links.Log()))
    out.append(tidy(g, f"(c) Gamma GLM log link, n={int(g.nobs)}", pct=False))
    d2 = d[d["n_recovered"] >= 2]                       # MTTR averaged over >= 2 recoveries (less noisy)
    if d2["group"].nunique() == 3:
        out.append(tidy(fit(f.format(y="log_mttr"), d2), f"(d) >=2 recoveries, n={len(d2)}", pct=False))
    res = pd.concat(out)
    res = res[res.index.isin(["is_devsecops", "is_secdevops"])]
    res["Ratio_exp(beta)"] = np.exp(res["Beta"])
    print("\n=== SENSITIVITY: group coefficients ===\n", res.round(4).to_string())
    res.to_csv(f"{OUT_DIR}/05_sensitivity_group_effects.csv")


def logit_table(d, formula, label, path):
    r = fit(formula, d, family=sm.families.Binomial())
    t = tidy(r, label, pct=False)
    t["Odds_ratio"], t["OR_CI_low"], t["OR_CI_high"] = np.exp(t["Beta"]), np.exp(t["CI_low"]), np.exp(t["CI_high"])
    print(f"\n=== {label} (n={int(r.nobs)}) ===\n", t.round(4).to_string())
    t.to_csv(path)


def hurdle_parts(valid, failed):
    d = add_common_vars(valid)
    d["any_failure"] = (d["episodes"] > 0).astype(int)
    ct = pd.crosstab(d["group"], d["any_failure"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f"\nAny failure by group:\n{ct}\nchi2 = {chi2:.3f}, p = {p:.4g}")
    # runs is included because more runs = more opportunity to fail
    logit_table(d, f"any_failure ~ is_devsecops + is_secdevops + {COVS}",
                "Part 1: P(any failure in window)", f"{OUT_DIR}/06_hurdle_any_failure.csv")
    f = add_common_vars(failed)
    f["recovered"] = (f["status"] == "recovered").astype(int)
    logit_table(f, f"recovered ~ is_devsecops + is_secdevops + {COVS}",
                "Part 1b: P(recovered | failure occurred)", f"{OUT_DIR}/06_recovery_given_failure.csv")
    print("(Part 2 = MTTR regression in the recovered sample, see 04_regression_main.csv)")


def success_rate_models(valid):
    d = add_common_vars(valid)
    d["success"] = d["success"].clip(0, 1)
    f = f"success ~ is_devsecops + is_secdevops + {COVS}"
    logit_table(d, f, "Success rate: fractional logit (HC3)", f"{OUT_DIR}/07_success_fractional_logit.csv")
    r = smf.glm(f, d, family=sm.families.Binomial(), var_weights=d["runs"]).fit(cov_type="HC3")
    t = tidy(r, "Success rate: binomial, weighted by runs (HC3)", pct=False)
    t["Odds_ratio"] = np.exp(t["Beta"])
    print("\n=== SUCCESS RATE weighted by runs ===\n", t.round(4).to_string())
    t.to_csv(f"{OUT_DIR}/07_success_weighted.csv")



# ----------------------------------------------------------------------------
# 6. Equivalence tests (TOST) and minimum detectable effect
# ----------------------------------------------------------------------------
EQUIV_RATIOS = [1.5, 2.0]          # equivalence margin: MTTR ratio within [1/r, r]; DECIDE BEFORE LOOKING
PS_BOUNDS = {"medium (0.36-0.64)": (0.36, 0.64), "small (0.44-0.56)": (0.44, 0.56)}


def tost_from_estimate(beta, se, df, delta):
    """Two one-sided t-tests that |beta| < delta (log scale). Returns p_TOST, 90% CI."""
    p_low = stats.t.sf((beta + delta) / se, df)        # H0: beta <= -delta
    p_up = stats.t.cdf((beta - delta) / se, df)        # H0: beta >= +delta
    tcrit = stats.t.ppf(0.95, df)
    return max(p_low, p_up), (beta - tcrit * se, beta + tcrit * se)


def equivalence_tests(rec):
    d = prep_regression_vars(rec)
    f2 = "log_mttr ~ is_devsecops + is_secdevops + log_lifespan + log_contributors"
    r = fit(f2, d)                                      # HC3
    rows = []
    contrasts = {"DevSecOps vs Basic": r.params["is_devsecops"], "SecDevOps vs Basic": r.params["is_secdevops"]}
    ses = {"DevSecOps vs Basic": r.bse["is_devsecops"], "SecDevOps vs Basic": r.bse["is_secdevops"]}
    tt = r.t_test("is_secdevops - is_devsecops = 0")    # SecDevOps vs DevSecOps
    contrasts["SecDevOps vs DevSecOps"] = float(np.squeeze(tt.effect))
    ses["SecDevOps vs DevSecOps"] = float(np.squeeze(tt.sd))
    dfres = r.df_resid
    for name, b in contrasts.items():
        se = ses[name]
        for ratio in EQUIV_RATIOS:
            p, (lo, hi) = tost_from_estimate(b, se, dfres, np.log(ratio))
            rows.append({"Contrast": name, "Margin_ratio": ratio, "MTTR_ratio": np.exp(b),
                         "CI90_ratio_low": np.exp(lo), "CI90_ratio_high": np.exp(hi),
                         "p_TOST": p, "Equivalent": p < 0.05,
                         "MDE_ratio_80pct_power": np.exp(2.8 * se)})   # (1.96+0.84)*SE on log scale
    res = pd.DataFrame(rows)
    print("\n=== EQUIVALENCE (TOST) on MTTR ratio, HC3 ===\n", res.round(4).to_string())
    res.to_csv(f"{OUT_DIR}/08_equivalence_tost_mttr.csv", index=False)


def equivalence_ps(d, value):
    """Bootstrap 90% CI of the probability of superiority vs. 'equivalence' bounds around 0.5."""
    rows = []
    for a, b in itertools.combinations(GROUP_ORDER, 2):
        x = d.loc[d["group"] == a, value].dropna().values
        y = d.loc[d["group"] == b, value].dropna().values
        ps = []
        for _ in range(N_BOOT):
            xb, yb = rng.choice(x, len(x)), rng.choice(y, len(y))
            ps.append(stats.mannwhitneyu(xb, yb)[0] / (len(x) * len(y)))
        lo, hi = np.percentile(ps, [5, 95])
        row = {"Group_1": a, "Group_2": b, "PS_CI90_low": lo, "PS_CI90_high": hi}
        for lab, (l, h) in PS_BOUNDS.items():
            row[f"Equivalent_{lab}"] = bool(lo > l and hi < h)
        rows.append(row)
    res = pd.DataFrame(rows)
    print(f"\n=== NON-PARAMETRIC EQUIVALENCE ({value}): bootstrap 90% CI of P(G1>G2) ===\n", res.round(3).to_string())
    res.to_csv(f"{OUT_DIR}/08_equivalence_ps_{value}.csv", index=False)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    valid, failed, recovered = build_samples(df)
    descriptives(valid, failed, recovered)

    kruskal_and_dunn(valid, "success")                  # success rate: valid pipelines
    pairwise_mwu(valid, "success")
    kruskal_and_dunn(recovered, "mttr")                 # MTTR: repositories with a recovered failure
    pairwise_mwu(recovered, "mttr")

    hurdle_parts(valid, failed)
    main_regressions(recovered)
    sensitivity(recovered)
    equivalence_tests(recovered)
    equivalence_ps(recovered, "mttr")
    success_rate_models(valid)
    print(f"\nDone. Tables saved in ./{OUT_DIR}/")
