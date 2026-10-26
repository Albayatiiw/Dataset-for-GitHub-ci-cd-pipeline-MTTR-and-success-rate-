#!/usr/bin/env python3
"""
Setup
  pip install PyGithub pyyaml
  PowerShell:  $env:GITHUB_TOKEN="your_token"      

Usage
  python mine_pipelines_v2.py --limit 20          # pilot run, check results by hand first
  python mine_pipelines_v2.py                     # full run (resumes automatically)
  python mine_pipelines_v2.py --repos-file old.csv   # re-measure an existing list (column: repository)
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, median

try:
    import yaml
except ImportError:          # pragma: no cover
    yaml = None
try:
    from github import Auth, Github, GithubException, RateLimitExceededException
except ImportError:          # pragma: no cover  (lets the pure functions be imported/tested without PyGithub)
    Github = Auth = None

    class GithubException(Exception):
        status = 0

    class RateLimitExceededException(GithubException):
        pass

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
#  here the queries run; 
BASE_FILTERS = "stars:>100 fork:false archived:false"
QUERIES = {
    "microservice": f"topic:microservice {BASE_FILTERS}",
    "devops":       f"topic:devops {BASE_FILTERS}",
    "ci-cd":        f"topic:ci-cd {BASE_FILTERS}",
}
SEARCH_SORT = "stars"            # the search API returns at most 1000 results per query
MAX_PER_QUERY = 1000

WINDOW_DAYS = 365                # observation window for workflow runs
RUN_EVENTS = ["push"]            # events counted on the default branch
MAX_RUNS = 300                   # safety cap on completed runs fetched per repository
MIN_RUNS_FOR_VALID = 10          # `valid_pipeline` = at least this many success/failure runs in the window
MAX_WORKFLOW_FILES = 40

MIN_TOOLS_FOR_SECDEVOPS = 1      # v1 required 2; set to 2 to mimic it
COUNT_DEPENDABOT_AS_TOOL = False # dependabot/renovate are recorded in their own columns

OUTPUT_CSV = "pipeline_analytics_v2.csv"
ERROR_CSV = "pipeline_errors_v2.csv"
CANDIDATES_CSV = "candidates_v2.csv"
SCRIPT_VERSION = "2.0"

POSTURE_BASIC = "Basic / No Automated Pipeline Security"
POSTURE_DEVSECOPS = "DevSecOps (Shift-Left Integrated)"
POSTURE_SECDEVOPS = "SecDevOps (Strict Security Gating)"

# ---------------------------------------------------------------------------
# SECURITY TOOL SIGNATURES  (regexes applied to the lower-cased `uses:` and `run:` text of each step)
# ---------------------------------------------------------------------------
SECURITY_TOOLS = {
    "codeql":        {"uses": [r"github/codeql-action"], "run": [r"\bcodeql\b"]},
    "snyk":          {"uses": [r"\bsnyk/"], "run": [r"\bsnyk\b"]},
    "trivy":         {"uses": [r"aquasecurity/trivy-action"], "run": [r"\btrivy\b"]},
    "semgrep":       {"uses": [r"semgrep/", r"returntocorp/semgrep"], "run": [r"\bsemgrep\b"]},
    "sonar":         {"uses": [r"sonarsource/", r"sonarqube", r"sonarcloud"],
                      "run": [r"\bsonar-scanner\b", r"\bsonarqube\b", r"\bsonarcloud\b", r"sonar\.projectkey"]},
    "gitleaks":      {"uses": [r"gitleaks/"], "run": [r"\bgitleaks\b"]},
    "trufflehog":    {"uses": [r"trufflesecurity/"], "run": [r"\btrufflehog\b"]},
    "checkov":       {"uses": [r"bridgecrewio/checkov"], "run": [r"\bcheckov\b"]},
    "tfsec":         {"uses": [r"aquasecurity/tfsec"], "run": [r"\btfsec\b"]},
    "bandit":        {"uses": [], "run": [r"\bbandit\b"]},
    "owasp-zap":     {"uses": [r"zaproxy/"], "run": [r"\bzap-(baseline|full-scan|api-scan)\b", r"\bzap\.sh\b"]},
    "owasp-depcheck": {"uses": [r"dependency-check/"], "run": [r"\bdependency-check\b"]},
    "dependency-review": {"uses": [r"actions/dependency-review-action"], "run": []},
    "grype/anchore": {"uses": [r"anchore/"], "run": [r"\bgrype\b"]},
    "sbom":          {"uses": [r"cyclonedx", r"anchore/sbom-action"], "run": [r"\bsyft\b", r"\bcyclonedx\b"]},
    "cosign":        {"uses": [r"sigstore/"], "run": [r"\bcosign\b"]},
    "scorecard":     {"uses": [r"ossf/scorecard-action"], "run": []},
    "npm-audit":     {"uses": [], "run": [r"\b(npm|yarn|pnpm) audit\b"]},
    "pip-audit":     {"uses": [], "run": [r"\bpip-audit\b", r"\bsafety (check|scan)\b"]},
    "govulncheck":   {"uses": [], "run": [r"\bgovulncheck\b"]},
    "cargo-audit":   {"uses": [], "run": [r"\bcargo[ -]audit\b", r"\bcargo deny\b"]},
}
# Tools whose default behaviour is to fail the step when findings exist.
FAILS_BY_DEFAULT = {"snyk", "bandit", "gitleaks", "checkov", "tfsec", "dependency-review",
                    "npm-audit", "pip-audit", "govulncheck", "cargo-audit"}
# Explicit "fail on findings" settings (for tools that do not fail by default).
GATE_PATTERNS = [
    r"exit-code['\"]?:\s*['\"]?[1-9]", r"--exit-code[ =]['\"]?[1-9]", r"--severity-threshold",
    r"fail[-_]on", r"fail[-_]build", r"failbuild", r"--error\b", r"--fail-on\b",
    r"sonar\.qualitygate\.wait\s*[:=]\s*['\"]?true", r"qualitygate",
]
# Settings that make a step non-blocking.
SOFT_PATTERNS = [
    r"soft[-_]fail['\"]?:\s*['\"]?true", r"--soft-fail", r"--exit-zero",
    r"exit-code['\"]?:\s*['\"]?0\b", r"--exit-code[ =]['\"]?0\b", r"\|\|\s*true", r"\|\|\s*exit 0",
]

# ---------------------------------------------------------------------------
#  FUNCTIONS (no network) 
# ---------------------------------------------------------------------------
def _truthy(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _norm_dt(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iter_steps(doc):
    """Yield (job_id, step_dict, job_continue_on_error) for every step of a parsed workflow."""
    for job_id, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        job_coe = _truthy(job.get("continue-on-error"))
        if "uses" in job:                                   # reusable workflow call
            yield job_id, {"uses": job["uses"], "with": job.get("with")}, job_coe
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                yield job_id, step, job_coe


def scan_workflow_text(text):
    """Return (findings, parsed_ok). Each finding: {tool, job, blocking, evidence}."""
    doc = None
    if yaml is not None:
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError:
            doc = None
    findings = []
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
        low = text.lower()                                  # fallback: never counted as blocking
        for tool, pats in SECURITY_TOOLS.items():
            if any(re.search(p, low) for p in pats["uses"] + pats["run"]):
                findings.append({"tool": tool, "job": "?", "blocking": False, "evidence": "unparsed"})
        return findings, False

    for job_id, step, job_coe in iter_steps(doc):
        uses = str(step.get("uses") or "").lower()
        run = str(step.get("run") or "").lower()
        w = step.get("with")
        with_txt = "\n".join(f"{k}: {v}" for k, v in w.items()).lower() if isinstance(w, dict) else ""
        blob = f"{uses}\n{run}\n{with_txt}"
        coe = job_coe or _truthy(step.get("continue-on-error"))
        for tool, pats in SECURITY_TOOLS.items():
            hit = any(re.search(p, uses) for p in pats["uses"]) or any(re.search(p, run) for p in pats["run"])
            if not hit:
                continue
            soft = coe or any(re.search(p, blob) for p in SOFT_PATTERNS)
            gate = tool in FAILS_BY_DEFAULT or any(re.search(p, blob) for p in GATE_PATTERNS)
            findings.append({"tool": tool, "job": str(job_id), "blocking": bool(gate and not soft),
                             "evidence": (uses or run.strip().replace("\n", " ")[:60])})
    return findings, True


def classify_posture(n_tools, n_blocking, has_dependabot=False):
    n = n_tools + (1 if (COUNT_DEPENDABOT_AS_TOOL and has_dependabot) else 0)
    if n >= MIN_TOOLS_FOR_SECDEVOPS and n_blocking >= 1:
        return POSTURE_SECDEVOPS
    if n >= 1:
        return POSTURE_DEVSECOPS
    return POSTURE_BASIC


FAIL_CONCLUSIONS = {"failure", "timed_out"}


def recovery_metrics(runs, window_end):
    """
    runs: iterable of dicts {created_at (tz-aware datetime), conclusion (str), workflow_id}.
    A failure episode starts at the first failed run of a workflow and ends at the next successful
    run of the SAME workflow. Duration = created_at(success) - created_at(first failure).
    Failures never followed by a success are right-censored and counted as `n_unrecovered`.
    """
    by_wf = defaultdict(list)
    for r in runs:
        if r["conclusion"] == "success" or r["conclusion"] in FAIL_CONCLUSIONS:
            by_wf[r["workflow_id"]].append(r)
    durations, censored = [], []
    n_success = n_fail = n_episodes = 0
    for rs in by_wf.values():
        rs.sort(key=lambda r: r["created_at"])
        open_start = None
        for r in rs:
            if r["conclusion"] == "success":
                n_success += 1
                if open_start is not None:
                    durations.append((r["created_at"] - open_start).total_seconds() / 3600.0)
                    open_start = None
            else:
                n_fail += 1
                if open_start is None:
                    open_start = r["created_at"]
                    n_episodes += 1
        if open_start is not None:
            censored.append((window_end - open_start).total_seconds() / 3600.0)
    n_total = n_success + n_fail
    if n_episodes == 0:
        status = "no_failures"
    elif durations:
        status = "recovered"
    else:
        status = "unrecovered_only"
    return {
        "n_completed_runs": n_total,
        "n_success": n_success,
        "n_failure": n_fail,
        "success_rate_pct": round(100.0 * n_success / n_total, 2) if n_total else None,
        "n_failure_episodes": n_episodes,
        "n_recovered": len(durations),
        "n_unrecovered": len(censored),
        "mttr_hours_mean": round(mean(durations), 3) if durations else None,
        "mttr_hours_median": round(median(durations), 3) if durations else None,
        "max_unrecovered_hours": round(max(censored), 1) if censored else None,
        "mttr_status": status,
    }


# ---------------------------------------------------------------------------
# GITHUB ACCESS
# ---------------------------------------------------------------------------
def wait_for_reset(gh, minimum=60):
    reset = getattr(gh, "rate_limiting_resettime", 0) or 0
    wait = max(reset - int(time.time()), minimum) + 5
    print(f"  [rate limit] sleeping {wait} s ...")
    time.sleep(wait)


def wait_if_low(gh, threshold=150):
    remaining = gh.rate_limiting[0]
    if 0 <= remaining < threshold:
        wait_for_reset(gh, minimum=1)


def analyze_workflows(repo):
    out = {"workflow_files": 0, "tools": set(), "blocking": [], "parse_failures": 0}
    try:
        items = repo.get_contents(".github/workflows")
    except GithubException as e:
        if e.status == 404:
            return out
        raise
    if not isinstance(items, list):
        items = [items]
    files = [c for c in items if c.name.lower().endswith((".yml", ".yaml"))][:MAX_WORKFLOW_FILES]
    out["workflow_files"] = len(files)
    for c in files:
        text = c.decoded_content.decode("utf-8", errors="ignore")
        findings, ok = scan_workflow_text(text)
        if not ok:
            out["parse_failures"] += 1
        for f in findings:
            out["tools"].add(f["tool"])
            if f["blocking"]:
                out["blocking"].append(f"{c.name}:{f['job']}:{f['tool']}")
    return out


def has_file(repo, directory, names):
    try:
        items = repo.get_contents(directory)
    except GithubException as e:
        if e.status == 404:
            return False
        raise
    if not isinstance(items, list):
        items = [items]
    return any(c.name.lower() in names for c in items)


def fetch_runs(repo, branch, now):
    since = (now - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    runs, capped = [], False
    for ev in RUN_EVENTS:
        for run in repo.get_workflow_runs(branch=branch, event=ev, status="completed",
                                          exclude_pull_requests=True, created=f">={since}"):
            runs.append({"created_at": _norm_dt(run.created_at), "conclusion": run.conclusion,
                         "workflow_id": run.workflow_id})
            if len(runs) >= MAX_RUNS:
                capped = True
                break
    return runs, capped


def process_repo(gh, repo, queries):
    now = datetime.now(timezone.utc)
    created, pushed = _norm_dt(repo.created_at), _norm_dt(repo.pushed_at)
    branch = repo.default_branch

    wf = analyze_workflows(repo)
    dependabot = has_file(repo, ".github", {"dependabot.yml", "dependabot.yaml"})
    renovate = has_file(repo, "", {"renovate.json", "renovate.json5", ".renovaterc", ".renovaterc.json"})

    runs, capped = fetch_runs(repo, branch, now)
    m = recovery_metrics(runs, window_end=now)

    try:
        contributors = repo.get_contributors(anon="true").totalCount
    except GithubException as e:
        if isinstance(e, RateLimitExceededException):
            raise
        contributors = None                 # e.g. "contributor list too large"
    try:
        commits = repo.get_commits().totalCount
    except GithubException as e:
        if isinstance(e, RateLimitExceededException):
            raise
        commits = None

    n_tools = len(wf["tools"])
    row = {
        "repository": repo.full_name, "url": repo.html_url, "source_queries": ";".join(sorted(queries)),
        "stars": repo.stargazers_count, "forks": repo.forks_count, "language": repo.language,
        "default_branch": branch,
        "created_at": created.isoformat() if created else None,
        "last_activity_at": pushed.isoformat() if pushed else None,
        "collected_at": now.date().isoformat(),
        "project_lifespan_days": (pushed - created).days if created and pushed else None,
        "repo_age_days": (now - created).days if created else None,
        "contributors_count": contributors, "total_commits": commits,
        "workflow_files": wf["workflow_files"],
        "security_tools_found": ", ".join(sorted(wf["tools"])) if wf["tools"] else "None",
        "security_tool_count": n_tools,
        "blocking_gate_count": len(wf["blocking"]),
        "blocking_gate_evidence": "; ".join(wf["blocking"][:10]),
        "dependabot_config": dependabot, "renovate_config": renovate,
        "workflow_parse_failures": wf["parse_failures"],
        "classified_posture": classify_posture(n_tools, len(wf["blocking"]), dependabot or renovate),
        "window_days": WINDOW_DAYS, "runs_capped": capped,
        "valid_pipeline": m["n_completed_runs"] >= MIN_RUNS_FOR_VALID,
        "script_version": SCRIPT_VERSION,
    }
    row.update(m)
    return row


FIELDS = ["repository", "url", "source_queries", "stars", "forks", "language", "default_branch",
          "created_at", "last_activity_at", "collected_at", "project_lifespan_days", "repo_age_days",
          "contributors_count", "total_commits", "workflow_files", "security_tools_found",
          "security_tool_count", "blocking_gate_count", "blocking_gate_evidence", "dependabot_config",
          "renovate_config", "workflow_parse_failures", "classified_posture", "window_days", "runs_capped",
          "valid_pipeline", "n_completed_runs", "n_success", "n_failure", "success_rate_pct",
          "n_failure_episodes", "n_recovered", "n_unrecovered", "mttr_hours_mean", "mttr_hours_median",
          "max_unrecovered_hours", "mttr_status", "script_version"]


def collect_candidates(gh, limit=None):
    cands = {}
    for name, q in QUERIES.items():
        print(f"Searching '{name}': {q}")
        n = 0
        for repo in gh.search_repositories(query=q, sort=SEARCH_SORT, order="desc"):
            if n >= MAX_PER_QUERY:
                break
            n += 1
            if repo.fork or repo.archived:
                continue
            cands.setdefault(repo.full_name, {"repo": repo, "queries": set()})["queries"].add(name)
        print(f"  -> {n} results; unique so far: {len(cands)}")
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repository", "source_queries"])
        for k, v in cands.items():
            w.writerow([k, ";".join(sorted(v["queries"]))])
    return cands


def read_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["repository"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="process only the first N repositories (pilot run)")
    ap.add_argument("--repos-file", help="CSV with a 'repository' column (owner/name) instead of searching")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set the GITHUB_TOKEN environment variable first.")
    if Github is None or yaml is None:
        sys.exit("Install dependencies: pip install PyGithub pyyaml")
    try:
        gh = Github(auth=Auth.Token(token), per_page=100)
    except Exception:
        gh = Github(token, per_page=100)

    if args.repos_file:
        with open(args.repos_file, newline="", encoding="utf-8-sig") as f:
            cands = {r["repository"].strip(): {"repo": None, "queries": {"repos_file"}}
                     for r in csv.DictReader(f) if r.get("repository")}
    else:
        cands = collect_candidates(gh)

    names = list(cands)[: args.limit] if args.limit else list(cands)
    done = read_done(OUTPUT_CSV)
    new_file = not os.path.exists(OUTPUT_CSV)
    print(f"{len(names)} repositories to process ({len(done)} already done).")

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fout, \
            open(ERROR_CSV, "a", newline="", encoding="utf-8") as ferr:
        writer = csv.DictWriter(fout, fieldnames=FIELDS, extrasaction="ignore")
        errw = csv.writer(ferr)
        if new_file:
            writer.writeheader()
        for i, name in enumerate(names, 1):
            if name in done:
                continue
            for attempt in range(3):
                try:
                    wait_if_low(gh)
                    repo = cands[name]["repo"] or gh.get_repo(name)
                    row = process_repo(gh, repo, cands[name]["queries"])
                    writer.writerow(row)
                    fout.flush()
                    print(f"[{i}/{len(names)}] {name} | {row['classified_posture'][:9]} | runs={row['n_completed_runs']}"
                          f" | success={row['success_rate_pct']} | MTTR={row['mttr_hours_mean']} ({row['mttr_status']})")
                    break
                except RateLimitExceededException:
                    wait_for_reset(gh)
                except Exception as e:                      # log and continue with the next repository
                    errw.writerow([name, type(e).__name__, str(e)[:300]])
                    ferr.flush()
                    print(f"[{i}/{len(names)}] {name} ERROR: {e}")
                    break
    print(f"Done. Results: {OUTPUT_CSV} | errors: {ERROR_CSV}")


if __name__ == "__main__":
    main()
