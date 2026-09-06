# Agentic Data Science — Plan

## Vision

An agentic data science workbench for **tabular data**. AutoML (Phases
1-3, below) was the first problem type: upload a dataset, get automatic
profiling, cleaning/feature suggestions, a trained-model leaderboard,
explainability, and predictions, the way DataRobot / H2O Driverless AI /
PyCaret's dashboard work — but with pipeline decisions made and narrated
by an LLM agent instead of a fixed script. Starting Phase 4, the workbench
grows a second kind of problem: classical statistics — hypothesis testing
and statistical modeling (GLM, Bayesian) — for when the question is "is
this difference real and how confident should I be", not "build me the
best predictor". A chat copilot is available to ask "why" at any stage,
across both.

## Reference architecture: Project-Shonku's Data Analyst agent

Still the base to build from — nothing below throws it away, it gets
reused in specific places called out per-page below.

- **`tools.py`** (`Project-Shonku/backend/agents/data_analyst/tools.py`) —
  `inspect_data`, `summarize_numeric`, `value_counts`, `correlation_matrix`,
  `detect_outliers`, `run_sql_query`, `smart_sql_query`, `run_python_code`,
  `generate_chart`. Plain pandas/sklearn/sqlite functions, no Shonku
  coupling — reused near-verbatim for the **Profile** page and the
  **Copilot** chat.
- **`sql_pipeline.py`** — full NL-to-SQL pipeline (schema context →
  TF-IDF few-shot retrieval → chain-of-thought SQL generation → execute
  with self-correction retries → memory write). Reused verbatim to power
  Copilot's "ask a question about the data" capability.
- **`agent.py`**'s supervisor+subagents pattern (`deepagents.create_deep_agent`)
  — the delegation shape (one agent per concern, a supervisor routing
  between them) is the template for the agent design below, retargeted
  from Shonku's chat-platform concerns to AutoML-lifecycle concerns.
- Known inherited limitation: `run_python_code`/`run_sql_query` execute
  LLM-generated code with no sandboxing. Fine for local single-user use;
  must be addressed before any shared/hosted deployment (tracked in
  Phase 3).

## Product shape: pages mirror the AutoML lifecycle

A Streamlit multipage app (same stack Shonku's frontend already proved
out — Python-only, fast to build, no separate frontend toolchain). Each
page has a clear human-in-the-loop point: the agent proposes, the user
confirms before anything irreversible (dropping columns, committing to a
model) happens.

| Page | Shows | Backed by | Human-in-loop point |
|---|---|---|---|
| **1. Upload & Target** | File upload or built-in dataset picker, row/column preview, dtypes | `inspect_data` | User confirms target column + task type (agent suggests classification/regression from target dtype + cardinality) |
| **2. Data Profile** | Per-column stats, missing %, distributions, correlation heatmap, outlier flags, a written narrative of what stands out | `summarize_numeric`, `value_counts`, `correlation_matrix`, `detect_outliers`, `generate_chart` + a new narrative tool | Read-only — nothing to confirm, just informs later steps |
| **3. Data Prep** | Agent's cleaning/feature plan (imputation strategy, encoding, outlier handling, derived features) as a diffable list | New `data-prep` tools | User approves/edits the plan before it's applied — never silent mutation |
| **4. Train** | Run config (metric, holdout split, which model families to try, time budget), progress log while training | New `modeling` tools | User launches the run; can cancel |
| **5. Leaderboard** | Ranked models: metric, train time, algorithm, params | New `modeling` tools | User picks a model to inspect or promote |
| **6. Model Detail** | Confusion matrix / ROC (classification) or residuals / R² (regression), feature importance, hyperparameters | New `explainer` tools | — |
| **7. Predict** | Upload new unlabeled data or fill a single-row form → predictions + confidence, downloadable CSV | Trained model + `inspect_data`-style validation | User reviews before download/export |
| **Copilot** (persistent side panel, every page) | Free-form Q&A about the data or the current run ("why is recall low on class B?") | `smart_sql_query`, `run_sql_query`, `run_python_code` (ported verbatim) | — |

## Agent design

Supervisor + subagents, same `deepagents` pattern as Shonku, retargeted to
AutoML stages instead of chat-platform concerns:

- **profiler** — EDA tools, powers the Data Profile page
- **data-prep** — cleaning/feature-engineering tools, propose-then-apply
- **modeling** — train/tune/leaderboard tools
- **explainer** — feature importance, confusion matrix/ROC/residual plots
- **copilot** — NL-to-SQL + Python (Shonku's `sql_pipeline.py`, ported)
- **report-writer** — reused pattern, now summarizing a full AutoML run
  (best model, key drivers, caveats) instead of a generic chat answer

**v1 simplification**: don't stand up all six on day one. Phase 1 ships as
a *single* agent holding the profiler + modeling tools directly — the
subagent split is real complexity that earns its keep once each concern
has enough tools/prompt weight to justify delegation (mirrors the same
call made in the previous version of this plan for the standalone
data-analyst agent). Split out data-prep, explainer, and copilot as their
own subagents in Phase 2, once each page actually exists to drive them.

## Modeling stack

Decisions made now so Phase 1 has no open blockers; revisit only if
evidence says otherwise (see Open Questions):

- **Tasks**: binary/multiclass classification and regression only for v1.
  Task type is auto-detected from the target column (object/low-cardinality
  numeric → classification, high-cardinality numeric → regression) and
  shown to the user to confirm/override on the Upload page — never silently
  assumed.
- **Model zoo**: scikit-learn baseline set (`LogisticRegression`/
  `LinearRegression`, `RandomForest`, `GradientBoosting`, `KNN`, `SVM`),
  plus `xgboost`/`lightgbm` if installed, treated as optional extras, not
  hard requirements. Chosen over `PyCaret`/`AutoGluon`: those are
  monolithic `setup()`-then-`compare_models()` APIs that fight the
  "LLM calls small plain-function tools" shape every other agent in this
  repo (and Shonku's) already uses — sklearn's per-model API fits that
  shape directly, and doesn't add a heavy opinionated dependency for what
  a granular tool-per-model already does.
- **Tuning**: `RandomizedSearchCV` on the leaderboard's top model(s) —
  already in scikit-learn, no new dependency. `optuna` is a candidate
  Phase 3 upgrade if search quality genuinely proves insufficient, not a
  day-one dependency.
- **Explainability**: `sklearn.inspection.permutation_importance` for v1
  (already in scikit-learn, model-agnostic, no new dependency). `shap` is
  a Phase 2/3 upgrade for deeper per-prediction explanations once the
  Model Detail page exists to show them.

## Tech stack

- **UI**: Streamlit, multipage (`st.navigation`, sectioned dict from
  Phase 4 on) — building this requires the `developing-with-streamlit`
  skill for styling/component work when implementation starts.
- **Agents**: `deepagents` (LangGraph), same as Shonku.
- **AutoML modeling**: pandas, scikit-learn, optional xgboost/lightgbm.
- **Statistics (Phase 4+)**: `scipy.stats` (hypothesis tests),
  `statsmodels` (GLM), `pymc` + `arviz` (Bayesian, Phase 6) — all verified
  installable and working in this repo's `.venv` before being committed
  to the plan.
- **LLM**: OpenRouter (matches Shonku, proven against the ported
  `sql_pipeline.py`), swappable behind one config function.

## Phases

**Phase 1 — MVP core loop**
Upload & Target → Profile → Train (sklearn baseline zoo only) →
Leaderboard → Predict. Single agent, no subagent split yet. Data prep is
automatic-and-basic (median/mode impute, one-hot encode) with no agent
decision-making yet — just enough to make Train runnable end to end. No
Copilot chat yet. Verify on a built-in dataset (`titanic` for
classification, `diabetes` for regression) plus one uploaded CSV before
calling this done.

**Phase 2 — Full agentic experience**
Split into the six subagents above. Add the Data Prep page with a real
propose→approve→apply flow. Add the Model Detail page's explainability
(permutation importance, confusion matrix/ROC/residual plots). Add the
Copilot side panel (port `sql_pipeline.py`). Add the report-writer's
end-of-run executive summary.

**Phase 3 — Depth & hardening**

Done:
- `xgboost`/`lightgbm` added as first-class leaderboard entries (optional —
  the zoo dicts only gain these keys if the packages import successfully).
  XGBoost needed a small wrapper (`_XGBClassifierSafe` in `modeling.py`):
  unlike every other classifier here, it errors on non-0..n-1 labels, so
  the wrapper label-encodes internally while still declaring real
  `__init__` params (not `**kwargs`) so `RandomizedSearchCV` tuning still
  works through `clone`/`get_params`/`set_params`.
- `shap_importance()` — a second, different-methodology explainability
  cross-check next to permutation importance on the Leaderboard page,
  computed on a small sample (model-agnostic SHAP calls the pipeline many
  times per row). On-demand via a button, not eager, since it takes a few
  seconds.
- Model export: download the trained `Pipeline` as pickle from the
  Leaderboard page. ONNX export not attempted (sklearn→ONNX conversion is
  finicky per-estimator-type; pickle covers the "get my model out" need
  for v1). Leaderboard CSV download added too.
- SQL sandboxing in `sql_pipeline.py`'s `_run_sql`: the LLM is asked for
  SELECT-only SQL, but a prompt isn't a security boundary. Added a
  regex guard rejecting anything that isn't a `SELECT`/`WITH` statement or
  that contains a schema/data-modifying or file-attaching keyword
  (`DROP`, `ATTACH`, `PRAGMA`, etc.) — sqlite3's DBAPI already refuses
  multi-statement injection on its own. This is a sandbox around what one
  query can do, not a full untrusted-code sandbox — `run_python_code`
  doesn't exist in this app at all (never ported), so that specific
  Shonku-inherited risk doesn't apply here.

Still deferred, not attempted:
- `optuna`-based tuning — `RandomizedSearchCV` (added in Phase 1) hasn't
  shown a real quality ceiling yet to justify the new dependency.
- Persistence of past runs across sessions — CSV/pickle export covers
  "get one run's results out"; a real run-history database is a bigger
  scope than this pass, revisit if actually needed.
- Additional data sources (databases/warehouses) beyond CSV/built-ins —
  no concrete need yet.

## Architecture change for Phase 4+: problem types

Everything through Phase 3 assumed one problem shape: pick a target,
predict it. Hypothesis tests don't have a "target" (a t-test takes two
columns, no prediction involved); GLM/Bayesian modeling have a
target+predictors shape but need a *family* choice and produce inference
output (coefficients, p-values, credible intervals), not a leaderboard.
Rather than force these into the AutoML flow, the app now has parallel
problem types after the shared data pages, and the Streamlit nav is
restructured into sections (`st.navigation` accepts a `{section: [pages]}`
dict — see the `developing-with-streamlit` skill's multipage-apps
reference) instead of one flat list:

- **Data** (shared by every problem type): Upload & target, Data profile,
  Data prep
- **AutoML** (Phases 1-3, unchanged): Train, Leaderboard, Predict
- **Statistics** (Phases 4-6, new): Hypothesis testing, Statistical
  modeling, Bayesian modeling
- **Copilot**: ungrouped, still available everywhere

The Statistics pages get their **own** column pickers rather than reusing
`st.session_state.target` — the column you'd run a t-test on isn't
necessarily the AutoML target, and coupling them would be a confusing,
unrequested link between two independent problem types.

## Modules and libraries for Phases 4-6

Verified before committing to any of these (installed + smoke-tested in
this repo's `.venv`, not assumed):

- **`agentic_ds/stats_tests.py`** (Phase 4) — `scipy.stats`. Already an
  installed transitive dependency (via scikit-learn); Phase 4 makes it a
  direct one since it's now called directly, not just through sklearn.
- **`agentic_ds/glm.py`** (Phase 5) — `statsmodels`. Chosen over trying to
  stretch sklearn's `PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`
  into this role: those fit coefficients but don't give p-values,
  confidence intervals, or deviance — and inference output *is* the point
  of a GLM page (the AutoML leaderboard already covers "just predict
  well"). Verified: installs cleanly, `sm.GLM(..., family=sm.families.Poisson()).fit()`
  runs and returns a real summary table.
- **`agentic_ds/bayesian.py`** (Phase 6) — `pymc` + `arviz`. Verified:
  both install in ~20s with no compiler issues, and a small 2-predictor
  Bayesian linear regression (500 draws, 500 tune, 2 chains) samples in
  ~9 seconds and recovers the true synthetic coefficients. Slow enough
  that every run stays behind an explicit button + spinner, never eager;
  fast enough that low-draw defaults keep the page usable.

## Phase 4 — Statistical hypothesis testing — Done

Shipped as planned, with the nav restructure and both new library/module
additions from above. One deviation from the original sketch: the
Compare-two-groups/3+-groups/paired/correlation/one-sample "modes" are
picked by the user via a segmented control rather than inferred purely
from column dtypes — cleaner than a fully automatic mode-detector, and
still leaves the *test choice within* a mode (parametric vs. non-
parametric) genuinely automatic, which is where the real judgment call is.

Concrete test catalog (not "all tests" — a specific, finite list):

- **Normality**: Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling
- **Variance/homogeneity**: Levene, Bartlett
- **Means, parametric**: one-sample t-test, two-sample t-test (Welch by
  default — doesn't assume equal variances — with Student's as an
  option), paired t-test, one-way ANOVA
- **Means, non-parametric**: Wilcoxon signed-rank (paired), Mann-Whitney U
  (two independent samples), Kruskal-Wallis (k independent samples)
- **Categorical association**: chi-square test of independence,
  chi-square goodness-of-fit, Fisher's exact (small-count 2×2 tables)
- **Correlation**: Pearson, Spearman, Kendall's tau
- **Proportions**: one-sample z-test, two-sample z-test

Agentic layer: user picks one or two columns (+ an optional group-by for
3+ groups). The agent runs assumption checks first (normality per group,
Levene for variance homogeneity), recommends the matching test (falls
back to the non-parametric equivalent when assumptions fail), runs it,
and narrates the result in plain English — statistic, p-value, an
appropriate effect size (Cohen's d / eta-squared / Cramér's V depending on
the test), and what it means practically. Same `llm.py` narrative pattern
as the rest of the app: degrades to a rule-based sentence
("p = 0.031 < 0.05 → reject H0 at the 5% level") with no API key, never a
hard requirement to get *a* result — only the phrasing improves with a key.

New: `app_pages/hypothesis_testing.py`, `agentic_ds/stats_tests.py`,
`tests/test_stats_tests.py` (verifies each test's output against
`scipy.stats` called directly, and that the assumption-check logic
actually switches branches — normal→t-test/ANOVA vs.
non-normal→Mann-Whitney/Kruskal-Wallis, small-count→Fisher's exact —
against fabricated examples designed to trigger each branch).

Ships with 6 realistic example questions grounded in the app's own
built-in datasets (not placeholders), one per test family, verified live:
"does Titanic survival depend on sex" (chi-square, real Cramér's V =
0.53), "does petal length differ by iris species" (Kruskal-Wallis — the
per-species samples aren't normal enough for ANOVA), "is wine alcohol
correlated with color intensity" (Spearman), "did Titanic survivors pay
higher fares" and "is breast-cancer mean radius associated with
diagnosis" (both fell back to Mann-Whitney U — real biological/social
data rarely satisfies the t-test's normality assumption), "does
passenger class affect fare". Each button loads its dataset and
pre-fills the form; clicking through all six against the live app
confirmed correct mode selection, correct assumption-driven test choice,
and sensible results throughout.

## Phase 5 — Statistical modeling (GLM) — Done

Shipped as planned. Same shape as Phase 4: own column pickers (target +
multiselect predictors, defaulting to the target's numeric siblings),
example questions grounded in real built-in-dataset columns, a rule-based
`interpret()` floor with an optional AI narrative on top.

One real limitation found and kept, not papered over: `detect_family`'s
"integer → Poisson" check is a shape heuristic, and it doesn't distinguish
*count* data (small non-negative integers, mean ≈ variance) from a
*continuous measurement that happens to be recorded in integer units* —
sklearn's `diabetes` target (disease progression, range 25-346) is the
latter, always whole numbers by construction, and gets auto-suggested as
Poisson even though Gaussian is the better call. The family selector's
manual override exists specifically for this — same "heuristic proposes,
human can override" pattern as `detect_task_type` in the AutoML flow — so
this was left as-is rather than chasing a more elaborate detector for one
edge case. Verified live by hitting it (the diabetes example) and
confirming the override works.

Negative binomial is deliberately never auto-suggested — it's reachable
only as the Poisson-overdispersion upgrade path (`check_overdispersion` +
a "Refit as negative binomial" button that appears when dispersion > 1.5),
matching how a statistician would actually work: fit Poisson, check the
assumption, upgrade if it fails. Inverse Gaussian is manual-only too — an
alternative to Gamma for heavier-tailed positive data, not a distinct
shape worth auto-detecting.

Families via `statsmodels`, each with a concrete trigger for when the
agent should suggest it:

- **Gaussian** (identity link) — continuous, roughly symmetric target
  (equivalent to OLS)
- **Binomial** (logit link) — binary or proportion target
- **Poisson** (log link) — count target
- **Negative binomial** — count target where variance ≫ mean (checked via
  a dispersion statistic after fitting Poisson; auto-suggested as the
  upgrade when Poisson is overdispersed)
- **Gamma** (log link) — positive continuous, right-skewed target (cost,
  duration)
- **Tweedie** — non-negative target with a point mass at zero plus a
  continuous positive part (e.g. insurance claim amounts)
- **Inverse Gaussian** — alternative to Gamma for positive continuous
  targets with heavier tails

Output: coefficient table (estimate, std err, z/t, p-value, 95% CI),
deviance, AIC/BIC, pseudo-R², a deviance-residuals-vs-fitted diagnostic
plot. The agent auto-suggests a family from the target's shape (integers
→ Poisson, then check dispersion; 0/1 → Binomial; positive + skewed →
Gamma) and the user can override before fitting.

New: `app_pages/statistical_modeling.py`, `agentic_ds/glm.py`,
`tests/test_glm.py` (Poisson and negative-binomial GLMs recover known
synthetic coefficients within tolerance; family auto-suggestion verified
for all 4 auto-detectable shapes — binomial/poisson/gamma/gaussian —
against fabricated examples designed to trigger each).

5 example questions, all grounded in real built-in-dataset columns and
verified live end-to-end (not just unit-tested): "what predicts Titanic
survival" (binomial), "what predicts family size aboard" — Titanic
`sibsp`, a genuine count column (poisson, dispersion 1.26 — not flagged
overdispersed on this particular slice), "what predicts breast-cancer
malignancy" (binomial), "what predicts breast-cancer area-measurement
error" — a real right-skewed column (`area error`, skew 5.45, verified
gamma), "what predicts diabetes disease progression" (see the
known-limitation note above — this one needed the manual override).

## Phase 6 — Bayesian modeling (stretch) — Done

Shipped as planned, sharing Phase 5's target/predictor picker shape.
`bayesian.suggest_family()` reuses `glm.detect_family()` and maps its
result onto the fixed 3-family set (gaussian→linear, binomial→logistic,
poisson→poisson) — so it inherits the same diabetes/Poisson heuristic
edge case documented in Phase 5, confirmed live rather than assumed away
(the example button surfaces it, same manual-override fix applies).

Real API surprise caught before it shipped wrong: `arviz` 1.3 renamed
`az.summary`'s `hdi_prob` parameter to `ci_prob`/`ci_kind`, and its output
columns changed shape (`hdi94_lb`/`hdi94_ub` rather than the `hdi_3%`/
`hdi_97%` names most online examples still show, since arviz now supports
both HDI and equal-tailed intervals via `ci_kind`). Found by running the
actual installed version rather than trusting memory of the API —
`agentic_ds/bayesian.py`'s docstring calls this out so it doesn't get
"fixed" back to the wrong names later.

Verified live end-to-end (Diabetes example, overridden to `linear` past
the same Poisson-heuristic quirk as Phase 5): converged (max r-hat
1.0049), correctly identified `bmi` and `s5` as credible predictors (94%
HDI excludes zero) while `bp` was not — a genuinely different call than
Phase 5's frequentist GLM made for the same data (`bp` had p = 0.00003
there), which is a real and expected divergence between the two
paradigms near a decision boundary, not a bug. Coefficient table,
convergence diagnostics, and the posterior-predictive-check chart
(observed vs. simulated distribution) all rendered correctly.

Bayesian counterparts to the three most common GLM families — linear
(Normal likelihood), logistic (Bernoulli likelihood), Poisson — via
`pymc`, with weakly-informative default priors (`Normal(0, 10)` on
coefficients, `HalfNormal(5)` on scale terms) so getting a first result
never requires specifying priors. Output: posterior mean/sd and 94% HDI
per coefficient (via `arviz`), convergence diagnostics (r_hat, effective
sample size — surfaced with an explicit warning if r_hat > 1.01, "results
may not be reliable, treat cautiously"), and a posterior-predictive-check
plot (observed vs. simulated outcome distribution).

Shares its target/predictor picker with the GLM page (same input shape,
different backend); default draws/tune/chains kept low (per the verified
~9s benchmark above) with an "advanced" expander for raising them.

New: `app_pages/bayesian_modeling.py`, `agentic_ds/bayesian.py`,
`tests/test_bayesian.py` (posterior means recover known synthetic
coefficients within a generous tolerance across all 3 families — MCMC is
stochastic, fixed seed — r_hat/ESS actually reported not dropped, and a
non-convergence warning is verified to actually fire).

## Phase 7 — Time series statistics — Done

Stationarity (ADF + KPSS, verdict only when they agree — flagged
"inconclusive" otherwise rather than picking one arbitrarily), seasonal
decomposition, ACF/PACF, Ljung-Box whiteness (used both on the raw series
and on ARIMA residuals — "is there structure" and "did the fit capture
it" are the same test applied twice), and a small ARIMA(p,d,q) fit +
forecast with confidence intervals. All via `statsmodels` — no new
dependency, reusing what Phase 5 already added.

**Real data gap found and handled honestly, not papered over**: none of
the app's five built-in datasets (iris/wine/breast_cancer/diabetes/
titanic) have a time axis — there's no honest "real" time-series example
to point to. Rather than fabricate one from memory (the classic
AirPassengers dataset is 144 hand-transcribed numbers — a genuine risk of
silent transcription errors, the same category of mistake as the titanic
row-count bug caught in Phase 4) or add a new dependency (seaborn, for
one more `load_dataset()` call) just for this, the page ships a clearly-
labeled *synthetic* monthly series (trend + yearly seasonality + noise,
deterministic seed) and says so in the UI — "not real data" is right
there in the caption. Users can still build a real series from any
numeric column of whatever they've already loaded (row order or an
actual date column as the index).

**API surprise caught before shipping wrong**: `adfuller`/`kpss` are
mid-deprecation on their old plain-tuple return in this statsmodels
version — used `result_object=True` for both to get named attributes
(`.statistic`, `.pvalue`) instead of chasing tuple positions that are
about to change meaning.

New: `agentic_ds/timeseries.py`, `app_pages/timeseries.py`,
`tests/test_timeseries.py` (a trending series is correctly flagged
non-stationary and reaches stationarity after differencing; a genuinely
autocorrelated series is distinguished from white noise via Ljung-Box;
decomposition recovers the synthetic series' known-positive trend
direction). Verified live: ADF/KPSS agreed "non-stationary" on the
synthetic series' trend, suggested d=1, and a fitted ARIMA(1,1,1)'s
residuals came back as white noise (Ljung-Box p=0.90) with a clean
12-step forecast + 95% CI table.

## Phase 8 — Survival analysis — Done

Kaplan-Meier survival curves, the log-rank test (compare survival between
groups), and Cox proportional hazards regression (which covariates affect
the hazard, with real inference — coefficients, hazard ratios, p-values,
same "inference is the point" reasoning as Phase 5's GLM choice).

Shipped as planned, and this phase didn't hit Phase 7's "no honest real
example data" problem — `lifelines` ships its own real, well-known
datasets, so the example button loads the actual `load_rossi()` study
(432 recently-released prisoners, real 1978 criminology data) rather than
anything synthetic. Verified live end to end: log-rank test on the `fin`
(financial aid) grouping matched the manually-computed value exactly
(p=0.05012), the two Kaplan-Meier curves plotted with the aid group
visibly staying higher, and the Cox fit reproduced the well-known
published result on this dataset — financial aid, age, and prior
convictions are the three significant predictors, aid and age lowering
the hazard of rearrest, priors raising it (concordance index 0.64).

One real bug caught by actually running the code, not just reading it:
`interpret_logrank` first referenced `result["group_labels"]` on the
log-rank sub-dict, but `compare_groups` only attached that key to the
outer result at the time — threw a `KeyError` on first run. Fixed by
moving `group_labels` into the log-rank sub-dict itself so
`interpret_logrank` is self-contained given just that one dict, not
dependent on being called with the right sibling data alongside it.

Proportional-hazards assumption checking (`CoxPHFitter.check_assumptions`)
deliberately not wired in — it dumps a long, plot-oriented, print-only
diagnostic that doesn't translate cleanly into an app JSON payload.
Concordance index already gives an honest "how good is this model"
number without it; revisit if someone specifically needs the assumption
check.

New: `agentic_ds/survival.py`, `app_pages/survival_analysis.py`,
`tests/test_survival.py` — Cox coefficients recovered within tolerance on
synthetic exponential-hazard data with a known true log-hazard-ratio, the
log-rank test verified to both catch a genuine 3x hazard difference and
correctly find no difference between identical groups, Kaplan-Meier's
survival function checked monotonically non-increasing, and the Cox fit
on real Rossi data checked against the dataset's own well-known direction
of effect (financial aid lowers hazard) rather than just an arbitrary
tolerance band.

## Non-goals

- Clustering, text/image tasks — tabular classification/regression/
  inference only unless a real need shows up.
- Power analysis / sample-size calculators — a natural companion to
  hypothesis testing, but not committed to this plan.
- A general-purpose Bayesian model builder with custom priors — Phase 6
  stays a small, fixed set of families with sensible defaults, not a PyMC
  authoring environment.
- Distributed/large-data processing (Spark/Dask) — assumes in-memory
  pandas-sized datasets.
- Shonku's other four agents (docqa, recommendation, story developer,
  mystery generator) — not relevant to this repo.
- Hosted multi-user deployment or auth — out of this plan's horizon.

## Open questions (defaults chosen, revisit if wrong)

- **Modeling library (AutoML)**: default sklearn + optional
  xgboost/lightgbm, not PyCaret/AutoGluon (see Modeling Stack above for
  why). Revisit if the leaderboard's quality/breadth genuinely proves
  insufficient.
- **UI framework**: default Streamlit (matches Shonku, fastest path to a
  working multipage dashboard). Revisit only if interactivity needs
  outgrow what Streamlit components can do.
- **Explainability library**: permutation importance first, SHAP as a
  cross-check (done, Phase 3).
- **GLM library**: `statsmodels`, not sklearn's limited GLM support —
  settled above, not really open (inference output is the requirement).
- **Bayesian library**: raw `pymc`+`arviz`, not `bambi` (a formula-
  interface wrapper on pymc). Default to raw pymc since the family set is
  small and fixed (3 families) — a formula-interface layer's convenience
  isn't earning its keep for that little surface area. Revisit if the
  family set grows enough that hand-writing each model gets repetitive.

## UI redesign — Done

The app functioned but looked like default unstyled Streamlit (flat page,
no visual hierarchy, raw `st.json` dumps for every fit-stats block).
Redesigned using the `developing-with-streamlit` skill's design/layouts/
theme references, no functional changes:

- **Theme**: `.streamlit/config.toml`, the skill's shadcn/ui-inspired
  template (zinc/near-black palette, Inter + JetBrains Mono, 6px radius,
  visible widget borders) — a neutral, data-table-friendly look rather
  than a colorful default.
- **Cards over dividers**: every logical section on every page moved from
  `st.subheader()` + `st.divider()` into `st.container(border=True)` —
  the skill's design.md explicitly flags dividers as heavy; bordered
  cards give the same section separation without the extra visual weight.
- **`agentic_ds/ui_helpers.py`**: one new shared helper, `metric_row()`,
  replacing the five raw `st.json(fit_stats)` dumps (Leaderboard, GLM,
  Bayesian, time series, survival) with proper `st.metric` KPI cards.
- **KPI headers**: Upload (rows/columns/missing), Leaderboard (best
  model/best score/models trained), Predict (model/task, then
  prediction/confidence for a single-row result) — each page now leads
  with the numbers that matter instead of burying them in a table.
- **Example-question buttons**: replaced the manual `st.columns(2)` +
  `cols[i % 2]` grid (Hypothesis testing, GLM, Bayesian) with
  `st.container(horizontal=True)`, per layouts.md's explicit guidance —
  buttons now size to their own label and wrap responsively instead of
  being forced into a rigid 2-column grid.
- **Conditional color**: result banners now use `st.warning`/`st.success`
  where the underlying result actually warrants it (GLM overdispersion,
  Bayesian non-convergence, ARIMA residuals failing the whiteness check,
  a significant log-rank test) instead of a flat `st.write` regardless of
  outcome.

Verified live end-to-end: Upload's new KPI row and cards, Profile's
outlier metric row (replacing raw JSON), Leaderboard's KPI header, GLM's
metric-row fit stats with the overdispersion warning correctly rendering
in orange, and Hypothesis testing's example buttons wrapping naturally
across 3 rows instead of a fixed grid. Full test suite re-run afterward
(unchanged — this was a display-only pass, no `agentic_ds/` logic
touched) to confirm nothing broke.

## Phase 9 — Skills & Tools architecture — 9a Done, 9b/9c/9d not yet built

**The debt this pays down**: the "v1 simplification" in Agent design
(above) said the subagent split — profiler/data-prep/modeling/explainer/
copilot/report-writer, each with its own tool set — would happen "in
Phase 2, once each page actually exists to drive them." Every page now
exists (12 of them, Phases 1-8), and the split never happened: Copilot
still only does NL-to-SQL, and none of the 25+ analysis functions built
since (GLM fitting, Bayesian sampling, ARIMA, Cox regression, every
hypothesis test) are reachable except by clicking through their specific
Streamlit page. Each page hand-codes its own sequence of function calls;
nothing is reusable as an *agent* capability. That's the actual gap
behind "add A/B testing" and "think about the developer journey" — the
missing piece isn't a new statistical method, it's a way for Copilot (or
any future entry point) to reach the methods that already exist.

**Two layers, not one**, because they solve different problems:

- **Tools** — LLM-callable wrappers around functions that already exist
  in `agentic_ds/`. A tool is "call `compare_two_groups`." No new
  analysis code, just a function-calling interface on top of what Phases
  1-8 already built.
- **Skills** — reusable multi-step playbooks (which tool(s) to call, in
  what order, how to interpret the result) — the same shape as this
  coding session's own Skills (a markdown file, a name, a description,
  instructions), not a Streamlit page's worth of hand-written Python. A
  Skill is "here's how to run an A/B test": pick a proportion z-test or a
  Welch t-test depending on the metric type, compute the lift and its CI,
  narrate — the actual reasoning currently hard-coded into each page.

Conflating these two would mean re-deriving playbook logic inside a
single giant tool, or duplicating it across many thin tools — keeping
them separate lets a Skill recombine existing Tools instead.

**Verified before committing to this** (in this repo's `.venv`, not
assumed): `langchain_core.tools.tool` — already installed transitively
via `langchain-openai`, zero new dependency — decorates a plain function
into a callable tool with an auto-derived schema. `ChatOpenAI.bind_tools()`
exists and accepts a list of such tools. Together these are enough for a
real tool-calling loop (call the model → check `.tool_calls` → run the
matching Python function → feed the result back as a `ToolMessage` →
repeat) without `deepagents`/LangGraph — confirming the same "v1
simplification" reasoning still holds after 8 phases: a single
tool-calling loop, not a supervisor+subagent graph, is what this actually
needs.

**Tool inventory** (wrapping existing functions, grouped by the module
that already implements them — no new analysis logic):

- `eda.py` → `schema_overview`, `numeric_summary`, `correlation_matrix`,
  `detect_outliers`, `top_value_counts`
- `stats_tests.py` → `compare_two_groups`, `compare_many_groups`,
  `compare_paired`, `test_correlation`, `test_categorical_association`,
  `one_sample_test`, both proportion z-tests
- `glm.py` → `detect_family`, `fit_glm`, `check_overdispersion`, `summarize`
- `bayesian.py` → `fit_bayesian`, `summarize`, `posterior_predictive_samples`
- `timeseries.py` → `check_stationarity`, `decompose`, `autocorrelation`,
  `whiteness_test`, `fit_arima`
- `survival.py` → `kaplan_meier`, `compare_groups`, `fit_cox`
- `sql_pipeline.py` → `nl_to_sql` (already Copilot's one existing tool)

**Design constraint**: every one of these functions takes a `pandas`
object as an argument, but LLM tool-calling args must be JSON-serializable
— the model can pass a column *name* (a string), never a DataFrame. Tools
take column names/strings and close over the currently-loaded
`st.session_state.df` at agent-construction time, resolved server-side —
the same shape Shonku's original `tools.py` used (`dataset_name` resolved
server-side, never a raw data blob passed through the LLM).

**Initial skill set** (as `skills/*.md`, mirroring this session's own
Skill file shape — frontmatter name/description + instructions), each
mapped to tools that already exist above:

- `ab_testing.md` — the feature from the "how to add A/B testing" question:
  identify metric type → proportion z-test or Welch t-test → lift % + CI →
  narrate. Needs one genuinely new tool first (sample-size/power
  calculation — flagged in Non-goals, not built) before this skill is
  complete; the comparison half is fully covered by existing tools today.
- `eda_report.md` — chain schema + stats + correlation + outliers into a
  written profile for any subset of columns, on demand, conversationally
  — mirrors the Data Profile page's logic but not tied to visiting it.
- `glm_diagnosis.md` — detect family → fit → check overdispersion → offer
  the negative-binomial upgrade → narrate — the Statistical modeling
  page's logic, reachable by asking Copilot "build me a model for X"
  instead of navigating to the GLM page.
- `trend_check.md` — stationarity + ACF/PACF → suggest an ARIMA order —
  the Time series page's logic, conversational.
- `survival_comparison.md` — Kaplan-Meier + log-rank between two groups —
  the Survival analysis page's group-comparison flow, conversational.

**Phasing**:
- **9a — Tools — Done**: `agentic_ds/agent_tools.py`, one `build_tools(df,
  source_label, model, memory_db)` factory returning 26 tools (27 with a
  model passed, for `smart_sql_query`) as closures bound to that
  DataFrame — the exact inventory planned above, one tool per bullet.
  Every tool returns a JSON string via a shared `_safe_json()` wrapper —
  including errors, as `{"error": ...}` instead of letting an exception
  kill a future tool-calling turn (verified: a bad column name and a
  wrong group count both come back as error JSON, not a raised
  exception). `fit_glm_model` and `fit_bayesian_model` are the two tools
  that don't map 1:1 to a single underlying function — `fit_glm`/
  `fit_bayesian` return a raw statsmodels/arviz object that can't cross
  the LLM boundary, so each tool composes fit+summarize (and, for
  Poisson, the overdispersion check) into one call. That's a Skill-shaped
  decision forced down into the Tool layer by the JSON-serializability
  constraint, not scope creep — noted here so 9b doesn't re-litigate it.

  `tests/test_agent_tools.py` (11 checks): every tool's schema is
  non-empty and valid; a representative tool per category is actually
  callable against a real synthetic DataFrame (EDA, a hypothesis test
  checked to match calling `stats_tests.py` directly, GLM, a Bayesian fit
  end to end via MCMC, time series, survival); tool count is exactly 26
  without a model and 27 with one; and the two error-path tests above.
  Full existing suite (8 other test modules) re-run alongside — all
  still green, confirming this was purely additive.
- **9b — Skill files**: author the five `skills/*.md` files above, plus a
  small loader (`agentic_ds/skills.py`) that parses frontmatter and lists
  what's available — same shape as this session's own skill-routing
  mechanism, deliberately, so it's a familiar pattern rather than a new
  one invented from scratch.
- **9c — Copilot becomes the real agent**: bind all Phase-9a tools to
  the model, put the Phase-9b skill menu in the system prompt, replace
  the current NL-to-SQL-only loop with a general tool-calling loop
  (`smart_sql_query`/`nl_to_sql` becomes one tool among many, not the
  only one). This is the phase that actually closes the journey gap.
- **9d — stretch, only if 9a-9c prove valuable**: offer the same
  tool-calling agent from other pages too (an "ask about this result"
  affordance on GLM/Bayesian/Survival, reusing one agent+tool set instead
  of each page's own narrative-only `generate_narrative` call).

**Non-goals for this phase**: a `deepagents`/LangGraph supervisor with
subagent routing — still not justified; explicitly re-verified above,
not just carried over from the original Phase 1 note. Tools that mutate
data — every tool here stays read-only/analysis, matching the
propose-then-apply pattern everywhere else in the app (Data Prep's
`apply_plan` stays button-driven, not agent-driven) and the same
reasoning behind Phase 3's SQL sandboxing: don't let an LLM call
anything destructive directly.

## Immediate next step

Phase 9a (Tools) is done. Next: 9b — author the five `skills/*.md`
playbooks and the small loader that lists them, per the initial skill set
above. 9c (Copilot becomes the real tool-calling agent) is the phase that
actually closes the journey gap, but it depends on 9b's skill menu
existing first. The eight numbered phases are otherwise all done; the
app covers two problem types end to end: AutoML
(Phases 1-3) and classical statistics — hypothesis testing, GLM, Bayesian
modeling, time series, and survival analysis (Phases 4-8) — sharing one
Data pipeline and one Copilot. Remaining scope decisions (see Non-goals
for what's deliberately not here yet: power analysis, database sources, run
persistence, sandboxed code execution for hosting), not obvious
follow-ups — say which direction matters most.
