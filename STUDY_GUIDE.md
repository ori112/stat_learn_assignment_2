# Statistical Learning — Comprehensive Study Guide
### HIT M.Sc. Data Science | Assignment 2 Topics

This guide covers every concept from the assignment at exam depth: theory, intuition, formulas, code snippets, and common pitfalls.  
Use the table of contents to jump to any topic.

---

## Table of Contents

1. [OLS Regression](#1-ols-regression)
2. [Categorical Predictors & Interactions](#2-categorical-predictors--interactions)
3. [Model Selection Criteria](#3-model-selection-criteria)
4. [Subset Selection Algorithms](#4-subset-selection-algorithms)
5. [Regularization: Ridge, LASSO, Elastic Net](#5-regularization-ridge-lasso-elastic-net)
6. [Bootstrap (Nonparametric)](#6-bootstrap-nonparametric)
7. [Cross-Validation](#7-cross-validation)
8. [Nonlinear Transformations](#8-nonlinear-transformations)
9. [Nonparametric Statistics](#9-nonparametric-statistics)
10. [Quick-Reference Formula Sheet](#10-quick-reference-formula-sheet)

---

## 1. OLS Regression

### Model

$$y_i = \beta_0 + \beta_1 x_{1i} + \cdots + \beta_p x_{pi} + \epsilon_i, \quad i = 1, \ldots, n$$

In matrix form: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$.

### Classical Assumptions

| # | Name | Statement |
|---|------|-----------|
| A1 | Linearity | $E[\epsilon_i \mid \mathbf{x}_i] = 0$ — the mean of the error is zero given any predictor values |
| A2 | Exogeneity | Predictors are uncorrelated with the error term ($\text{Cov}(x_{ji}, \epsilon_i) = 0$) |
| A3 | Homoskedasticity | $\text{Var}(\epsilon_i \mid \mathbf{x}) = \sigma^2$ — constant across all observations |
| A4 | No autocorrelation | $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$ |
| A5 | No perfect multicollinearity | $(\mathbf{X}^\top\mathbf{X})$ is invertible (full column rank) |
| A6 | Normality (for inference) | $\epsilon_i \overset{iid}{\sim} N(0, \sigma^2)$ |

A1–A5 are the Gauss-Markov conditions guaranteeing BLUE (Best Linear Unbiased Estimator).  
A6 is additionally needed for valid t/F-tests in small samples.

### Closed-Form Solution

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

**Geometric interpretation:** $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}$ where the **hat matrix** $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ orthogonally projects $\mathbf{y}$ onto the column space of $\mathbf{X}$.

### Key Properties

- **Unbiased:** $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$  
- **Variance:** $\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$  
- **Consistency:** $\hat{\boldsymbol{\beta}} \xrightarrow{p} \boldsymbol{\beta}$ as $n \to \infty$  

### Inference

**t-test for a single coefficient:**
$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p-1} \quad \text{under } H_0: \beta_j = 0$$

**Overall F-test** ($H_0$: all slope coefficients are zero):
$$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)} \sim F_{p,\, n-p-1}$$

**Partial F-test** (nested model comparison — does adding variables improve fit?):
$$F = \frac{(RSS_{\text{reduced}} - RSS_{\text{full}})/q}{RSS_{\text{full}}/(n-p-1)} \sim F_{q,\, n-p-1}$$
where $q$ = number of additional parameters in the full model.

### R² and Adjusted R²

$$R^2 = 1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}, \qquad \bar{R}^2 = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$$

$R^2$ always increases when you add a predictor (even useless ones). $\bar{R}^2$ penalises for complexity and can decrease. **Use $\bar{R}^2$ for model selection, not $R^2$.**

### Code: Fitting and Reading the Summary

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Formula API (easiest for interactions and categoricals)
model = smf.ols("spend ~ income + C(sex) + income:C(sex)", data=df).fit()
print(model.summary())

# Key attributes
model.params          # coefficient estimates
model.bse             # standard errors
model.tvalues         # t-statistics
model.pvalues         # p-values
model.conf_int()      # 95% CIs
model.rsquared        # R²
model.rsquared_adj    # Adjusted R²
model.aic             # AIC
model.ssr             # Residual sum of squares (RSS)
model.fittedvalues    # ŷ
model.resid           # residuals ε̂
```

### Model Diagnostics

| Plot | What it checks | Good pattern |
|------|---------------|--------------|
| Residuals vs Fitted | Linearity, homoskedasticity | Points randomly scattered around 0, no funnel |
| Q-Q of residuals | Normality (A6) | Points on the 45° line |
| Histogram of residuals | Normality | Bell-shaped, symmetric around 0 |
| Cook's Distance | Influential observations | Most points well below 4/n threshold |

```python
import scipy.stats as stats
from statsmodels.stats.outliers_influence import OLSInfluence

residuals  = model.resid
fitted     = model.fittedvalues
cooks_d    = OLSInfluence(model).cooks_distance[0]

# Residuals vs fitted
plt.scatter(fitted, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')

# Q-Q plot
stats.probplot(residuals, dist='norm', plot=plt)

# Cook's distance threshold: 4/n
threshold = 4 / len(df)
n_influential = (cooks_d > threshold).sum()
```

**Common pitfalls:**
- Forgetting that $R^2 = 1$ when the number of predictors equals $n-1$ (perfect fit = overfitting).
- Interpreting a non-significant interaction as "no difference" — low power, not absence of effect.
- Not standardising before comparing coefficient magnitudes.

---

## 2. Categorical Predictors & Interactions

### Dummy / Indicator Coding

A categorical variable with $k$ levels needs $k-1$ dummy variables (one is the **reference category**, absorbed into the intercept).

**Example:** `city` with levels {Haifa, TLV, Jerusalem}. Drop Haifa (reference):

| city | `TLV` | `Jerusalem` |
|------|-------|-------------|
| Haifa | 0 | 0 |
| TLV | 1 | 0 |
| Jerusalem | 0 | 1 |

The intercept represents the mean response for Haifa (the reference).

### Interpretation of Interaction Coefficients

Model: `spend ~ income * C(sex)` expands to `spend ~ income + C(sex) + income:C(sex)`

$$\hat{\text{spend}} = \hat{\beta}_0 + \hat{\beta}_1 \cdot \text{income} + \hat{\beta}_2 \cdot \text{sex} + \hat{\beta}_3 \cdot (\text{income} \times \text{sex})$$

- $\hat{\beta}_0$: intercept for `sex=0` (Female)
- $\hat{\beta}_1$: income slope for Female
- $\hat{\beta}_2$: intercept **shift** for Male vs Female (at `income=0`)
- $\hat{\beta}_3$: slope **shift** for Male vs Female — **this is the interaction**

Male line: $(\hat{\beta}_0 + \hat{\beta}_2) + (\hat{\beta}_1 + \hat{\beta}_3) \cdot \text{income}$

### Testing Whether Slopes Differ (Interaction Test)

$$H_0: \beta_3 = 0 \quad \text{vs} \quad H_1: \beta_3 \neq 0$$

Two equivalent approaches:
1. **t-test** on $\hat{\beta}_3$ directly from the summary.
2. **Partial F-test**: compare the full model (with interaction) to the reduced model (without interaction).

```python
full_model    = smf.ols("spend ~ income * C(sex)", data=df).fit()
reduced_model = smf.ols("spend ~ income + C(sex)", data=df).fit()

# Partial F-test
import statsmodels.api as sm
anova_table = sm.stats.anova_lm(reduced_model, full_model)
print(anova_table)
```

### Code: Explicit Interaction Column

```python
df["income_x_sex"] = df["income"] * df["sex"]   # manual interaction column
model = smf.ols("spend ~ income + sex + income_x_sex", data=df).fit()
```

**Common pitfalls:**
- Including an interaction without the corresponding main effects violates the **marginality principle** (the interaction effect is uninterpretable without main effects).
- Forgetting that changing the reference category changes all coefficients (but not fitted values or predictions).

---

## 3. Model Selection Criteria

### R² vs Adjusted R² vs AIC vs BIC

| Criterion | Formula | Penalty | Best when |
|-----------|---------|---------|-----------|
| $R^2$ | $1 - RSS/TSS$ | None | Never use for selection |
| $\bar{R}^2$ | $1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$ | df penalty | Small $p$, interpretability priority |
| AIC | $-2\ell + 2p$ | $2p$ | Prediction, asymptotically optimal |
| BIC | $-2\ell + p\log n$ | $p\log n$ | Consistency (selects true model as $n\to\infty$) |

For $n > 8$: BIC penalises more heavily than AIC → tends to select smaller models.  
For prediction tasks: AIC is preferred (it minimises expected prediction error asymptotically).

### AIC for the Gaussian Linear Model (Q5 Derivation)

**Claim:** $AIC = n\log(\hat{\sigma}^2_{MLE}) + C_{n,p}$

**Derivation:**

1. Log-likelihood:  
   $\ell(\hat{\boldsymbol{\beta}}, \hat{\sigma}^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\hat{\sigma}^2_{MLE}) - \frac{n}{2}$  
   (after plugging in MLEs: $\hat{\sigma}^2_{MLE} = RSS/n$, $RSS / (2\hat{\sigma}^2_{MLE}) = n/2$)

2. $-2\ell = n\log(2\pi) + n\log(\hat{\sigma}^2_{MLE}) + n$

3. $AIC = -2\ell + 2(p+2) = n\log(\hat{\sigma}^2_{MLE}) + \underbrace{n\log(2\pi) + n + 2(p+2)}_{C_{n,p}}$

Since $C_{n,p}$ only depends on $n$ and $p$, **comparing AICs across models with the same $n$ is equivalent to comparing $\log(\hat{\sigma}^2_{MLE}) + 2p/n$** (penalised log-variance).

### LOOCV MSE

Leave-one-out cross-validation: train on $n-1$ observations, predict the held-out one, average the squared errors.

$$LOOCV = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_{-i})^2$$

**Shortcut for OLS** (no re-fitting needed):
$$\hat{y}_{-i} = \frac{y_i - h_{ii}\hat{y}_i}{1 - h_{ii}}, \quad h_{ii} = [\mathbf{H}]_{ii}$$

where $h_{ii}$ is the $i$-th diagonal of the hat matrix (leverage). This is the **PRESS statistic**.

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression

loo     = LeaveOneOut()
lr      = LinearRegression()
cv_neg_mse = cross_val_score(lr, X, y, cv=loo, scoring='neg_mean_squared_error')
loocv_mse  = -cv_neg_mse.mean()
```

**Common pitfalls:**
- Comparing AICs across datasets with different $n$ — meaningless.
- Using plain $R^2$ to select among models with different numbers of predictors.
- Forgetting that AIC/BIC as implemented in `statsmodels` include the $\hat{\sigma}^2$ parameter in $p$ — so `model.aic` has $p$ = (number of regressors) + 1 (intercept) + 1 ($\sigma^2$).

---

## 4. Subset Selection Algorithms

### Global Best (All-Subsets)

Enumerate all $2^p - 1$ non-empty subsets of $p$ candidate predictors. For each subset, fit the model and record the criterion (e.g., $\bar{R}^2$ or AIC). Select the globally optimal subset.

**Complexity:** $O(2^p)$ — only feasible for small $p$ (say $p \leq 15$).

```python
import itertools
import statsmodels.formula.api as smf

candidates = ["income", "sex", "income_x_sex"]
best_adj_r2 = -np.inf
for k in range(1, len(candidates) + 1):
    for subset in itertools.combinations(candidates, k):
        formula  = "y ~ " + " + ".join(subset)
        model    = smf.ols(formula, data=df).fit()
        adj_r2   = model.rsquared_adj
        if adj_r2 > best_adj_r2:
            best_adj_r2   = adj_r2
            best_formula  = formula
```

### Forward Selection

**Start:** empty model.  
**Each step:** add the single variable that most improves AIC.  
**Stop:** no addition improves AIC.

```python
def forward_aic(data, response, candidates):
    current_vars, current_aic = [], smf.ols(f"{response} ~ 1", data=data).fit().aic
    improved = True
    while improved:
        improved, best_var = False, None
        for var in candidates:
            if var in current_vars: continue
            aic = smf.ols(f"{response} ~ " + " + ".join(current_vars + [var]), data=data).fit().aic
            if aic < current_aic:
                current_aic, best_var, improved = aic, var, True
        if improved:
            current_vars.append(best_var)
    return smf.ols(f"{response} ~ " + (" + ".join(current_vars) or "1"), data=data).fit()
```

### Backward Elimination

**Start:** full model (all candidates).  
**Each step:** remove the single variable whose removal most improves AIC.  
**Stop:** no removal improves AIC.

### Stepwise (Bidirectional)

**Each step:** consider both adding and removing one variable; take the single move (add or remove) that most improves AIC. Can recover from mistakes made by pure forward/backward search.

### Comparison of Methods

| Method | Strength | Weakness |
|--------|----------|---------|
| Global best | Guaranteed optimal | Exponential in $p$ |
| Forward | Fast; works well when true model is sparse | Can miss interactions; early mistakes are not corrected |
| Backward | Starts with full model, sees all predictors | Expensive for very large $p$ |
| Stepwise | Corrects forward mistakes; good compromise | Still greedy; not globally optimal |

**Key fact:** All greedy methods can produce different final models from the same data.

**Common pitfalls:**
- Treating the stepwise-selected model as if it were pre-specified (inflated t-statistics, underestimated p-values — post-selection inference problem).
- Not considering that the selection criterion (AIC vs BIC vs $\bar{R}^2$) can lead to different best models.

---

## 5. Regularization: Ridge, LASSO, Elastic Net

### Motivation

OLS is unbiased but can have high variance when predictors are correlated or $p$ is large relative to $n$. Regularisation **shrinks** coefficients toward zero, trading some bias for a large reduction in variance.

### Ridge Regression

**Objective:**
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \|y - X\beta\|^2 + \lambda\|\boldsymbol{\beta}\|^2_2$$

**Closed-form solution:**
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

For the simple case $y = \beta x + \epsilon$:
$$\hat{\beta}_{\text{Ridge}} = \frac{\sum x_i y_i}{\sum x_i^2 + \lambda}$$

- **Always has a solution** (even when $\mathbf{X}^\top\mathbf{X}$ is singular).  
- **Biased:** $E[\hat{\boldsymbol{\beta}}_{\text{Ridge}}] \neq \boldsymbol{\beta}$ in general.  
- **Never produces exact zeros** — keeps all predictors.  
- **Consistent** if $\lambda/n \to 0$.

### LASSO

**Objective:**
$$\hat{\boldsymbol{\beta}}_{\text{LASSO}} = \arg\min_{\boldsymbol{\beta}} \|y - X\beta\|^2 + \lambda\|\boldsymbol{\beta}\|_1$$

**Solution (soft-thresholding for orthogonal predictors):**
$$\hat{\beta}_{j,\text{LASSO}} = \text{sign}(\hat{\beta}_{j,\text{OLS}}) \cdot \max\left(|\hat{\beta}_{j,\text{OLS}}| - \frac{\lambda}{2}, 0\right)$$

- **Produces exact zeros** — performs **variable selection**.  
- **Biased** (shrinks surviving coefficients toward zero).  
- **Consistent** if $\lambda/n \to 0$.  
- Geometric intuition: the L1 ball has corners on the coordinate axes; the optimal solution tends to land on a corner (sparse solution).

### Why LASSO Produces Sparsity but Ridge Does Not

The L1 ball is a **diamond** with sharp corners aligned with the axes; the quadratic objective ellipsoid touches a corner (zero coordinate) with positive probability.  
The L2 ball is a **sphere** with no corners; the ellipsoid touches the smooth surface where all coordinates are non-zero.

### Relaxed LASSO

1. Run LASSO with CV-tuned $\lambda$ → identify survivors (non-zero coefficients).
2. Refit plain OLS on the survivors only.

**Why:** LASSO shrinks surviving coefficients toward zero (bias). The relaxed step removes that bias while retaining LASSO's variable selection.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

# Step 1: LASSO with 10-fold CV
lasso_pipe = make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=0))
lasso_pipe.fit(X, y)

lasso_coefs = lasso_pipe.named_steps["lassocv"].coef_
survivor_names = [name for name, c in zip(feature_names, lasso_coefs) if c != 0]

# Step 2: OLS refit on survivors
X_surv = sm.add_constant(df[survivor_names])
relaxed_model = sm.OLS(y, X_surv).fit()
print(relaxed_model.summary())
```

### Elastic Net

**Objective:**
$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\|y - X\beta\|^2 + \lambda\left[\alpha\|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|^2_2\right]$$

- $\alpha = 1$: pure LASSO; $\alpha = 0$: pure Ridge.  
- $\alpha = 0.7$ (as in this assignment): 70% LASSO + 30% Ridge.  
- **Advantages over LASSO:** handles groups of correlated predictors better (LASSO picks one arbitrarily; Elastic Net tends to include or exclude the whole group together).

```python
from sklearn.linear_model import ElasticNetCV

# alpha = l1_ratio in sklearn; cv=15-fold
en_pipe = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=0.7, cv=15, random_state=0, max_iter=10_000)
)
en_pipe.fit(X, y)
best_lambda = en_pipe.named_steps["elasticnetcv"].alpha_
en_coefs    = en_pipe.named_steps["elasticnetcv"].coef_
```

### Tuning λ via Cross-Validation

`LassoCV` and `ElasticNetCV` automatically search a grid of $\lambda$ values (from $\lambda_{max}$ down by a factor `eps` to $\lambda_{min}$) and select the one with the lowest CV loss.

```python
lasso_cv = lasso_pipe.named_steps["lassocv"]
print(f"λ range: [{lasso_cv.alphas_.min():.5f}, {lasso_cv.alphas_.max():.5f}]")
print(f"Chosen λ: {lasso_cv.alpha_:.5f}")
```

### Always Standardise Before Regularisation

The L1/L2 penalties treat all coefficients symmetrically. A predictor measured in large units (e.g., income in ILS vs thousands ILS) will have a smaller coefficient — the penalty unfairly shrinks it less. **Standardise to zero mean, unit variance before regularising.**  
Using `StandardScaler` inside a `Pipeline` prevents data leakage (the scaler is fit only on training folds).

### Bias-Variance Trade-off

| Method | Bias | Variance | Sparsity |
|--------|------|----------|---------|
| OLS | 0 | High (large $p$) | No |
| Ridge | Low-Medium | Lower | No |
| LASSO | Medium | Lower | Yes |
| Elastic Net | Low-Medium | Lower | Yes |

**Common pitfalls:**
- Comparing LASSO coefficients across different $\lambda$ values without re-standardising.
- Forgetting the relaxed step (reporting LASSO coefficients as final estimates — they are biased downward).
- Using `lasso_path` without standardisation.

---

## 6. Bootstrap (Nonparametric)

### The Bootstrap Principle

The empirical distribution $\hat{F}_n$ (the distribution that puts weight $1/n$ on each observed data point) is our best approximation to the true unknown distribution $F$. Resampling from $\hat{F}_n$ mimics sampling from $F$.

**Nonparametric bootstrap algorithm:**
1. Draw $B$ resamples of size $n$ with replacement from the data.
2. Compute the statistic of interest on each resample → $\hat{\theta}^{*1}, \ldots, \hat{\theta}^{*B}$.
3. Use the distribution of $\{\hat{\theta}^{*b}\}$ to approximate the sampling distribution of $\hat{\theta}$.

### Confidence Intervals

**Percentile CI:**
$$[\hat{\theta}^*_{(\alpha/2)},\; \hat{\theta}^*_{(1-\alpha/2)}]$$
Take the $\alpha/2$ and $1-\alpha/2$ quantiles of the bootstrap distribution.  
Simple, widely used, works well when the sampling distribution is roughly symmetric.

**Basic (Reflected) CI:**
$$[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)},\; 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}]$$
Corrects for bias; performs better when the distribution is skewed.

**Analytical (Wald) CI:**
$$\hat{\theta} \pm z_{\alpha/2} \cdot \text{SE}(\hat{\theta})$$
Assumes normality of the sampling distribution. Valid in large samples; can break for complex statistics.

### When Bootstrap is Better Than Analytical CIs

- The sampling distribution of $\hat{\theta}$ is **not normal** (e.g., for ratios, products, maximum likelihood estimators with bounded support).
- **Small samples** where the central limit theorem has not kicked in.
- **Complex estimators** (LASSO, Elastic Net) where there is no closed-form variance formula.

### Code: Bootstrap for Regression Coefficients

```python
np.random.seed(42)
B = 1000
bootstrap_coefs = []

for b in range(B):
    sample = df.sample(n=len(df), replace=True, random_state=b)
    boot_model = smf.logit("SURVIVED ~ C(PCLASS) + SEX + AGE + SEX:AGE",
                            data=sample).fit(disp=False)
    bootstrap_coefs.append(boot_model.params.values)

boot_df = pd.DataFrame(bootstrap_coefs, columns=coef_names)

# Percentile CI
boot_ci_lower = boot_df.quantile(0.025)
boot_ci_upper = boot_df.quantile(0.975)
```

### Bootstrap for Elastic Net

```python
en_boot_coefs = []
for b in range(B):
    idx     = np.random.choice(len(X), size=len(X), replace=True)
    pipe_b  = make_pipeline(StandardScaler(),
                             ElasticNetCV(l1_ratio=0.7, cv=15, random_state=b))
    pipe_b.fit(X[idx], y[idx])
    en_boot_coefs.append(pipe_b.named_steps["elasticnetcv"].coef_)

en_boot_df = pd.DataFrame(en_boot_coefs, columns=feature_names)
```

**Common pitfalls:**
- Bootstrapping when there is **dependence** in the data (time series) — the iid bootstrap breaks; use block bootstrap instead.
- Small $B$ (use at least $B = 1000$; for BCa use $B \geq 5000$).
- For LASSO/EN bootstrap: different resamples may produce different sparsity patterns, leading to non-zero coefficients for some resamples that were zero in the original fit — handle this by focusing on originally non-zero coefficients.

---

## 7. Cross-Validation

### k-Fold CV

Split data into $k$ equal folds. For each fold, train on the other $k-1$ folds and evaluate on the held-out fold. Average the scores.

$$CV_{(k)} = \frac{1}{k}\sum_{j=1}^k \text{MSE}_j$$

**Typical choice:** $k = 5$ or $k = 10$.

### LOOCV (Leave-One-Out)

Special case of $k$-fold CV with $k = n$.  

- **Pro:** lowest bias (train on nearly all data each time).  
- **Con:** high variance (each model is trained on nearly the same data); expensive for large $n$ (unless using the PRESS shortcut for OLS).

### Bias-Variance of Different CV Schemes

| Scheme | Bias | Variance | Cost |
|--------|------|----------|------|
| LOOCV | Low | High | $O(n)$ model fits |
| 10-fold CV | Moderate | Lower | 10 model fits |
| 5-fold CV | Moderate+ | Lower+ | 5 model fits |

### Nested CV (When to Use)

If you are **both selecting a model and estimating its error**, use nested CV:
- **Outer loop:** $k$-fold CV to estimate prediction error.
- **Inner loop:** $k'$-fold CV to select the model (e.g., tune $\lambda$).

Using the same CV loop for both selection and evaluation leads to **optimistic bias**.

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression

loo    = LeaveOneOut()
scores = cross_val_score(LinearRegression(), X, y, cv=loo,
                          scoring="neg_mean_squared_error")
loocv_mse = -scores.mean()
```

---

## 8. Nonlinear Transformations

### Concave vs Convex

| Transform | Shape | Use case | Example |
|-----------|-------|---------|---------|
| $\log(x)$ | Concave (diminishing returns) | Income, population | Wage vs income, psychometric vs family income |
| $\sqrt{x}$ | Concave | Milder than log | Same; less extreme saturation |
| $x^2$ | Convex | Accelerating growth | Quadratic trends |
| $e^x$ | Convex | Explosive growth | Viral spread |

**Decision rule for income:** If the scatter plot of $y$ vs $\log(\text{income})$ looks more linear than $y$ vs income, the concave transform is appropriate.

**Psychometric grades vs income:** The relationship saturates at high income — a child from a family earning 50k ILS does not score much higher than one from 30k ILS. Therefore **concave ($\log$, $\sqrt$) is more plausible**.

### Effect on Interpretation

After fitting $y = \beta_0 + \beta_1 \log(x) + \epsilon$:  
A 1% increase in $x$ corresponds to an increase of $\beta_1 / 100$ units in $y$ (semi-elasticity).

After $y = \beta_0 + \beta_1 \sqrt{x} + \epsilon$:  
The marginal effect of $x$ is $\beta_1 / (2\sqrt{x})$ — decreasing as $x$ increases.

### Comparing Models with Transformations

Once you transform $y$ (e.g., $\log y$), you cannot directly compare $R^2$ or AIC with a model on untransformed $y$. You must compute the criterion on the **same outcome scale**.

**Common pitfalls:**
- Applying $\log$ to zero or negative values — check `df['income'].min() > 0` before taking logs.
- Forgetting to add the transformed columns to the dataframe before running stepwise/LASSO.

---

## 9. Nonparametric Statistics

### Discrete Uniform Distribution $X \sim DU[1, N]$

$$E[X] = \frac{N+1}{2}, \qquad \text{Var}(X) = \frac{N^2-1}{12}$$

**Derivation of Var(X):**

$$E[X^2] = \frac{1}{N}\sum_{k=1}^N k^2 = \frac{(N+1)(2N+1)}{6}$$
$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{(N+1)(2N+1)}{6} - \frac{(N+1)^2}{4} = \frac{N^2-1}{12}$$

### Rank Sum Identity

With $N = n_1 + n_2$ total observations, the sum of all ranks is:
$$R_1 + R_2 = \sum_{k=1}^N k = \frac{N(N+1)}{2}$$

### Expected Rank Sums Under $H_0$ (Wilcoxon)

Under the null hypothesis of no group difference, all $\binom{N}{n_1}$ rank assignments are equally likely. Each rank $k$ has probability $n_g/N$ of belonging to group $g$:

$$E[R_g] = \frac{n_g(N+1)}{2}, \qquad g = 1, 2$$

**Cross-check:** $E[R_1] + E[R_2] = \frac{(n_1+n_2)(N+1)}{2} = \frac{N(N+1)}{2}$ ✓

### Wilcoxon Rank-Sum Test (Mann-Whitney U)

Used when the normality assumption is violated or the data is ordinal.  
$H_0$: the two groups come from the same distribution (equal medians).  
Test statistic: $W = R_1 - \frac{n_1(N+1)}{2}$ (deviation of $R_1$ from its expected value under $H_0$).  

Under $H_0$, for large $n_1, n_2$:
$$Z = \frac{W}{\sqrt{n_1 n_2 (N+1)/12}} \approx N(0,1)$$

```python
from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
```

**Common pitfalls:**
- Confusing $W$ (Wilcoxon statistic) with $U$ (Mann-Whitney U) — they are linearly related: $U = W - n_1(n_1+1)/2$.
- Using the rank-sum test when data are paired — use Wilcoxon signed-rank instead.

---

## 10. Quick-Reference Formula Sheet

### OLS
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}, \quad E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}, \quad \text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$$

### Simple OLS ($y = \beta x + \epsilon$, no intercept)
$$\hat{\beta}_{OLS} = \frac{\sum x_i y_i}{\sum x_i^2}, \quad \text{Var}(\hat{\beta}) = \frac{\sigma^2}{\sum x_i^2}$$

### Ridge ($y = \beta x + \epsilon$)
$$\hat{\beta}_{Ridge} = \frac{\sum x_i y_i}{\sum x_i^2 + \lambda}, \quad \text{Bias} = -\frac{\lambda\beta}{\sum x_i^2 + \lambda}, \quad \text{Var} = \frac{\sigma^2 \sum x_i^2}{(\sum x_i^2 + \lambda)^2}$$

### LASSO ($y = \beta x + \epsilon$)
$$\hat{\beta}_{LASSO} = \text{sign}(\hat{\beta}_{OLS}) \cdot \max\left(|\hat{\beta}_{OLS}| - \frac{\lambda}{2\sum x_i^2},\; 0\right)$$

### Goodness of Fit
$$R^2 = 1 - \frac{RSS}{TSS}, \quad \bar{R}^2 = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$$

### AIC (General)
$$AIC = -2\log(\hat{L}) + 2p$$

### AIC (Gaussian Linear Model)
$$AIC = n\log(\hat{\sigma}^2_{MLE}) + C_{n,p}, \quad \hat{\sigma}^2_{MLE} = \frac{RSS}{n}, \quad C_{n,p} = n\log(2\pi) + n + 2(p+2)$$

### F-tests
$$F_{overall} = \frac{(TSS - RSS)/p}{RSS/(n-p-1)} \sim F_{p,n-p-1}$$
$$F_{partial} = \frac{(RSS_{red} - RSS_{full})/q}{RSS_{full}/(n-p_{full}-1)} \sim F_{q,n-p_{full}-1}$$

### Elastic Net Objective
$$\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \lambda\left[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|^2_2\right]$$

### Bootstrap Percentile CI
$$[\hat{\theta}^*_{(\alpha/2)},\; \hat{\theta}^*_{(1-\alpha/2)}]$$

### Discrete Uniform $DU[1,N]$
$$E[X] = \frac{N+1}{2}, \quad \text{Var}(X) = \frac{N^2-1}{12}$$

### Rank Sum Identity
$$R_1 + R_2 = \frac{N(N+1)}{2}, \quad E[R_g] = \frac{n_g(N+1)}{2} \text{ under } H_0$$

---

*End of Study Guide — good luck on the exam!*
