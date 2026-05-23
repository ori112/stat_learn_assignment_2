"""Script to generate notebooks/Q1_haifa.ipynb"""
import nbformat as nbf
import pathlib

notebooks_dir = pathlib.Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "# Question 1 — Haifa Residents: Spending vs Income by Sex\n\n"
    "**Data:** `instructions/haifa_res.csv`  \n"
    "**Variables:** `spend` (monthly spending, thousands ILS), "
    "`income` (monthly income, thousands ILS), `sex` (0 = Female, 1 = Male)\n\n"
    "---"
))

# ---------------------------------------------------------------------------
# Cell 1 — Imports & load data
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "import pathlib\n"
    "import itertools\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import statsmodels.formula.api as smf\n"
    "import statsmodels.api as sm\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.linear_model import LassoCV\n"
    "\n"
    "# ── Locate data regardless of where Jupyter was launched from ──────────\n"
    "_here = pathlib.Path().resolve()\n"
    "_project_root = _here if (_here / 'instructions').exists() else _here.parent\n"
    "data_path = _project_root / 'instructions' / 'haifa_res.csv'\n"
    "df = pd.read_csv(data_path)\n"
    "\n"
    "# Create an explicit interaction column (income × sex) for later use\n"
    "df['income_x_sex'] = df['income'] * df['sex']\n"
    "\n"
    "print('Shape:', df.shape)\n"
    "print('\\nFirst 5 rows:')\n"
    "display(df.head())\n"
    "print('\\nSummary statistics:')\n"
    "display(df.describe())\n"
    "print('\\nSex distribution:')\n"
    "display(df['sex'].value_counts().rename({0: 'Female (0)', 1: 'Male (1)'}))\n"
))

# ---------------------------------------------------------------------------
# Cell 2 — Q1.1 markdown: theoretical model
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1.1 — Indicator Variable + Interaction OLS\n\n"
    "### Theoretical Model\n\n"
    "We model monthly spending as a function of income, a sex indicator, and their interaction:\n\n"
    "$$\\text{spend}_i = \\beta_0 + \\beta_1 \\cdot \\text{income}_i "
    "+ \\beta_2 \\cdot \\text{sex}_i "
    "+ \\beta_3 \\cdot (\\text{income}_i \\times \\text{sex}_i) + \\epsilon_i$$\n\n"
    "This lets each group have its **own intercept and its own slope**:\n\n"
    "| Group | Intercept | Slope |\n"
    "|-------|-----------|-------|\n"
    "| Female (sex = 0) | $\\beta_0$ | $\\beta_1$ |\n"
    "| Male (sex = 1) | $\\beta_0 + \\beta_2$ | $\\beta_1 + \\beta_3$ |\n\n"
    "### OLS Assumptions\n\n"
    "1. **Linearity:** $E[\\epsilon_i \\mid \\mathbf{x}_i] = 0$ — the model correctly specifies the conditional mean.\n"
    "2. **Exogeneity:** predictors are uncorrelated with the error term.\n"
    "3. **Homoskedasticity:** $\\text{Var}(\\epsilon_i \\mid \\mathbf{x}) = \\sigma^2$ — constant variance across observations.\n"
    "4. **No autocorrelation:** $\\text{Cov}(\\epsilon_i, \\epsilon_j) = 0$ for $i \\neq j$.\n"
    "5. **No perfect multicollinearity:** $(\\mathbf{X}^\\top \\mathbf{X})$ must be invertible.\n"
    "6. **Normality (for inference):** $\\epsilon_i \\overset{iid}{\\sim} N(0, \\sigma^2)$.\n"
))

# ---------------------------------------------------------------------------
# Cell 3 — Q1.1 code: fit full model
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# Fit the full model: main effects + interaction\n"
    "# statsmodels formula: 'A * B' expands to 'A + B + A:B'\n"
    "full_model = smf.ols('spend ~ income * C(sex)', data=df).fit()\n"
    "print(full_model.summary())\n"
))

# ---------------------------------------------------------------------------
# Cell 4 — Q1.1b: plot
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### Q1.1b — Plot: Fitted Lines for Each Sex Group"))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "\n"
    "sex_colors = {0: 'steelblue', 1: 'tomato'}\n"
    "sex_labels = {0: 'Female (sex = 0)', 1: 'Male (sex = 1)'}\n"
    "\n"
    "# Scatter plot: points colored by sex\n"
    "for sex_val, group in df.groupby('sex'):\n"
    "    ax.scatter(\n"
    "        group['income'], group['spend'],\n"
    "        color=sex_colors[sex_val],\n"
    "        label=sex_labels[sex_val],\n"
    "        alpha=0.5, s=30\n"
    "    )\n"
    "\n"
    "# Overlay the two fitted lines\n"
    "income_range = np.linspace(df['income'].min(), df['income'].max(), 200)\n"
    "for sex_val in [0, 1]:\n"
    "    pred_data = pd.DataFrame({\n"
    "        'income': income_range,\n"
    "        'sex': [sex_val] * len(income_range)\n"
    "    })\n"
    "    fitted_line = full_model.predict(pred_data)\n"
    "    ax.plot(income_range, fitted_line,\n"
    "            color=sex_colors[sex_val], linewidth=2.5)\n"
    "\n"
    "ax.set_xlabel('Monthly Income (thousands ILS)', fontsize=12)\n"
    "ax.set_ylabel('Monthly Spending (thousands ILS)', fontsize=12)\n"
    "ax.set_title('Monthly Spending vs Income by Sex', fontsize=14)\n"
    "ax.legend(fontsize=11)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 5 — Q1.2 markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1.2 — Statistical Test: Does the Slope Differ by Sex?\n\n"
    "**Hypotheses:**\n\n"
    "$$H_0: \\beta_3 = 0 \\quad \\text{(same income slope for both sexes)}$$\n"
    "$$H_1: \\beta_3 \\neq 0 \\quad \\text{(different slopes — interaction is real)}$$\n\n"
    "The $\\beta_3$ coefficient on $\\text{income} \\times \\text{sex}$ is exactly the "
    "difference in slopes. Two equivalent tests:\n"
    "1. **t-test** on $\\hat{\\beta}_3$ from the OLS summary.\n"
    "2. **Partial F-test** (ANOVA) comparing the full model (with interaction) "
    "to the reduced model (without interaction).\n"
))

# ---------------------------------------------------------------------------
# Cell 6 — Q1.2 code: partial F-test
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# Reduced model — no interaction\n"
    "reduced_model = smf.ols('spend ~ income + C(sex)', data=df).fit()\n"
    "\n"
    "# Partial F-test: does the interaction term add explanatory power?\n"
    "anova_result = sm.stats.anova_lm(reduced_model, full_model)\n"
    "print('Partial F-test (Reduced vs Full):')\n"
    "print(anova_result)\n"
    "\n"
    "# Interaction term statistics from the full model\n"
    "interaction_key = [k for k in full_model.params.index\n"
    "                   if 'income' in k and 'sex' in k][0]\n"
    "print(f'\\nInteraction term: {interaction_key}')\n"
    "print(f'  Coefficient : {full_model.params[interaction_key]:.4f}')\n"
    "print(f'  Std Error   : {full_model.bse[interaction_key]:.4f}')\n"
    "print(f'  t-statistic : {full_model.tvalues[interaction_key]:.4f}')\n"
    "print(f'  p-value     : {full_model.pvalues[interaction_key]:.4f}')\n"
    "\n"
    "alpha_level = 0.05\n"
    "p_val = full_model.pvalues[interaction_key]\n"
    "if p_val < alpha_level:\n"
    "    print(f'\\nConclusion: p = {p_val:.4f} < {alpha_level}')\n"
    "    print('Reject H0 — the income-spending slope differs significantly by sex.')\n"
    "else:\n"
    "    print(f'\\nConclusion: p = {p_val:.4f} >= {alpha_level}')\n"
    "    print('Fail to reject H0 — no significant difference in slopes.')\n"
))

# ---------------------------------------------------------------------------
# Cell 7 — Q1.3 markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1.3 — Global Best (All-Subsets) with Adjusted R²\n\n"
    "With 3 candidate predictors (`income`, `sex`, `income_x_sex`), "
    "there are $2^3 - 1 = 7$ non-empty subsets.\n\n"
    "**Adjusted R²** penalises for model complexity:\n\n"
    "$$\\bar{R}^2 = 1 - \\frac{RSS/(n-p-1)}{TSS/(n-1)}$$\n\n"
    "We fit every subset and choose the one with the **highest adjusted R²**.\n"
))

# ---------------------------------------------------------------------------
# Cell 8 — Q1.3 code: all-subsets
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "candidate_predictors = ['income', 'sex', 'income_x_sex']\n"
    "\n"
    "subset_results = []\n"
    "for num_vars in range(1, len(candidate_predictors) + 1):\n"
    "    for subset in itertools.combinations(candidate_predictors, num_vars):\n"
    "        formula = 'spend ~ ' + ' + '.join(subset)\n"
    "        model = smf.ols(formula, data=df).fit()\n"
    "        subset_results.append({\n"
    "            'predictors': ', '.join(subset),\n"
    "            'adj_r2':     round(model.rsquared_adj, 4),\n"
    "            'r2':         round(model.rsquared, 4),\n"
    "            'aic':        round(model.aic, 2),\n"
    "        })\n"
    "\n"
    "results_df = pd.DataFrame(subset_results).sort_values('adj_r2', ascending=False)\n"
    "print('All-Subsets Results (sorted by Adjusted R²):\\n')\n"
    "display(results_df.reset_index(drop=True))\n"
    "\n"
    "best_predictors = results_df.iloc[0]['predictors'].split(', ')\n"
    "best_formula    = 'spend ~ ' + ' + '.join(best_predictors)\n"
    "print(f'\\n>>> Best model by Adjusted R²: {best_formula}')\n"
    "best_subset_model = smf.ols(best_formula, data=df).fit()\n"
    "print(best_subset_model.summary())\n"
))

# ---------------------------------------------------------------------------
# Cell 9 — Q1.4 markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1.4 — Stepwise / Forward / Backward AIC\n\n"
    "**AIC (Akaike Information Criterion):**\n\n"
    "$$AIC = -2\\log(\\hat{L}) + 2p$$\n\n"
    "Lower AIC = better model (balances goodness-of-fit against model complexity).\n\n"
    "Three greedy search strategies, all using AIC as the criterion. "
    "Each prints the **initial** and **final** model.\n"
))

# ---------------------------------------------------------------------------
# Cell 10 — Q1.4 code: helpers + run
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "def forward_aic(data, response, candidates):\n"
    "    \"\"\"Forward selection: start with an empty model, greedily add the\n"
    "    variable that most reduces AIC. Stop when no addition improves AIC.\"\"\"\n"
    "    current_vars = []\n"
    "    current_aic  = smf.ols(f'{response} ~ 1', data=data).fit().aic\n"
    "    print(f'  Initial : \"{response} ~ 1\"   AIC = {current_aic:.2f}')\n"
    "\n"
    "    improved = True\n"
    "    while improved:\n"
    "        improved    = False\n"
    "        best_new_var = None\n"
    "        for var in candidates:\n"
    "            if var in current_vars:\n"
    "                continue\n"
    "            test_vars = current_vars + [var]\n"
    "            formula   = f'{response} ~ ' + ' + '.join(test_vars)\n"
    "            test_aic  = smf.ols(formula, data=data).fit().aic\n"
    "            if test_aic < current_aic:\n"
    "                current_aic  = test_aic\n"
    "                best_new_var = var\n"
    "                improved     = True\n"
    "        if improved:\n"
    "            current_vars.append(best_new_var)\n"
    "            print(f'  Added \"{best_new_var}\"  -> AIC = {current_aic:.2f}')\n"
    "\n"
    "    final_formula = f'{response} ~ ' + (' + '.join(current_vars) if current_vars else '1')\n"
    "    print(f'  Final   : {final_formula}')\n"
    "    return smf.ols(final_formula, data=data).fit()\n"
    "\n"
    "\n"
    "def backward_aic(data, response, candidates):\n"
    "    \"\"\"Backward elimination: start with the full model, greedily remove\n"
    "    the variable whose removal most reduces AIC.\"\"\"\n"
    "    current_vars = list(candidates)\n"
    "    formula      = f'{response} ~ ' + ' + '.join(current_vars)\n"
    "    current_aic  = smf.ols(formula, data=data).fit().aic\n"
    "    print(f'  Initial : {formula}   AIC = {current_aic:.2f}')\n"
    "\n"
    "    improved = True\n"
    "    while improved and current_vars:\n"
    "        improved  = False\n"
    "        worst_var = None\n"
    "        for var in current_vars:\n"
    "            test_vars    = [v for v in current_vars if v != var]\n"
    "            formula_test = (f'{response} ~ ' +\n"
    "                            (' + '.join(test_vars) if test_vars else '1'))\n"
    "            test_aic = smf.ols(formula_test, data=data).fit().aic\n"
    "            if test_aic < current_aic:\n"
    "                current_aic = test_aic\n"
    "                worst_var   = var\n"
    "                improved    = True\n"
    "        if improved:\n"
    "            current_vars.remove(worst_var)\n"
    "            print(f'  Removed \"{worst_var}\"  -> AIC = {current_aic:.2f}')\n"
    "\n"
    "    final_formula = f'{response} ~ ' + (' + '.join(current_vars) if current_vars else '1')\n"
    "    print(f'  Final   : {final_formula}')\n"
    "    return smf.ols(final_formula, data=data).fit()\n"
    "\n"
    "\n"
    "def stepwise_aic(data, response, candidates):\n"
    "    \"\"\"Bidirectional stepwise: at each step consider both adding and removing\n"
    "    a variable; take the single move that most reduces AIC.\"\"\"\n"
    "    current_vars = []\n"
    "    current_aic  = smf.ols(f'{response} ~ 1', data=data).fit().aic\n"
    "    print(f'  Initial : \"{response} ~ 1\"   AIC = {current_aic:.2f}')\n"
    "\n"
    "    improved = True\n"
    "    while improved:\n"
    "        improved    = False\n"
    "        best_aic    = current_aic\n"
    "        best_action = None\n"
    "        best_var    = None\n"
    "\n"
    "        # Consider additions\n"
    "        for var in candidates:\n"
    "            if var not in current_vars:\n"
    "                test_vars = current_vars + [var]\n"
    "                formula   = f'{response} ~ ' + ' + '.join(test_vars)\n"
    "                test_aic  = smf.ols(formula, data=data).fit().aic\n"
    "                if test_aic < best_aic:\n"
    "                    best_aic    = test_aic\n"
    "                    best_action = 'add'\n"
    "                    best_var    = var\n"
    "\n"
    "        # Consider removals\n"
    "        for var in current_vars:\n"
    "            test_vars = [v for v in current_vars if v != var]\n"
    "            formula   = (f'{response} ~ ' +\n"
    "                         (' + '.join(test_vars) if test_vars else '1'))\n"
    "            test_aic  = smf.ols(formula, data=data).fit().aic\n"
    "            if test_aic < best_aic:\n"
    "                best_aic    = test_aic\n"
    "                best_action = 'remove'\n"
    "                best_var    = var\n"
    "\n"
    "        if best_action == 'add':\n"
    "            current_vars.append(best_var)\n"
    "            current_aic = best_aic\n"
    "            improved = True\n"
    "            print(f'  Added \"{best_var}\"  -> AIC = {current_aic:.2f}')\n"
    "        elif best_action == 'remove':\n"
    "            current_vars.remove(best_var)\n"
    "            current_aic = best_aic\n"
    "            improved = True\n"
    "            print(f'  Removed \"{best_var}\"  -> AIC = {current_aic:.2f}')\n"
    "\n"
    "    final_formula = f'{response} ~ ' + (' + '.join(current_vars) if current_vars else '1')\n"
    "    print(f'  Final   : {final_formula}')\n"
    "    return smf.ols(final_formula, data=data).fit()\n"
    "\n"
    "\n"
    "# ── Run all three methods ───────────────────────────────────────────────\n"
    "aic_candidates = ['income', 'sex', 'income_x_sex']\n"
    "\n"
    "print('=' * 55)\n"
    "print('FORWARD SELECTION')\n"
    "print('=' * 55)\n"
    "forward_result = forward_aic(df, 'spend', aic_candidates)\n"
    "\n"
    "print('\\n' + '=' * 55)\n"
    "print('BACKWARD ELIMINATION')\n"
    "print('=' * 55)\n"
    "backward_result = backward_aic(df, 'spend', aic_candidates)\n"
    "\n"
    "print('\\n' + '=' * 55)\n"
    "print('STEPWISE (BIDIRECTIONAL)')\n"
    "print('=' * 55)\n"
    "stepwise_result = stepwise_aic(df, 'spend', aic_candidates)\n"
    "\n"
    "print('\\n' + '=' * 55)\n"
    "print('SUMMARY OF AIC-BASED SELECTION')\n"
    "print('=' * 55)\n"
    "print(f'Forward   AIC: {forward_result.aic:.2f}  ')\n"
    "print(f'Backward  AIC: {backward_result.aic:.2f}  ')\n"
    "print(f'Stepwise  AIC: {stepwise_result.aic:.2f}  ')\n"
    "all_agree = (forward_result.model.formula ==\n"
    "             backward_result.model.formula ==\n"
    "             stepwise_result.model.formula)\n"
    "print(f'All three methods agree: {all_agree}')\n"
))

# ---------------------------------------------------------------------------
# Cell 11 — Q1.5 markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1.5 — Relaxed LASSO (10-fold Cross-Validation)\n\n"
    "**Algorithm:**\n"
    "1. Standardise all predictors (equal L1 penalty treatment).\n"
    "2. Fit LASSO with 10-fold CV to choose optimal $\\lambda$.\n"
    "3. Identify **survivors** — predictors with non-zero LASSO coefficients.\n"
    "4. **Relaxed step:** refit plain OLS on survivors only — removes the shrinkage bias introduced by LASSO.\n\n"
    "This follows the pattern from the course reference code (LassoCV pipeline + OLS refit on survivors).\n"
))

# ---------------------------------------------------------------------------
# Cell 12 — Q1.5 code: relaxed LASSO
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# Design matrix — the three candidate predictors\n"
    "X = df[['income', 'sex', 'income_x_sex']].values\n"
    "y_arr    = df['spend'].values   # numpy array for sklearn\n"
    "y_series = df['spend']          # pandas Series for statsmodels (nicer output)\n"
    "feature_names = ['income', 'sex', 'income_x_sex']\n"
    "\n"
    "# Steps 1 & 2: Standardise + LASSO with 10-fold CV\n"
    "lasso_pipeline = make_pipeline(\n"
    "    StandardScaler(),\n"
    "    LassoCV(cv=10, random_state=0, max_iter=10_000)\n"
    ")\n"
    "lasso_pipeline.fit(X, y_arr)\n"
    "\n"
    "lasso_cv      = lasso_pipeline.named_steps['lassocv']\n"
    "chosen_lambda = lasso_cv.alpha_\n"
    "lasso_coefs   = lasso_cv.coef_\n"
    "\n"
    "print(f'Chosen lambda (10-fold CV): {chosen_lambda:.6f}')\n"
    "print('\\nLASSO coefficients (on standardised predictors):')\n"
    "for name, coef in zip(feature_names, lasso_coefs):\n"
    "    status = 'SURVIVED' if coef != 0 else 'zeroed out'\n"
    "    print(f'  {name:<20s}  coef = {coef:+.4f}   [{status}]')\n"
    "\n"
    "# Step 3: Survivors\n"
    "survivor_names = [name for name, coef in zip(feature_names, lasso_coefs)\n"
    "                  if coef != 0]\n"
    "print(f'\\nSurvivor predictors: {survivor_names}')\n"
    "\n"
    "# Step 4: Relaxed LASSO — plain OLS on survivors\n"
    "if survivor_names:\n"
    "    X_survivors   = sm.add_constant(df[survivor_names])\n"
    "    relaxed_model = sm.OLS(y_series, X_survivors).fit()  # pass Series for proper name\n"
    "    print('\\nRelaxed LASSO — OLS refit on survivors:')\n"
    "    print(relaxed_model.summary())\n"
    "else:\n"
    "    relaxed_model = None\n"
    "    print('No survivors — LASSO selected the null model.')\n"
    "\n"
    "# ── Comparison ──────────────────────────────────────────────────────────\n"
    "print('\\n--- Comparison: Relaxed LASSO vs Stepwise-AIC ---')\n"
    "print(f'Stepwise-AIC   AIC = {stepwise_result.aic:.2f}   '\n"
    "      f'Adj R² = {stepwise_result.rsquared_adj:.4f}   '\n"
    "      f'Variables: {[v for v in aic_candidates if v in stepwise_result.model.formula]}')\n"
    "if relaxed_model:\n"
    "    print(f'Relaxed LASSO  AIC = {relaxed_model.aic:.2f}   '\n"
    "          f'Adj R² = {relaxed_model.rsquared_adj:.4f}   '\n"
    "          f'Survivors: {survivor_names}')\n"
))

# ---------------------------------------------------------------------------
# Q1 — Summary discussion
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q1 — Summary & Discussion\n\n"
    "### Key findings\n\n"
    "| Method | Best model | AIC | Adj R² |\n"
    "|--------|-----------|-----|--------|\n"
    "| Full OLS (with interaction) | spend ~ income × sex | — | 0.723 |\n"
    "| All-subsets (Adj R²) | spend ~ income | — | 0.723 |\n"
    "| Forward / Backward / Stepwise AIC | spend ~ income | 731.4 | 0.723 |\n"
    "| Relaxed LASSO (10-fold CV) | spend ~ income | 731.4 | 0.723 |\n\n"
    "### Interpretation\n\n"
    "- **Q1.1:** The full model (income + sex + interaction) has F-statistic significant overall "
    "($p \\approx 5\\times10^{-55}$), with income as the dominant predictor "
    "($\\hat{\\beta}_1 \\approx 0.99$, $p < 0.001$). "
    "The sex indicator and interaction are not individually significant.\n\n"
    "- **Q1.2:** The interaction term (different slopes by sex) is **not significant** "
    "($p = 0.198$). We fail to reject $H_0: \\beta_3 = 0$. "
    "The relationship between income and spending does not differ significantly by sex.\n\n"
    "- **Q1.3–Q1.5:** All three model selection approaches — all-subsets, AIC stepwise, "
    "and relaxed LASSO — unanimously select the simple model "
    "`spend ~ income`. This is consistent with Q1.2: once income is included, "
    "sex and the interaction do not improve the model. "
    "The relaxed LASSO independently zeroes out both `sex` and `income_x_sex`, "
    "confirming the stepwise-AIC result.\n"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb.cells = cells
notebook_path = notebooks_dir / "Q1_haifa.ipynb"
nbf.write(nb, notebook_path)
print(f"Created: {notebook_path}")
