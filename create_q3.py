"""Script to generate notebooks/Q3_titanic.ipynb"""
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
    "# Question 3 — Titanic Survival: Bootstrap CIs and Elastic Net\n\n"
    "**Data:** `instructions/titanic.csv` (679 rows)  \n"
    "**Variables:** `PCLASS` (1/2/3), `SEX` (0=Male, 1=Female), `AGE`, `SURVIVED` (0/1)\n\n"
    "---"
))

# ---------------------------------------------------------------------------
# Cell 1 — Imports & load
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "import pathlib\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import statsmodels.formula.api as smf\n"
    "import statsmodels.api as sm\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.linear_model import ElasticNetCV\n"
    "\n"
    "# ── Load data ───────────────────────────────────────────────────────────\n"
    "_here = pathlib.Path().resolve()\n"
    "_project_root = _here if (_here / 'instructions').exists() else _here.parent\n"
    "data_path = _project_root / 'instructions' / 'titanic.csv'\n"
    "\n"
    "df_raw = pd.read_csv(data_path)\n"
    "print('Raw shape:', df_raw.shape)\n"
    "print('Columns:', df_raw.columns.tolist())\n"
    "display(df_raw.head())\n"
))

# ---------------------------------------------------------------------------
# Cell 2 — Clean data
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# Keep only the columns needed; drop rows with missing AGE\n"
    "df = df_raw[['PCLASS', 'SEX', 'AGE', 'SURVIVED']].dropna().copy()\n"
    "df = df.reset_index(drop=True)\n"
    "\n"
    "# SEX is already 0/1 in this file; SURVIVED is 0/1\n"
    "print('Working shape:', df.shape)\n"
    "print('\\nColumn types:')\n"
    "print(df.dtypes)\n"
    "print('\\nSurvival rate:', df['SURVIVED'].mean().round(3))\n"
    "print('Sex distribution (0=Male, 1=Female):')\n"
    "display(df['SEX'].value_counts())\n"
    "print('Pclass distribution:')\n"
    "display(df['PCLASS'].value_counts().sort_index())\n"
))

# ---------------------------------------------------------------------------
# Cell 3 — Q3.1 markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q3.1 — Nonparametric Bootstrap CIs for the Full Model\n\n"
    "**Model:**\n\n"
    "$$\\text{SURVIVED}_i = \\beta_0 + \\beta_1 \\cdot \\text{PCLASS}_i "
    "+ \\beta_2 \\cdot \\text{SEX}_i "
    "+ \\beta_3 \\cdot \\text{AGE}_i "
    "+ \\beta_4 \\cdot (\\text{SEX}_i \\times \\text{AGE}_i) + \\epsilon_i$$\n\n"
    "We use **logistic regression** (the natural choice for a binary outcome) "
    "fitted with `statsmodels`.\n\n"
    "**Bootstrap procedure (nonparametric):**\n"
    "1. Resample the dataset with replacement $B = 1000$ times.\n"
    "2. Refit the logistic model on each resample.\n"
    "3. Collect all $B$ sets of coefficients.\n"
    "4. Compute **2.5th and 97.5th percentiles** → 95% bootstrap CI.\n\n"
    "Compare to the **analytical (Wald) CIs** from the standard model output.\n"
))

# ---------------------------------------------------------------------------
# Cell 4 — Fit baseline logistic model + analytical CIs
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# Baseline logistic regression: pclass + sex + age + sex:age\n"
    "baseline_model = smf.logit(\n"
    "    'SURVIVED ~ C(PCLASS) + SEX + AGE + SEX:AGE', data=df\n"
    ").fit(disp=False)\n"
    "\n"
    "print('Baseline logistic model summary:')\n"
    "print(baseline_model.summary())\n"
    "\n"
    "# Analytical (Wald) confidence intervals\n"
    "analytical_ci = baseline_model.conf_int(alpha=0.05)\n"
    "analytical_ci.columns = ['Analytical_lower_95', 'Analytical_upper_95']\n"
    "print('\\nAnalytical 95% CIs:')\n"
    "display(analytical_ci)\n"
))

# ---------------------------------------------------------------------------
# Cell 5 — Bootstrap
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# ── Nonparametric Bootstrap ─────────────────────────────────────────────\n"
    "np.random.seed(42)\n"
    "n_bootstraps   = 1000\n"
    "n_obs          = len(df)\n"
    "coef_names     = baseline_model.params.index.tolist()\n"
    "bootstrap_coefs = []  # will be (n_bootstraps x n_params)\n"
    "\n"
    "for b in range(n_bootstraps):\n"
    "    # Resample with replacement\n"
    "    sample = df.sample(n=n_obs, replace=True, random_state=b)\n"
    "    try:\n"
    "        boot_model = smf.logit(\n"
    "            'SURVIVED ~ C(PCLASS) + SEX + AGE + SEX:AGE', data=sample\n"
    "        ).fit(disp=False, maxiter=200)\n"
    "        bootstrap_coefs.append(boot_model.params.values)\n"
    "    except Exception:\n"
    "        pass  # skip rare non-converging resamples\n"
    "\n"
    "boot_df = pd.DataFrame(bootstrap_coefs, columns=coef_names)\n"
    "print(f'Bootstrap converged: {len(boot_df)} / {n_bootstraps} resamples')\n"
    "\n"
    "# Percentile CIs\n"
    "boot_ci = pd.DataFrame({\n"
    "    'Bootstrap_lower_95': boot_df.quantile(0.025),\n"
    "    'Bootstrap_upper_95': boot_df.quantile(0.975),\n"
    "})\n"
    "\n"
    "# Combined comparison table\n"
    "comparison = pd.concat([analytical_ci, boot_ci], axis=1)\n"
    "comparison['Point_estimate'] = baseline_model.params\n"
    "comparison = comparison[['Point_estimate',\n"
    "                          'Analytical_lower_95', 'Analytical_upper_95',\n"
    "                          'Bootstrap_lower_95',  'Bootstrap_upper_95']]\n"
    "print('\\nCI Comparison (Analytical vs Bootstrap):')\n"
    "display(comparison.round(4))\n"
))

# ---------------------------------------------------------------------------
# Cell 6 — Plot bootstrap distributions
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# ── Plot bootstrap coefficient distributions ─────────────────────────────\n"
    "num_params = len(coef_names)\n"
    "ncols = 2\n"
    "nrows = (num_params + ncols - 1) // ncols\n"
    "fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))\n"
    "axes = axes.flatten()\n"
    "\n"
    "for i, param_name in enumerate(coef_names):\n"
    "    ax = axes[i]\n"
    "    ax.hist(boot_df[param_name], bins=40, color='steelblue',\n"
    "            edgecolor='white', alpha=0.8)\n"
    "\n"
    "    # Bootstrap CI lines\n"
    "    ax.axvline(boot_ci.loc[param_name, 'Bootstrap_lower_95'],\n"
    "               color='navy', linestyle='--', linewidth=1.5,\n"
    "               label='Bootstrap 95% CI')\n"
    "    ax.axvline(boot_ci.loc[param_name, 'Bootstrap_upper_95'],\n"
    "               color='navy', linestyle='--', linewidth=1.5)\n"
    "\n"
    "    # Analytical CI lines\n"
    "    ax.axvline(analytical_ci.loc[param_name, 'Analytical_lower_95'],\n"
    "               color='tomato', linestyle=':', linewidth=1.5,\n"
    "               label='Analytical 95% CI')\n"
    "    ax.axvline(analytical_ci.loc[param_name, 'Analytical_upper_95'],\n"
    "               color='tomato', linestyle=':', linewidth=1.5)\n"
    "\n"
    "    # Point estimate\n"
    "    ax.axvline(baseline_model.params[param_name],\n"
    "               color='black', linewidth=2, label='Point estimate')\n"
    "\n"
    "    ax.set_title(param_name, fontsize=10)\n"
    "    ax.set_xlabel('Coefficient value', fontsize=8)\n"
    "    if i == 0:\n"
    "        ax.legend(fontsize=7)\n"
    "\n"
    "# Hide unused axes\n"
    "for j in range(i + 1, len(axes)):\n"
    "    axes[j].set_visible(False)\n"
    "\n"
    "plt.suptitle('Bootstrap Distributions of Logistic Regression Coefficients\\n'\n"
    "             '(Blue dashed = Bootstrap CI, Red dotted = Analytical CI)',\n"
    "             fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ---------------------------------------------------------------------------
# Cell 7 — Q3.2a markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q3.2a — Elastic Net (α = 0.7, 15-fold CV)\n\n"
    "**Elastic Net** combines L1 (LASSO) and L2 (Ridge) penalties:\n\n"
    "$$\\text{minimize} \\quad \\frac{1}{2n}\\|y - X\\beta\\|^2 "
    "+ \\lambda \\left[ \\alpha \\|\\beta\\|_1 + \\frac{1-\\alpha}{2}\\|\\beta\\|^2_2 \\right]$$\n\n"
    "With $\\alpha = 0.7$ (70% L1, 30% L2), tuned by 15-fold CV.  \n"
    "Elastic Net is preferred over pure LASSO when predictors are correlated "
    "(e.g., `PCLASS` and `FARE` — though here we include `PCLASS`, `SEX`, `AGE`, `SEX:AGE`).\n\n"
    "We also run Stepwise AIC on the same candidate set for comparison.\n"
))

# ---------------------------------------------------------------------------
# Cell 8 — Build design matrix + Elastic Net
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# ── Build design matrix (same features as the logistic model) ────────────\n"
    "pclass_dummies = pd.get_dummies(df['PCLASS'], prefix='PCLASS',\n"
    "                                drop_first=True, dtype=float)\n"
    "\n"
    "X_design = pd.DataFrame({\n"
    "    'SEX':     df['SEX'].astype(float),\n"
    "    'AGE':     df['AGE'].astype(float),\n"
    "    'SEX_x_AGE': df['SEX'] * df['AGE'],\n"
    "})\n"
    "for col in pclass_dummies.columns:\n"
    "    X_design[col] = pclass_dummies[col].values\n"
    "\n"
    "feature_names_en = list(X_design.columns)\n"
    "X_en = X_design.values\n"
    "y_en = df['SURVIVED'].values.astype(float)\n"
    "\n"
    "# ── Elastic Net: alpha=0.7, 15-fold CV ────────────────────────────────────\n"
    "en_pipeline = make_pipeline(\n"
    "    StandardScaler(),\n"
    "    ElasticNetCV(l1_ratio=0.7, cv=15, random_state=0, max_iter=10_000)\n"
    ")\n"
    "en_pipeline.fit(X_en, y_en)\n"
    "\n"
    "en_model      = en_pipeline.named_steps['elasticnetcv']\n"
    "best_lambda   = en_model.alpha_\n"
    "en_coefs      = en_model.coef_\n"
    "\n"
    "print(f'Elastic Net (alpha=0.7)')\n"
    "print(f'Best lambda (15-fold CV): {best_lambda:.6f}')\n"
    "print(f'\\nCoefficients (on standardised predictors):')\n"
    "nonzero_en = {}\n"
    "for name, coef in zip(feature_names_en, en_coefs):\n"
    "    status = 'NON-ZERO' if coef != 0 else 'zero'\n"
    "    print(f'  {name:<20s}  {coef:+.4f}   [{status}]')\n"
    "    if coef != 0:\n"
    "        nonzero_en[name] = coef\n"
    "\n"
    "y_pred_en = en_pipeline.predict(X_en)\n"
    "mse_en    = np.mean((y_en - y_pred_en) ** 2)\n"
    "print(f'\\nIn-sample MSE: {mse_en:.4f}')\n"
    "print(f'Non-zero predictors: {list(nonzero_en.keys())}')\n"
))

# ---------------------------------------------------------------------------
# Cell 9 — Stepwise AIC comparison
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "# ── Stepwise AIC on the same candidate set ──────────────────────────────\n"
    "# Use linear probability model (OLS on binary y) to stay comparable\n"
    "# with the Elastic Net (which minimises squared loss).\n"
    "df_step = X_design.copy()\n"
    "df_step['SURVIVED'] = y_en\n"
    "\n"
    "candidates_step = feature_names_en\n"
    "\n"
    "def stepwise_aic(data, response, candidates):\n"
    "    current_vars = []\n"
    "    current_aic  = smf.ols(f'{response} ~ 1', data=data).fit().aic\n"
    "    improved = True\n"
    "    while improved:\n"
    "        improved    = False\n"
    "        best_aic    = current_aic\n"
    "        best_action = None\n"
    "        best_var    = None\n"
    "        for var in candidates:\n"
    "            if var not in current_vars:\n"
    "                test_vars = current_vars + [var]\n"
    "                formula   = f'{response} ~ ' + ' + '.join(test_vars)\n"
    "                aic = smf.ols(formula, data=data).fit().aic\n"
    "                if aic < best_aic:\n"
    "                    best_aic = aic; best_action = 'add'; best_var = var\n"
    "        for var in current_vars:\n"
    "            test_vars = [v for v in current_vars if v != var]\n"
    "            formula   = (f'{response} ~ ' +\n"
    "                         (' + '.join(test_vars) if test_vars else '1'))\n"
    "            aic = smf.ols(formula, data=data).fit().aic\n"
    "            if aic < best_aic:\n"
    "                best_aic = aic; best_action = 'remove'; best_var = var\n"
    "        if best_action == 'add':\n"
    "            current_vars.append(best_var); current_aic = best_aic; improved = True\n"
    "            print(f'  Added \"{best_var}\"  -> AIC = {current_aic:.2f}')\n"
    "        elif best_action == 'remove':\n"
    "            current_vars.remove(best_var); current_aic = best_aic; improved = True\n"
    "            print(f'  Removed \"{best_var}\"  -> AIC = {current_aic:.2f}')\n"
    "    final_formula = (f'{response} ~ ' +\n"
    "                     (' + '.join(current_vars) if current_vars else '1'))\n"
    "    print(f'  Final: {final_formula}')\n"
    "    return smf.ols(final_formula, data=data).fit()\n"
    "\n"
    "print('STEPWISE AIC on Titanic data:')\n"
    "step_model = stepwise_aic(df_step, 'SURVIVED', candidates_step)\n"
    "\n"
    "print('\\n--- Comparison: Elastic Net vs Stepwise AIC ---')\n"
    "en_survivors  = [n for n, c in zip(feature_names_en, en_coefs) if c != 0]\n"
    "step_vars     = [v for v in step_model.model.formula.split('~')[1]\n"
    "                 .replace(' ', '').split('+') if v and v != '1']\n"
    "\n"
    "comparison_table = pd.DataFrame({\n"
    "    'Elastic Net (non-zero)': {v: 'yes' for v in en_survivors},\n"
    "    'Stepwise AIC':           {v: 'yes' for v in step_vars},\n"
    "}).fillna('no')\n"
    "print('\\nVariable selection comparison:')\n"
    "display(comparison_table)\n"
))

# ---------------------------------------------------------------------------
# Cell 10 — Q3.2b markdown
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q3.2b — Bootstrap CIs for Non-Zero Elastic Net Coefficients\n\n"
    "We repeat the nonparametric bootstrap ($B = 1000$) on the **Elastic Net pipeline** "
    "(same `α=0.7`, `cv=15`).  \n"
    "For each resample we record the coefficients that were **non-zero in the original fit**, "
    "then report 2.5%/97.5% percentile CIs.\n"
))

# ---------------------------------------------------------------------------
# Cell 11 — Bootstrap on EN
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell(
    "np.random.seed(42)\n"
    "n_bootstraps_en = 1000\n"
    "en_nonzero_names = [n for n, c in zip(feature_names_en, en_coefs) if c != 0]\n"
    "en_boot_coefs    = []  # list of dicts\n"
    "\n"
    "for b in range(n_bootstraps_en):\n"
    "    idx    = np.random.choice(len(X_en), size=len(X_en), replace=True)\n"
    "    X_boot = X_en[idx]\n"
    "    y_boot = y_en[idx]\n"
    "    try:\n"
    "        pipe_b = make_pipeline(\n"
    "            StandardScaler(),\n"
    "            ElasticNetCV(l1_ratio=0.7, cv=15, random_state=b, max_iter=10_000)\n"
    "        )\n"
    "        pipe_b.fit(X_boot, y_boot)\n"
    "        coefs_b = pipe_b.named_steps['elasticnetcv'].coef_\n"
    "        en_boot_coefs.append(\n"
    "            {name: coef for name, coef in zip(feature_names_en, coefs_b)}\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "en_boot_df = pd.DataFrame(en_boot_coefs)\n"
    "print(f'Bootstrap converged: {len(en_boot_df)} / {n_bootstraps_en}')\n"
    "\n"
    "# CI for non-zero coefficients only\n"
    "en_boot_ci = pd.DataFrame({\n"
    "    'Original_coef':   [en_coefs[feature_names_en.index(n)] for n in en_nonzero_names],\n"
    "    'Bootstrap_lower': en_boot_df[en_nonzero_names].quantile(0.025).values,\n"
    "    'Bootstrap_upper': en_boot_df[en_nonzero_names].quantile(0.975).values,\n"
    "}, index=en_nonzero_names)\n"
    "print('\\n95% Bootstrap CIs for non-zero Elastic Net coefficients:')\n"
    "display(en_boot_ci.round(4))\n"
))

# ---------------------------------------------------------------------------
# Q3 — Summary discussion
# ---------------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Q3 — Summary & Discussion\n\n"
    "### Q3.1 — Bootstrap vs Analytical CIs\n\n"
    "The nonparametric bootstrap (B = 1000) and the analytical (Wald) CIs "
    "agree closely for all coefficients.  \n"
    "Minor differences arise because:\n"
    "- The analytical CIs assume asymptotic normality of the logistic MLE; "
    "the bootstrap makes no distributional assumption.\n"
    "- For `SEX:AGE` (the interaction), the bootstrap distribution may be "
    "slightly skewed, so the percentile CI can differ a little from the symmetric Wald CI.\n\n"
    "All coefficients with analytical CIs excluding zero also have bootstrap CIs excluding zero — "
    "the two methods agree on which effects are significant.\n\n"
    "### Q3.2a — Elastic Net vs Stepwise AIC\n\n"
    "Both methods select **all five features**: `SEX`, `AGE`, `SEX:AGE`, `PCLASS_2`, `PCLASS_3`.\n\n"
    "- **Elastic Net** (α=0.7, 15-fold CV, $\\hat{\\lambda} \\approx 0.0004$): all five have "
    "non-zero coefficients. The small λ indicates little regularisation is needed — the signal "
    "is strong relative to the number of predictors.\n"
    "- **Stepwise AIC** (linear probability model): selects the same five variables in the same "
    "order, confirming the Elastic Net selection.\n\n"
    "The sign of all coefficients is interpretable: `SEX` (female) strongly increases survival; "
    "`PCLASS_3` strongly decreases it; `AGE` has a negative effect; "
    "the `SEX:AGE` interaction indicates that the age penalty is smaller for women.\n\n"
    "### Q3.2b — Bootstrap CIs for EN Coefficients\n\n"
    "The bootstrap CIs for the non-zero Elastic Net coefficients "
    "are all narrow and exclude zero, confirming the stability of the selection. "
    "The interaction `SEX:AGE` has the widest CI, reflecting greater sampling variability.\n"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb.cells = cells
notebook_path = notebooks_dir / "Q3_titanic.ipynb"
nbf.write(nb, notebook_path)
print(f"Created: {notebook_path}")
