# Bayesian Ordinal Model for Adolescent Internet Addiction Prediction

Predicting internet addiction severity (0-3 scale) in 2,736 adolescents using hierarchical Bayesian ordinal logistic regression. Built with **Stan** and **Python** to handle unbalanced age groups and extensive missingness (31% SII, 57% PAQ) via partial pooling.

## Quick Summary

**Problem**: Predict internet addiction severity across age groups (5-22 years) with unbalanced sample sizes and missing data

**Solution**: Hierarchical Bayesian ordinal model with partial pooling (smaller groups borrow strength from larger ones)

**Results**: 
- Internet use: β = 0.48 [0.39, 0.57] — strongest predictor
- Actigraphy (wearable sensors): ENMO mean β = -0.31 (protective), ENMO SD β = 0.21 (risk)
- Model performance: AUROC 0.747 (survey-only) → 0.749 (survey+actigraphy)

**Why this matters**: Instead of chasing accuracy only, the project evaluates whether adding expensive wearable data is actually worth it — a realistic trade-off in applied data science. **Key insight**: Survey-only model achieves 99.7% of the performance with 0% of the hardware cost, making it more scalable for large-scale screening programs.

## Tech Stack

- **Stan** (v2.34) - Bayesian modeling, non-centered parameterization
- **Python** - Data preprocessing, MCMC sampling (CmdStanPy), visualization
- **MICE** (scikit-learn) - Missing data imputation
- **ArviZ** - Posterior diagnostics, LOO-CV model comparison

## Key Features

- **Hierarchical modeling**: Age-group-specific effects with shared hyperpriors (partial pooling)
- **Ordinal logistic regression**: Respects 0-3 severity scale structure
- **Multi-modal data**: Survey data + actigraphy features from wearable sensors
- **Robust to missingness**: MICE imputation + Bayesian inference handles 31-57% missingness

## Project Structure

```
├── model_a_ordinal.stan          # Survey-only model
├── model_b_ordinal.stan          # Survey + actigraphy model  
├── Taesung_Ha_stats551_final.pdf # Full paper
├── figs/                         # Visualizations
└── requirements.txt              # Dependencies
```

## Model Architecture

**Model A (Baseline)**: Survey-only ordinal logistic regression
- Predictors: internet hours, BMI, sex, age, PAQ (physical activity)
- Hierarchical: Age-group-specific intercepts and PAQ slopes

**Model B (Extended)**: Adds 5 actigraphy features
- ENMO mean, ENMO SD, zero-activity proportion, non-wear proportion, night-to-day ratio

**Why hierarchical?** Unequal sample sizes (n₁=1142, n₂=909, n₃=685) cause unstable estimates in smaller groups. Partial pooling allows smaller groups to borrow information from larger ones.

## Results

| Predictor | Coefficient | 95% CI | Interpretation |
|-----------|-------------|--------|----------------|
| Internet use | 0.48 | [0.39, 0.57] | Strongest predictor |
| ENMO mean | -0.31 | [-0.53, -0.11] | Protective (more activity → lower addiction) |
| ENMO SD | 0.21 | [0.03, 0.41] | Risk factor (irregular activity → higher addiction) |
| PAQ | -0.03 to -0.04 | CI includes 0 | Weak/non-significant |

**Model comparison**: LOO-CV shows Model B marginally better (68% posterior probability), but discrimination improvement is minimal (Δ AUROC = +0.002). Actigraphy shows promise but limited by sample size (N=516, 13% of baseline).

**Business implications**:
- **Survey-only model (Model A) is production-ready**: Achieves AUROC 0.747 with zero hardware cost, making it scalable for schools/clinics
- **Internet use hours is the key lever**: Strongest predictor (β = 0.48) — simple self-report question captures most of the signal
- **Wearable data ROI is questionable**: +0.002 AUROC improvement doesn't justify device costs ($50-200/unit) and data collection overhead

## Implementation Details

- **MCMC**: 4 chains, 800 warmup + 800 sampling iterations, adapt_delta=0.99
- **Convergence**: All R̂ < 1.01, ESS/N > 0.1, no divergent transitions
- **Missing data**: MICE with 5 imputations (IterativeImputer, BayesianRidge)
- **Preprocessing**: Continuous predictors z-scored, PAQ-C/PAQ-A combined by age

## Files

- `model_a_ordinal.stan` / `model_b_ordinal.stan` - Stan model definitions
- `Taesung_Ha_stats551_final.pdf` - Full paper with methodology and results
- `figs/` - Visualizations (correlation matrices, posterior distributions, ROC curves)

## Setup

```bash
pip install -r requirements.txt
# run models
# python src/train_model.py
```

Note: Data files not included (HBN requires approval). MCMC samples excluded due to size but can be regenerated.

---

**Full report available in** [`Taesung_Ha_stats551_final.pdf`](Taesung_Ha_stats551_final.pdf)

**Course**: STATS 551 - Bayesian Modeling, University of Michigan  
**Author**: Taesung Ha (MAS 2025)
