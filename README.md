# **ROS Method in Python**
### *Regression on Order Statistics (ROS) for Censored Data*

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Personal%20Use-lightgrey)


This repository contains a **Python implementation of the ROS method**, which is used for handling
**left-censored environmental data**, inspired by the **NADA package in R**.

---

## **Background**

ROS is a **parametric method** designed to estimate summary statistics (mean, standard deviation,
quantiles) for datasets that contain **non-detects** (values below a detection or reporting limit).
Rather than discarding non-detects or substituting them with arbitrary constants (e.g., LOQ/2),
ROS leverages the distributional structure of the detected values to extrapolate plausible values
for the censored observations.

The algorithm proceeds as follows:
1. Assign **Helsel-Cohn plotting positions** to all data points (both detected and non-detected),
   accounting for multiple detection limits if present.
2. Fit a **linear regression** of the transformed detected values against their normal quantiles
   (i.e., fit the data on a probability plot).
3. Use the fitted regression line to **predict transformed values** for the censored observations
   based on their plotting positions.
4. **Back-transform** censored predictions to the original scale.
5. Compute **summary statistics** using the combined set of detected and modeled censored values.

---

## **Files Included**

| File | Description |
|------|-------------|
| `ros_implementation.py` | Python script implementing the ROS method (v2.0) |
| `ros_notebook.ipynb` | Jupyter Notebook explaining the methodology & running examples |
| `MPCA_benz.csv` | Single-substance example dataset (Benzene concentration data) |
| `multi_substance_example.csv` | Multi-substance PFAS example dataset for `fit_dataframe()` demo |
| `environment.yml` | Conda environment file for setting up dependencies |

---

## **Installation & Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/Aliiivalipoor/ROS-Method-Python.git
   cd ROS-Method-Python
   ```

2. Set up the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ros_env
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter lab
   ```
   Then open **`ros_notebook.ipynb`** to explore the analysis.

---

## **How to Use the ROS Method**

### **Single Dataset**

```python
from ros_implementation import ROS
import pandas as pd

# Load dataset
df = pd.read_csv("MPCA_benz.csv")
obs = df["Benzene"]
censored = df["BenzCen"].astype(bool)

# Initialize and fit ROS model
ros = ROS(distribution="log-normal")
ros.fit(obs, censored)

# Print summary statistics
ros.summary()

# Generate Q-Q plot
ros.plot_qq()
```

### **Batch Processing: Applying ROS to a Multi-Substance DataFrame**

When your dataset contains multiple substances, use `fit_dataframe()` to apply ROS
to each substance independently in a single call:

```python
from ros_implementation import ROS
import pandas as pd

# Load multi-substance dataset
df = pd.read_csv("multi_substance_example.csv")

# Initialize ROS
ros = ROS(distribution="log-normal")

# Apply ROS to all substances
df_result = ros.fit_dataframe(
    df,
    concentration_col="Concentration",
    censored_col="Below LOQ",
    loq_col="LOQ",
    loq_replacement=True,   # Fall back to LOQ * loq_factor when ROS is not applicable
    loq_factor=0.5          # Use LOQ / 2 as the substitution value (default)
)

print(df_result.head())
```

The DataFrame must contain a `Substance` column. When a substance has fewer than
3 detected values, the method automatically falls back to `LOQ × loq_factor`
substitution (if `loq_replacement=True`).

---

## **Example Datasets**

### `MPCA_benz.csv`
Contains **Benzene concentration measurements** from the Minnesota Pollution Control Agency (MPCA),
where some values are censored (below detection limits).

| Column | Description |
|--------|-------------|
| `Benzene` | Measured benzene concentration |
| `BenzCen` | Boolean flag (`True` if censored) |

---

### `multi_substance_example.csv`

> ⚠️ **This dataset is entirely artificial and was created for demonstration purposes only.**
> The concentration values, detection frequencies, and LOQ values do **not** represent
> real environmental measurements and should not be used for any scientific or regulatory conclusions.

Contains **600 simulated water quality measurements** for six
**per- and polyfluoroalkyl substances (PFAS)** — a group of synthetic chemicals of significant
environmental and public health concern, commonly detected in drinking water, groundwater,
and surface water worldwide. Each substance has **100 samples**, with concentrations drawn
from lognormal distributions with substance-specific parameters.

Censoring was applied by designating the lowest measured values as non-detects (i.e., values
reported as "below the LOQ"), consistent with how real environmental non-detects arise in practice.
Detected values are guaranteed to exceed their respective LOQ.

The six substances were deliberately chosen to cover **all scenarios** that `fit_dataframe()` can encounter:

| Substance | Full Name | LOQ (ng/L) | Detected | Censored | % Censored | Expected Behavior |
|-----------|-----------|:----------:|:--------:|:--------:|:----------:|-------------------|
| **PFOA** | Perfluorooctanoic acid | 1.00 | 53 | 47 | 47% | ✅ Full ROS applied |
| **PFOS** | Perfluorooctane sulfonic acid | 0.50 | 33 | 67 | 67% | ✅ Full ROS applied |
| **PFHxA** | Perfluorohexanoic acid | 0.40 | 30 | 70 | 70% | ⚠️ ROS applied with robustness warning |
| **PFHxS** | Perfluorohexane sulfonic acid | 0.25 | 30 | 70 | 70% | ⚠️ ROS applied with robustness warning |
| **PFBS** | Perfluorobutane sulfonic acid | 0.30 | 2 | 98 | 98% | ⚠️ LOQ/2 fallback (< 3 detected) |
| **PFBA** | Perfluorobutanoic acid | 0.20 | 0 | 100 | 100% | ⚠️ LOQ/2 fallback (no detections) |

| Column | Type | Description |
|--------|------|-------------|
| `Substance` | `str` | PFAS compound name |
| `Concentration` | `float` | Reported concentration in ng/L. For censored values, this equals the LOQ |
| `Below LOQ` | `bool` | `True` if the value is a non-detect (censored), `False` if detected |
| `LOQ` | `float` | Limit of Quantification in ng/L |

---

## **Why Use This Implementation?**

✔️ Python alternative to the **NADA package in R**  
✔️ Handles **left-censored data** efficiently  
✔️ Handles **multiple detection limits** via Helsel-Cohn plotting positions  
✔️ Supports **log-normal & normal distributions**  
✔️ Generates **Q-Q plots & summary statistics**  
✔️ Provides **model diagnostics**: AIC, BIC, PPCC, Shapiro-Francia W  
✔️ Supports **batch processing** of multi-substance DataFrames via `fit_dataframe()`  
✔️ Falls back to **LOQ × factor substitution** when ROS is not applicable  
✔️ Issues **warnings** when results may be unreliable (high censoring rate, few detections)  

---

## **Model Diagnostics**

After fitting, the following diagnostics are available via `ros.summary()`:

| Diagnostic | Description |
|------------|-------------|
| **AIC** | Akaike Information Criterion — lower is better |
| **BIC** | Bayesian Information Criterion — lower is better |
| **PPCC** | Probability Plot Correlation Coefficient — measures goodness-of-fit to the assumed distribution; values close to 1 indicate a good fit |
| **Shapiro-Francia W** | Normality test applied to the (log-)transformed modeled data; values close to 1 indicate normality |

---

## **Applicability and Limitations**

- ROS requires a **detection frequency > 50%** and at least **3 detected values** to produce
  statistically valid estimates. With fewer detected values, results should be interpreted
  with great caution or avoided altogether.
- For best results, a minimum of **8–10 observations** is recommended.
- ROS is designed for **left-censored data** (non-detects below a detection limit). It is **not**
  appropriate for right-censored or interval-censored data without modification.
- The distributional assumption (log-normal or normal) should be evaluated using the **PPCC**
  or **Q-Q plot** before relying on summary statistics.

---

## **Differences from R's NADA Package**

| Aspect | This Implementation | R's NADA |
|--------|---------------------|----------|
| Regression | `numpy.linalg.lstsq` + `sklearn.LinearRegression` | `lm()` — results numerically equivalent |
| Normal quantiles | `scipy.stats.norm.ppf()` | `qnorm()` — negligible floating-point differences |
| Shapiro-Francia W | Approximated via `scipy.stats.shapiro` (Shapiro-Wilk) | Exact Shapiro-Francia statistic |
| Confidence intervals for `pexceed` | ❌ Not yet implemented | ✅ Available |

---

## **References**

- Helsel, D. R. (2005). *Nondetects and Data Analysis: Statistics for Censored Environmental Data*.
- Helsel, D.R. (2012). *Statistics for Censored Environmental Data Using Minitab and R*.
- **NADA R Package**: [CRAN NADA](https://cran.r-project.org/web/packages/NADA/index.html)

---

## **Contributions**

Feel free to **open an issue** or **submit a pull request** if you:
- Find any **bugs** or **errors**
- Have **suggestions for improvements**
- Want to **add new features**

---

## **License**

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
You are free to use, modify, and distribute this code, provided you give appropriate credit to the author.

---

## **Contact**

If you have any questions, feel free to reach out!  
✉️ **Email:** [Aliiivalipoor@gmail.com](mailto:Aliiivalipoor@gmail.com)  
📌 **GitHub Issues:** Open a new issue in this repository  

---

🎯 **If you find this repository useful, please ⭐ star it on GitHub!** 🚀
