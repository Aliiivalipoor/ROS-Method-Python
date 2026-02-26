# **ROS Method in Python**
### *Regression on Order Statistics (ROS) for Censored Data*

This repository contains a **Python implementation of the ROS method**, which is used for handling
**left-censored environmental data**, inspired by the **NADA package in R**.

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

### `multi_substance_example.csv`

> ⚠️ **This dataset is entirely artificial and was created for demonstration purposes only.**
> The concentration values, detection frequencies, and LOQ values do **not** represent
> real environmental measurements and should not be used for any scientific or regulatory conclusions.

Contains **600 simulated water quality measurements** for six
**per- and polyfluoroalkyl substances (PFAS)** — 100 samples per substance.
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
✔️ Supports **multiple detection limits** via Helsel-Cohn plotting positions  
✔️ Supports **log-normal & normal distributions**  
✔️ Generates **Q-Q plots & summary statistics**  
✔️ Supports **batch processing** of multi-substance DataFrames via `fit_dataframe()`  
✔️ Falls back to **LOQ × factor substitution** when ROS is not applicable  

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

This project is for **personal use** and does not have a specific license.
If you wish to use or modify this code, please contact me.

---

## **Contact**

If you have any questions, feel free to reach out!  
✉️ **Email:** [Aliiivalipoor@gmail.com](mailto:Aliiivalipoor@gmail.com)  
📌 **GitHub Issues:** Open a new issue in this repository  

---

🎯 **If you find this repository useful, please ⭐ star it on GitHub!** 🚀
