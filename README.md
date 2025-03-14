# **ROS Method in Python**
### *Regression on Order Statistics (ROS) for Censored Data*

This repository contains a **Python implementation of the ROS method**, which is used for handling **left-censored environmental data**, inspired by the **NADA package in R**.

## **Files Included**
- `ros_implementation.py` – Python script implementing the ROS method.
- `ros_notebook.ipynb` – Jupyter Notebook explaining the methodology & running an example.
- `MPCA_benz.csv` – Sample dataset (Benzene Concentration data).
- `environment.yml` – Conda environment file for setting up dependencies.

## **Installation & Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ROS-Method-Python.git
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
   Then, open **`ros_notebook.ipynb`** to explore the analysis.

## **How to Use the ROS Method**
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

## **Example Dataset: `MPCA_benz.csv`**
The dataset contains **Benzene concentration measurements**, where some values are **censored (below detection limits)**.

| Column  | Description |
|---------|------------|
| `Benzene` | Measured benzene concentration |
| `BenzCen` | Boolean flag (`True` if censored) |

## **Why Use This Implementation?**
✔️ Python alternative to the **NADA package in R**  
✔️ Handles **left-censored data** efficiently  
✔️ Supports **log-normal & normal distributions**  
✔️ Generates **Q-Q plots & summary statistics**    

## **References**
- - Helsel, D. R. (2005). *Nondetects and Data Analysis: Statistics for Censored Environmental Data*.
- Helsel, D.R. (2012). *Statistics for Censored Environmental Data Using Minitab and R*  
- **NADA R Package**: [CRAN NADA](https://cran.r-project.org/web/packages/NADA/index.html)

## **Contributions**
Feel free to **open an issue** or **submit a pull request** if you:
- Find any **bugs** or **errors**
- Have **suggestions for improvements**
- Want to **add new features**

## **License**
This project is licensed under the **MIT License** – you are free to use, modify, and distribute it.

## **Contact**
If you have any questions, feel free to reach out!  
✉️ **Email:** [Your Email]  
📌 **GitHub Issues:** Open a new issue in this repository  

---
🎯 **If you find this repository useful, please ⭐ star it on GitHub!** 🚀

