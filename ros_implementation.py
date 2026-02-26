import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tabulate import tabulate
import logging

class ROS:
    """
    Regression on Order Statistics (ROS) for left-censored data.

    This is a Python implementation of the ROS method, which is used for handling 
    environmental data with detection limits. The method is inspired by the NADA package in R 
    and follows the same statistical principles.

    Author: Ali Valipoor
    Email: Aliiivalipoor@gmail.com
    Date: February 2026
    Version: 2.0

    Reference:
    ----------
    - Helsel, D. R. (2005). Nondetects and data analysis. Statistics for censored environmental data.
    - Helsel, D.R. (2012). Statistics for Censored Environmental Data Using Minitab and R.
    - NADA package in R: https://cran.r-project.org/web/packages/NADA/index.html

    Background:
    -----------
    ROS is a parametric method designed to estimate summary statistics (mean, standard deviation,
    quantiles) for datasets that contain non-detects (values below a detection or reporting limit).
    Rather than discarding non-detects or substituting them with arbitrary constants (e.g., LOQ/2),
    ROS leverages the distributional structure of the detected values to extrapolate plausible values
    for the censored observations.

    The algorithm proceeds as follows:
    1. Assign Helsel-Cohn plotting positions to all data points (both detected and non-detected),
       accounting for multiple detection limits if present.
    2. Fit a linear regression of the transformed detected values against their normal quantiles
       (i.e., fit the data on a probability plot).
    3. Use the fitted regression line to predict transformed values for the censored observations
       based on their plotting positions.
    4. Back-transform censored predictions to the original scale.
    5. Compute summary statistics using the combined set of detected and modeled censored values.

    Key Features:
    -------------
    - Supports both **log-normal** and **normal** distributions.
    - Handles **multiple detection limits** using Helsel-Cohn plotting position calculations.
    - Automatically drops censored values exceeding the maximum uncensored value (as these
      would be inconsistent with left-censored data and could distort the regression).
    - Issues **warnings** when the proportion of censored data exceeds 80%, or when fewer than
      3 detected values are available, as ROS estimates become unreliable under these conditions.
    - Provides **model diagnostics** including:
        - AIC (Akaike Information Criterion)
        - BIC (Bayesian Information Criterion)
        - PPCC (Probability Plot Correlation Coefficient): measures goodness-of-fit to the
          assumed distribution; values close to 1 indicate good fit.
        - Shapiro-Francia W: a normality test applied to the (log-)transformed modeled data.
    - Provides **statistical summary functions**: mean, median, standard deviation, and quantiles
      of the combined detected + modeled dataset.
    - Supports **batch processing** of multi-substance DataFrames via `fit_dataframe()`, with
      optional fallback to LOQ/2 substitution when ROS is not applicable.
    - Allows **graphical visualization** using Q-Q plots comparing observed and modeled data.

    Differences from R's NADA:
    --------------------------
    - The regression is implemented using `numpy.linalg.lstsq` and `sklearn.linear_model.LinearRegression`,
      rather than R's built-in `lm()` function — results should be numerically equivalent.
    - Normal quantile calculations are performed using `scipy.stats.norm.ppf()`, which may introduce
      negligible floating-point differences compared to R's `qnorm()`.
    - Confidence intervals for exceedance probabilities (`pexceed`) are **not yet implemented**.
    - The Shapiro-Francia W is approximated here using `scipy.stats.shapiro` (Shapiro-Wilk), which
      is closely related but not identical to the Shapiro-Francia statistic used in R's NADA package.

    Applicability and Limitations:
    -------------------------------
    - ROS requires a **detection frequency > 50%** and at least **3 detected values** to produce
      statistically valid estimates. With fewer detected values, results should be interpreted
      with great caution or avoided altogether.
    - For best results, a minimum of **8–10 observations** is recommended.
    - ROS is designed for **left-censored data** (non-detects below a detection limit). It is not
      appropriate for right-censored or interval-censored data without modification.
    - The distributional assumption (log-normal or normal) should be evaluated using the PPCC
      or Q-Q plot before relying on summary statistics.

    Usage:
    ------
    Basic fit:
    ```python
    ros = ROS(distribution="log-normal")  # or "normal"
    ros.fit(obs=[1.2, 0.5, 0.3, 2.1, 5.0], censored=[False, True, True, False, False])
    print(ros.summary())       # Summary of the model
    ros.plot_qq()              # Q-Q plot of modeled vs. observed values
    ```

    Batch processing over a DataFrame:
    ```python
    df_result = ros.fit_dataframe(
        df,
        concentration_col="Concentration",
        censored_col="Below LOQ",
        loq_col="LOQ",
        loq_replacement=True,   # Fall back to LOQ/2 when ROS is not applicable
        loq_factor=0.5          # Use LOQ * 0.5 as the substitution value (default)
    )
    ```

    Attributes:
    -----------
    - `obs` (np.ndarray): Sorted original observations (including censored values) after fitting.
    - `censored` (np.ndarray): Boolean array indicating censored status (True = censored).
    - `pp` (np.ndarray): Helsel-Cohn plotting positions assigned to each observation.
    - `modeled` (np.ndarray): Combined array of detected values and ROS-modeled censored values.
    - `model` (LinearRegression): Fitted sklearn LinearRegression object (intercept + slope on
      the probability plot of the detected data).
    - `aic` (float): Akaike Information Criterion of the fitted model (based on detected values only).
    - `bic` (float): Bayesian Information Criterion of the fitted model.
    - `ppcc` (float): Probability Plot Correlation Coefficient for the full modeled dataset.
    - `shapiro_w` (float): Shapiro-Wilk W statistic for normality of the (log-)transformed modeled data.
    - `distribution` (str): Distribution assumption used ('log-normal' or 'normal').
    - `forward_transform` (callable): Transformation applied to data before regression (log for
      log-normal, identity for normal).
    - `reverse_transform` (callable): Inverse of the forward transform.

    Methods:
    --------
    - `fit(obs, censored)`: Fit the ROS model to the given dataset. Returns self.
    - `predict(new_quantiles)`: Predict back-transformed values for given normal quantiles
      using the fitted regression line.
    - `mean()`: Compute the mean of the modeled dataset.
    - `median()`: Compute the median of the modeled dataset.
    - `sd()`: Compute the standard deviation of the modeled dataset (using n-1 denominator).
    - `quantile(probs)`: Compute quantiles of the modeled dataset at the specified probabilities.
    - `pexceed(newdata)`: Compute exceedance probabilities for new data values using the fitted model.
    - `summary()`: Print and return a formatted statistical summary table.
    - `to_dataframe()`: Return a DataFrame with obs, censored, pp, and modeled columns.
    - `fit_dataframe(df, ...)`: Fit ROS across all substances in a DataFrame; falls back to
      LOQ-based substitution when ROS is not applicable (if enabled).
    - `plot_qq(figsize, title)`: Generate a Q-Q plot comparing observed and modeled data.
    """

    
    def __init__(self, distribution='log-normal'):
        """
        Initialize ROS object with distribution model.
        
        Parameters:
        -----------
        distribution : str
            Distribution model to use. Options are:
            - 'log-normal' (default)
            - 'normal'
        """
        if distribution not in ['log-normal', 'normal']:
            raise ValueError(f"Unknown distribution: {distribution}. Choose from 'log-normal' or 'normal'.")
            
        self.distribution = distribution
        self.model = None
        self.obs = None
        self.censored = None
        self.pp = None
        self.modeled = None
        self.aic = None
        self.bic = None
        self.ppcc = None
        self.shapiro_w = None
        
        # Set transformations based on distribution
        self._set_transformations()
    
    def _set_transformations(self):
        """Set appropriate transformation functions based on the distribution."""
        if self.distribution == 'log-normal':
            self.forward_transform = np.log
            self.reverse_transform = np.exp
        elif self.distribution == 'normal':
            self.forward_transform = lambda x: x
            self.reverse_transform = lambda x: x
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}. Choose from 'log-normal' or 'normal'.")
    
    def fit(self, obs, censored):
        """
        Fit ROS model to data.
        
        Parameters:
        -----------
        obs : array-like
            Observations (including censored values)
        censored : array-like of bool
            Boolean indicator of censoring (True for censored)
            
        Returns:
        --------
        self : ROS object
        """

        # Convert inputs to numpy arrays
        obs = np.asarray(obs, dtype=float)
        censored = np.asarray(censored, dtype=bool)
        
        # Check for missing values
        if np.isnan(obs).any() or np.isnan(censored).any():
            mask = ~(np.isnan(obs) | np.isnan(censored))
            obs = obs[mask]
            censored = censored[mask]
        
        # Warning for high proportion of censored values
        cen_proportion = np.sum(censored) / len(censored)
        if cen_proportion > 0.8:
            if np.sum(~censored)== 0:
                logging.warning(f"⚠️ WARNING: Input {cen_proportion:.1%} censored and only {np.sum(~censored)} detected data -- ROS is not applicable due to insufficient detected data!\n   → Consider that to compute ROS, at a minimum, there must be at least three detected values and a detection frequency greater than 50%. More realistically, you should have at least 8-10 measurements.")
            if np.sum(~censored)<3:
                logging.warning(f"⚠️ WARNING: Input {cen_proportion:.1%} censored and only {np.sum(~censored)} detected data -- Results are not robust!\n   → Consider that to compute ROS, at a minimum, there must be at least three detected values and a detection frequency greater than 50%. More realistically, you should have at least 8-10 measurements.")
            else:
                logging.warning(f"⚠️ WARNING: Input {cen_proportion:.1%} censored -- Results are not robust!\n   → Consider that to compute ROS, at a minimum, there must be at least three detected values and a detection frequency greater than 50%. More realistically, you should have at least 8-10 measurements.")
            
        
        # Drop censored values that exceed max of uncensored values
        if np.any(~censored):
            max_uncensored = np.max(obs[~censored])
            drop_mask = (obs > max_uncensored) & censored
            if np.any(drop_mask):
                logging.warning(f"⚠️ WARNING:  Dropped censored values that exceed max of uncensored values ({obs[obs > max_uncensored]})!\n   → Consider that ROS is for left-censored data (values below a detection limit).")
                keep_mask = ~drop_mask
                obs = obs[keep_mask]
                censored = censored[keep_mask]
        
        # Sort the data
        sort_idx = np.argsort(obs)
        obs = obs[sort_idx]
        censored = censored[sort_idx]
        
        # Calculate plotting positions
        pp = self._hc_ppoints(obs, censored)
        
        # Get normal quantiles for plotting positions of uncensored values
        pp_nq = stats.norm.ppf(pp[~censored])
        
        # Transform uncensored observations
        obs_transformed = self.forward_transform(obs[~censored])
        
        # Fit linear regression
        X = np.column_stack((np.ones(len(pp_nq)), pp_nq))
        beta = np.linalg.lstsq(X, obs_transformed, rcond=None)[0]
        
        # Create a LinearRegression model with the fitted coefficients
        self.model = LinearRegression()
        self.model.intercept_ = beta[0]
        self.model.coef_ = np.array([beta[1]])
        
        # Store results
        self.obs = obs
        self.censored = censored
        self.pp = pp
        
        # Calculate modeled values for all observations
        self.modeled = np.copy(obs)
        
        # For censored values, predict using the regression model
        if np.any(censored):
            censored_pp_nq = stats.norm.ppf(pp[censored])
            X_censored = np.column_stack((np.ones(len(censored_pp_nq)), censored_pp_nq))
            predicted_transformed = X_censored @ beta
            self.modeled[censored] = self.reverse_transform(predicted_transformed)
        
        # Calculate model diagnostics
        self._calculate_diagnostics()
        
        return self
    
    def _calculate_diagnostics(self):
        """Calculate model diagnostic metrics like AIC, BIC, PPCC, and Shapiro-Francia W."""
        if np.any(~self.censored):
            # Transform all modeled data for diagnostics
            transformed_data = self.forward_transform(self.modeled)
            n = len(self.modeled)
            
            # Calculate residuals for uncensored data only
            obs_transformed = self.forward_transform(self.obs[~self.censored])
            pp_nq = stats.norm.ppf(self.pp[~self.censored])
            X = np.column_stack((np.ones(len(pp_nq)), pp_nq))
            predicted = X @ np.array([self.model.intercept_, self.model.coef_[0]])
            residuals = obs_transformed - predicted
            
            # Calculate AIC and BIC
            k = 2  # Number of parameters: intercept and slope
            rss = np.sum(residuals**2)
            sigma2 = rss / len(residuals)
            loglik = -len(residuals)/2 * np.log(2*np.pi*sigma2) - rss/(2*sigma2)
            
            self.aic = -2 * loglik + 2 * k
            self.bic = -2 * loglik + k * np.log(n)
            
            # Calculate PPCC (Probability Plot Correlation Coefficient)
            # This measures the correlation between sample quantiles and theoretical quantiles
            if len(self.modeled) > 2:
                sorted_data = np.sort(transformed_data)
                pp_sorted = (np.arange(1, n+1) - 0.5) / n
                theoretical_quantiles = stats.norm.ppf(pp_sorted)
                self.ppcc = np.corrcoef(sorted_data, theoretical_quantiles)[0, 1]
            else:
                self.ppcc = np.nan
            
            # Calculate Shapiro-Francia W test
            if len(self.modeled) >= 3:  # Need at least 3 points for the test
                self.shapiro_w, _ = stats.shapiro(transformed_data)
            else:
                self.shapiro_w = np.nan
    
    def predict(self, new_quantiles):
        """
        Predict values for new normal quantiles.
        
        Parameters:
        -----------
        new_quantiles : array-like
            Normal quantiles to predict values for
            
        Returns:
        --------
        array-like : Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reshape input if needed
        if np.isscalar(new_quantiles):
            new_quantiles = np.array([new_quantiles])
        
        # Create design matrix with intercept
        X = np.column_stack((np.ones(len(new_quantiles)), new_quantiles))
        
        # Make predictions
        preds = X @ np.array([self.model.intercept_, self.model.coef_[0]])
        
        # Apply reverse transformation
        return self.reverse_transform(preds)
    
    def _cohn(self, obs, censored):
        """
        Calculate Cohn numbers as described by Helsel and Cohn (1988).
        
        Parameters:
        -----------
        obs : array-like
            Observations (including censored values)
        censored : array-like of bool
            Boolean indicator of censoring (True for censored)
            
        Returns:
        --------
        dict : Dictionary of Cohn numbers
        """
        # Extract uncensored and censored values
        uncen = obs[~censored]
        cen = obs[censored]
        
        if len(cen) == 0:
            return {'A': np.array([]), 'B': np.array([]), 
                    'C': np.array([]), 'P': np.array([]), 'limit': np.array([])}
        
        # Get unique censoring limits (sorted)
        limit = np.sort(np.unique(cen))
        
        # Check if there are uncensored values below the lowest limit
        if len(uncen) > 0 and np.any(uncen < limit[0]):
            limit = np.concatenate(([0], limit))
        
        n = len(limit)
        A = np.zeros(n)
        B = np.zeros(n)
        C = np.zeros(n)
        P = np.zeros(n)
        
        # Calculate for highest limit first
        i = n - 1
        A[i] = np.sum(uncen >= limit[i])
        B[i] = np.sum(obs <= limit[i]) - np.sum(uncen == limit[i])
        C[i] = np.sum(cen == limit[i])
        
        if A[i] + B[i] > 0:
            P[i] = A[i] / (A[i] + B[i])
        
        # Calculate for remaining limits
        for i in range(n-2, -1, -1):
            # A[i]: uncensored obs between current and next higher limit
            if i < n-1:
                A[i] = np.sum((uncen >= limit[i]) & (uncen < limit[i+1]))
            else:
                A[i] = np.sum(uncen >= limit[i])
                
            B[i] = np.sum(obs <= limit[i]) - np.sum(uncen == limit[i])
            C[i] = np.sum(cen == limit[i])
            
            if A[i] + B[i] > 0:
                P[i] = P[i+1] + ((A[i]/(A[i] + B[i])) * (1 - P[i+1]))
            else:
                P[i] = P[i+1]
        
        return {'A': A, 'B': B, 'C': C, 'P': P, 'limit': limit}
    
    def _hc_ppoints_uncen(self, obs, censored, cn=None):
        """
        Calculate plotting positions for uncensored data.
        
        Parameters:
        -----------
        obs : array-like
            Observations
        censored : array-like of bool
            Boolean indicator of censoring
        cn : dict, optional
            Cohn numbers, calculated if not provided
            
        Returns:
        --------
        array-like : Plotting positions for uncensored data
        """
        if cn is None:
            cn = self._cohn(obs, censored)
        
        # Filter out entries where A is zero
        nonzero = cn['A'] != 0
        A = cn['A'][nonzero]
        P = cn['P'][nonzero]
        limit = cn['limit'][nonzero]
        
        pp = []
        n = len(limit)
        
        for i in range(n):
            R = np.arange(1, A[i]+1)
            
            k = P[i+1] if i+1 < len(P) else 0
            
            for r in R:
                pp.append((1 - P[i]) + ((P[i] - k) * r) / (A[i] + 1))
        
        return np.array(pp)
    
    def _hc_ppoints_cen(self, obs, censored, cn=None):
        """
        Calculate plotting positions for censored data.
        
        Parameters:
        -----------
        obs : array-like
            Observations
        censored : array-like of bool
            Boolean indicator of censoring
        cn : dict, optional
            Cohn numbers, calculated if not provided
            
        Returns:
        --------
        array-like : Plotting positions for censored data
        """
        if cn is None:
            cn = self._cohn(obs, censored)
        
        C = cn['C']
        P = cn['P']
        limit = cn['limit']
        
        # Handle special case when P[0] == 1
        if len(P) > 0 and P[0] == 1:
            C = C[1:]
            P = P[1:]
            limit = limit[1:]
        
        pp = []
        for i in range(len(limit)):
            c_i = C[i]
            for r in range(1, int(c_i)+1):
                pp.append((1 - P[i]) * r / (c_i + 1))
        
        return np.array(pp)
    
    def _hc_ppoints(self, obs, censored):
        """
        Calculate Helsel-Cohn plotting positions for mixed censored/uncensored data.
        
        Parameters:
        -----------
        obs : array-like
            Observations
        censored : array-like of bool
            Boolean indicator of censoring
            
        Returns:
        --------
        array-like : Plotting positions for all data
        """
        if not np.any(censored):
            # If no censored values, use standard plotting positions
            n = len(obs)
            pp = (np.arange(1, n+1) - 0.5) / n
            return pp
        
        # Calculate Cohn numbers
        cn = self._cohn(obs, censored)
        
        # Calculate plotting positions for uncensored and censored data
        pp_uncen = self._hc_ppoints_uncen(obs, censored, cn)
        pp_cen = self._hc_ppoints_cen(obs, censored, cn)
        
        # Initialize result array
        pp = np.zeros(len(obs))
        
        # Assign values
        pp[~censored] = pp_uncen
        pp[censored] = pp_cen
        
        return pp
    
    def mean(self):
        """Calculate mean of modeled values."""
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.mean(self.modeled)
    
    def median(self):
        """Calculate median of modeled values."""
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.median(self.modeled)
    
    def sd(self):
        """
        Calculate standard deviation of modeled values.
        
        Uses R-style calculation (n-1 denominator) to match R's sd function.
        """
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use ddof=1 to match R's variance calculation (n-1 denominator)
        return np.sqrt(np.var(self.modeled, ddof=1))
    
    def quantile(self, probs=None):
        """
        Calculate quantiles of modeled values.
        
        Parameters:
        -----------
        probs : array-like, optional
            Probabilities for quantiles
            
        Returns:
        --------
        array-like : Quantiles
        """
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if probs is None:
            probs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        
        # Use type=7 interpolation to match R's default
        return np.quantile(self.modeled, probs, interpolation='linear')
    
    def pexceed(self, newdata, conf_int=False, conf_level=0.95):
        """
        Calculate probability of exceeding given values.
        
        Parameters:
        -----------
        newdata : array-like
            Values to calculate exceedance probabilities for
        conf_int : bool, optional
            Whether to return confidence intervals
        conf_level : float, optional
            Confidence level
            
        Returns:
        --------
        array-like or DataFrame : Exceedance probabilities
        """
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Fit reversed model (normal quantiles predicted from transformed values)
        pp_nq = stats.norm.ppf(self.pp[~self.censored])
        obs_transformed = self.forward_transform(self.obs[~self.censored])
        
        # Fit using numpy to match R behavior
        X = np.column_stack((np.ones(len(obs_transformed)), obs_transformed))
        beta = np.linalg.lstsq(X, pp_nq, rcond=None)[0]
        
        # Transform new data
        nd_transformed = self.forward_transform(np.asarray(newdata)).reshape(-1)
        X_new = np.column_stack((np.ones(len(nd_transformed)), nd_transformed))
        
        if not conf_int:
            # Calculate predictions
            pred = X_new @ beta
            # Return exceedance probabilities
            return 1 - stats.norm.cdf(pred)
        else:
            # TODO: Implement confidence intervals
            # This would require more complex calculations
            # For now, just returning the point estimates
            print("Warning: Confidence intervals not yet implemented")
            pred = X_new @ beta
            return 1 - stats.norm.cdf(pred)
    
    def to_dataframe(self):
        """
        Convert ROS results to DataFrame.
        
        Returns:
        --------
        pandas.DataFrame : DataFrame with observations, censoring indicators,
                          plotting positions, and modeled values
        """
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return pd.DataFrame({
            'obs': self.obs,
            'censored': self.censored,
            'pp': self.pp,
            'modeled': self.modeled
        })


    def fit_dataframe(self, df, concentration_col="Concentration", censored_col="Below LOQ",
                      loq_col="LOQ", loq_replacement=True, loq_factor=0.5):
        """
        Fit ROS to an entire DataFrame while preserving row order and handling censored data.

        For each unique substance in the DataFrame, ROS is fitted independently. When ROS
        cannot be applied (e.g., too few detected values), the method can optionally fall
        back to substituting censored values with a fraction of the LOQ (e.g., LOQ / 2).

        Preprocessing:
        --------------
        - Converts invalid values (negative concentrations) to NaN.
        - Drops censored values exceeding the max uncensored value (inconsistent with
          left-censoring; these would distort the ROS regression).
        - Maintains original DataFrame row order throughout.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input data containing concentration measurements, censoring flags,
            and optionally the LOQ values.
        concentration_col : str, optional
            Column name containing the concentration values. Default: "Concentration".
        censored_col : str, optional
            Column name indicating whether a value is censored (True/False or 1/0).
            Default: "Below LOQ".
        loq_col : str, optional
            Column name containing the Limit of Quantification (LOQ) values.
            Required when `loq_replacement=True`. Default: "LOQ".
        loq_replacement : bool, optional
            Whether to fall back to LOQ-based substitution when ROS cannot be applied
            (i.e., fewer than 3 detected values or all values are censored).
            - If True (default): censored values are replaced with `LOQ * loq_factor`,
              and detected values are kept as-is (or replaced with their mean if ROS
              is not available).
            - If False: rows that cannot be modeled by ROS are left as NaN in the
              'modeled' column.
        loq_factor : float, optional
            Multiplication factor applied to LOQ when falling back to LOQ substitution.
            Default is 0.5 (i.e., LOQ / 2). Only used when `loq_replacement=True`.

        Returns:
        --------
        pandas.DataFrame
            A copy of the input DataFrame with an additional 'modeled' column containing:
            - ROS-modeled values for substances where ROS was applicable.
            - LOQ * loq_factor substitutions for censored values of substances where
              ROS was not applicable (if `loq_replacement=True`).
            - NaN for rows that could not be modeled and `loq_replacement=False`.

        Notes:
        ------
        - The method expects a column named "Substance" in the DataFrame to group observations
          by substance. Each substance is modeled independently.
        - When `loq_replacement=True` and a substance has some (but fewer than 3) detected
          values, the detected values are assigned their observed mean and censored values
          are assigned LOQ * loq_factor.
        - When `loq_replacement=True` and no detected values exist at all, all rows for
          that substance are assigned LOQ * loq_factor.
        - A warning is printed for each substance where LOQ replacement is triggered.

        Examples:
        ---------
        ```python
        ros = ROS(distribution="log-normal")

        # With LOQ/2 fallback (default behavior)
        df_result = ros.fit_dataframe(df, loq_replacement=True, loq_factor=0.5)

        # Strict ROS only — no substitution, unmodelable values left as NaN
        df_result = ros.fit_dataframe(df, loq_replacement=False)

        # Custom substitution factor (e.g., LOQ / 3)
        df_result = ros.fit_dataframe(df, loq_replacement=True, loq_factor=1/3)
        ```
        """
        # Create a copy of the original DataFrame
        df_result = df.copy()
        df_result["modeled"] = np.nan  # Initialize new column
        
        # Store original indices to maintain order
        df_result["original_index"] = df_result.index
        
        for substance in df["Substance"].unique():
            # Filter for specific substance
            df_sub = df[df["Substance"] == substance].copy()
            df_sub["original_index"] = df_sub.index
            
            # Preprocess data
            obs = df_sub[concentration_col].values
            censored = df_sub[censored_col].values
            loq = df_sub[loq_col].values if loq_col in df_sub.columns else None
            
            # Mask for valid observations (non-negative and non-NaN)
            valid_mask = ~np.isnan(obs) & (obs >= 0)
            obs = obs[valid_mask]
            censored = censored[valid_mask]
            loq = loq[valid_mask] if loq is not None else None
            original_indices = df_sub.loc[valid_mask, "original_index"].values
            
            # Warning for high proportion of censored values
            cen_proportion = np.sum(censored) / len(censored) if len(censored) > 0 else 0
            if cen_proportion > 0.8:
                if np.sum(~censored) == 0:
                    print(f"⚠️ WARNING: No detected values for {substance}.")
                    
                    if loq_replacement:
                        # If all values are censored, use LOQ * loq_factor for all values
                        if loq is not None:
                            df_result.loc[df_result["Substance"] == substance, "modeled"] = loq / (1 / loq_factor)
                        else:
                            print(f"   → Cannot apply LOQ replacement for {substance}: no LOQ column provided.")
                    continue
                elif np.sum(~censored) < 3:
                    print(f"⚠️ WARNING: Insufficient detected values for {substance}.")
            
            # Handling censored values
            if np.any(~censored):
                max_uncensored = np.max(obs[~censored])
                
                # Drop censored values that exceed max of uncensored values
                drop_mask = (obs > max_uncensored) & censored
                if np.any(drop_mask):
                    print(f"⚠️ WARNING: Dropped censored values that exceed max of uncensored values for {substance}.")
                    
                    # Mark dropped censored values as NaN
                    obs[drop_mask] = np.nan
                    censored[drop_mask] = False
            
            # Separate valid data
            valid_mask = ~np.isnan(obs)
            obs_valid = obs[valid_mask]
            censored_valid = censored[valid_mask]
            original_indices_valid = original_indices[valid_mask]
            
            # Check if there are enough uncensored values for ROS
            if np.sum(~censored_valid) >= 3:
                # Sort data as in original ROS implementation
                sort_idx = np.argsort(obs_valid)
                obs_sorted = obs_valid[sort_idx]
                censored_sorted = censored_valid[sort_idx]
                orig_indices_sorted = original_indices_valid[sort_idx]
                
                # Fit ROS
                self.fit(obs_sorted, censored_sorted)
                ros_model = self.to_dataframe()
                
                # Add original indices to ros_model
                ros_model["original_index"] = orig_indices_sorted
                
                # Merge modeled values
                df_sub = df_sub.merge(ros_model[["original_index", "modeled"]], on="original_index", how="left")
                
                # Update result DataFrame
                df_result.loc[df_result["Substance"] == substance, "modeled"] = df_result.loc[
                    df_result["Substance"] == substance, "original_index"
                ].map(df_sub.set_index("original_index")["modeled"])
            
            else:
                # Not enough uncensored values for ROS
                if loq_replacement:
                    if loq is not None:
                        modeled_values = np.full(len(df_sub), np.nan)
                        
                        # Replace censored values with LOQ * loq_factor
                        censored_positions = np.where(censored_valid)[0]
                        modeled_values[censored_positions] = loq[censored_positions] * loq_factor
                        
                        # If some uncensored values exist, use their mean for non-censored positions
                        if np.sum(~censored_valid) > 0:
                            uncensored_mean = np.mean(obs_valid[~censored_valid])
                            uncensored_positions = np.where(~censored_valid)[0]
                            modeled_values[uncensored_positions] = uncensored_mean
                        
                        # Assign to result DataFrame using original indices
                        df_result.loc[df_result["Substance"] == substance, "modeled"] = df_result.loc[
                            df_result["Substance"] == substance, "original_index"
                        ].map(dict(zip(df_sub["original_index"], modeled_values)))
                    else:
                        print(f"⚠️ WARNING: Cannot apply LOQ replacement for {substance}: no LOQ column provided.")
                else:
                    print(f"ℹ️ INFO: ROS not applicable for {substance} (fewer than 3 detected values). "
                          f"Rows left as NaN (loq_replacement=False).")
        
        # Drop temporary index column
        df_result.drop(columns=["original_index"], inplace=True)
        
        return df_result
    
    
    def plot_qq(self, figsize=(10, 6), title=None):
        """
        Create a Q-Q plot showing both observed and modeled data.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        title : str, optional
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure : Figure object
        """
        if self.modeled is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normal quantiles
        q = stats.norm.ppf(self.pp)
        
        # Apply transformations for y-axis based on distribution
        y_obs = self.forward_transform(self.obs)
        
        # Plot observed uncensored data with one marker
        ax.scatter(q[~self.censored], y_obs[~self.censored], 
                   color='blue', marker='o', label='Observed (Uncensored)')
        
        # Plot modeled censored data with different marker
        ax.scatter(q[self.censored], self.forward_transform(self.modeled[self.censored]), 
                   color='red', marker='x', label='Modeled (Censored)')
        
        # Fit line for all data
        X = np.column_stack((np.ones(len(q)), q))
        beta = np.linalg.lstsq(X, self.forward_transform(self.modeled), rcond=None)[0]
        
        # Create x values for line
        x_line = np.array([np.min(q), np.max(q)])
        X_line = np.column_stack((np.ones(len(x_line)), x_line))
        y_line = X_line @ beta
        
        # Plot line
        ax.plot(x_line, y_line, 'k--', linewidth=1)
        
        # Set labels based on distribution
        if self.distribution == 'log-normal':
            ax.set_ylabel('log(Concentration)')
        elif self.distribution == 'normal':
            ax.set_ylabel('Concentration')
        
        ax.set_xlabel('Normal Quantiles')
        
        # Set title
        if title is None:
            title = f'Q-Q Plot for {self.distribution.capitalize()} Distribution'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def summary(self):
        """
        Print a formatted summary of the ROS results.
        
        Returns:
        --------
        str : Summary text
        """
        if self.modeled is None:
            return "ROS object (not fitted)"
        
        # Section 1: Model information and diagnostic tests
        model_info = [
            ["Distribution", self.distribution.capitalize()],
            ["AIC", f"{self.aic:.4f}" if self.aic is not None else "N/A"],
            ["BIC", f"{self.bic:.4f}" if self.bic is not None else "N/A"],
            ["PPCC", f"{self.ppcc:.4f}" if self.ppcc is not None else "N/A"],
            ["Shapiro-Francia W", f"{self.shapiro_w:.4f}" if self.shapiro_w is not None else "N/A"]
        ]
        
        # Section 2: Data information
        n = len(self.modeled)
        n_cen = np.sum(self.censored)
        pct_cen = (n_cen / n) * 100
        
        data_info = [
            ["Total observations", f"{n}"],
            ["Censored observations", f"{n_cen} ({pct_cen:.1f}%)"],
            ["Uncensored observations", f"{n - n_cen} ({100 - pct_cen:.1f}%)"]
        ]
        
        # Section 3: Statistical summary
        probs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        quants = self.quantile(probs)
        
        stat_info = [
            ["\033[1m Mean", f"{self.mean():.4f}"],
            ["\033[1m Median", f"{self.median():.4f}"],
            ["\033[1m Standard Deviation", f"{self.sd():.4f} \033[0m"]
        ]
        
        for i, prob in enumerate(probs):
            stat_info.append([f"{int(prob*100)}th Percentile", f"{quants[i]:.4f}"])
        
        # Format the sections with tabulate
        summary_text = "\n" + "=" * 60 + "\n"
        summary_text += "ROS SUMMARY\n"
        summary_text += "=" * 60 + "\n\n"
        
        summary_text += "MODEL INFORMATION\n"
        summary_text += "-" * 60 + "\n"
        summary_text += tabulate(model_info, tablefmt="plain") + "\n\n"
        
        summary_text += "DATA INFORMATION\n"
        summary_text += "-" * 60 + "\n"
        summary_text += tabulate(data_info, tablefmt="plain") + "\n\n"
        
        summary_text += "STATISTICAL SUMMARY\n"
        summary_text += "-" * 60 + "\n"
        summary_text += tabulate(stat_info, tablefmt="plain") + "\n"
        
        print(summary_text)
        return summary_text
    
    def __str__(self):
        """String representation of ROS object."""
        if self.modeled is None:
            return "ROS object (not fitted)"
        
        n = len(self.modeled)
        n_cen = np.sum(self.censored)
        
        return (f"ROS Results ({self.distribution} distribution):\n"
                f"  n: {n}\n"
                f"  n_censored: {n_cen} ({(n_cen/n)*100:.1f}%)\n"
                f"  median: {self.median():.4f}\n"
                f"  mean: {self.mean():.4f}\n"
                f"  sd: {self.sd():.4f}")
