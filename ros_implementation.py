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
    Date: March 2025
    Version: 1.0

    Reference:
    ----------
    - Helsel, D. R. (2005). Nondetects and data analysis. Statistics for censored environmental data.
    - Helsel, D.R. (2012). Statistics for Censored Environmental Data Using Minitab and R.
    - NADA package in R: https://cran.r-project.org/web/packages/NADA/index.html

    Key Features:
    -------------
    - Supports both **log-normal** and **normal** distributions.
    - Implements Helsel-Cohn plotting position calculations.
    - Provides model diagnostics such as AIC, BIC, PPCC, and Shapiro-Francia W.
    - Includes statistical summary functions (mean, median, SD, quantiles).
    - Allows graphical visualization using Q-Q plots.

    Differences from R’s NADA:
    --------------------------
    - The regression is implemented using `scipy.stats` and `sklearn.linear_model.LinearRegression`.
    - Normal quantile calculations are performed using `scipy.stats.norm.ppf()`, which may introduce slight variations.
    - Confidence intervals for exceedance probabilities are **not yet implemented**.
    - Fitting uses NumPy-based linear regression (`np.linalg.lstsq`) rather than R's built-in statistical functions.
    - The Shapiro-Francia W test is implemented using `scipy.stats`.

    Usage:
    ------
    ```
    ros = ROS(distribution="log-normal")  # or "normal"
    ros.fit(obs=[1.2, 0.5, 0.3, 2.1, 5.0], censored=[False, True, True, False, False])
    print(ros.summary())  # Summary of the model
    ros.plot_qq()  # Q-Q plot of modeled vs. observed values
    ```

    Attributes:
    -----------
    - `obs` (np.array): Original observations.
    - `censored` (np.array): Boolean mask of censored data.
    - `pp` (np.array): Calculated plotting positions.
    - `modeled` (np.array): Estimated values for censored observations.
    - `aic`, `bic`, `ppcc`, `shapiro_w`: Model diagnostics.

    Methods:
    --------
    - `fit(obs, censored)`: Fit the ROS model to the given dataset.
    - `predict(new_quantiles)`: Predict values for new normal quantiles.
    - `summary()`: Print a detailed statistical summary.
    - `to_dataframe()`: Convert results to a Pandas DataFrame.
    - `plot_qq()`: Generate a Q-Q plot comparing observed and modeled data.
    - `quantile(probs)`: Compute quantiles of the modeled dataset.
    - `pexceed(newdata)`: Compute exceedance probabilities for new data.
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