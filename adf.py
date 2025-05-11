import numba
import numpy as np
import math

spec = [
    ("tau_star_c", numba.float64[:]),
    ("tau_min_c", numba.float64[:]),
    ("tau_max_c", numba.float64[:]),
    ("tau_c_smallp", numba.float64[:, :]),
    ("tau_c_largep", numba.float64[:, :]),
]


@numba.experimental.jitclass(spec)
class ADF_Test:
    def __init__(self):
        self.tau_star_c = np.array(
            [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93], dtype=np.float64
        )
        self.tau_min_c = np.array(
            [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27], dtype=np.float64
        )
        self.tau_max_c = np.array([2.74, 0.92, 0.55, 0.61, 0.79, 1.0], dtype=np.float64)

        small_scaling = np.array([1.0, 1.0, 1e-2], dtype=np.float64)
        tau_c_smallp = np.array(
            [
                [2.1659, 1.4412, 3.8269],
                [2.92, 1.5012, 3.9796],
                [3.4699, 1.4856, 3.164],
                [3.9673, 1.4777, 2.6315],
                [4.5509, 1.5338, 2.9545],
                [5.1399, 1.6036, 3.4445],
            ],
            dtype=np.float64,
        )
        self.tau_c_smallp = tau_c_smallp * small_scaling

        large_scaling = np.array([1.0, 1e-1, 1e-1, 1e-2], dtype=np.float64)
        tau_c_largep = np.array(
            [
                [1.7339, 9.3202, -1.2745, -1.0368],
                [2.1945, 6.4695, -2.9198, -4.2377],
                [2.5893, 4.5168, -3.6529, -5.0074],
                [3.0387, 4.5452, -3.3666, -4.1921],
                [3.5049, 5.2098, -2.9158, -3.3468],
                [3.9489, 5.8933, -2.5359, -2.721],
            ],
            dtype=np.float64,
        )
        self.tau_c_largep = tau_c_largep * large_scaling

    def norm_cdf(self, x, mean=0, std_dev=1):
        return 0.5 * (1 + math.erf((x - mean) / (std_dev * math.sqrt(2))))

    def horner_eval(self, coeffs, x):
        result = 0.0
        for c in coeffs:
            result = result * x + c
        return result

    def mackinnonp(self, teststat, N=1):
        maxstat = self.tau_max_c
        minstat = self.tau_min_c
        starstat = self.tau_star_c
        if teststat > maxstat[N - 1]:
            return 1.0
        elif teststat < minstat[N - 1]:
            return 0.0
        if teststat <= starstat[N - 1]:
            tau_coef = self.tau_c_smallp[N - 1]
        else:
            # Note: above is only for z stats
            tau_coef = self.tau_c_largep[N - 1]
        return self.norm_cdf(self.horner_eval(tau_coef[::-1], teststat))

    def narrow(self, input, dim, start, length):
        if dim == 0:
            return input[start : start + length]
        # elif dim == 1:
        #     return input[:, start : start + length]

    def ad_fuller(self, series, maxlag=None):
        """Get series and return the p-value and the t-stat of the coefficient"""
        if maxlag is None:
            n = int((len(series) - 1) ** (1.0 / 3))
        elif maxlag < 1:
            n = 1
        else:
            n = maxlag

        # Putting the X values on a Tensor with Double as type
        X = series

        # Generating the lagged tensor to calculate the difference
        X_1 = self.narrow(X, 0, 1, X.shape[0] - 1)

        # Re-sizing the x values to get the difference
        X_diff = self.narrow(X, 0, 0, X.shape[0] - 1)
        dX = X_1 - X_diff

        # Generating the lagged difference tensors
        # and concatenating the lagged tensors into a single one
        for i in range(1, n + 1):
            lagged_n = self.narrow(dX, 0, n - i, (dX.shape[0] - n))
            lagged_reshape = np.reshape(lagged_n, (lagged_n.shape[0], 1))
            if i == 1:
                lagged_tensors = lagged_reshape
            else:
                lagged_tensors = np.concatenate((lagged_tensors, lagged_reshape), 1)

        # Reshaping the X and the difference tensor
        # to match the dimension of the lagged ones
        X_match = self.narrow(X_diff, 0, 0, X_diff.shape[0] - n)
        dX = self.narrow(dX, 0, n, dX.shape[0] - n)
        dX = np.reshape(dX, (dX.shape[0], 1))

        # Concatenating the lagged tensors to the X one
        # and adding a column full of ones for the Linear Regression
        X_linreg = X_match.copy()
        X_linreg = np.concatenate(
            (np.reshape(X_linreg, (X_linreg.shape[0], 1)), lagged_tensors), 1
        )
        ones_columns = np.ones((X_linreg.shape[0], 1))
        X_ = np.concatenate((X_linreg, np.ones_like(ones_columns, dtype=np.float64)), 1)

        # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
        coeff = np.linalg.inv(X_.T @ X_) @ X_.T @ dX

        std_error = self.get_std_error(X_, dX, coeff)
        coeff_std_err = self.get_coeff_std_error(X_, std_error, coeff)[0]
        t_stat = (coeff[0] / coeff_std_err).item()

        p_value = self.mackinnonp(t_stat, N=1)

        return t_stat, p_value

    def get_coeff_std_error(self, X, std_error, p):
        """Receive the regression standard error
        and calculate for the coefficient p"""
        std_coeff = []
        temp = np.linalg.inv(X.T @ X)
        for i in range(len(p)):
            s = temp[i][i] * (std_error**2)
            s = np.sqrt(s)
            std_coeff.append(s)
        return std_coeff

    def get_std_error(self, X, label, p):
        """Get the regression standard error"""
        std_error = 0
        y_new = X @ p
        std_error = np.sum((label[:, 0] - y_new[:, 0]) ** 2)
        return np.sqrt(std_error / X.shape[0])
