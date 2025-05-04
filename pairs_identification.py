from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations


class cointegration_pairs:
    def __init__(self, df, p_val_cutoff=0.05):
        self.df = df
        self.tickers = df.columns
        self.potential_pairs = combinations(self.tickers, r=2)
        self.pairs = {}
        self.p_val_cutoff = p_val_cutoff

    def adf_test(self, series):
        result = adfuller(series)
        if result[1] > self.p_val_cutoff:  # likely non-stationary (I(1))
            return True
        else:
            return False

    def coint_test(self, x1, x2):
        coint_stat, pval, _ = coint(x1, x2)
        return coint_stat, pval

    def identify_pairs(
        self,
    ):
        for x1, x2 in self.potential_pairs:
            x1_arr, x2_arr = self.df[[x1]], self.df[[x2]]
            if self.adf_test(x1_arr) and self.adf_test(x2_arr):
                coint_stat, pval = self.coint_test(x1=x1_arr, x2=x2_arr)
                if pval <= self.p_val_cutoff:
                    self.pairs[(x1, x2)] = coint_stat

    def get_top_pairs(self, n=20):
        if len(self.pairs) > 0:
            self.pairs = {
                k: v
                for k, v in sorted(
                    self.pairs.items(),
                    key=lambda item: item[1],
                )
            }

        return list(self.pairs.keys())[:n]
