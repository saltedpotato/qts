import numpy as np
from numba import prange, njit
from numba.typed import List
from adf import ADF_Test


class cointegration_pairs:
    def __init__(self, df, p_val_cutoff=0.05, cluster_pairs=None):
        self.tickers = list(df.columns)
        self.pairs = {}

        if cluster_pairs is not None:
            self.cluster_pairs = list(cluster_pairs.values())
            nb_list = List()
            for lst in self.cluster_pairs:
                nb_list.append(lst)

            self.cluster_pairs = nb_list

            flatten_list = [
                item for sublist in list(cluster_pairs.values()) for item in sublist
            ]
            self.df_arr = df.select(flatten_list).to_numpy().T
        else:
            self.cluster_pairs = cluster_pairs
            self.df_arr = df.to_numpy().T

        self.p_val_cutoff = p_val_cutoff
        self.adf = ADF_Test()

    def identify_pairs(self):
        pairs = self._identify_pairs(
            tickers=self.tickers,
            df_arr=self.df_arr,
            adf=self.adf,
            coint=self.coint_test,
            p_val_cutoff=self.p_val_cutoff,
            cluster_pairs=self.cluster_pairs,
        )
        
        self.pairs = dict(pairs)

    @staticmethod
    @njit(fastmath=True)
    def coint_test(x1, x2):
        X = np.column_stack((np.ones(x1.shape[0]), x1))
        beta = np.linalg.inv(X.T @ X) @ X.T @ x2

        pred = X @ beta
        resid = x2 - pred
        return resid

    @staticmethod
    @njit(parallel=True)
    def _identify_pairs(tickers, df_arr, adf, coint, p_val_cutoff, cluster_pairs):
        pairs = {}

        if cluster_pairs is not None:
            start = 0
            n_clusters = len(cluster_pairs)
            for cluster in range(n_clusters):
                n_this_cluster = len(cluster_pairs[cluster])
                if n_this_cluster != 1:
                    for i in prange(start + n_this_cluster):
                        x1_arr = df_arr[i]
                        for j in range(start + i + 1, start + n_this_cluster + 1):
                            x2_arr = df_arr[j]
                            if adf.ad_fuller(x1_arr) and adf.ad_fuller(x2_arr):
                                resid = coint(x1=x1_arr, x2=x2_arr)
                                t, pval = adf.ad_fuller(resid)
                                if pval <= p_val_cutoff:
                                    pairs[(tickers[i], tickers[j])] = t

                start += n_clusters
        else:
            n = len(tickers)
            for i in prange(n):
                x1_arr = df_arr[i]
                for j in range(i + 1, n + 1):
                    x2_arr = df_arr[j]
                    if adf.ad_fuller(x1_arr) and adf.ad_fuller(x2_arr):
                        resid = coint(x1=x1_arr, x2=x2_arr)
                        t, pval = adf.ad_fuller(resid)
                        if pval <= p_val_cutoff:
                            pairs[(tickers[i], tickers[j])] = t
        return pairs

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
