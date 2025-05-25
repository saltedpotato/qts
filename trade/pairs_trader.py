import polars as pl
import numpy as np
from numba import njit, prange
from typing import Any, Dict, Optional, List, Tuple

from data.market_data import market_data
from utils.params import PARAMS
from utils.market_time import market_hours


class PairsTrader:
    """
    Class for executing a pairs trading strategy.

    Parameters
    ----------
    data : market_data
        The market data source containing price and timestamp information.
    pairs : list
        List of tuples representing the asset pairs to trade.
    params : dict
        Dictionary containing strategy parameters for each pair.
    trade_hour : market_hours, optional
        Hours during which trading is allowed.
    """

    def __init__(
        self,
        data: market_data,
        pairs: list[tuple[str, str]],
        params: Dict[Any, Any],
        trade_hour: market_hours = None,
    ):
        self.params = params
        self.data = data
        self.trade_hour = trade_hour
        self.df = data.filter_hours(hours=self.trade_hour).select(["date", "time"])

        self.pairs = pairs

    def update_data(self, data: market_data):
        """
        Updates the internal data and filters by trading hours.

        Parameters
        ----------
        data : market_data
            The new market data.
        """
        self.data = data
        self.df = data.filter_hours(hours=self.trade_hour).select(["date", "time"])

    def compute_beta(
        self,
    ):
        """
        Computes rolling beta for the selected pair using rolling covariance and variance.

        Sets a new column in `self.resampled_df` with the computed beta.
        """
        self.resampled_df = self.resampled_df.with_columns(
            (
                pl.rolling_cov(
                    pl.col(self.p1),
                    pl.col(self.p2),
                    window_size=self.this_param[PARAMS.beta_win],
                )
                / pl.col(self.p2).rolling_var(
                    window_size=self.this_param[PARAMS.beta_win]
                )
            ).alias(f"BETA_{self.p1}_ON_{self.p2}")  # p1 (Y) = b * p2 (X)
        )

    def compute_spread(self):
        """
        Computes the spread between the two assets in a pair.

        Sets a new column in `self.resampled_df` with the computed spread.
        """
        self.resampled_df = self.resampled_df.with_columns(
            (
                pl.col(self.p1)
                - pl.col(self.p2) * pl.col(f"BETA_{self.p1}_ON_{self.p2}")
            ).alias(f"SPREAD_{self.p1}_ON_{self.p2}")  # p1 (Y) = b * p2 (X)
        )

    def compute_z_score(self):
        """
        Computes the z-score of the spread using a rolling window.

        Sets a new column in `self.resampled_df` with the z-score.
        """
        self.resampled_df = self.resampled_df.with_columns(
            (
                (
                    pl.col(f"SPREAD_{self.p1}_ON_{self.p2}")
                    - pl.col(f"SPREAD_{self.p1}_ON_{self.p2}").rolling_mean(
                        window_size=self.this_param[PARAMS.z_win]
                    )
                )
                / pl.col(f"SPREAD_{self.p1}_ON_{self.p2}").rolling_std(
                    window_size=self.this_param[PARAMS.z_win]
                )
            ).alias(f"Z_{self.p1}_ON_{self.p2}")
        )

    def compute_stock_returns(self) -> pl.LazyFrame:
        """
        Computes log returns of all stocks excluding the date and time columns.

        Returns
        -------
        pl.LazyFrame
            LazyFrame containing the log returns of all assets.
        """
        returns_df = self.data.filter_hours(hours=self.trade_hour)
        returns_df = returns_df.with_columns(
            pl.all().exclude(["date", "time"]).log().diff()
        )
        return returns_df

    @staticmethod
    @njit
    def compute_pos(
        Z_arr: np.ndarray,
        beta_arr: np.ndarray,
        n_pairs: int,
        z_entry_arr: np.ndarray,
        z_exit_arr: np.ndarray,
        market_close_flag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute trading signals, positions, and associated betas for each pair.

        Parameters
        ----------
        Z_arr : np.ndarray
            Z-score time series, shape (n_rows, n_pairs).
        beta_arr : np.ndarray
            Estimated beta time series, shape (n_rows, n_pairs).
        n_pairs : int
            Number of trading pairs.
        z_entry_arr : np.ndarray
            Z-score entry thresholds for each pair, shape (n_pairs,).
        z_exit_arr : np.ndarray
            Z-score exit thresholds for each pair, shape (n_pairs,).
        market_close_flag : np.ndarray
            Binary flags indicating when to flatten positions, shape (n_rows,).

        Returns
        -------
        signal_arr : np.ndarray
            Trade signals (-1: short, 0: no signal, 1: long), shape (n_rows, n_pairs).
        pos_arr : np.ndarray
            Open position per pair (-1: short, 0: flat, 1: long), shape (n_rows, n_pairs).
        pos_beta_arr : np.ndarray
            Position beta value when open, 0 otherwise, shape (n_rows, n_pairs).
        """
        n_rows = Z_arr.shape[0]

        signal_arr = np.zeros((n_rows, n_pairs))  # shape: n rows, n pairs
        pos_arr = np.zeros((n_rows, n_pairs))  # shape: n rows, n pairs
        pos_beta_arr = np.zeros((n_rows, n_pairs))  # shape: n rows, n pairs

        for i in range(1, n_rows):
            curr_Z = Z_arr[i]
            prev_pos = pos_arr[i - 1]
            curr_beta = beta_arr[i]

            long_cond = curr_Z > z_entry_arr
            short_cond = curr_Z < -z_entry_arr

            signal_arr[i] = np.where(
                (long_cond) & (~np.isnan(curr_Z)),
                1,
                np.where((short_cond) & (~np.isnan(curr_Z)), -1, 0),
            )

            curr_signal = signal_arr[i]

            exit_long_cond = curr_Z <= z_exit_arr
            exit_short_cond = curr_Z >= -z_exit_arr

            pos_arr[i] = np.where(
                (
                    (
                        # if long and Z reverts below exit and not NA
                        (prev_pos == 1) & (exit_long_cond) & (~np.isnan(curr_Z))
                    )
                    | (
                        # if short and Z reverts above exit and not NA
                        (prev_pos == -1) & (exit_short_cond) & (~np.isnan(curr_Z))
                    )
                    | (market_close_flag[i] == 1)  # market close flatten
                ),
                0,
                np.where(
                    # if no pos and short spread and market not close
                    (prev_pos == 0)
                    & (curr_signal == -1)
                    & (~np.isnan(curr_beta))
                    & (market_close_flag[i] == 0)
                    & ((-2 <= curr_beta) & (curr_beta <= 2)),  # restrict beta
                    -1,
                    np.where(
                        # if no pos and long spread and market not close
                        (prev_pos == 0)
                        & (curr_signal == 1)
                        & (~np.isnan(curr_beta))
                        & (market_close_flag[i] == 0)
                        & ((-2 <= curr_beta) & (curr_beta <= 2)),
                        1,
                        prev_pos,  # unchanged pos
                    ),
                ),
            )

            new_pos = (prev_pos == 0) & (pos_arr[i] != 0)
            close_pos = (prev_pos != 0) & (pos_arr[i] == 0)
            for j in prange(n_pairs):
                if new_pos[j]:
                    # if new long / short then beta
                    pos_beta_arr[i][j] = curr_beta[j]
                elif close_pos[j]:
                    # if exit, beta = 0
                    pos_beta_arr[i][j] = 0
                else:
                    pos_beta_arr[i][j] = pos_beta_arr[i - 1][j]

        return signal_arr, pos_arr, pos_beta_arr

    @staticmethod
    @njit
    def capital_allocation_ret(
        pair_ret_arr: np.ndarray, pos_arr: np.ndarray, n_pairs: int, cost: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute capital allocation and capital over time.

        Parameters
        ----------
        pair_ret_arr : np.ndarray
            Returns per pair, shape (n_rows, n_pairs).
        pos_arr : np.ndarray
            Position matrix (-1, 0, 1), shape (n_rows, n_pairs).
        n_pairs : int
            Number of pairs.
        cost : float
            market impact per entry/exit as a proportion (e.g. 0.001 for 0.1%).

        Returns
        -------
        capital_allocation_arr : np.ndarray
            Capital allocated per pair over time, shape (n_rows, n_pairs).
        remaining_capital : np.ndarray
            Remaining float capital not currently deployed, shape (n_rows,).
        """
        n_rows = pos_arr.shape[0]
        capital_allocation_arr = np.zeros((n_rows, n_pairs))  # shape: n rows, n pairs
        remaining_capital = np.ones(n_rows)

        for i in range(1, n_rows):
            prev_pos = pos_arr[i - 1]
            new_pos = ((prev_pos == 0) & (pos_arr[i] != 0)).astype(np.int32)
            close_pos = ((prev_pos != 0) & (pos_arr[i] == 0)).astype(np.int32)

            this_remaining_capital = remaining_capital[i - 1]

            if np.any(close_pos):
                # if close position, return capital
                for j in prange(n_pairs):
                    if close_pos[j] == 1:
                        this_remaining_capital += (
                            (1 + pair_ret_arr[i][j])
                            * capital_allocation_arr[i - 1][j]
                            * (1 - cost)
                        )
                        capital_allocation_arr[i - 1][j] = 0

            if np.any(new_pos):
                # if multiple signals, then equally divide capital
                deployable_capital = (this_remaining_capital) / sum(new_pos)
                for j in prange(n_pairs):
                    if new_pos[j] == 1:
                        capital_allocation_arr[i - 1][j] = deployable_capital * (
                            1 - cost
                        )
                        this_remaining_capital -= deployable_capital
            # float capital returns
            capital_allocation_arr[i] = (1 + pair_ret_arr[i]) * capital_allocation_arr[
                i - 1
            ]
            remaining_capital[i - 1] = this_remaining_capital
            remaining_capital[i] = this_remaining_capital

        return capital_allocation_arr, remaining_capital

    def generate_backtest_df(self) -> pl.DataFrame:
        """
        Generate a backtest dataframe by computing beta, spread, and z-score for each pair.

        Resamples the raw market data for each pair's trading frequency,
        calculates indicators, and joins them into a unified dataframe.

        Returns
        -------
        pl.DataFrame
            A merged dataframe containing all computed columns across pairs.
        """
        df = self.df
        # each pair has different trading freq, loop and resample to compute beta and Zs
        for pair in self.pairs:
            self.p1, self.p2 = pair[0], pair[1]
            self.resampled_df = self.data.filter(
                resample_freq=self.params[(self.p1, self.p2)][PARAMS.trade_freq],
                hours=self.trade_hour,
            ).select(["date", "time", self.p1, self.p2])
            self.this_param = self.params[(self.p1, self.p2)]
            self.compute_beta()
            self.compute_spread()
            self.compute_z_score()

            df = df.join(
                self.resampled_df.drop([self.p1, self.p2]),
                how="left",
                left_on=["date", "time"],
                right_on=["date", "time"],
            )

            del self.resampled_df
        return df

    def backtest(
        self,
        start: pl.Expr,
        end: pl.Expr,
        cost: float = 0.0005,
    ) -> pl.DataFrame:
        """
        Run the pairs trading backtest using defined strategy parameters and signals.

        Parameters
        ----------
        start : pl.Expr
        Polars expression for start filter (e.g., pl.col("date") >= some_date).
        end : pl.Expr
            Polars expression for end filter (e.g., pl.col("date") <= some_date).
        cost : float, optional
            Transaction cost per entry/exit (default is 0.0005, or 0.05%).

        Returns
        -------
        pl.DataFrame
            The final dataframe containing trades, positions, returns, and capital evolution.
        """
        returns = self.compute_stock_returns().filter(
            pl.col("date").is_between(start, end)
        )

        df = self.generate_backtest_df()

        # market close flag, this is to flatten position and avoid gap risk
        last_ = df.group_by("date", maintain_order=True).last()
        df = df.with_columns(
            pl.when(
                pl.col("date").is_in(last_["date"])
                & pl.col("time").is_in(last_["time"])
            )
            .then(1)
            .otherwise(0)
            .alias("market_close")
        ).filter(pl.col("date").is_between(start, end))

        Z_arr = df.select(
            # select reorders the columns
            [f"Z_{p1}_ON_{p2}" for p1, p2 in self.pairs]
        ).to_numpy()  # shape: n rows, n pairs

        beta_arr = df.select(
            # select reorders the columns
            [f"BETA_{p1}_ON_{p2}" for p1, p2 in self.pairs]
        ).to_numpy()  # shape: n rows, n pairs

        market_close_flag = df.select("market_close").to_numpy().flatten()

        z_entry_arr = np.array(
            [self.params[pair][PARAMS.z_entry] for pair in self.pairs]
        )
        z_exit_arr = np.array([self.params[pair][PARAMS.z_exit] for pair in self.pairs])

        signal_arr, pos_arr, pos_beta_arr = self.compute_pos(
            Z_arr=Z_arr,
            beta_arr=beta_arr,
            n_pairs=len(self.pairs),
            z_entry_arr=z_entry_arr,
            z_exit_arr=z_exit_arr,
            market_close_flag=market_close_flag,
        )
        pos_arr = np.roll(pos_arr, 1, axis=0)  # shift down 1
        pos_arr[0] = 0
        pos_beta_arr = np.roll(pos_beta_arr, 1, axis=0)  # shift down 1
        pos_beta_arr[0] = 0

        signal_df = pl.DataFrame(
            signal_arr, schema=[f"SIGNAL_{p1}_ON_{p2}" for p1, p2 in self.pairs]
        )
        pos_df = pl.DataFrame(
            pos_arr, schema=[f"POS_{p1}_ON_{p2}" for p1, p2 in self.pairs]
        )
        pos_beta_df = pl.DataFrame(
            pos_beta_arr, schema=[f"POS_BETA_{p1}_ON_{p2}" for p1, p2 in self.pairs]
        )

        backtest_df = (
            pl.concat(
                [returns, signal_df, pos_df, pos_beta_df],
                how="horizontal",
            )
            .with_columns(
                *[
                    # y's return
                    pl.when(pl.col(f"POS_{p1}_ON_{p2}") == 1)
                    .then(pl.col(p1))
                    .when(pl.col(f"POS_{p1}_ON_{p2}") == -1)
                    .then(-pl.col(p1))
                    .otherwise(0)
                    .alias(f"{p1}_RET_{p1}_ON_{p2}")
                    for p1, p2 in self.pairs
                ],
                *[
                    # x's return
                    pl.when(pl.col(f"POS_{p1}_ON_{p2}") == 1)
                    .then(pl.col(p2) * -pl.col(f"POS_BETA_{p1}_ON_{p2}"))
                    .when(pl.col(f"POS_{p1}_ON_{p2}") == -1)
                    .then(pl.col(p2) * pl.col(f"POS_BETA_{p1}_ON_{p2}"))
                    .otherwise(0)
                    .alias(f"{p2}_RET_{p1}_ON_{p2}")
                    for p1, p2 in self.pairs
                ],
                *[
                    # pair's return
                    pl.when(pl.col(f"POS_{p1}_ON_{p2}") == 1)
                    .then(pl.col(p1) + pl.col(p2) * -pl.col(f"POS_BETA_{p1}_ON_{p2}"))
                    .when(pl.col(f"POS_{p1}_ON_{p2}") == -1)
                    .then(-pl.col(p1) + pl.col(p2) * pl.col(f"POS_BETA_{p1}_ON_{p2}"))
                    .otherwise(0)
                    .alias(f"{p1}_{p2}_PAIR_RET")
                    for p1, p2 in self.pairs
                ],
            )
            .fill_null(0)
        )

        capital_allocation_arr, remaining_capital = self.capital_allocation_ret(
            pair_ret_arr=backtest_df.select(
                [col for col in backtest_df.columns if "PAIR_RET" in col]
            ).to_numpy(),
            pos_arr=pos_arr,
            n_pairs=len(self.pairs),
            cost=cost,
        )
        backtest_df = pl.concat(
            [
                backtest_df,
                pl.DataFrame(
                    capital_allocation_arr,
                    schema=[f"CAPITAL_{p1}_ON_{p2}" for p1, p2 in self.pairs],
                ),
                pl.DataFrame(remaining_capital, schema=["REMAINING_CAPITAL"]),
            ],
            how="horizontal",
        ).with_columns(
            pl.sum_horizontal(
                [f"CAPITAL_{p1}_ON_{p2}" for p1, p2 in self.pairs]
                + ["REMAINING_CAPITAL"]
            ).alias("CAPITAL")
        )

        return backtest_df
