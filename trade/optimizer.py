import optuna
from optuna.samplers import TPESampler
import optuna.study.study
import polars as pl
import numpy as np
from typing import Any, Dict

from utils.params import PARAMS

from pairs_finding.pairs_identification import cointegration_pairs
from trade.pairs_trader import PairsTrader


class optimizer:
    """
    Hyperparameter optimizer for a pairs trading strategy using Optuna.

    Parameters
    ----------
    backtester : PairsTrader
        The backtesting engine used to evaluate strategy performance.
    find_pairs : cointegration_pairs
        The object responsible for identifying cointegrated pairs.
    start : pl.Expr
        Polars expression for start filter (e.g., pl.col("date") >= some_date).
    end : pl.Expr
        Polars expression for end filter (e.g., pl.col("date") <= some_date).
    """

    def __init__(
        self,
        backtester: PairsTrader,
        find_pairs: cointegration_pairs,
        start: pl.Expr,
        end: pl.Expr,
    ):
        self.backtester = backtester
        self.find_pairs = find_pairs
        self.start, self.end = start, end

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function to optimize with Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object that suggests hyperparameter values.

        Returns
        -------
        float
            The Sharpe ratio resulting from the backtest using suggested parameters.
        """
        trade_n_pairs = trial.suggest_int("pairs_to_trade", 1, 5, 1)
        pairs = self.find_pairs.get_top_pairs(n=trade_n_pairs)
        self.backtester.pairs = pairs

        params = {
            (p1, p2): {
                PARAMS.beta_win: trial.suggest_int(
                    f"{p1}_{p2}_beta_win", 10, 500, step=5
                ),
                # PARAMS.beta_freq: "1d",  # Assuming this is fixed as you mentioned
                PARAMS.z_win_mean: trial.suggest_int(
                    f"{p1}_{p2}_z_win_mean", 10, 500, step=5
                ),
                PARAMS.z_win_std: trial.suggest_int(
                    f"{p1}_{p2}_z_win_std", 10, 500, step=5
                ),
                PARAMS.z_entry: trial.suggest_float(
                    f"{p1}_{p2}_z_entry", 1, 3.5, step=0.1
                ),
                PARAMS.z_exit: trial.suggest_float(
                    f"{p1}_{p2}_z_exit", -3.5, -1, step=0.1
                ),
                PARAMS.trade_freq: trial.suggest_categorical(
                    f"{p1}_{p2}_trade_freq",
                    [str(i) + "m" for i in range(1, 15)],
                ),
                # PARAMS.stop_loss: trial.suggest_float(
                #     f"{p1}_{p2}_stop_loss", 0.001, 0.01, step=0.001
                # ),
            }
            for p1, p2 in pairs
        }

        self.backtester.params = params
        bt_df = self.backtester.backtest(
            start=self.start,
            end=self.end,
            cost=0.0005,
            stop_loss=None 
            # np.array(
            #     [params[(p1, p2)][PARAMS.stop_loss] for p1, p2 in pairs]
            # ),
        )
        bt_df = (
            bt_df.select("CAPITAL")
            .with_columns(pl.all().pct_change())
            .fill_null(0)
            .to_numpy()
            .flatten()
        )

        sharpe = np.nanmean(bt_df) / np.nanstd(bt_df) * np.sqrt(390 * 252)  # min level

        return sharpe

    def optimize(self, n_trials: int = 200) -> optuna.study.study:
        """
        Runs Optuna optimization over the defined search space.

        Parameters
        ----------
        n_trials : int, optional
            Number of trials to run, by default 200.

        Returns
        -------
        optuna.study.study
            study object
        """
        sampler = TPESampler(seed=621)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        return study
