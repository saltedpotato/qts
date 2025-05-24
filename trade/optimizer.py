import optuna
from optuna.samplers import TPESampler
import polars as pl
import numpy as np

from utils.params import PARAMS
from trade.pairs_trader import PairsTrader


class optimizer:
    def __init__(self, backtester: PairsTrader, pairs, start, end):
        self.backtester = backtester
        self.pairs = pairs
        self.start, self.end = start, end

    def objective(self, trial):
        params = {
            (p1, p2): {
                PARAMS.beta_win: trial.suggest_int(f"{p1}_{p2}_beta_win", 5, 100, 5),
                # PARAMS.beta_freq: "1d",  # Assuming this is fixed as you mentioned
                PARAMS.z_win: trial.suggest_int(f"{p1}_{p2}_z_win", 5, 100, 5),
                PARAMS.z_entry: trial.suggest_float(
                    f"{p1}_{p2}_z_entry", 0, 3.5, step=0.1
                ),
                PARAMS.z_exit: trial.suggest_float(
                    f"{p1}_{p2}_z_exit", -3.5, 0, step=0.1
                ),
                PARAMS.trade_freq: trial.suggest_categorical(
                    f"{p1}_{p2}_trade_freq",
                    ["1m", "3m", "5m", "15m"],
                ),
            }
            for p1, p2 in self.pairs
        }

        self.backtester.params = params
        bt_df = self.backtester.backtest(start=self.start, end=self.end)
        bt_df = (
            bt_df.select("CAPITAL")
            .with_columns(pl.all().pct_change())
            .fill_null(0)
            .to_numpy()
            .flatten()
        )

        sharpe = np.mean(bt_df) / np.std(bt_df) * np.sqrt(len(bt_df) * 252)

        return sharpe

    def optimize(self, n_trials: int = 200):
        sampler = TPESampler(seed=627)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params
