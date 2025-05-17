
class PairsTrader:
    def __init__(self, df, pairs, lookback_period = 7, beta_freq = '1D'):
        """
        INPUTS TO THIS BAD ASS MTFKER
        df = wide df like roh's usual. cols names r symbols
        pairs = from edmunds pair output is list of tuples [('CRWD', 'DXCM'), ('AMD', 'IDXX')]
        lookback_period for # of days for the OLS beta calculations
        beta_freq = frequency to use for beta calc. just leave as 1D
        """
        self.df = df.copy()
        self.pairs = pairs
        self.lookback_period = lookback_period
        self.beta_freq = beta_freq
        self.df.index = pd.to_datetime(self.df.index)
        self.capital = 1000


    def calc_beta_series(self, stonk1, stonk2):
        """
        CALS THE BETA FOR LOOKBACK PERIOD BEFORE CURRENT DAY NOT LOOK AHEAD BIAS
        RETURNS A PD SERIES 
        """
        
        ## RESAMPLE TO THE TIMEFRAME YOU WANT. DEFAULT 1 DAY OK JUST ADD MORE DAYS
        resampled = df[[stonk1, stonk2]].resample(self.beta_freq).last().dropna() # DROP THEM WEEKENDS
        beta_series = pd.Series(index = resampled.index)
        
        for current_time in resampled.index:
            window_start = current_time - timedelta(days = self.lookback_period)
            window_data = resampled.loc[(resampled.index >= window_start) & (resampled.index < current_time)] # LESS THAN SO NO LOOK AHEAD BIAS BITCH

            if window_data.empty:
                continue

            days_covered = (window_data.index.max() - window_data.index.min()).days

            if days_covered < self.lookback_period - 1:
                continue
            
            cov = window_data[stonk1].cov(window_data[stonk2])
            var = window_data[stonk2].var()
            
            if var == 0 or np.isnan(cov) or np.isnan(var):
                continue
    
            beta_series[current_time] = cov / var
    
        return beta_series.ffill().dropna()

    def merge_beta_df_time(self, beta_series):
        """
        CREATES THEM TIMESERIES IN BETA SERIES TO ALIGN WITH MAIN DF
        """
        
        beta_reindexed = beta_series.reindex(self.df.index, method = "ffill")
        
        return beta_reindexed

    def calc_beta_and_spread(self):
        """
        FOR EACH TUPLE OF STONK PAIRS, COMPUTE HEDGE RATIO AND SPREAD AND APPEND TO THE MAIN DF
        """
        output_cols = []
        
        for stonk1, stonk2 in self.pairs:
            
            beta_series = self.calc_beta_series(stonk1, stonk2)
            beta_reindexed = self.merge_beta_df_time(beta_series)

            # fuk this idk how to label the pairs tgt just gonna do this
            beta_col = "beta_" + str(stonk1) + "_" + str(stonk2)
            spread_col = "spread_" + str(stonk1) + "_" + str(stonk2)

            self.df[beta_col] = beta_reindexed
            self.df[spread_col] = self.df[stonk1] - beta_reindexed * self.df[stonk2]

            output_cols.extend([stonk1, stonk2, beta_col, spread_col])
            
        output_cols = list(set(output_cols))
        
        self.df = self.df[output_cols].dropna() # drop all periods where not warmed up

    def calc_zscore_signals(self, z_window  = 60, entry_threshold = 2.0, exit_threshold = 0.5):
        """
        CALCULATES Z-SCORE OF SPREAD FOR EACH PAIR AND GENERATES ENTRY/EXIT SIGNALS
        
        z_window: how many time periods to use for rolling mean/std
        entry_threshold & exit_threshold: z-score to enter and exit 
        
        """
        
        for stonk1, stonk2 in self.pairs:
            spread_col = "spread_" + str(stonk1) + "_" + str(stonk2)
            z_col = "zscore_" + str(stonk1) + "_" + str(stonk2)
            signal_col = "signal_" + str(stonk1) + "_" + str(stonk2)
    
            spread = self.df[spread_col]
    
            rolling_mean = spread.rolling(window=z_window).mean()
            rolling_std = spread.rolling(window=z_window).std()
    
            zscore = (spread - rolling_mean) / rolling_std
            self.df[z_col] = zscore
    
            # 🔥 Vectorized signal logic using np.where
            signal_values = np.where(
                zscore > entry_threshold, -1,
                np.where(zscore < -entry_threshold, 1,
                         np.where(zscore.abs() < exit_threshold, 0, np.nan))
            )
    
            signal = pd.Series(signal_values, index=self.df.index).ffill().fillna(0)
            self.df[signal_col] = signal


    
    def simulate_trades(self, 
                        trading_cost = 1.0, 
                        borrow_rate = 0.0001, 
                        stop_loss_threshold = -0.10):
 
        cash = self.capital
        have_position = False
        trade_log = []
        open_trade = {}
        position_col = []
        daily_pnl_col = []
        cash_col = []
    
        # Compute returns input as cols
        unique_tickers = list({s for pair in self.pairs for s in pair})
        returns = self.df[unique_tickers].pct_change()
    
        for ticker in unique_tickers:
            self.df[f"ret_{ticker}"] = returns[ticker]
    
        # start backtesting
        for idx in range(1, len(self.df)):
            if have_position:
                
                # need to find yolo pair and get their retns
                row = self.df.iloc[idx]
                prev_row = self.df.iloc[idx - 1]
    
                open_trade = open_trade
                stonk1 = open_trade["stonk1"]
                stonk2 = open_trade["stonk2"]
                direction = open_trade["direction"]
                
                
                r1 = row[f"ret_{stonk1}"]
                r2 = row[f"ret_{stonk2}"]
                
                # need to flip returns depending on long/short directions
                # edmund might wanna acct for borrowing cost
                if direction == 1:
                    short_price = prev_row[stonk2]
                    long_leg_return = r1
                    short_leg_return = r2
                    
                else:
                    short_price = prev_row[stonk1]
                    long_leg_return = r2
                    short_leg_return = r1
    
                short_cost = borrow_rate * short_price
                notional = cash
                pnl = (long_leg_return - short_leg_return - short_cost) * notional
                cash += pnl
                
                #cache this shit
                open_trade["log"].append({
                    "time": self.df.index[idx],
                    "pnl": pnl,
                    "cash": cash
                })
    
                position_col.append(1)
                daily_pnl_col.append(pnl)
                cash_col.append(cash)
    
                # check stop losses 
                total_trade_pnl = 0.0
                
                for log_entry in open_trade["log"]:
                    total_trade_pnl += log_entry["pnl"]
    
                if total_trade_pnl < stop_loss_threshold * cash:
                    have_position = False
                    open_trade["exit_time"] = self.df.index[idx]
                    open_trade["reason"] = "stop_loss"
                    trade_log.append(open_trade)
                    open_trade = {}
                
                # check if signal no more
                elif row[open_trade["signal_col"]] == 0:
                    have_position = False
                    open_trade["exit_time"] = self.df.index[idx]
                    open_trade["reason"] = "mean_reversion"
                    trade_log.append(open_trade)
                    open_trade = {}
    
            # if not current position
            else:
    
                prev_row = self.df.iloc[idx - 1]
                position_col.append(0)
                daily_pnl_col.append(0.0)
                cash_col.append(cash)
    
                # fk what if more than 1 signal?
                candidate_entries = []
    
                # compare zscores
                for pair in self.pairs:
                    stonk1 = pair[0]
                    stonk2 = pair[1]
                    signal_col = f"signal_{stonk1}_{stonk2}"
                    zscore_col = f"zscore_{stonk1}_{stonk2}"  
    
                    signal = prev_row[signal_col]
                    
                    #combined the zscores
                    if signal != 0:
                        
                        zscore = prev_row[zscore_col]
                        candidate_entries.append({
                                                "pair": (stonk1, stonk2),
                                                "signal": signal,
                                                "zscore": abs(zscore),
                                                "zscore_raw": zscore,
                                                "signal_col": signal_col,
                                                "beta_col": f"beta_{stonk1}_{stonk2}"
                                                })
    
                # choose best pair all other pairs suck
                if candidate_entries:
                    candidate_entries.sort(key=lambda x: x["zscore"], reverse=True)
                    top_pick = candidate_entries[0]
    
                    open_trade = {
                        "entry_time": self.df.index[idx],
                        "stonk1": top_pick["pair"][0],
                        "stonk2": top_pick["pair"][1],
                        "beta_col": top_pick["beta_col"],
                        "signal_col": top_pick["signal_col"],
                        "direction": top_pick["signal"],
                        "log": []
                    }
    
                    have_position = True
                    cash -= trading_cost
    
        # Finalize tracking columns
        idx_range = self.df.index[:len(position_col) + 1]
        self.df["position"] = pd.Series([0] + position_col, index=idx_range).ffill().astype(int)
        self.df["strategy_pnl"] = pd.Series([0.0] + daily_pnl_col, index=idx_range)
        self.df["strategy_cash"] = pd.Series([cash] + cash_col, index=idx_range)
        self.trade_log = trade_log



    def get_output(self):
        return self.df

backtester = PairsTrader(df, pairs)
backtester.calc_beta_and_spread()
backtester.calc_zscore_signals()
backtester.simulate_trades()
df_results = backtester.df
trades = backtester.trade_log
