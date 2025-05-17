
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



    def calc_beta_series(self, stonk1, stonk2):
        """
        CALS THE BETA FOR LOOKBACK PERIOD BEFORE CURRENT DAY NOT LOOK AHEAD BIAS
        RETURNS A PD SERIES 
        """
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
        
    def get_output(self):
        return self.df

backtester = PairsTrader(df, pairs, lookback_period=7, beta_freq='1D')
backtester.calc_beta_and_spread()
result_df = backtester.get_output()
