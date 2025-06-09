import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit

class PairsBacktester():

    @staticmethod
    @njit
    def compute_rolling_zscore(price_array: np.ndarray, 
                               window_size: int
                               ) -> tuple[np.ndarray, np.ndarray]:
        
        """
        Computes the rolling beta, spread, and z-score for multiple asset pairs over time.

        The input `price_array` is expected to have shape (T, 2 * N), where T is the number
        of time steps and N is the number of pairs. Each pair consists of two columns:
        - Even-indexed columns (0, 2, 4, ...) are treated as the long leg (x)
        - Odd-indexed columns (1, 3, 5, ...) are treated as the short leg (y)

        For each pair, the method:
        1. Computes the rolling beta of y with respect to x
        2. Calculates the spread as `y - beta * x`
        3. Computes the rolling z-score of the spread

        Parameters
        ----------
        price_array : np.ndarray
            2D array of shape (n_timesteps, 2 * n_pairs), where each consecutive pair of
            columns corresponds to (x, y) price data for a trading pair.
        
        window_size : int
            Rolling window size used to compute the beta, spread, and z-score.

        Returns
        -------
        spread_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling spread series
            for each pair.
        
        zscore_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling z-scores of
            the spread series for each pair.
        """

        #######################################################################
        #################### STEP 1 : INITIALIZE AND LOOP #####################
        #######################################################################
        
        n_timesteps = price_array.shape[0]
        n_pairs     = price_array.shape[1] // 2
        shape       = (n_timesteps - window_size + 1, window_size)# get shape to pass to as_strided. can be used for all cus i expect all same size
        
        spread_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        zscore_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        
        for j in range(n_pairs):
            x = price_array[:, 2 * j]
            y = price_array[:, 2 * j + 1]
        
            
        #######################################################################
        #################### STEP 2 : COMPUTE ROLLING BETA ####################
        #######################################################################
            
            beta = np.full(n_timesteps, np.nan, dtype = np.float64)
            
            stride      = x.strides[0] # gets number of bytes to go next element
            x_rolling   = as_strided(x, shape = shape, strides = (stride, stride)) # use x, create shape, then move by stride bytes for each element
            
            stride      = y.strides[0]
            y_rolling   = as_strided(y, shape = shape, strides = (stride, stride))
            
            # compute beta for each timestep
            for i in range(n_timesteps - window_size + 1):
                x_window = x_rolling[i]
                y_window = y_rolling[i]
                
                x_mean = np.nanmean(x_window)
                y_mean = np.nanmean(y_window)
                
                cov = np.nanmean((x_window - x_mean) * (y_window - y_mean))
                var = np.nanmean((x_window - x_mean) ** 2)
                
                # insert computed beta into initialized array
                if var > 0:
                    beta[window_size + i - 1] = (cov/var)
        
        #######################################################################
        ################### STEP 3 : COMPUTE ROLLING SPREAD ###################
        #######################################################################
    
            spread = np.full(n_timesteps, np.nan, dtype = np.float64)
            
            #start from window_size - 1 since already np.nan intialized
            for i in range(window_size - 1, n_timesteps): 
                spread[i] = y[i] - (beta[i] - x[i])
                
            spread_matrix[:, j] = spread
        #######################################################################
        ################### STEP 4 : COMPUTE ROLLING Z-SCORE ##################
        #######################################################################
            
            zscore = np.full(n_timesteps, np.nan, dtype = np.float64)
    
            stride          = spread.strides[0]
            spread_rolling  = as_strided(spread, shape = shape, strides = (stride, stride))
        
            for i in range(n_timesteps - window_size + 1):
                spread_window   = spread_rolling[i]
                spread_mean     = np.nanmean(spread_window)
                spread_std      = np.nanstd(spread_window)
                
                if spread_std > 0:
                    # insert computed zscore into initialized array
                    zscore[i + window_size - 1] = (spread[i + window_size - 1] - spread_mean) / np.nanstd(spread_window)
                else:
                    zscore[i + window_size - 1] = 0
        
            zscore_matrix[:, j] = zscore

        return spread_matrix, zscore_matrix ## [2 * window_size - 3:, :] #weiird slice but trust me its to warm-up the signals
    

    @staticmethod
    @njit
    def simulate_portfolio(spread_matrix: np.ndarray,
                           zscore_matrix: np.ndarray,
                           window_size: int,
                           max_positions: int,
                           entry_zscore: float,
                           take_profit_zscore: float,
                           stop_loss_zscore: float,
                           reentry_delay: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates a rule-based long-short pairs trading strategy using z-score signals, stop-loss/take-profit rules, and capital allocation constraints.
        
        This method iterates over time, dynamically assigning positions and rebalancing weights based on pair-specific z-scores and prior portfolio states.
        Capital is allocated to a maximum number of pairs each day, and trade delays are enforced after stop-loss exits.
        
        Parameters
        ----------
        spread_matrix : np.ndarray
            A 2D array of shape (T, N), where T is the number of time steps and N is the number of trading pairs.
            Each column contains the spread values for one pair over time.
        
        zscore_matrix : np.ndarray
            A 2D array of shape (T, N), corresponding to the z-scores of the spread series for each pair at each time step.
        
        window_size : int
            The number of periods used to warm up the strategy (e.g., to wait for z-score calculation stability).
        
        max_positions : int
            Maximum number of simultaneous pair trades allowed in the portfolio at any given time.
        
        entry_zscore : float
            Z-score threshold used to trigger entry signals. Pairs are entered when the absolute z-score exceeds this threshold.
        
        take_profit_zscore : float
            Z-score level at which positions are exited for profit. For long spreads, this is triggered when z-score ≥ -take_profit_zscore.
            For short spreads, it is triggered when z-score ≤ take_profit_zscore.
        
        stop_loss_zscore : float
            Z-score level at which positions are forcefully exited to stop losses. For long spreads, this is triggered when z-score ≤ -stop_loss_zscore.
            For short spreads, it is triggered when z-score ≥ stop_loss_zscore.
        
        reentry_delay : int
            Number of periods to wait before re-entering a trade after a stop-loss has been triggered for that pair.
        
        Returns
        -------
        positions_matrix : np.ndarray
            A 2D array of shape (T, N) with values in {1, 0, -1}. Each row represents portfolio positions:
            1 = long spread (buy A, sell B), -1 = short spread (sell A, buy B), 0 = no position.
        
        weights_matrix : np.ndarray
            A 2D array of shape (T, N + 1). Each row contains portfolio weights:
            the first N elements are weights allocated to pairs, the last element is cash weight.
        
        returns_matrix : np.ndarray
            A 1D array of shape (T,), representing daily portfolio return at each time step,
            computed as the weighted return of all active positions.
        """

        #######################################################################
        ################ STEP 1 : COMPUTE DAILY SPREAD RETURNS ################
        #######################################################################
        
        n_timesteps = spread_matrix.shape[0]
        n_pairs = spread_matrix.shape[1]
        spread_rtn_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        
        # compute spread return by using (T spread / T-1 spread)
        for i in range(window_size, n_timesteps): # window_size because need 2 days to get returns.
            spread_rtn_matrix[i, :] = spread_matrix[i, :] / spread_matrix[i - 1, :]
        
            
        #######################################################################
        ################### STEP 2 : FIGURE OUT PORTF VAL #####################
        #######################################################################

        allocation = 1/max_positions

        # initialize for positions, wgts, rtns tracking
        delay_tracker         = np.full(n_pairs, 0, dtype = int) # to prevent too early re-enter same position
        returns_matrix        = np.full(n_timesteps, np.nan, dtype = np.float64)
        positions_matrix      = np.full((n_timesteps, n_pairs), 0, dtype = int)
        weights_matrix        = np.full((n_timesteps, n_pairs + 1), 0.0, dtype = np.float64) # +1 col for cash
        weights_matrix[:, -1] = 1 # for cash => 100% in cash
        
        
        for i in range(window_size, n_timesteps): # FOR EACH NEW FUCKING TIME STEP
            
            #########################################################################
            ################ STEP 2.1 : INITIALIZATION & PREPERATION ################
            #########################################################################
            
            tm1_weights     = weights_matrix[i - 1, :] # get prev weights
            tm1_positions   = positions_matrix[i - 1, :] # get prev positions
            
            t_weights       = tm1_weights.copy() # get t weights. just copy first
            t_positions     = tm1_positions.copy() # initialize array for T positions. use t-1 as template
            
            tm1_zscore      = zscore_matrix[i - 1, :]

            # CHECK TP/SL TO GET CLEARER PICTURE
            for j in range(n_pairs): # for each pair,
                
                if tm1_positions[j] == 1: # if long position, check tp/sl

                    if tm1_zscore[j] >= -take_profit_zscore: # if COULD HAVE take profit
                        t_positions[j] = 0 # take profit!
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        t_weights[j] = 0.0 # clear out position
                        
                    elif tm1_zscore[j] <= -stop_loss_zscore: # if reach stop loss!
                        t_positions[j] = 0 # stop loss
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        t_weights[j] = 0.0 # clear out position
                        
                        delay_tracker[j] = reentry_delay # CANNOT BUY INSTANTLY NEXT PERIOD

                elif tm1_positions[j] == -1: # if short position, check tp/sl

                    if tm1_zscore[j] <= take_profit_zscore: # if could have taken profit
                        t_positions[j] = 0 # take profit!
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        t_weights[j] = 0.0 # clear out position
                        
                    elif tm1_zscore[j] >= stop_loss_zscore: # if reach stop loss ALREADY DIE LAST PERIOD

                        t_positions[j] = 0 # stop loss
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        t_weights[j] = 0.0 # clear out position
                        
                        delay_tracker[j] = reentry_delay # CANNOT BUY INSTANTLY NEXT PERIOD
            
            # get number of currently open positions, after checking SL/TP but BEFORE new entry in current timestep
            #n_open_positions = np.sum(np.abs(positions_matrix[i - 1, :]) > 0)
            n_open_positions = np.sum(np.abs(t_positions))
            
            # get number of eligible pairs for current timestep
            temp_array = np.full(n_pairs, 0, dtype = int) # temp variable to count new positions
            
            for k in range(n_pairs):# ignore all pairs that recently exited
                if tm1_positions[k] ==0 and np.abs(tm1_zscore[k]) >= entry_zscore and delay_tracker[k] == 0:
                    temp_array[k] = 1
            
            
            n_new_positions = np.sum(temp_array)
            
            #########################################################################
            ##################### STEP 2.2 : MANAGING POSITIONS #####################
            #########################################################################
            
            # NOTHING TO BUY / SELL     
            if n_new_positions == 0:
                pass
                   
            elif n_open_positions >= max_positions: # if too many positions, dont enter

                pass # just proceed to float weights

            # WANT TO TRADE BUT CONSTRAINED
            elif (n_new_positions + n_open_positions) > max_positions: # if EXCEED max limit of open positions

                # gotta rank by zscore cus i aint got no money
                max_new_positions = max_positions - n_open_positions # HOW MANY CAN WE BUY ACTUALLY
                
                temp_array = np.abs(tm1_zscore) # temp variable to rank zscores
                for m in range(n_pairs): # ignore all pairs that recently exited
                    if delay_tracker[m] != 0:
                        temp_array[m] = 0
                    if t_positions[m] != 0:
                        temp_array[m] = 0
                        
                sorted_indices = np.argsort(temp_array)[-max_new_positions:] # get the index of highest zscores
                sorted_indices = sorted_indices[::-1] # reverse the array. before it was weakest zscore to strongest zscore

                #prepare to buy/sell motherfucker
                for a in sorted_indices: # we're restricted. only buy best zscores
                    if tm1_positions[a] == 0 and tm1_zscore[a] >= entry_zscore: # if no active posn + short signal cus spread too wide. dont check delay cus done in temp_array
                        t_positions[a] = -1 # short the spread
                        t_weights[a] = allocation if allocation <= t_weights[-1] else t_weights[-1]
                        t_weights[-1] -= t_weights[a] # remove weight from cash
                
                    elif tm1_positions[a] == 0 and tm1_zscore[a] <= -entry_zscore: # if no active posn + long signal cus spread too narrow
                        t_positions[a] = 1 # long the spread
                        t_weights[a] = allocation if allocation <= t_weights[-1] else t_weights[-1]
                        t_weights[-1] -= t_weights[a] # remove weight from cash



            # YOLO TRADE LETS GO
            else: # means (n_new_positions + n_open_positions) <= max_positions. nothing breaking. within limits

                #prepare to buy/sell motherfucker
                for b in range(n_pairs): # for each pair,
                
                    if tm1_positions[b] == 0 and tm1_zscore[b] >= entry_zscore and delay_tracker[b] == 0: # if no active posn + short signal + not recent sl cus spread too wide

                        t_positions[b] = -1 # short the spread
                        t_weights[b] = allocation if allocation <= t_weights[-1] else t_weights[-1]# ENTER
                        t_weights[-1] -= t_weights[b]# DEDUCT CASH

                        
                    elif tm1_positions[b] == 0 and tm1_zscore[b] <= -entry_zscore and delay_tracker[b] == 0: # if no active posn + long signal + not recent slcus spread too narrow

                        t_positions[b] = 1 # long the spread
                        t_weights[b] = allocation if allocation <= t_weights[-1] else t_weights[-1] # ENTER
                        t_weights[-1] -= t_weights[b] # DEDUCT CASH


            #########################################################################
            #################### STEP 2.3 : COMPUTE WGTS & RTNS #####################
            #########################################################################
            
            daily_returns =  np.sum((spread_rtn_matrix[i, :] - 1) * t_weights[:-1]) # compute t_weights before float!!
            
            t_weights[:-1] = (((1 - spread_rtn_matrix[i, :]) * t_positions) + 1) * t_weights[:-1] # float weights of pairs only! fuck cash

            t_weights = np.round(t_weights, 6)
            total = np.sum(t_weights)
            t_weights /= total
                
            #########################################################################
            #################### STEP 2.x : APPEND TO MAIN MATRIX ###################
            #########################################################################
            # decrease delay_tracker
            
            for d in range(n_pairs):
                if delay_tracker[d] > 0:
                    delay_tracker[d] -= 1
                    
            positions_matrix[i, :] = t_positions
            weights_matrix[i, :] = t_weights
            returns_matrix[i] = daily_returns

        return positions_matrix, weights_matrix, returns_matrix


max_positions = 4
entry_zscore = 2.0
take_profit_zscore = 0.5
stop_loss_zscore = 3
reentry_delay = 30
window_size = 60
price_array = pl.read_parquet()
price_array = price_array['AAL', 'AAPL', 
                          'ADBE', 'ADI', 
                          'ADP', 'ADSK', 
                          'ALGN', 'ALXN', 
                          'AMAT', 'AMGN', 
                          'AMZN', 'ATVI', 
                          'BIIB', 'BKNG']
price_array = price_array.to_numpy()

spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(price_array, window_size = window_size)
positions_matrix, weights_matrix, returns_matrix = PairsBacktester.simulate_portfolio(spread_matrix, zscore_matrix, window_size, max_positions = 2, entry_zscore = 2.0, take_profit_zscore = 0.5, stop_loss_zscore = 3, reentry_delay = 60)
