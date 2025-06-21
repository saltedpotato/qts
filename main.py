from datetime import datetime
import polars as pl
import pandas as pd

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange

from itertools import combinations

# note! need to ensure to get (2xwindow_size - 2) more datapoints so it neatly dropna into start of quarter
# drop 1 LESS for returns calculation


# to do - after compute_rolling_zscore, trim x and y (should be [2 * window_size - 3:])
# to do - need to drop first row after compute_positions?
# !!!! REMOVE WINDOW_SIZE WILL NOT NEED AFTERWARDS IN COMPUTE_POSITIONS !!!!!!
class PairsBacktester():
    
    @staticmethod
    @njit(parallel = True)
    def rank_cointegrated_pairs(price_data: np.ndarray,
                                pair_indices: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate ADF-like test statistics in parallel for multiple stock pairs.
    
        This function approximates cointegration by regressing the first difference of the spread
        on the lagged spread for each pair and extracting the autoregressive coefficient (tau).
        More negative tau implies stronger mean-reversion.
    
        Parameters
        ----------
        price_data : np.ndarray
            A 2D array of shape (T, N) where T is the number of time points and N is the number of assets.
            Each column represents a time series of one asset's prices.
        pair_indices : np.ndarray
            A 2D array of shape (M, 2), where each row contains the (i, j) index pair for the price_data columns to be tested.
    
        Returns
        -------
        tau_stats : np.ndarray
            Array of shape (M,) containing the estimated AR(1) coefficients from the ADF-like test for each pair.
            More negative values suggest stronger evidence of cointegration.
        valid_flags : np.ndarray
            Binary array of shape (M,), where 1 indicates a valid stat (no NaNs encountered), 0 means skipped.
        """
    
        # M = number of pairs, T = number of time periods
        n_pairs     = pair_indices.shape[0]
        n_timesteps = price_data.shape[0]
    
        # initialize output arrays
        tau_stats   = np.full(n_pairs, 999.0)     # use 999.0 as a placeholder for "invalid"
        valid_flags = np.zeros(n_pairs)         # 0 = invalid, 1 = valid
    
        # loop through each pair in parallel
        for k in prange(n_pairs):
            
            i, j = pair_indices[k]              # get indices for this pair
            x    = price_data[:, i]                # time series for asset i
            y    = price_data[:, j]                # time series for asset j
    
            # skip any pair that contains NaNs
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
    
            # compute hedge ratio using simple OLS: beta = cov(x, y) / var(x)
            mean_x = x.mean()
            mean_y = y.mean()
    
            cov = np.mean((x - mean_x) * (y - mean_y))      # sample covariance
            var = np.mean((x - mean_x) ** 2)                # sample variance
    
            # fallback if variance is zero (flat series)
            beta = cov / var if var > 0 else 1.0
    
            # construct spread: y - beta * x
            spread = y - beta * x
    
            # compute first difference of spread (dy)
            difference = spread[1:] - spread[:-1]
    
            # lagged spread (y_{t-1})
            y_lag = spread[:-1]
    
            # build regression matrix X = [y_lag, 1]
            X       = np.empty((n_timesteps - 1, 2))
            X[:, 0] = y_lag                   # lagged spread
            X[:, 1] = 1.0                     # constant intercept term
    
            # manually compute beta = (X'X)^-1 X'y using closed-form 2x2 inversion
            Xt  = X.T                         # transpose X: shape (2, T-1)
            XtX = Xt @ X                     # X'X: shape (2, 2)
            Xty = Xt @ difference            # X'y: shape (2,)
    
            # compute determinant of XtX for inversion
            det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
    
            if np.abs(det) > 1e-8:           # make sure matrix is invertible
                inv_XtX = np.empty((2, 2))   # initialize inverse matrix
    
                # 2x2 matrix inversion
                inv_XtX[0, 0] = XtX[1, 1] / det
                inv_XtX[1, 1] = XtX[0, 0] / det
                inv_XtX[0, 1] = -XtX[0, 1] / det
                inv_XtX[1, 0] = -XtX[1, 0] / det
    
                # estimated OLS betas = [tau, intercept]
                beta_ols = inv_XtX @ Xty
                tau      = beta_ols[0]            # only care about coefficient of y_{t-1}
            else:
                tau = 999.0                  # fallback for singular matrix
    
            # store result
            tau_stats[k]   = tau
            valid_flags[k] = 1               # mark as valid
    
        return tau_stats, valid_flags

    
    @staticmethod
    @njit
    def compute_rolling_zscore(price_array: np.ndarray, 
                               window_size: int
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        beta_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling hedge ratios (betas)
            used to neutralize x vs y for each pair.
        
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
        
        beta_matrix   = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        spread_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        zscore_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        
        for j in prange(n_pairs):
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
            for i in prange(n_timesteps - window_size + 1):
                x_window = x_rolling[i]
                y_window = y_rolling[i]
                
                x_mean = np.nanmean(x_window)
                y_mean = np.nanmean(y_window)
                
                cov = np.nanmean((x_window - x_mean) * (y_window - y_mean))
                var = np.nanmean((x_window - x_mean) ** 2)
                
                # insert computed beta into initialized array
                if var > 0:
                    beta[window_size + i - 1] = (cov/var)
                else:
                    beta[window_size + i - 1] = beta[window_size + i - 2] # forward fill beta
            
            beta_matrix[:, j] = beta
            
        #######################################################################
        ################### STEP 3 : COMPUTE ROLLING SPREAD ###################
        #######################################################################
    
            spread = np.full(n_timesteps, np.nan, dtype = np.float64)
            
            #start from window_size - 1 since already np.nan intialized
            for i in prange(window_size - 1, n_timesteps): 
                spread[i] = y[i] - (beta[i] * x[i])
                
            spread_matrix[:, j] = spread
            
        #######################################################################
        ################### STEP 4 : COMPUTE ROLLING Z-SCORE ##################
        #######################################################################
            
            zscore = np.full(n_timesteps, np.nan, dtype = np.float64)
    
            stride          = spread.strides[0]
            spread_rolling  = as_strided(spread, shape = shape, strides = (stride, stride))
        
            for i in prange(n_timesteps - window_size + 1):
                spread_window   = spread_rolling[i]
                spread_mean     = np.nanmean(spread_window)
                spread_std      = np.nanstd(spread_window)
                
                if spread_std > 0:
                    # insert computed zscore into initialized array
                    zscore[i + window_size - 1] = (spread[i + window_size - 1] - spread_mean) / np.nanstd(spread_window)
                else:
                    zscore[i + window_size - 1] = 0
        
            zscore_matrix[:, j] = zscore

        return beta_matrix, spread_matrix, zscore_matrix ## [2 * window_size - 3:, :] #weiird slice but trust me its to warm-up the signals
    

    @staticmethod
    @njit
    def simulate_portfolio(price_array: np.ndarray,
                            beta_matrix: np.ndarray,
                            spread_matrix: np.ndarray,
                            zscore_matrix: np.ndarray,
                            window_size: int,
                            max_positions: int,
                            entry_zscore: float,
                            take_profit_zscore: float,
                            stop_loss_zscore: float,
                            reentry_delay: int,
                            trading_cost: float,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates a rule-based long-short pairs trading strategy using z-score signals,
        stop-loss/take-profit rules, and capital allocation constraints.
        
        This method iterates over time, dynamically assigning positions and rebalancing weights
        based on pair-specific z-scores and prior portfolio states.
        Capital is allocated to a maximum number of pairs each day, and trade delays are enforced
        after stop-loss exits.

        Parameters
        ----------
        price_array : np.ndarray
            2D array of shape (n_timesteps, 2 * n_pairs), where each consecutive pair of
            columns corresponds to (x, y) price data for a trading pair.
            
        beta_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs), where each value is the rolling hedge ratio
            from regressing y on x in a pair (i.e., y ~ x).
            
        spread_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs), with the residual (spread) between
            y and beta * x for each pair.

        zscore_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs), containing standardized z-scores
            of the spread for each pair and time step.

        window_size : int
            The number of periods used to warm up the strategy (e.g., to wait for z-score
            calculation stability).

        max_positions : int
            Maximum number of simultaneous pair trades allowed in the portfolio at any given time.

        entry_zscore : float
            Z-score threshold used to trigger entry signals. Pairs are entered when the absolute
            z-score exceeds this threshold:
            - zscore >= entry_zscore triggers a short spread (sell A, buy B)
            - zscore <= -entry_zscore triggers a long spread (buy A, sell B)

        take_profit_zscore : float
            Z-score level at which positions are exited for profit:
            - Long spread: zscore ≥ -take_profit_zscore
            - Short spread: zscore ≤ take_profit_zscore

        stop_loss_zscore : float
            Z-score level at which positions are forcefully exited to stop losses:
            - Long spread: zscore ≤ -stop_loss_zscore
            - Short spread: zscore ≥ stop_loss_zscore

        reentry_delay : int
            Number of periods to wait before re-entering a trade after a stop-loss
            has been triggered for that pair.

        Returns
        -------
        positions_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) with values in {1, 0, -1}. Each row represents portfolio positions:
            1 = long spread (buy A, sell B), -1 = short spread (sell A, buy B), 0 = no position.

        weights_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs + 1). Each row contains portfolio weights:
            the first n_pairs elements are weights allocated to active positions, and the final
            column represents cash weight.

        returns_matrix : np.ndarray
            1D array of shape (n_timesteps,), representing the daily portfolio return at each time step,
            computed as the weighted return of all active positions.
        """

        #######################################################################
        #################### STEP 1 : COMPUTE DAILY RETURNS ###################
        #######################################################################
        
        n_timesteps    = spread_matrix.shape[0]
        n_pairs        = spread_matrix.shape[1]

        x_price_matrix = np.empty((n_timesteps, n_pairs), dtype = np.float64)
        y_price_matrix = np.empty((n_timesteps, n_pairs), dtype = np.float64)
        
        for col in prange(n_pairs):
            x_price_matrix[:, col] = price_array[:, 2 * col]
            y_price_matrix[:, col] = price_array[:, 2 * col + 1]

        #spread_rtn_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype = np.float64)
        
        #compute spread return by using (T spread / T-1 spread)
        #for i in range(window_size, n_timesteps): # window_size because need 2 days to get returns.
        #    spread_rtn_matrix[i, :] = spread_matrix[i, :] / spread_matrix[i - 1, :]
        
        #######################################################################
        ################### STEP 2 : FIGURE OUT PORTF VAL #####################
        #######################################################################

        allocation            = 1 / max_positions

        # initialize for positions, wgts, rtns tracking
        delay_tracker         = np.full(n_pairs, 0, dtype = int) # to prevent too early re-enter same position
        beta_tracker          = np.full(n_pairs, 0.0, dtype = np.float64) # track entry beta
        
        returns_matrix        = np.full(n_timesteps, np.nan, dtype = np.float64) # storage matrix of returns
        positions_matrix      = np.full((n_timesteps, n_pairs), 0, dtype = int) # storage matrix of positions
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
            
            tm1_beta        = beta_matrix[i - 1, :]
            tm1_zscore      = zscore_matrix[i - 1, :]

            tm1_x_price     = x_price_matrix[i - 1, :]
            tm1_y_price     = y_price_matrix[i - 1, :]
            
            t_x_price       = x_price_matrix[i, :]
            t_y_price       = y_price_matrix[i, :]
            
            # CHECK TP/SL TO GET CLEARER PICTURE
            for j in range(n_pairs): # for each pair,
                
                if tm1_positions[j] == 1: # if long position, check tp/sl

                    if tm1_zscore[j] >= -take_profit_zscore: # if COULD HAVE take profit
                        t_positions[j] = 0 # take profit!
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        
                        t_weights[j] = 0.0 # clear out position
                        beta_tracker[j] = 0.0 # clear beta cache

                        tm1_y_price[j] *= (1 - trading_cost) # short Y at lower price
                        tm1_x_price[j] *= (1 + trading_cost) # long X at higher price
                        
                    elif tm1_zscore[j] <= -stop_loss_zscore: # if reach stop loss!
                        t_positions[j] = 0 # stop loss
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        
                        t_weights[j] = 0.0 # clear out position
                        beta_tracker[j] = 0.0 # clear beta cache
                        delay_tracker[j] = reentry_delay # CANNOT BUY INSTANTLY NEXT PERIOD
                        
                        tm1_y_price[j] *= (1 - trading_cost) # short Y at lower price
                        tm1_x_price[j] *= (1 + trading_cost) # long X at higher price
                        
                elif tm1_positions[j] == -1: # if short position, check tp/sl

                    if tm1_zscore[j] <= take_profit_zscore: # if could have taken profit
                        t_positions[j] = 0 # take profit!
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        
                        t_weights[j] = 0.0 # clear out position
                        beta_tracker[j] = 0.0 # clear beta cache

                        tm1_y_price[j] *= (1 + trading_cost) # buy Y at higher price
                        tm1_x_price[j] *= (1 - trading_cost) # short X at lower price
                        
                    elif tm1_zscore[j] >= stop_loss_zscore: # if reach stop loss ALREADY DIE LAST PERIOD

                        t_positions[j] = 0 # stop loss
                        t_weights[-1] += tm1_weights[j] # assume sell at prev close, add back to cash
                        
                        t_weights[j] = 0.0 # clear out position
                        beta_tracker[j] = 0.0 # clear beta cache
                        delay_tracker[j] = reentry_delay # CANNOT BUY INSTANTLY NEXT PERIOD
                        
                        tm1_y_price[j] *= (1 + trading_cost) # buy Y at higher price
                        tm1_x_price[j] *= (1 - trading_cost) # short X at lower price
                        
            
            # get number of currently open positions, after checking SL/TP but BEFORE new entry in current timestep
            #n_open_positions = np.sum(np.abs(positions_matrix[i - 1, :]) > 0)
            n_open_positions = np.sum(np.abs(t_positions))
            
            # get number of eligible pairs for current timestep
            temp_array = np.full(n_pairs, 0, dtype = int) # temp variable to count new positions
            
            for k in range(n_pairs):# ignore all pairs that recently exited
                if tm1_positions[k] ==0 and np.abs(tm1_zscore[k]) >= entry_zscore and delay_tracker[k] == 0 and abs(tm1_beta[k]) < 2:
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
                for m in prange(n_pairs): # ignore all pairs that recently exited
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
                        beta_tracker[a] = tm1_beta[a] # track entry beta
                        
                        tm1_y_price[a] *= (1 - trading_cost) # short Y at lower price
                        tm1_x_price[a] *= (1 + trading_cost) # long X at higher price
                        
                    elif tm1_positions[a] == 0 and tm1_zscore[a] <= -entry_zscore: # if no active posn + long signal cus spread too narrow
                        t_positions[a] = 1 # long the spread
                        t_weights[a] = allocation if allocation <= t_weights[-1] else t_weights[-1]
                        t_weights[-1] -= t_weights[a] # remove weight from cash
                        beta_tracker[a] = tm1_beta[a] # track entry beta

                        tm1_y_price[a] *= (1 + trading_cost) # buy Y at higher price
                        tm1_x_price[a] *= (1 - trading_cost) # short X at lower price
                        
            # YOLO TRADE LETS GO
            else: # means (n_new_positions + n_open_positions) <= max_positions. nothing breaking. within limits

                #prepare to buy/sell motherfucker
                for b in range(n_pairs): # for each pair,
                
                    if tm1_positions[b] == 0 and tm1_zscore[b] >= entry_zscore and delay_tracker[b] == 0: # if no active posn + short signal + not recent sl cus spread too wide

                        t_positions[b] = -1 # short the spread
                        t_weights[b] = allocation if allocation <= t_weights[-1] else t_weights[-1]# ENTER
                        t_weights[-1] -= t_weights[b]# DEDUCT CASH
                        beta_tracker[b] = tm1_beta[b] # track entry beta
                        
                        tm1_y_price[b] *= (1 - trading_cost) # short Y at lower price
                        tm1_x_price[b] *= (1 + trading_cost) # long X at higher price
                        
                    elif tm1_positions[b] == 0 and tm1_zscore[b] <= -entry_zscore and delay_tracker[b] == 0: # if no active posn + long signal + not recent slcus spread too narrow

                        t_positions[b] = 1 # long the spread
                        t_weights[b] = allocation if allocation <= t_weights[-1] else t_weights[-1] # ENTER
                        t_weights[-1] -= t_weights[b] # DEDUCT CASH
                        beta_tracker[b] = tm1_beta[b] # track entry beta

                        tm1_y_price[b] *= (1 + trading_cost) # buy Y at higher price
                        tm1_x_price[b] *= (1 - trading_cost) # short X at lower price
                        
            #########################################################################
            #################### STEP 2.3 : COMPUTE WGTS & RTNS #####################
            #########################################################################

            tm1_spread_price = (tm1_y_price - (beta_tracker * tm1_x_price))
            t_spread_price = (t_y_price - (beta_tracker * t_x_price))
            net_spread_price = (tm1_y_price + np.abs(beta_tracker * tm1_x_price))
            
            returns_array = t_positions * ((t_spread_price - tm1_spread_price)/net_spread_price)
            daily_returns =  np.sum(returns_array * t_weights[:-1]) # compute t_weights before float!!

            #np.sum((spread_rtn_matrix[i, :] - 1) * t_weights[:-1]) # compute t_weights before float!!

            t_weights[:-1] = (returns_array + 1) * t_weights[:-1] # float weights of pairs only! fuck cash

            t_weights = np.round(t_weights, 6)
            total = np.sum(t_weights)
            t_weights /= total

            #########################################################################
            #################### STEP 2.4 : APPEND TO MAIN MATRIX ###################
            #########################################################################
            # decrease delay_tracker
            
            for d in range(n_pairs):
                if delay_tracker[d] > 0:
                    delay_tracker[d] -= 1
                    
            positions_matrix[i, :] = t_positions
            weights_matrix[i, :] = t_weights
            returns_matrix[i] = daily_returns

        return positions_matrix, weights_matrix, returns_matrix



    
quarters = [
    (datetime(2020, 6, 1), datetime(2020, 9, 1)),
    (datetime(2020, 9, 1), datetime(2020, 12, 1)),
    (datetime(2020, 12, 1), datetime(2021, 3, 1)), 
    (datetime(2021, 3, 1), datetime(2021, 6, 1)),
    (datetime(2021, 6, 1), datetime(2021, 9, 1)),
    (datetime(2021, 9, 1), datetime(2021, 12, 1)),
    (datetime(2021, 12, 1), datetime(2022, 3, 1)),
    (datetime(2022, 3, 1), datetime(2022, 6, 1)),
    (datetime(2022, 6, 1), datetime(2022, 9, 1)),
    (datetime(2022, 9, 1), datetime(2022, 12, 1)),
    (datetime(2022, 12, 1), datetime(2023, 3, 1)),
    (datetime(2023, 3, 1), datetime(2023, 6, 1)),
    (datetime(2023, 6, 1), datetime(2023, 9, 1)),
    (datetime(2023, 9, 1), datetime(2023, 12, 1)),
    (datetime(2023, 12, 1), datetime(2024, 3, 1)),
    (datetime(2024, 3, 1), datetime(2024, 6, 1)),
    (datetime(2024, 6, 1), datetime(2024, 9, 1)),
    (datetime(2024, 9, 1), datetime(2024, 12, 1)),
    (datetime(2024, 12, 1), datetime(2025, 3, 1)),
    (datetime(2025, 3, 1), datetime(2025, 6, 1))
    ]


n_coint_pairs = 10
max_positions = 4
entry_zscore = 2.5
take_profit_zscore = 0.5
stop_loss_zscore = 3.0
reentry_delay = 10
window_size = 10
trading_cost = 0.0001

flattened_list = []
returns_array = np.array([])

for period in quarters:
    print(period)
    
    # slice the file to qtr
    main_df = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")
    main_df = main_df[(main_df["t"] >= period[0]) & (main_df["t"] < period[1])]
    main_df = main_df.drop(columns=['t'])
    main_df = main_df.ffill().dropna(axis = 1).dropna(axis = 0).astype(np.float64)
    
    # Filter price columns
    symbols = [col for col in main_df.columns if col != "t"]  # exclude timestamp
    price_array = main_df[symbols].to_numpy()  # T x N matrix (prices only)
    
    # Create all unique pair index combinations
    pair_indices = np.array(list(combinations(range(len(symbols)), 2)))  # shape (M, 2)
    
    # Run the Numba-parallel ADF proxy
    tau_stats, valid_flags = PairsBacktester.rank_cointegrated_pairs(price_array, pair_indices)
    
    # Extract top N pairs by most negative tau
    valid_idx = np.where(valid_flags == 1)[0]
    sorted_idx = valid_idx[np.argsort(tau_stats[valid_idx])]

    top_pairs = [pair_indices[i] for i in sorted_idx[:n_coint_pairs]]
    
    # Convert index pairs to ticker symbols and show tau
    top_pairs_named = [(symbols[i], symbols[j], tau_stats[idx]) for idx, (i, j) in zip(sorted_idx[:n_coint_pairs], top_pairs)]
    top_pair_tuples = [(i, j) for i, j, _ in top_pairs_named]
    
    if flattened_list == []:
        flattened_list = [item for tup in top_pair_tuples for item in tup]
        continue
    
    flattened_list = [item for tup in top_pair_tuples for item in tup]
    
    main_df = main_df[flattened_list]
    main_df = main_df.ffill().dropna(axis = 1).dropna(axis = 0).astype(np.float64)
    
    price_array = main_df.to_numpy()
    del main_df
    
    beta_matrix, spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(price_array, window_size = window_size)
    positions_matrix, weights_matrix, returns_matrix = PairsBacktester.simulate_portfolio(price_array, beta_matrix, spread_matrix, zscore_matrix, window_size, 
                                                                                          max_positions = max_positions, entry_zscore = entry_zscore, 
                                                                                          take_profit_zscore = take_profit_zscore, stop_loss_zscore = stop_loss_zscore, 
                                                                                          reentry_delay = reentry_delay, trading_cost = trading_cost)
    
    returns_array = np.concatenate((returns_array, returns_matrix))
    # Convert to DataFrame
    df = pd.DataFrame(returns_array)
    
    # Save to Parquet
    #df.to_parquet("my_array2.parquet", index=False)
    
import matplotlib.pyplot as plt
plt.plot(np.cumprod(1+returns_matrix[60:]))
plt.show()

#############

c = Clustering(df=main_df.select(pl.all().exclude(["t"])))
c.run_clustering(method=Clustering_methods.agnes, min_clusters=2, max_clusters=5)
    
find_pairs = cointegration_pairs(
    df = main_df.select(pl.all().exclude(["date", "time"])),
    p_val_cutoff=0.01,
    cluster_pairs=c.cluster_pairs,
)
find_pairs.identify_pairs()
pairs = find_pairs.get_top_pairs()
flattened_list = [item for tup in pairs for item in tup]

############################ fake data 1
import polars as pl
import numpy as np




price_array = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")

price_array = price_array[["AAPL", "GOOG"]]

price_array = price_array.to_numpy()

########################### fake data 2

# Create sample datetime index
p = 30
ts_event = pd.date_range("2023-01-01", periods=p, freq="D")

data = {
    "BKNG": np.random.uniform(2000, 2100, p),
    "DASH": np.random.uniform(100, 110, p),
    "CRWD": np.random.uniform(150, 160, p),
    "FTNT": np.random.uniform(50, 55, p),
    "AAA": np.random.uniform(150, 160, p),
    "BB": np.random.uniform(50, 55, p),
    "SS": np.random.uniform(150, 160, p),
    "XX": np.random.uniform(50, 55, p),
    "FF": np.random.uniform(150, 160, p),
    "QWE": np.random.uniform(50, 55, p),
}

df = pd.DataFrame(data, index=ts_event)
df.index.name = "ts_event"
price_array = df.to_numpy()

window_size = 60
beta_matrix, spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(price_array, window_size = window_size)
positions_matrix, weights_matrix, returns_matrix = PairsBacktester.simulate_portfolio(price_array, beta_matrix, spread_matrix, zscore_matrix, window_size, 
                                                                                      max_positions = max_positions, entry_zscore = entry_zscore, 
                                                                                      take_profit_zscore = take_profit_zscore, stop_loss_zscore = stop_loss_zscore, 
                                                                                      reentry_delay = reentry_delay, trading_cost = trading_cost)
#aa = aa[2 * window_size - 3:, :]
#bb = bb[2 * window_size - 3:, :]

import matplotlib.pyplot as plt
plt.plot(np.cumprod(1+returns_matrix[60:]))

plt.show()


#test = pd.read_json(r"C:\Users\Zeke\Downloads\j.json")
