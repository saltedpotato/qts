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
                               beta_window_size: int,
                               zscore_window_size: int
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the rolling beta, spread, and z-score for multiple asset pairs over time.
    
        The input `price_array` is expected to have shape (T, 2 * N), where T is the number
        of time steps and N is the number of pairs. Each pair consists of two columns:
        - Even-indexed columns (0, 2, 4, ...) are treated as the long leg (x)
        - Odd-indexed columns (1, 3, 5, ...) are treated as the short leg (y)
    
        Parameters
        ----------
        price_array : np.ndarray
            2D array of shape (n_timesteps, 2 * n_pairs), where each consecutive pair of
            columns corresponds to (x, y) price data for a trading pair.
            
        beta_window_size : int
            Rolling window size used to compute the hedge ratio beta.
    
        zscore_window_size : int
            Rolling window size used to compute the z-score of the spread.
            
        Returns
        -------
        beta_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling hedge ratios.
        
        spread_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling spreads.
        
        zscore_matrix : np.ndarray
            2D array of shape (n_timesteps, n_pairs) containing the rolling z-scores.
        """
    
        #######################################################################
        #################### STEP 1 : INITIALIZE AND LOOP #####################
        #######################################################################
    
        n_timesteps = price_array.shape[0]
        n_pairs     = price_array.shape[1] // 2
    
        beta_shape   = (n_timesteps - beta_window_size + 1, beta_window_size)
        zscore_shape = (n_timesteps - zscore_window_size + 1, zscore_window_size)
    
        beta_matrix   = np.full((n_timesteps, n_pairs), np.nan, dtype=np.float64)
        spread_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype=np.float64)
        zscore_matrix = np.full((n_timesteps, n_pairs), np.nan, dtype=np.float64)
    
        for j in prange(n_pairs):
            x = price_array[:, 2 * j]
            y = price_array[:, 2 * j + 1]
    
            #######################################################################
            #################### STEP 2 : COMPUTE ROLLING BETA ####################
            #######################################################################
    
            beta = np.full(n_timesteps, np.nan, dtype=np.float64)
    
            stride = x.strides[0]
            x_rolling = as_strided(x, shape=beta_shape, strides=(stride, stride))
    
            stride = y.strides[0]
            y_rolling = as_strided(y, shape=beta_shape, strides=(stride, stride))
    
            for i in prange(n_timesteps - beta_window_size + 1):
                x_window = x_rolling[i]
                y_window = y_rolling[i]
    
                x_mean = np.nanmean(x_window)
                y_mean = np.nanmean(y_window)
    
                cov = np.nanmean((x_window - x_mean) * (y_window - y_mean))
                var = np.nanmean((x_window - x_mean) ** 2)
    
                if var > 0:
                    beta[beta_window_size + i - 1] = cov / var
                else:
                    beta[beta_window_size + i - 1] = beta[beta_window_size + i - 2]  # forward fill
    
            beta_matrix[:, j] = beta
    
            #######################################################################
            ################### STEP 3 : COMPUTE ROLLING SPREAD ###################
            #######################################################################
    
            spread = np.full(n_timesteps, np.nan, dtype=np.float64)
    
            for i in prange(beta_window_size - 1, n_timesteps):
                spread[i] = y[i] - (beta[i] * x[i])
    
            spread_matrix[:, j] = spread
    
            #######################################################################
            ################### STEP 4 : COMPUTE ROLLING Z-SCORE ##################
            #######################################################################
    
            zscore = np.full(n_timesteps, np.nan, dtype=np.float64)
    
            stride = spread.strides[0]
            spread_rolling = as_strided(spread, shape=zscore_shape, strides=(stride, stride))
    
            for i in prange(n_timesteps - zscore_window_size + 1):
                spread_window = spread_rolling[i]
                spread_mean   = np.nanmean(spread_window)
                spread_std    = np.nanstd(spread_window)
    
                if spread_std > 0:
                    zscore[i + zscore_window_size - 1] = (spread[i + zscore_window_size - 1] - spread_mean) / spread_std
                else:
                    zscore[i + zscore_window_size - 1] = 0.0
    
            zscore_matrix[:, j] = zscore
    
        return beta_matrix, spread_matrix, zscore_matrix

    

    @staticmethod
    @njit
    def simulate_portfolio(price_array: np.ndarray,
                           beta_matrix: np.ndarray,
                           spread_matrix: np.ndarray,
                           zscore_matrix: np.ndarray,
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
        
        
        for i in range(1, n_timesteps): # FOR EACH NEW FUCKING TIME STEP
            
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
                if tm1_positions[k] ==0 and np.abs(tm1_zscore[k]) >= entry_zscore and delay_tracker[k] == 0 and abs(tm1_beta[k]) < 1.0:
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
max_positions = 2
entry_zscore = 2.5
take_profit_zscore = 0.5
stop_loss_zscore = 3.0
reentry_delay = 60
beta_window_size = 6000
zscore_window_size = 1000
warmup_size = beta_window_size + zscore_window_size - 2
trading_cost = 0.0005

flattened_list = []
returns_array = np.array([])

main_df = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")

for period in quarters:
    print(period)

    ############### ADF ###############
    adf_df = main_df[(main_df["t"] >= period[0]) & (main_df["t"] < period[1])]
    adf_df = adf_df.set_index('t').resample('1h').last()
    adf_df = adf_df.ffill().dropna(axis = 1).dropna(axis = 0).astype(np.float64)
    
    # Filter price columns
    symbols = [col for col in adf_df.columns if col != "t"]  # exclude timestamp
    adf_array = adf_df[symbols].to_numpy()  # T x N matrix (prices only)
    
    # Create all unique pair index combinations
    pair_indices = np.array(list(combinations(range(len(symbols)), 2)))  # shape (M, 2)
    
    # Run the Numba-parallel ADF proxy
    tau_stats, valid_flags = PairsBacktester.rank_cointegrated_pairs(adf_array, pair_indices)
    
    # Extract top N pairs by most negative tau
    valid_idx = np.where(valid_flags == 1)[0]
    sorted_idx = valid_idx[np.argsort(tau_stats[valid_idx])]

    top_pairs = [pair_indices[i] for i in sorted_idx[:n_coint_pairs]]
    
    # Convert index pairs to ticker symbols and show tau
    top_pairs_named = [(symbols[i], symbols[j], tau_stats[idx]) for idx, (i, j) in zip(sorted_idx[:n_coint_pairs], top_pairs)]
    top_pair_tuples = [(i, j) for i, j, _ in top_pairs_named]
    
    # if first period no pairs yet. 
    if flattened_list == []:
        flattened_list = [item for tup in top_pair_tuples for item in tup]
        continue
    
    mask = (main_df["t"] >= period[0]) & (main_df["t"] < period[1])
    date_range_indices = main_df[mask].index
    start_idx = max(0, date_range_indices[0] - warmup_size)
    end_idx = date_range_indices[-1] + 1  # inclusive
    price_array = main_df.iloc[start_idx:end_idx]

    price_array = price_array[flattened_list]
    price_array = price_array.to_numpy()
    
    beta_matrix, spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(price_array, beta_window_size = beta_window_size, zscore_window_size = zscore_window_size)
    
    positions_matrix, weights_matrix, returns_matrix = PairsBacktester.simulate_portfolio(price_array[warmup_size:], beta_matrix[warmup_size:], spread_matrix[warmup_size:], zscore_matrix[warmup_size:], 
                                                                                          max_positions = max_positions, entry_zscore = entry_zscore, 
                                                                                          take_profit_zscore = take_profit_zscore, stop_loss_zscore = stop_loss_zscore, 
                                                                                          reentry_delay = reentry_delay, trading_cost = trading_cost)
    
    returns_array = np.concatenate((returns_array, returns_matrix))
    
    flattened_list = [item for tup in top_pair_tuples for item in tup]

    
import matplotlib.pyplot as plt
plt.plot(np.cumprod(1+returns_array[1:]))
plt.show()

abc = '''
[I 2025-06-22 12:09:16,818] Trial 192 finished with value: 1.8997156811319735 
and parameters: {'n_coint_pairs': 10, 'max_positions': 1, 'entry_zscore': 1.5, 'take_profit_zscore': 0.0, 'stop_loss_zscore': 3.0, 'reentry_delay': 60, 'beta_window_size': 6500, 'zscore_window_size': 8000}. 
Best is trial 192 with value: 1.8997156811319735.
'''

############################################# OPTUNA EDMUND METHOD #############################################


import optuna

# === Load full data ===
main_df = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")

# === Define objective function factory ===
def make_objective(main_df, prev_pairs, prev_symbols, current_period):
    def objective(trial):
        n_coint_pairs = 10
        max_positions = trial.suggest_int("max_positions", 1, 4)
        entry_zscore = trial.suggest_categorical("entry_zscore", [1.5, 2.0, 2.5, 3.0])
        take_profit_zscore = trial.suggest_categorical("take_profit_zscore", [0.0, 0.5])
        stop_loss_zscore = trial.suggest_categorical("stop_loss_zscore", [3.0, 3.5, 4.0])
        reentry_delay = trial.suggest_categorical("reentry_delay", [60, 120, 180, 240])
        beta_window_size = trial.suggest_int("beta_window_size", 100, 8000, step=100)
        zscore_window_size = trial.suggest_int("zscore_window_size", 100, 8000, step=100)
        trading_cost = 0.0005
        warmup_size = beta_window_size + zscore_window_size - 2

        mask = (main_df["t"] >= current_period[0]) & (main_df["t"] < current_period[1])
        date_range_indices = main_df[mask].index
        if len(date_range_indices) == 0:
            return -999.0

        start_idx = max(0, date_range_indices[0] - warmup_size)
        end_idx = date_range_indices[-1] + 1
        price_array = main_df.iloc[start_idx:end_idx][prev_symbols].to_numpy()

        beta_matrix, spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(
            price_array, beta_window_size=beta_window_size, zscore_window_size=zscore_window_size
        )

        try:
            _, _, returns_matrix = PairsBacktester.simulate_portfolio(
                price_array[warmup_size:], beta_matrix[warmup_size:], spread_matrix[warmup_size:], zscore_matrix[warmup_size:],
                max_positions=max_positions, entry_zscore=entry_zscore,
                take_profit_zscore=take_profit_zscore, stop_loss_zscore=stop_loss_zscore,
                reentry_delay=reentry_delay, trading_cost=trading_cost
            )
        except:
            return -999.0

        returns_array = returns_matrix[~np.isnan(returns_matrix)]
        if len(returns_array) < 2:
            return -999.0

        total_return = np.prod(1 + returns_array[1:]) - 1
        trial.set_user_attr("total_return", total_return)

        #sharpe = returns_array.mean() / (returns_array.std() + 1e-8)
        #annualized_sharpe = sharpe * np.sqrt(252 * 390)
        return total_return

    return objective


# === Load and prepare data ===
main_df = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")

# Define quarterly periods
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
results = []

for i in range(1, len(quarters)):  # start from Q2
    prev_period = quarters[i - 1]
    current_period = quarters[i]

    print(f"\n🔍 Optimizing Q{i+1} using Q{i} pairs: {prev_period[0].date()} → {prev_period[1].date()}")

    # --- Step 1: get pairs from previous quarter ---
    adf_df = main_df[(main_df["t"] >= prev_period[0]) & (main_df["t"] < prev_period[1])]
    adf_df = adf_df.set_index("t").resample("1h").last().ffill().dropna(axis=1).astype(np.float64)

    symbols = adf_df.columns.tolist()
    adf_array = adf_df[symbols].to_numpy()
    pair_indices = np.array(list(combinations(range(len(symbols)), 2)))

    tau_stats, valid_flags = PairsBacktester.rank_cointegrated_pairs(adf_array, pair_indices)
    valid_idx = np.where(valid_flags == 1)[0]
    sorted_idx = valid_idx[np.argsort(tau_stats[valid_idx])]
    top_pairs = [pair_indices[i] for i in sorted_idx[:20]]

    used_symbols = sorted(set(i for pair in top_pairs for i in pair))
    prev_symbols = [symbols[i] for i in used_symbols]
    prev_pairs = [(symbols[i], symbols[j]) for i, j in top_pairs]

    # --- Step 2: optimize current quarter using prev_pairs ---
    objective = make_objective(main_df, prev_pairs, prev_symbols, current_period)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best = {
        "quarter": f"{current_period[0].date()} to {current_period[1].date()}",
        "sharpe": study.best_value,
        "total_return": study.best_trial.user_attrs.get("total_return", np.nan),
        "params": study.best_params
    }
    results.append(best)
    print(f"✅ Best Sharpe: {best['sharpe']:.4f} | Total Return: {best['total_return']:.4%}")

# === Dump as DataFrame ===
results_df = pd.DataFrame(results)
results_df.to_csv("optuna_quarter_results.csv", index=False)
print("\n🎯 Optimization complete. Results saved to 'optuna_quarter_results.csv'")

plt.plot(np.cumprod(1+results_df["total_return"]))
plt.show()

optuna_returns = results_df["total_return"].to_numpy()
returns_array = returns_array[~np.isnan(returns_array)]
bt_returns = np.cumprod(1 + returns_array[1:]) - 1
quarters = len(optuna_returns)

for i in range(quarters):
    r_optuna = optuna_returns[i]
    r_bt = bt_returns[i]

    print(f"Q{i+2}: optuna={r_optuna:.4%}, backtest={r_bt:.4%}, diff={abs(r_bt - r_optuna):.4%}")

##############################################################################



combined_configs = [
    # Q1 – for pair selection only (no backtest)
    {"quarter_start": datetime(2020, 6, 1), "quarter_end": datetime(2020, 9, 1), "config": None},

    # Q2 – Q20
    {"quarter_start": datetime(2020, 9, 1), "quarter_end": datetime(2020, 12, 1),
     "max_positions": 2, "entry_zscore": 2.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.0,
     "reentry_delay": 180, "beta_window_size": 2200, "zscore_window_size": 700},

    {"quarter_start": datetime(2020, 12, 1), "quarter_end": datetime(2021, 3, 1),
     "max_positions": 1, "entry_zscore": 1.5, "take_profit_zscore": 0.0, "stop_loss_zscore": 4.0,
     "reentry_delay": 60, "beta_window_size": 500, "zscore_window_size": 3700},

    {"quarter_start": datetime(2021, 3, 1), "quarter_end": datetime(2021, 6, 1),
     "max_positions": 2, "entry_zscore": 2.0, "take_profit_zscore": 0.0, "stop_loss_zscore": 4.0,
     "reentry_delay": 240, "beta_window_size": 3100, "zscore_window_size": 3700},

    {"quarter_start": datetime(2021, 6, 1), "quarter_end": datetime(2021, 9, 1),
     "max_positions": 2, "entry_zscore": 2.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.0,
     "reentry_delay": 120, "beta_window_size": 6200, "zscore_window_size": 5100},

    {"quarter_start": datetime(2021, 9, 1), "quarter_end": datetime(2021, 12, 1),
     "max_positions": 4, "entry_zscore": 3.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 4.0,
     "reentry_delay": 120, "beta_window_size": 6700, "zscore_window_size": 6900},

    {"quarter_start": datetime(2021, 12, 1), "quarter_end": datetime(2022, 3, 1),
     "max_positions": 1, "entry_zscore": 2.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 4.0,
     "reentry_delay": 120, "beta_window_size": 6500, "zscore_window_size": 1800},

    {"quarter_start": datetime(2022, 3, 1), "quarter_end": datetime(2022, 6, 1),
     "max_positions": 1, "entry_zscore": 2.0, "take_profit_zscore": 0.0, "stop_loss_zscore": 3.5,
     "reentry_delay": 120, "beta_window_size": 4100, "zscore_window_size": 6800},

    {"quarter_start": datetime(2022, 6, 1), "quarter_end": datetime(2022, 9, 1),
     "max_positions": 1, "entry_zscore": 1.5, "take_profit_zscore": 0.5, "stop_loss_zscore": 4.0,
     "reentry_delay": 60, "beta_window_size": 3000, "zscore_window_size": 5700},

    {"quarter_start": datetime(2022, 9, 1), "quarter_end": datetime(2022, 12, 1),
     "max_positions": 2, "entry_zscore": 3.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.5,
     "reentry_delay": 180, "beta_window_size": 5100, "zscore_window_size": 200},

    {"quarter_start": datetime(2022, 12, 1), "quarter_end": datetime(2023, 3, 1),
     "max_positions": 3, "entry_zscore": 2.5, "take_profit_zscore": 0.0, "stop_loss_zscore": 4.0,
     "reentry_delay": 120, "beta_window_size": 5400, "zscore_window_size": 3800},

    {"quarter_start": datetime(2023, 3, 1), "quarter_end": datetime(2023, 6, 1),
     "max_positions": 1, "entry_zscore": 2.0, "take_profit_zscore": 0.0, "stop_loss_zscore": 3.0,
     "reentry_delay": 240, "beta_window_size": 7500, "zscore_window_size": 2500},

    {"quarter_start": datetime(2023, 6, 1), "quarter_end": datetime(2023, 9, 1),
     "max_positions": 1, "entry_zscore": 2.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 4.0,
     "reentry_delay": 120, "beta_window_size": 4900, "zscore_window_size": 4800},

    {"quarter_start": datetime(2023, 9, 1), "quarter_end": datetime(2023, 12, 1),
     "max_positions": 1, "entry_zscore": 1.5, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.0,
     "reentry_delay": 240, "beta_window_size": 3900, "zscore_window_size": 6400},

    {"quarter_start": datetime(2023, 12, 1), "quarter_end": datetime(2024, 3, 1),
     "max_positions": 2, "entry_zscore": 2.0, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.0,
     "reentry_delay": 60, "beta_window_size": 1900, "zscore_window_size": 400},

    {"quarter_start": datetime(2024, 3, 1), "quarter_end": datetime(2024, 6, 1),
     "max_positions": 3, "entry_zscore": 1.5, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.0,
     "reentry_delay": 60, "beta_window_size": 5200, "zscore_window_size": 2500},

    {"quarter_start": datetime(2024, 6, 1), "quarter_end": datetime(2024, 9, 1),
     "max_positions": 1, "entry_zscore": 1.5, "take_profit_zscore": 0.0, "stop_loss_zscore": 3.5,
     "reentry_delay": 120, "beta_window_size": 4800, "zscore_window_size": 5400},

    {"quarter_start": datetime(2024, 9, 1), "quarter_end": datetime(2024, 12, 1),
     "max_positions": 4, "entry_zscore": 2.5, "take_profit_zscore": 0.0, "stop_loss_zscore": 3.0,
     "reentry_delay": 240, "beta_window_size": 3700, "zscore_window_size": 6700},

    {"quarter_start": datetime(2024, 12, 1), "quarter_end": datetime(2025, 3, 1),
     "max_positions": 2, "entry_zscore": 1.5, "take_profit_zscore": 0.5, "stop_loss_zscore": 3.5,
     "reentry_delay": 120, "beta_window_size": 2400, "zscore_window_size": 700},

    {"quarter_start": datetime(2025, 3, 1), "quarter_end": datetime(2025, 6, 1),
     "max_positions": 2, "entry_zscore": 1.5, "take_profit_zscore": 0.0, "stop_loss_zscore": 4.0,
     "reentry_delay": 120, "beta_window_size": 3100, "zscore_window_size": 2600}
]


flattened_list = []
returns_array = np.array([])

main_df = pd.read_parquet(r"C:\Users\Zeke\Desktop\Data Hoarder\Polygon\full_qqq.parquet")

for i in range(1, len(combined_configs)):  # start from Q2
    prev_cfg = combined_configs[i - 1]
    curr_cfg = combined_configs[i]

    q_start = curr_cfg["quarter_start"]
    q_end = curr_cfg["quarter_end"]

    print(f"\n=== Backtesting {q_start.date()} to {q_end.date()} ===")

    # === Step 1: ADF pair selection using *previous quarter*
    adf_df = main_df[(main_df["t"] >= prev_cfg["quarter_start"]) & (main_df["t"] < prev_cfg["quarter_end"])]
    adf_df = adf_df.set_index("t").resample("1h").last()
    adf_df = adf_df.ffill().dropna(axis=1).dropna(axis=0).astype(np.float64)

    symbols = adf_df.columns.tolist()
    adf_array = adf_df.to_numpy()
    pair_indices = np.array(list(combinations(range(len(symbols)), 2)))

    tau_stats, valid_flags = PairsBacktester.rank_cointegrated_pairs(adf_array, pair_indices)
    valid_idx = np.where(valid_flags == 1)[0]
    sorted_idx = valid_idx[np.argsort(tau_stats[valid_idx])]

    n_coint_pairs = 10
    top_pairs = [pair_indices[i] for i in sorted_idx[:n_coint_pairs]]
    top_pair_tuples = [(symbols[i], symbols[j]) for (i, j) in top_pairs]

    # flatten unique symbols
    flattened_list = list(dict.fromkeys([item for tup in top_pair_tuples for item in tup]))

    # === Step 2: Load backtest window
    warmup_size = curr_cfg["beta_window_size"] + curr_cfg["zscore_window_size"] - 2
    mask = (main_df["t"] >= q_start) & (main_df["t"] < q_end)
    date_range_indices = main_df[mask].index
    start_idx = max(0, date_range_indices[0] - warmup_size)
    end_idx = date_range_indices[-1] + 1

    price_array = main_df.iloc[start_idx:end_idx][flattened_list].to_numpy()

    # === Step 3: Run rolling calculations
    beta_matrix, spread_matrix, zscore_matrix = PairsBacktester.compute_rolling_zscore(
        price_array,
        beta_window_size=curr_cfg["beta_window_size"],
        zscore_window_size=curr_cfg["zscore_window_size"]
    )

    # === Step 4: Simulate strategy
    positions_matrix, weights_matrix, returns_matrix = PairsBacktester.simulate_portfolio(
        price_array[warmup_size:],
        beta_matrix[warmup_size:],
        spread_matrix[warmup_size:],
        zscore_matrix[warmup_size:],
        max_positions=curr_cfg["max_positions"],
        entry_zscore=curr_cfg["entry_zscore"],
        take_profit_zscore=curr_cfg["take_profit_zscore"],
        stop_loss_zscore=curr_cfg["stop_loss_zscore"],
        reentry_delay=curr_cfg["reentry_delay"],
        trading_cost=0.0004
    )

    # === Step 5: Store results
    returns_array = np.concatenate((returns_array, returns_matrix))

returns_array = returns_array[~np.isnan(returns_array)]

import matplotlib.pyplot as plt
plt.plot(np.cumprod(1+returns_array[1:]))
plt.show()


returns_df = pd.DataFrame({
    "run_1": returns_array_1,#0
    "run_2": returns_array_2,#1
    "run_3": returns_array_3,#2
    "run_4": returns_array_4,#3
    "run_5": returns_array_5,#4
    "run_6": returns_array_6#5
})

returns_df.to_csv("rtns different costs.csv")
