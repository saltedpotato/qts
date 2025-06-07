
class PairsBacktester:

    @staticmethod
    @njit
    def rolling_beta(x: np.ndarray, y: np.ndarray, beta_window: int) -> np.ndarray:
        """
        calc rolling beta
        used in compute_zscore! dont call it yoself

        Parameters
        ----------
        x : np.ndarray
            time series of asset x.
        y : np.ndarray
            time series of asset y.
        beta_window : int
            number of periods for rolling window.

        Returns
        -------
        np.ndarray
            rolling beta values aligned with x and y.
        """
        n = x.shape[0]
        beta = np.full(n, np.nan)
        if n < beta_window:
            return beta

        stride = x.strides[0]
        x_rolling = as_strided(x, shape=(n - beta_window + 1, beta_window), strides=(stride, stride))
        y_rolling = as_strided(y, shape=(n - beta_window + 1, beta_window), strides=(stride, stride))

        for i in range(n - beta_window + 1):
            x_win = x_rolling[i]
            y_win = y_rolling[i]
            mean_x = x_win.mean()
            mean_y = y_win.mean()
            cov = ((x_win - mean_x) * (y_win - mean_y)).mean()
            var = ((x_win - mean_x) ** 2).mean()
            beta[i + beta_window - 1] = cov / var if var > 0 else np.nan

        return beta

    @staticmethod
    @njit
    def compute_zscore(x: np.ndarray, y: np.ndarray, beta_window: int, zscore_window: int) -> tuple[np.ndarray, np.ndarray]:
        """
        compute spread and zscore

        Parameters
        ----------
        x : np.ndarray
            time series of asset x
        y : np.ndarray
            time series of asset y
        beta_window : int
            rolling window size for computing beta
        zscore_window : int
            rolling window size for zscore computation

        Returns
        -------
        tuple
            zscore : zscore of spread.
            spread : raw spread values.
        """
        n = x.shape[0]
        zscore = np.full(n, np.nan)
        spread = np.full(n, np.nan)

        beta = PairsBacktester.rolling_beta(x, y, beta_window)

        for i in range(n):
            if not np.isnan(beta[i]):
                spread[i] = y[i] - beta[i] * x[i]

        if n < zscore_window:
            return zscore, spread

        stride = spread.strides[0]
        spread_rolling = as_strided(spread, shape=(n - zscore_window + 1, zscore_window), strides=(stride, stride))

        for i in range(n - zscore_window + 1):
            window = spread_rolling[i]
            mean = window.mean()
            std = window.std()
            if std > 0:
                zscore[i + zscore_window - 1] = (spread[i + zscore_window - 1] - mean) / std

        return zscore, spread

    @staticmethod
    @njit
    def check_new_entries(zscores: np.ndarray, positions_tm1: np.ndarray, entry_threshold: float,
                           max_active_pairs: int, method: str = "equal", cash_available: float = 1.0) -> np.ndarray:
        """
        find and allocate capital to new signals based on zscores
        used in backtest_weights_loop !
        

        Parameters
        ----------
        zscores : np.ndarray
            signal array for all pairs.
        positions_tm1 : np.ndarray
            position array from previous time step.
        entry_threshold : float
            entry entry theshold
        max_active_pairs : int
            max number of allowed positions
        method : str
            weighting method. 
            ('equal' or 'yolo')
        cash_available : float
            aamt of cash

        Returns
        -------
        allocations : np.ndarray
            wgts allocated to new entries.
        """
        N = zscores.shape[0]
        allocations = np.zeros(N)

        is_new = (positions_tm1 == 0) & (np.abs(zscores) > entry_threshold)
        num_current_open = np.sum(positions_tm1)
        slots_available = max_active_pairs - num_current_open

        if slots_available <= 0:
            return allocations

        candidate_indices = np.where(is_new)[0]

        if len(candidate_indices) > slots_available:
            ranked_idx = np.argsort(-np.abs(zscores))
            selected_idx = []
            for i in ranked_idx:
                if is_new[i]:
                    selected_idx.append(i)
                    if len(selected_idx) >= slots_available:
                        break
        else:
            selected_idx = candidate_indices

        if method == "equal":
            fixed_alloc = 1.0 / max_active_pairs
            for i in selected_idx:
                allocations[i] = fixed_alloc

        elif method == "yolo":
            if len(selected_idx) > 0:
                allocations[selected_idx[0]] = cash_available

        return allocations

    @staticmethod
    @njit
    def compute_portfolio_weights(price_t: np.ndarray, price_tm1: np.ndarray, weights_tm1: np.ndarray,
                                  positions_t: np.ndarray, new_allocations: np.ndarray) -> tuple[np.ndarray, float]:
        """
        update portf wgts by floating + entries.
        used in backtest_weights_loop !

        Parameters
        ----------
        price_t : np.ndarray
            prices at current time T
        price_tm1 : np.ndarray
            prices at previous time T-1
        weights_tm1 : np.ndarray
            wgts from t-1
        positions_t : np.ndarray
            current open positions
        new_allocations : np.ndarray
            capital to assign if new position

        Returns
        -------
        portfolio_weights : np.ndarray
            portf wgts
        sum_floated_weights : float
            portfolio value
        """
        N = price_t.shape[0]
        floated_weights = np.zeros(N)
        cash_tm1 = weights_tm1[-1]  # extract last element = previous cash

        for i in range(N):
            if positions_t[i] == 1 and weights_tm1[i] > 0:
                ratio = price_t[i] / price_tm1[i]
                floated_weights[i] = weights_tm1[i] * ratio

        floated_weights = floated_weights + new_allocations
        floated_weights = np.append(floated_weights, cash_tm1)
        sum_floated_weights = np.sum(floated_weights)
        portfolio_weights = floated_weights / sum_floated_weights

        return portfolio_weights, sum_floated_weights

    @staticmethod
    @njit
    def backtest_weights_loop(prices: np.ndarray, zscores: np.ndarray, positions: np.ndarray,
                               entry_threshold: float, max_active_pairs: int, method: str = "equal") -> tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        prices : np.ndarray
            T x N price matrix
        zscores : np.ndarray
            T x N zscore signals matrix
        positions : np.ndarray
            T x N array of  positions
        entry_threshold : float
            threshold zscore for new positions
        max_active_pairs : int
            max number of trades allowed
        method : str
            weighting method. 
            ('equal' or 'yolo')

        Returns
        -------
        weights : np.ndarray
            T x (N+1) portfolio weights. RMB LAST COL IS CASH
        portfolio_values : np.ndarray
            portf value at each timestep
        """
        T, N = prices.shape
        weights = np.zeros((T, N + 1))  # include cash column
        portfolio_values = np.zeros(T)

        weights[0, -1] = 1.0
        portfolio_values[0] = 1.0

        for t in range(1, T):
            z_t = zscores[t]
            pos_tm1 = positions[t - 1]
            pos_t = positions[t]
            w_tm1 = weights[t - 1]
            p_tm1 = prices[t - 1]
            p_t = prices[t]

            new_alloc = PairsBacktester.check_new_entries(z_t, pos_tm1, entry_threshold,
                                                          max_active_pairs, method, w_tm1[-1])

            weights[t], portfolio_values[t] = PairsBacktester.compute_portfolio_weights(
                price_t=p_t,
                price_tm1=p_tm1,
                weights_tm1=w_tm1,
                positions_t=pos_t,
                new_allocations=new_alloc
            )

        return weights, portfolio_values

    @staticmethod
    def export_results_to_csv(time_index: np.ndarray, output_path: str = "backtest_output.csv", **data_arrays):
        """
        
        Parameters
        ----------
        time_index : np.ndarray
            array of index timestamps
        output_path : str
            filepath to save the CSV
        **data_arrays : dict
            Keyword arguments of named arrays (e.g., zscore=arr, weights=arr).

        Returns
        -------
        None
            Writes CSV to disk.
        """
        df = pd.DataFrame(index=time_index)

        for name, arr in data_arrays.items():
            if arr.ndim == 1:
                df[name] = arr
            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    df[f"{name}_{i}"] = arr[:, i]
            else:
                raise ValueError(f"Unsupported array dimension for '{name}': {arr.ndim}")

        df.to_csv(output_path)
        print(f"Results exported to {output_path}")
