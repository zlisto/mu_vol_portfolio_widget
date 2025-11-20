"""
Trade Simulator Module

This module provides comprehensive tools for backtesting trading strategies with realistic
cost modeling and performance metrics. It includes helper functions for signal processing,
risk metrics calculation, and visualization of trading results.

Key Features:
- Robust z-score calculation using median/MAD
- Signal persistence filtering
- Realistic backtesting with fees and slippage
- Performance metrics (Sharpe ratio, max drawdown, annualized returns)
- Price data loading from parquet files
- Equity curve visualization

Author: Trading Strategy Development
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import glob

# -----------------------------
# Classes
# -----------------------------


@dataclass
class Params:
    """
    Configuration parameters for backtesting and strategy execution.
    
    This dataclass contains all the key parameters needed for running backtests,
    including signal processing settings, trading costs, and time calibration.
    
    Attributes:
        persistence (int): Number of consecutive bars required for signal generation.
                          Defaults to 1 (no persistence filter).
        fee_bps (float): Trading fee per side in basis points (1 bps = 0.01%).
                        Defaults to 5.0 bps.
        slip_bps (float): Slippage per side in basis points. Defaults to 0.0 bps.
        bars_per_year (float): Number of bars per year for annualization.
                              Defaults to 525,600 (1-minute data: 365*24*60).
    
    Note:
        All cost parameters are applied per trade side (entry/exit), not round-trip.
    """
    # Smoothing    

    # Sampling / calendar
    bars_per_year: float = 365.0 * 24.0 * 60.0  # if 1-minute data
    window_days: int = 1
    window_vol_days: int = 10
    hold_hours: int = 12
    trade_fee: float = 0.0005
    tau_mu: float = 1
    tau_sigma: float = 1
    take_profit: float = 0.05
    stop_loss: float = -0.05
    leverage: float = 1

class TradeSimulator:
    """
    A comprehensive trading simulator for backtesting strategies with realistic costs.
    
    This class provides methods for signal processing, backtesting, performance analysis,
    and visualization of trading strategies. It handles realistic trading costs including
    fees and slippage, and provides comprehensive performance metrics.
    
    Attributes:
        params (Params): Configuration parameters for backtesting and signal processing
    """
    
    def __init__(self, params: Params = None):
        """
        Initialize the TradeSimulator with configuration parameters.
        
        Args:
            params (Params, optional): Configuration parameters. If None, uses default Params().
        """
        self.params = params if params is not None else Params()
    # -----------------------------
    # -----------------------------
    def list_all_pairs(self, folder: str = "data/binance/klein/1m/") -> list:
        """
        List all pairs in the data/binance/klein/1m/ folder.
        """
        import os
        pairs = os.listdir(folder)
        pairs = [pair.split("-1m")[0] for pair in pairs]
        pairs = list(set(pairs))
        return pairs

    def find_pair_file(self, pair_name: str) -> str:
        """
        Find the parquet file for a given pair name in the data/binance/klein/1m/ folder.
        
        Args:
            pair_name (str): Trading pair name (e.g., 'DOGEUSDT')
        
        Returns:
            str: Full filepath to the parquet file, or None if not found
        """
        import os
        # Construct the search pattern using os.path.join for cross-platform compatibility
        search_pattern = os.path.join("data", "binance", "klein", "1m", f"{pair_name}-1m-*.parquet")
        
        # Find matching files
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            # Return the first match (should be only one)
            return matching_files[0]
        else:
            print(f"No file found for pair: {pair_name}")
            return None

    def load_price(self, pair: str) -> pd.DataFrame:
        """
        Load price data for a single trading pair from parquet file.
        
        Args:
            pair (str): Trading pair name (e.g., 'BTCUSDT')
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and datetime index
        
        Raises:
            FileNotFoundError: If the parquet file for the pair cannot be found
            Exception: If there are issues reading or processing the parquet file
        """
        filepath = self.find_pair_file(pair)
        if filepath is None:
            raise FileNotFoundError(f"Could not find data file for {pair}")
        
        try:
            df = pd.read_parquet(filepath)
            df.index = pd.to_datetime(df.open_time, unit='ms')
            df = df[~df.index.isna()]    
            return df
        except Exception as e:
            raise Exception(f"Error loading price data for {pair}: {str(e)}")

    def load_prices(self, pairs: list) -> pd.DataFrame:
        """
        Load price data for multiple trading pairs from parquet files.
        
        This function loads historical price data for a list of trading pairs from
        the data/binance/klein/1m/ directory. It processes the data to extract
        close prices and align them by timestamp.
        
        Args:
            pairs (list): List of trading pair names (e.g., ['BTCUSDT', 'ETHUSDT'])
        
        Returns:
            pd.DataFrame: DataFrame with trading pairs as columns and timestamps as index.
                        Contains close prices for each pair.
        
        Raises:
            FileNotFoundError: If any pair file cannot be found
            Exception: If there are issues loading or processing any pair data
        
        Note:
            - Expects parquet files in data/binance/klein/1m/ directory
            - Converts open_time from milliseconds to datetime index
            - Filters out rows with invalid timestamps
            - Returns only close prices as float type
        """
        dict_list = []
        missing_pairs = []
        
        for pair_name in pairs:
            try:
                df = self.load_price(pair_name)
                price = df["close"].astype(float)
                price.name = pair_name
                dict_list.append(price)
            except FileNotFoundError:
                missing_pairs.append(pair_name)
            except Exception as e:
                raise Exception(f"Error processing {pair_name}: {str(e)}")
        
        if missing_pairs:
            raise FileNotFoundError(f"Could not find data files for pairs: {missing_pairs}")
        
        if not dict_list:
            raise ValueError("No valid price data loaded")
            
        df_prices = pd.concat(dict_list, axis=1)
        return df_prices


    
    def robust_z(self, x: pd.Series, window: int, eps: float = 1e-12) -> pd.Series:
        """
        Calculate robust z-score using mean and interquartile range (IQR).
        
        This function computes a robust version of the z-score that is less sensitive
        to outliers compared to the traditional mean/std z-score. The IQR is converted
        to standard deviation using the factor 1.4826.
        
        Args:
            x (pd.Series): Input time series data
            window (int): Rolling window size for mean and IQR calculation
            eps (float, optional): Small epsilon value to avoid division by zero. Defaults to 1e-12.
        
        Returns:
            pd.Series: Robust z-score series with same index as input
        
        Raises:
            ValueError: If window is not positive or x is empty
        
        Note:
            IQR to sigma conversion factor: 1.4826 (for normal distribution)
        """
        if window <= 0:
            raise ValueError("Window size must be positive")
        if x.empty:
            raise ValueError("Input series cannot be empty")
        mu = x.rolling(window).mean()
        q25 = x.rolling(window).quantile(0.25)
        q75 = x.rolling(window).quantile(0.75)
        iqr = q75 - q25
        sigma = iqr / 1.4826
        return (x - mu) / (sigma.replace(0, np.nan) + eps)

    def apply_persistence(self, sig: pd.Series, n: int = None) -> pd.Series:
        """
        Apply persistence filter to require n consecutive bars of the same nonzero sign.
        
        This function filters trading signals by requiring a minimum number of consecutive
        bars with the same sign before generating a signal. This helps reduce noise and
        false signals in trading strategies.
        
        Args:
            sig (pd.Series): Input signal series (can contain positive, negative, or zero values)
            n (int, optional): Minimum number of consecutive bars required for signal generation.
                              If None, uses self.params.persistence. Defaults to None.
        
        Returns:
            pd.Series: Filtered signal series with persistence requirement applied
        
        Raises:
            ValueError: If n is not positive or sig is empty
        
        Example:
            If n=3 and input is [0, 1, 1, 1, -1, -1, 1], output would be [0, 0, 0, 1, 0, 0, 0]
        """
        if n is not None and n <= 0:
            raise ValueError("Persistence value must be positive")
        if sig.empty:
            raise ValueError("Input signal series cannot be empty")
        if n is None:
            n = self.params.persistence
            
        v = np.sign(sig.fillna(0.0).values)
        out = np.zeros_like(v)
        run = 0
        last = 0
        for i, s in enumerate(v):
            if s == 0:
                run = 0; last = 0; out[i] = 0
            else:
                if s == last:
                    run += 1
                else:
                    run = 1; last = int(s)
                out[i] = s if run >= n else 0
        return pd.Series(out, index=sig.index, dtype=float)

    def annualized_sharpe(self, r: pd.Series, bars_per_year: float = None) -> float:
        """
        Calculate annualized Sharpe ratio from return series.
        
        The Sharpe ratio measures risk-adjusted returns by dividing the excess return
        (mean return) by the standard deviation of returns, then annualizing the result.
        
        Args:
            r (pd.Series): Return series (log returns or simple returns)
            bars_per_year (float, optional): Number of bars per year for annualization.
                                            If None, uses self.params.bars_per_year. Defaults to None.
        
        Returns:
            float: Annualized Sharpe ratio, or NaN if std is zero or NaN
        
        Raises:
            ValueError: If bars_per_year is not positive or r is empty
        
        Note:
            Assumes risk-free rate is zero. For non-zero risk-free rate, subtract it
            from the mean return before calculation.
        """
        if r.empty:
            raise ValueError("Return series cannot be empty")
        if bars_per_year is not None and bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")
        if bars_per_year is None:
            bars_per_year = self.params.bars_per_year
            
        r = r.dropna()
        if r.std() == 0 or np.isnan(r.std()):
            return np.nan
        return (r.mean() / r.std()) * np.sqrt(bars_per_year)

    def annualized_returns(self, r: pd.Series, bars_per_year: float = None) -> float:
        """
        Calculate annualized returns from return series.
        
        This function compounds the average return over the specified number of periods
        per year to get the annualized return rate.
        
        Args:
            r (pd.Series): Return series (log returns or simple returns)
            bars_per_year (float, optional): Number of bars per year for annualization.
                                            If None, uses self.params.bars_per_year. Defaults to None.
        
        Returns:
            float: Annualized return rate
        
        Raises:
            ValueError: If bars_per_year is not positive or r is empty
        
        Note:
            This assumes simple compounding. For log returns, use: np.exp(r.mean() * bars_per_year) - 1
        """
        if r.empty:
            raise ValueError("Return series cannot be empty")
        if bars_per_year is not None and bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")
        if bars_per_year is None:
            bars_per_year = self.params.bars_per_year
            
        r = r.dropna()
        n = len(r)
        return (np.prod(1+r))**(bars_per_year/n) - 1

    def max_drawdown(self, r: pd.Series) -> float:
        """
        Calculate maximum drawdown from a return series.
        
        Maximum drawdown is the largest peak-to-trough decline in the cumulative
        returns over the entire period. It measures the worst-case loss an investor
        would have experienced.
        
        Args:
            r (pd.Series): Return series (log returns or simple returns)
        
        Returns:
            float: Maximum drawdown as a negative percentage (e.g., -0.15 for 15% drawdown)
        
        Raises:
            ValueError: If r is empty
        
        Note:
            Works with both log returns and simple returns. The result is always negative
            or zero, representing the maximum loss from a peak.
        """
        if r.empty:
            raise ValueError("Return series cannot be empty")
        cum = (1 + r).cumprod()  # equity curve
        peak = cum.cummax()
        dd = cum / peak - 1
        return dd.min()

    def backtest(self, price: pd.Series, pos: pd.Series) -> pd.DataFrame:
        """
        Execute a comprehensive backtest with realistic trading costs.
        
        This function simulates trading based on position signals, applying realistic
        trading costs (fees and slippage) on position changes. It calculates strategy
        returns, cumulative performance, and compares against buy-and-hold.
        
        Args:
            price (pd.Series): Price series for the asset being traded
            pos (pd.Series): Position signals (-1, 0, 1 for short, flat, long)
        
        Returns:
            pd.DataFrame: Comprehensive backtest results containing:
                - price: Original price series
                - log_price: Log-transformed prices
                - log_ret: Log returns
                - strat_ret: Strategy returns (after costs)
                - cum_strat: Cumulative strategy performance
                - cum_hodl: Cumulative buy-and-hold performance
                - pos_exec: Executed positions (shifted by 1 bar)
                - costs: Trading costs incurred
        
        Raises:
            ValueError: If price or pos series are empty or have mismatched indices
        
        Note:
            - Positions are executed on the next bar (realistic delay)
            - Costs are charged per position change, not round-trip
            - Log returns are used for more accurate compounding
        """
        if price.empty or pos.empty:
            raise ValueError("Price and position series cannot be empty")
        if not price.index.equals(pos.index):
            raise ValueError("Price and position series must have matching indices")
        leverage  = self.params.leverage
        lp = np.log(price)
        r = lp.diff().fillna(0.0)

        # Execute next bar
        pos_exec = pos.shift(1).fillna(0.0)

        # Trading cost per change in position (round turn not assumed; per change).
        # Convert bps to log-return deduction approximating small-cost regime.
        tc_per_side = self.params.trade_fee
        dpos = pos_exec.diff().abs().fillna(pos_exec.abs())  # first bar change = |pos_exec|
        costs = dpos * tc_per_side

        strat_r = (pos_exec * r - costs)*leverage
        cum_strat = np.exp(strat_r.cumsum())-1
        cum_bh = np.exp(r.cumsum())-1

        return pd.DataFrame({
            "price": price,
            "log_price": lp,
            "log_ret": r,
            "strat_ret": strat_r,
            "cum_strat": cum_strat,
            "cum_hodl": cum_bh,
            "pos_exec": pos_exec,
            "costs": costs
        })


    def create_trades_dataframe(self, signal: pd.Series, price: pd.Series) -> pd.DataFrame:
        stop_loss = self.params.stop_loss
        take_profit = self.params.take_profit
        trade_fee = self.params.trade_fee


        assert stop_loss is None or (-1.0 <= stop_loss < 0.0), "stop_loss must be in [-1,0)"
        assert take_profit is None or (take_profit > 0.0), "take_profit must be > 0"
        if not signal.index.equals(price.index):
            price = price.reindex(signal.index).ffill()
        

        trade_signals = signal.diff().dropna()
        entries = trade_signals[trade_signals > 0].index
        exits   = trade_signals[trade_signals < 0].index
        trades_list = []

        if signal.iloc[0] == 1 and not exits.empty:
            entries = entries[entries > exits[0]]

        if entries.empty:
            return pd.DataFrame(columns=['time_enter','price_enter','time_exit','price_exit','returns','trade_duration','trade_duration_hours'])

        for entry_time in entries:
            possible_exits = exits[exits > entry_time]
            if possible_exits.empty:
                continue
            exit_time   = possible_exits[0]
            price_enter = float(price.loc[entry_time])
            price_exit  = float(price.loc[exit_time])

            raw_return = (price_exit / price_enter) - 1.0
            actual_exit_time = exit_time
            actual_price_exit = price_exit
            #print(f"Entry at {entry_time}, price: {price_enter}")

            # Path-wise TP/SL (first-hit wins) with trailing stop-loss
            exit_reason = None  # Initialize exit_reason
            if (stop_loss is not None) or (take_profit is not None):
                path = price.loc[entry_time:exit_time]
                path_cum = (path / price_enter) - 1.0

                sl_time = None
                tp_time = None
                current_stop_loss = stop_loss  # Start with original stop_loss
                max_return_so_far = -np.inf
                
                # Iterate through path chronologically to track trailing stop
                for idx, cum_return in path_cum.items():
                    max_return_so_far = max(max_return_so_far, cum_return)

                    # Check if stop-loss is hit (using current_stop_loss which was just updated)
                    if stop_loss is not None and cum_return <= current_stop_loss and sl_time is None:
                        sl_time = idx
                        exit_reason = "STOP LOSS"
                    
                    # Check if take-profit is hit
                    if take_profit is not None and cum_return >= take_profit and tp_time is None:
                        tp_time = idx
                        exit_reason = "TAKE PROFIT"
                    
                    # If both are hit, we can break early (first-hit logic)
                    if sl_time is not None and tp_time is not None:
                        exit_reason = "BOTH"
                        break
                     # Progressive trailing stop-loss: move stop up as profit increases
                    # Update this after we survie the minute without exiting
                    if stop_loss is not None:
                        if max_return_so_far >= 0.05 and current_stop_loss < 0.05:
                            current_stop_loss = 0.05  # Lock in 5% profit
                            #print(f"\tStop-loss locked in at 5% profit at {idx}, price: {price.loc[idx]}, entry price: {price_enter}")
                        elif max_return_so_far >= 0.04 and current_stop_loss < 0.04:
                            current_stop_loss = 0.04  # Lock in 4% profit
                            #print(f"\tStop-loss locked in at 4% profit at {idx}, price: {price.loc[idx]}, entry price: {price_enter}")
                        elif max_return_so_far >= 0.03 and current_stop_loss < 0.03:
                            current_stop_loss = 0.03  # Lock in 3% profit
                            #print(f"\tStop-loss locked in at 3% profit at {idx}, price: {price.loc[idx]}, entry price: {price_enter}")
                        elif max_return_so_far >= 0.02 and current_stop_loss < 0.02:
                            current_stop_loss = 0.02  # Lock in 2% profit
                            #print(f"\tStop-loss locked in at 2% profit at {idx}, price: {price.loc[idx]}, entry price: {price_enter}")
                        elif max_return_so_far >= 0.01 and current_stop_loss < 0.01:
                            current_stop_loss = 0.01  # Lock in 1% profit
                        #    print(f"\tStop-loss locked in at 1% profit at {idx}, price: {price.loc[idx]}")

                    # Stop-loss hit first
                    if sl_time is not None and (tp_time is None or sl_time <= tp_time):
                        raw_return = current_stop_loss  # stop hit (could be original, +1%, +2%, +3%, +4%, or +5%)
                        actual_exit_time = sl_time
                        bar_price = float(price.loc[sl_time])
                        actual_price_exit = price_enter*(1+current_stop_loss)
                        exit_reason = "TRAILING STOP LOSS"
                        #bar_return = (bar_price / price_enter) - 1.0
                        #print(f"Stop-loss hit at {sl_time}, actual exit price: {actual_price_exit:.4f}, bar price:{bar_price}, entry price: {price_enter}, return: {current_stop_loss:.2%}, bar return: {bar_return:.2%}")
                        break
                    # Take-profit hit first
                    elif tp_time is not None and (sl_time is None or tp_time < sl_time):
                        raw_return = take_profit  # TP hit first
                        actual_exit_time = tp_time
                        actual_price_exit = float(price.loc[tp_time])
                        exit_reason = "TAKE PROFIT"
                        break
                #if exit_reason is not set, set it to "HOLD PERIOD"
                if exit_reason is None:
                    exit_reason = "HOLD PERIOD"
            else:
                # No stop-loss or take-profit, so exit reason is "HOLD PERIOD"
                exit_reason = "HOLD PERIOD"

            leverage = self.params.leverage
            trade_return_net = (raw_return - 2.0 * trade_fee)*leverage
            trades_list.append({
                'time_enter': entry_time,
                'time_exit':  actual_exit_time,
                'price_enter': price_enter,
                'price_exit':  actual_price_exit,
                'returns': trade_return_net,
                'exit_reason': exit_reason
            })

        if not trades_list:
            return pd.DataFrame(columns=['time_enter','price_enter','time_exit','price_exit','returns','trade_duration','trade_duration_hours'])

        df_trades = pd.DataFrame(trades_list)
        df_trades['trade_duration'] = df_trades['time_exit'] - df_trades['time_enter']
        df_trades['trade_duration_hours'] = df_trades['trade_duration'].dt.total_seconds() / 3600.0
        return df_trades




    def apply_min_hold_period(self, signal: pd.Series, min_hold_bars: int) -> pd.Series:
        """
        Modifies a trading signal to enforce a minimum holding period.

        Once a position is entered (signal flips to 1), this function ensures
        the signal remains 1 for at least `min_hold_bars`, overwriting any
        subsequent exit signals during that window. This is a fast approach
        suitable for long time series.

        Args:
            signal (pd.Series): The original trading signal (0s and 1s).
            min_hold_bars (int): The minimum number of bars to hold a position.

        Returns:
            pd.Series: The modified signal with the holding period enforced.
        """
        if min_hold_bars <= 1:
            return signal

        # Make a copy to avoid modifying the original series
        modified_signal = signal.copy()

        # Find the integer locations of entries (0 -> 1 transitions)
        # Using np.where is much faster than filtering a pandas index for this task
        entry_indices = np.where(modified_signal.diff() == 1)[0]

        # A loop over entries is much faster than iterating over the whole series,
        # as the number of trades is typically orders of magnitude smaller than
        # the number of data points.
        current_hold_end_idx = -1
        for entry_idx in entry_indices:
            # If this entry is inside an existing hold period from a prior signal, skip it
            if entry_idx < current_hold_end_idx:
                continue

            # Define the end of the new holding period
            hold_end_idx = entry_idx + min_hold_bars

            # Set the signal to 1 for the entire duration of the hold
            modified_signal.iloc[entry_idx:hold_end_idx] = 1

            # Update the end of the current hold period to prevent overlapping holds
            current_hold_end_idx = hold_end_idx

        return modified_signal


    def plot_equity(self, bt: pd.DataFrame, title: str = "Cumulative Returns") -> None:
        """
        Create comprehensive equity curve visualization.
        
        This function generates a two-panel plot showing:
        1. Cumulative returns comparison (strategy vs buy-and-hold)
        2. Price chart with position markers (long/short signals)
        
        Args:
            bt (pd.DataFrame): Backtest results from the backtest() method or df_output from simulate_mu_vol_trade()
            title (str, optional): Title for the plot. Defaults to "Cumulative Returns".
        
        Returns:
            None: Displays the plot using matplotlib
        
        Note:
            - Data is downsampled by 60 for better visualization (shows every 60th point)
            - Green dots indicate long positions, red dots indicate short positions
            - Grid lines and zero line are included for better readability
            - Works with both backtest() output (cum_strat, pos_exec) and simulate_mu_vol_trade() output (cum_signal, signal)
        """
        plt.figure(figsize=(9,5))
        plt.subplot(2, 1, 1)
        plt.plot(bt["cum_hodl"][::60], label="Hodl")
        
        # Handle different column names for strategy returns
        if "cum_strat" in bt.columns:
            plt.plot(bt["cum_strat"][::60], label="Strategy")
        elif "cum_signal" in bt.columns:
            plt.plot(bt["cum_signal"][::60], label="Strategy")
        else:
            raise ValueError("DataFrame must contain either 'cum_strat' or 'cum_signal' column")
            
        plt.title(title)
        plt.xlabel("Date"); plt.ylabel("Cum")
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend(); plt.tight_layout(); 
        
        plt.subplot(2, 1, 2)
        
        # Handle different column names for price and positions
        if "price" in bt.columns:
            price = bt["price"]
            pos = bt["pos_exec"] if "pos_exec" in bt.columns else bt["signal"] if "signal" in bt.columns else None
        else:
            # If no price column, we can't plot the price chart
            plt.text(0.5, 0.5, 'No price data available for plotting', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.xlabel("Date"); plt.ylabel("Price")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return
            
        plt.plot(price[::60], label="Price")
        if pos is not None:
            plt.plot(price[pos==1][::60], marker = ".", color = "green", linewidth = 0, label="Long")
            plt.plot(price[pos==-1][::60], marker = ".", color = "red", linewidth = 0, label="Short")
        plt.xlabel("Date"); plt.ylabel("Price")
        plt.grid(True)
        plt.legend(); 
        plt.tight_layout(); 
        plt.show()

    def compute_metrics(self, returns: pd.Series, bars_per_year: float = None) -> dict:
        """
        Calculate comprehensive performance metrics for a return series.
        
        Args:
            returns (pd.Series): Return series
            bars_per_year (int, optional): Number of bars per year for annualization.
                                         If None, uses self.params.bars_per_year. Defaults to None.
        
        Returns:
            pd.DataFrame: Single-row DataFrame containing:
                - annualized_returns_strat: Strategy annualized returns
                - annualized_returns_bh: Buy-and-hold annualized returns
                - sharpe_ratio_strat: Strategy Sharpe ratio
                - sharpe_ratio_bh: Buy-and-hold Sharpe ratio
                - max_drawdown_strat: Strategy maximum drawdown
                - max_drawdown_bh: Buy-and-hold maximum drawdown
        
        Note:
            All metrics are calculated using the same methodology for fair comparison.
            Sharpe ratios assume zero risk-free rate.
        """
        if bars_per_year is None:
            bars_per_year = self.params.bars_per_year

        sh = self.annualized_sharpe(returns, bars_per_year)
        ann = self.annualized_returns(returns, bars_per_year)
        max_drawdown = self.max_drawdown(returns)
        return {
            'annualized_returns': ann,
            'sharpe_ratio': sh,
            'max_drawdown': max_drawdown,
        }

    def cum_trades_series(self, df_trades: pd.DataFrame) -> pd.Series:
        """
        Create a cumulative returns series from a trades DataFrame with exact trade times.
        
        This function creates a step function representing cumulative returns over time,
        with timestamps at trade entry and exit points. This shows the exact timing of
        trades and how cumulative returns change at each trade.
        
        Args:
            df_trades (pd.DataFrame): DataFrame containing trade information with columns:
                - time_enter: Entry timestamp
                - time_exit: Exit timestamp
                - returns: Net trade return (after fees and risk management)
        
        Returns:
            pd.Series: Series with datetime index and cumulative returns as values.
                      The index contains: time_enter[0], time_exit[0], time_enter[1], time_exit[1], ...
                      The values are: 0, cum1, cum1, cum2, cum2, cum3, cum3, ...
                      where cum_i is the cumulative return after trade i (compounded).
        
        Example:
            >>> df_trades = pd.DataFrame({
            ...     'time_enter': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-05')],
            ...     'time_exit': [pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-07')],
            ...     'returns': [0.02, 0.01]
            ... })
            >>> cum_series = ts.cum_trades_series(df_trades)
            >>> # Returns Series with index: [2023-01-01, 2023-01-03, 2023-01-05, 2023-01-07]
            >>> # and values: [0.0, 0.02, 0.02, 0.0302] (compounded)
        """
        if df_trades.empty:
            return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        
        # Create cumulative returns (compounded)
        cum_trades = (1 + df_trades.returns).cumprod() - 1
        
        # Create step function: time_enter[0], time_exit[0], time_enter[1], time_exit[1], ...
        # with cumulative values: 0, cum1, cum1, cum2, cum2, cum3, cum3, ...
        time_trades_plot = []
        cum_trades_plot = []
        
        # First point: start at 0 at first entry
        time_trades_plot.append(df_trades.iloc[0]['time_enter'])
        cum_trades_plot.append(0.0)
        
        # For each trade: add exit point, then next entry point (both at same cumulative value)
        for i in range(len(df_trades)):
            # Exit point: cumulative value after this trade
            time_trades_plot.append(df_trades.iloc[i]['time_exit'])
            cum_trades_plot.append(cum_trades.iloc[i])
            
            # Next entry point (if not last trade): same cumulative value as previous exit
            if i < len(df_trades) - 1:
                time_trades_plot.append(df_trades.iloc[i+1]['time_enter'])
                cum_trades_plot.append(cum_trades.iloc[i])
        
        # Create and return Series
        return pd.Series(cum_trades_plot, index=time_trades_plot)

    def returns_trades_series(self, df_trades: pd.DataFrame) -> pd.Series:
        """
        Create a daily returns series from a trades DataFrame.
        
        This function converts individual trades into daily returns by creating a step function
        of cumulative returns, resampling to daily frequency, and then computing daily returns.
        
        Args:
            df_trades (pd.DataFrame): DataFrame containing trade information with columns:
                - time_enter: Entry timestamp
                - time_exit: Exit timestamp
                - returns: Net trade return (after fees and risk management)
        
        Returns:
            pd.Series: Series with daily datetime index and daily returns as values.
                      Returns are computed from the cumulative returns of trades, resampled to daily.
        
        Example:
            >>> df_trades = pd.DataFrame({
            ...     'time_enter': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-05')],
            ...     'time_exit': [pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-07')],
            ...     'returns': [0.02, 0.01]
            ... })
            >>> returns_series = ts.returns_trades_series(df_trades)
            >>> # Returns Series with daily index and daily returns
        """
        if df_trades.empty:
            return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        
        # Create cumulative returns (compounded)
        cum_trades = (1 + df_trades.returns).cumprod() - 1
        
        # Create step function: time_enter[0], time_exit[0], time_enter[1], time_exit[1], ...
        # with cumulative values: 0, cum1, cum1, cum2, cum2, cum3, cum3, ...
        time_trades_plot = []
        cum_trades_plot = []
        
        # First point: start at 0 at first entry
        time_trades_plot.append(df_trades.iloc[0]['time_enter'])
        cum_trades_plot.append(0.0)
        
        # For each trade: add exit point, then next entry point (both at same cumulative value)
        for i in range(len(df_trades)):
            # Exit point: cumulative value after this trade
            time_trades_plot.append(df_trades.iloc[i]['time_exit'])
            cum_trades_plot.append(cum_trades.iloc[i])
            
            # Next entry point (if not last trade): same cumulative value as previous exit
            if i < len(df_trades) - 1:
                time_trades_plot.append(df_trades.iloc[i+1]['time_enter'])
                cum_trades_plot.append(cum_trades.iloc[i])
        
        # Create Series with cumulative returns indexed by time
        cum_trades_series = pd.Series(cum_trades_plot, index=time_trades_plot)
        
        # Resample to daily, taking the last value of each day (forward fill for intraday)
        cum_trades_daily = cum_trades_series.resample('D').last().ffill()
        
        # Calculate daily returns: change in cumulative returns
        # Since cum_trades is compounded, convert to daily returns:
        # daily_return = (1 + cum_today) / (1 + cum_yesterday) - 1
        cum_trades_daily_1plus = 1 + cum_trades_daily
        returns_daily = (cum_trades_daily_1plus / cum_trades_daily_1plus.shift(1) - 1).fillna(0.0)
        
        return returns_daily

    def compute_metrics_trades(self, df_trades: pd.DataFrame, bars_per_year: float = 365.0) -> dict:
        """
        Compute performance metrics from a trades DataFrame by converting to daily returns.
        
        This function takes a DataFrame of individual trades (with time_enter, time_exit, returns)
        and computes performance metrics at the daily level. It accounts for the actual trade
        execution times and stop-loss/take-profit logic embedded in the trades.
        
        Args:
            df_trades (pd.DataFrame): DataFrame containing trade information with columns:
                - time_enter: Entry timestamp
                - time_exit: Exit timestamp
                - returns: Net trade return (after fees and risk management)
            bars_per_year (float, optional): Number of bars per year for annualization.
                                           Defaults to 365.0 (daily bars, crypto trades 365 days/year).
        
        Returns:
            dict: Dictionary containing performance metrics:
                - annualized_returns: Annualized returns
                - sharpe_ratio: Annualized Sharpe ratio
                - max_drawdown: Maximum drawdown
        
        Note:
            - Trades are assumed to be compounded (cumulative product of (1 + returns))
            - Daily returns are computed by resampling the cumulative returns to daily frequency
            - Metrics are calculated at the daily level, then annualized
        """
        if df_trades.empty:
            return {
                'annualized_returns': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
            }
        
        # Get daily returns from trades
        returns_daily = self.returns_trades_series(df_trades)
        
        # Compute metrics at daily level
        return self.compute_metrics(returns_daily, bars_per_year=bars_per_year)


    def simulate_mu_vol_trade(self, price: pd.Series, params: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate a mean-volatility momentum trading strategy with risk management.
        
        This function implements a sophisticated trading strategy that combines mean reversion
        and volatility momentum signals. It generates trading signals based on z-scores of
        exponentially weighted moving averages of returns and volatility, with optional
        stop-loss and take-profit levels.
        
        Strategy Logic:
        1. Calculate exponentially weighted moving averages of returns (mu_t) and volatility (sigma_t)
        2. Compute z-scores for both mu_t and sigma_t using rolling windows
        3. Generate signals when both z-scores exceed their respective thresholds
        4. Apply minimum holding period to reduce overtrading
        5. Execute trades with realistic costs and optional risk management
        
        Args:
            price (pd.Series): Price series for the asset being traded
            params (dict): Dictionary containing strategy parameters:
                - window_days (int): Days for return mean calculation window
                - window_vol_days (int): Days for volatility calculation window  
                - bars_per_year (float): Number of bars per year for time conversion
                - hold_hours (float): Minimum holding period in hours
                - trade_fee (float): Trading fee per trade (as decimal, e.g., 0.0005 for 0.05%)
                - tau_mu (float): Threshold for mean z-score signal generation
                - tau_sigma (float): Threshold for volatility z-score signal generation
                - stop_loss (float, optional): Stop-loss level (negative decimal, e.g., -0.05 for 5%)
                - take_profit (float, optional): Take-profit level (positive decimal, e.g., 0.10 for 10%)
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - df_output: DataFrame containing strategy performance and intermediate calculations:
                    - cum_hodl: Cumulative buy-and-hold returns
                    - cum_signal: Cumulative strategy returns
                    - returns: Asset returns
                    - signal_raw: Raw trading signals before holding period filter
                    - signal: Final trading signals after holding period filter
                    - mu_t: Exponentially weighted moving average of returns
                    - sigma_t: Exponentially weighted moving average of volatility
                    - mu_t_z: Z-score of return mean
                    - sigma_t_z: Z-score of volatility
                    - price: Original price series
                - df_trades: DataFrame containing individual trade details:
                    - time_enter: Entry timestamp
                    - time_exit: Exit timestamp  
                    - price_enter: Entry price
                    - price_exit: Exit price
                    - returns: Net trade returns (after fees and risk management)
                    - trade_duration: Trade duration as timedelta
                    - trade_duration_hours: Trade duration in hours
        
        Raises:
            ValueError: If required parameters are missing or invalid
            KeyError: If required parameter keys are not found in params dict
        
        Note:
            - Strategy enters long positions when both mu_t_z > tau_mu AND sigma_t_z > tau_sigma
            - Positions are held for minimum hold_hours to reduce transaction costs
            - Stop-loss and take-profit are applied on a first-hit basis during trade execution
            - All returns are calculated net of trading fees
        """
        if params is None:
            params = self.params
        window_days = params.window_days
        window_vol_days = params.window_vol_days
        bars_per_year = params.bars_per_year
        hold_hours = params.hold_hours
        trade_fee = params.trade_fee
        tau_mu = params.tau_mu
        tau_sigma = params.tau_sigma
        leverage = params.leverage


        returns = price.pct_change().fillna(0)
        cum_hodl = (1 + returns).cumprod() - 1

        window = int(window_days*bars_per_year/365)
        window_vol = int(window_vol_days*bars_per_year/365)
        hold_period = int(hold_hours*bars_per_year/365/24)  

        mu_t = returns.ewm(span=window, adjust=False).mean()
        sigma_t = returns.ewm(span=window_vol, adjust=False).std()

        # Cache rolling window objects to avoid recomputation
        mu_t_rolling = mu_t.rolling(window=window)
        sigma_t_rolling = sigma_t.rolling(window=window)
        mu_t_z = (mu_t - mu_t_rolling.mean()) / mu_t_rolling.std()
        sigma_t_z = (sigma_t - sigma_t_rolling.mean()) / sigma_t_rolling.std()

        signal_raw = ((mu_t_z>tau_mu) & (sigma_t_z>tau_sigma)).astype(int)
        signal = self.apply_min_hold_period(signal_raw, hold_period)

        df_trades = self.create_trades_dataframe(signal, price)

        # Update signal to reflect actual trade durations (including early exits from SL/TP)
        # Optimized approach using pandas index methods for datetime indexing
        signal_updated = pd.Series(0, index=signal.index, dtype=np.int8)
        if not df_trades.empty:
            signal_values = signal_updated.values
            idx = signal_updated.index
            
            # Use pandas index get_indexer for fast datetime lookups
            for _, trade in df_trades.iterrows():
                time_enter = trade['time_enter']
                time_exit = trade['time_exit']
                # Try to find exact matches first, then fall back to nearest
                try:
                    start_idx = idx.get_loc(time_enter)
                    if isinstance(start_idx, slice):
                        start_idx = start_idx.start
                except (KeyError, TypeError):
                    # If exact match not found, use get_indexer
                    start_idx = idx.get_indexer([time_enter], method='pad')[0]
                
                try:
                    end_idx = idx.get_loc(time_exit)
                    if isinstance(end_idx, slice):
                        end_idx = end_idx.start
                except (KeyError, TypeError):
                    # If exact match not found, use get_indexer
                    end_idx = idx.get_indexer([time_exit], method='backfill')[0]
                
                if start_idx >= 0 and end_idx >= 0 and start_idx <= end_idx:
                    # Set signal to 1 from entry bar (inclusive) to exit bar (exclusive)
                    # Note: signal_updated.shift(1) means signal at t affects returns at t+1
                    # - If we enter at t_enter, signal[t_enter]=1 captures returns at t_enter+1
                    # - If we exit at t_exit, we exit at the start of that bar
                    #   The return at t_exit is from price[t_exit-1] to price[t_exit]
                    #   To capture this, we need signal[t_exit-1]=1
                    #   We don't want signal[t_exit]=1 because that would capture returns at t_exit+1
                    # So we set signal from start_idx to end_idx (exclusive of end_idx)
                    signal_values[start_idx:end_idx] = 1  # Don't include exit bar
            
            signal_updated = pd.Series(signal_values, index=signal_updated.index)
                
        trades = signal_updated.diff().fillna(0).abs()
        #print(f"Trade fee = {trade_fee}")
        returns_signal = (returns*signal_updated.shift(1)-trade_fee*trades)*leverage
        cum_signal = (1+returns_signal).cumprod() - 1
        
        # Create DataFrame directly instead of using concat (faster)
        df_output = pd.DataFrame({
            'cum_hodl': cum_hodl,
            'cum_signal': cum_signal,
            'returns': returns,
            'returns_signal': returns_signal,
            'signal_raw': signal_raw,
            'signal': signal_updated,
            'mu_t': mu_t,
            'sigma_t': sigma_t,
            'mu_t_z': mu_t_z,
            'sigma_t_z': sigma_t_z,
            'price': price
        })
        return df_output, df_trades
        
    def print_dict_pretty(self, label: str, d: dict, indent: int = 4) -> None:
        """Print a dictionary in a pretty format with one item per line, tabbed."""
        print(f"{label}:")
        if isinstance(d, pd.DataFrame):
            # Convert DataFrame to dict (first row)
            d = d.iloc[0].to_dict()
        
        # Keys that should be displayed as percentages
        percent_keys = {'annualized_returns', 'max_drawdown', 'annualized_returns_strat', 
                       'max_drawdown_strat', 'annualized_returns_bh', 'max_drawdown_bh'}
        
        for key, value in d.items():
            # Convert numpy types to Python native types for cleaner printing
            if isinstance(value, (np.integer, np.floating, np.number)):
                value = float(value.item())
            
            # Handle NaN values
            if pd.isna(value):
                print(f"{' ' * indent}{key}: NaN")
            # Format as percentage if it's a returns or drawdown metric
            elif isinstance(value, (int, float)) and key in percent_keys:
                print(f"{' ' * indent}{key}: {value*100:.2f}%")
            # Format numeric values with 2 decimal places
            elif isinstance(value, (int, float)):
                print(f"{' ' * indent}{key}: {value:.2f}")
            # Print other types as-is
            else:
                print(f"{' ' * indent}{key}: {value}")

    def format_metrics_label(self, d: dict) -> str:
        """
        Format a metrics dictionary into a string suitable for plot labels.
        
        Args:
            d: Dictionary with metric names and values (can be DataFrame or dict)
        
        Returns:
            str: Formatted string like "AR=0.27, SR=0.71, MD=-0.68"
        
        Example:
            >>> ts.format_metrics_label({'annualized_returns': 0.27, 'sharpe_ratio': 0.71, 'max_drawdown': -0.68})
            'AR=0.27, SR=0.71, MD=-0.68'
        """
        if isinstance(d, pd.DataFrame):
            # Convert DataFrame to dict (first row)
            d = d.iloc[0].to_dict()
        
        # Abbreviation mapping for common metrics
        abbrev_map = {
            'annualized_returns': 'AR',
            'sharpe_ratio': 'SR',
            'max_drawdown': 'MD',
            'annualized_returns_strat': 'AR_strat',
            'sharpe_ratio_strat': 'SR_strat',
            'max_drawdown_strat': 'MD_strat',
            'annualized_returns_bh': 'AR_bh',
            'sharpe_ratio_bh': 'SR_bh',
            'max_drawdown_bh': 'MD_bh',
        }
        
        # Keys that should be displayed as percentages
        percent_keys = {'annualized_returns', 'max_drawdown', 'annualized_returns_strat', 
                       'max_drawdown_strat', 'annualized_returns_bh', 'max_drawdown_bh'}
        
        parts = []
        for key, value in d.items():
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating, np.number)):
                value = float(value.item())
            
            # Get abbreviation or use key name
            label = abbrev_map.get(key, key)
            
            # Format the value
            if pd.isna(value):
                parts.append(f"{label}=NaN")
            # Format as percentage if it's a returns or drawdown metric
            elif isinstance(value, (int, float)) and key in percent_keys:
                parts.append(f"{label}={value*100:.2f}%")
            elif isinstance(value, (int, float)):
                parts.append(f"{label}={value:.2f}")
            else:
                parts.append(f"{label}={value}")
        
        return ", ".join(parts)