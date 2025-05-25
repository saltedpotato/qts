import datetime
import polars as pl
import numpy as np
from typing import List, Optional
from utils.market_time import market_hours


class market_data:
    """
    A class for managing market data, including reading from parquet files,
    preparing time-series data, and processing market timings for different sessions.

    Attributes:
    ----------
    file_path : str
        Path to the directory where the parquet files are stored.
    df : pl.LazyFrame | None
        Polars DataFrame containing market data.
    market_timing : dict
        Dictionary containing market timings for both 'daylight' and 'normal' sessions.
    times : dict
        Dictionary storing lists of times for 'daylight' and 'normal' sessions.
    full_date_time : pl.DataFrame | None
        DataFrame containing the full date and time information after processing.
    """

    def __init__(self, file_path: str):
        """
        Initializes the market_data object by setting the file path and preparing market session times.

        Parameters:
        ----------
        file_path : str
            Path to the directory where the parquet files are stored.
        """

        self.file_path = file_path
        self.df = pl.LazyFrame()

        self.market_timing = {
            market_hours.DAYLIGHT: {
                market_hours.PRE: datetime.time(8, 0, 0),
                market_hours.MARKET_OPEN: datetime.time(13, 30, 0),
                market_hours.MARKET_CLOSE: datetime.time(19, 59, 0),
                market_hours.POST: datetime.time(23, 59, 0),
            },
            market_hours.NORMAL: {
                market_hours.PRE: datetime.time(9, 0, 0),
                market_hours.MARKET_OPEN: datetime.time(14, 30, 0),
                market_hours.MARKET_CLOSE: datetime.time(20, 59, 0),
                market_hours.POST: datetime.time(23, 59, 0),
            },
        }

        self.times = {market_hours.DAYLIGHT: [], market_hours.NORMAL: []}

        start = datetime.datetime.combine(datetime.date.today(), datetime.time(8, 0, 0))
        end = datetime.datetime.combine(datetime.date.today(), datetime.time(23, 59, 0))
        daylight_start = datetime.time(9, 0, 0)

        current = start
        while current <= end:
            time_only = current.time()
            self.times[market_hours.NORMAL].append(time_only)
            if time_only >= daylight_start:
                self.times[market_hours.DAYLIGHT].append(time_only)
            current += datetime.timedelta(minutes=1)

        self.times[market_hours.DAYLIGHT] = pl.LazyFrame(
            {"time": self.times[market_hours.DAYLIGHT]}
        )
        self.times[market_hours.NORMAL] = pl.LazyFrame(
            {"time": self.times[market_hours.NORMAL]}
        )

        self.full_date_time = None

    def read(self, cons: List[str], start: str, end: str) -> bool:
        """
        Reads market data from parquet files, filters by date range, and prepares the final DataFrame.

        Parameters:
        ----------
        cons : List[str]
            A list of symbols to filter the market data for.
        start : str
            Start date of the data in 'YYYY-MM-DD' format.
        end : str
            End date of the data in 'YYYY-MM-DD' format.

        Returns:
        -------
        bool
            True if read successfully
        """
        year_to_read = list(set([start[:4], end[:4]]))
        pl_start = pl.lit(start).str.strptime(pl.Date, "%Y-%m-%d")
        pl_end = pl.lit(end).str.strptime(pl.Date, "%Y-%m-%d")

        if len(year_to_read) > 1:
            df = pl.concat(
                [
                    pl.read_parquet(
                        f"{self.file_path}{year_to_read[0]}.parquet",
                        columns=[
                            "ts_event",
                            "close",
                            "symbol",
                        ],
                    ),
                    pl.read_parquet(
                        f"{self.file_path}{year_to_read[1]}.parquet",
                        columns=[
                            "ts_event",
                            "close",
                            "symbol",
                        ],
                    ),
                ],
                how="vertical",
            ).lazy()
        else:
            df = pl.read_parquet(
                f"{self.file_path}{year_to_read[0]}.parquet",
                columns=[
                    "ts_event",
                    "close",
                    "symbol",
                ],
            ).lazy()

        self.df = (
            df.with_columns(date=pl.col("ts_event").str.strptime(pl.Datetime))
            .with_columns(
                date=pl.col("date").dt.date(),
                time=pl.col("date").dt.time(),
            )
            .filter(
                (
                    pl.col("date").is_between(
                        lower_bound=pl_start
                        - pl.duration(
                            days=5
                        ),  # filter few days behind to get the ffill price
                        upper_bound=pl_end,
                    )
                    & (pl.col("symbol").is_in(cons))
                )
            )
            .collect()
            .pivot(on="symbol", index=["date", "time"], values="close")
            .sort(by=["date", "time"])
            .lazy()
        )

        self.prep_time()

        self.df = (
            self.full_date_time.join(
                self.df, how="left", left_on=["date", "time"], right_on=["date", "time"]
            )
            .fill_null(strategy="forward")
            .filter(pl.col("date") >= pl_start)
        ).collect()

        return True

    def filter(self, resample_freq, hours):
        df = self.filter_hours(hours=hours)
        df = df.with_columns(
            (pl.Series(name="max_time", values=np.array([0] * len(df)))),
            (pl.Series(name="min_time", values=np.array([0] * len(df)))),
        )
        df = self.resample_df(df=df, resample_freq=resample_freq)

        return df

    def resample_df(
        self, df: pl.DataFrame = None, resample_freq: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Resample the DataFrame (`self.df`) based on the specified frequency.

        Parameters
        ----------
        df : pl.DataFrame, optional
            DataFrame for filtering, if None, defaults to member variable
        resample_freq : str, optional
            The frequency to which the DataFrame should be resampled.
            If `None`, the DataFrame will not be resampled and the original DataFrame will be returned.
            Example frequencies include:
            - '1H' for hourly resampling
            - '30m' for 30-minute resampling
            - '15m' for 15-minute resampling
            - 'D' for daily resampling

        Returns
        -------
        pl.DataFrame
            A DataFrame with the time column adjusted to the new frequency.
            The `min_time` and `max_time` columns are dropped after resampling.

        Raises
        ------
        ValueError
            If an invalid `resample_freq` is provided.

        Example
        -------
        # Resample the DataFrame to a frequency of 30 minutes
        resampled_df = market_data.resample_df("30m")

        # Display the resampled DataFrame
        print(resampled_df)
        """
        if df is None:
            df = self.df
        if resample_freq is not None:
            resampled_df = (
                df.with_columns(
                    pl.col("time").dt.to_string().str.to_datetime("%H:%M:%S")
                )
                .upsample(
                    time_column="time",
                    every=resample_freq,
                    group_by="date",
                    maintain_order=True,
                )
                .with_columns(pl.col("time").dt.time())
            )

            return resampled_df.drop(["min_time", "max_time"])
        return df

    def filter_hours(
        self, df: pl.DataFrame = None, hours: market_hours = market_hours.MARKET
    ) -> pl.DataFrame:
        """
        Filter the DataFrame (`self.df`) based on the specified market hours.

        Parameters
        ----------
        df : pl.DataFrame, optional
            DataFrame for filtering, if None, defaults to member variable
        hours : str, optional
            Defines the time period for filtering:
            - 'MARKET' (default) : Filters data during normal market hours.
            - 'PRE' : Filters data during pre-market hours (before market open).
            - 'POST' : Filters data during post-market hours (after market close).

        Returns
        -------
        pl.DataFrame
            A filtered DataFrame containing only the rows where the `time` is within the
            specified market range based on the `min_time` column.

        Raises
        ------
        ValueError
            If the `hours` parameter is not one of 'MARKET', 'PRE', or 'POST'.
        """
        if df is None:
            df = self.df
        if hours == market_hours.MARKET:
            filtered_df = df.filter(
                (
                    pl.col("time")
                    >= pl.when(
                        # if normal then filter based on normal, if daylight then filter based on daylight
                        pl.col("min_time")
                        == self.market_timing[market_hours.NORMAL][market_hours.PRE]
                    )
                    .then(
                        pl.lit(
                            self.market_timing[market_hours.NORMAL][
                                market_hours.MARKET_OPEN
                            ]
                        )
                    )
                    .otherwise(
                        pl.lit(
                            self.market_timing[market_hours.DAYLIGHT][
                                market_hours.MARKET_OPEN
                            ]
                        )
                    )
                )
                & (
                    pl.col("time")
                    <= pl.when(
                        pl.col("min_time")
                        == self.market_timing[market_hours.DAYLIGHT][market_hours.PRE]
                    )
                    .then(
                        pl.lit(
                            self.market_timing[market_hours.NORMAL][
                                market_hours.MARKET_CLOSE
                            ]
                        )
                    )
                    .otherwise(
                        pl.lit(
                            self.market_timing[market_hours.DAYLIGHT][
                                market_hours.MARKET_CLOSE
                            ]
                        )
                    )
                )
            )

        elif hours == market_hours.PRE:
            filtered_df = df.filter(
                (
                    pl.col("time")
                    <= pl.when(
                        pl.col("min_time")
                        == self.market_timing[market_hours.NORMAL][market_hours.PRE]
                    )
                    .then(
                        pl.lit(
                            self.market_timing[market_hours.NORMAL][
                                market_hours.MARKET_OPEN
                            ]
                        )
                    )
                    .otherwise(
                        pl.lit(
                            self.market_timing[market_hours.DAYLIGHT][
                                market_hours.MARKET_OPEN
                            ]
                        )
                    )
                )
            )

        elif hours == market_hours.POST:
            filtered_df = df.filter(
                (
                    pl.col("time")
                    >= pl.when(
                        pl.col("min_time")
                        == self.market_timing[market_hours.NORMAL][market_hours.PRE]
                    )
                    .then(
                        pl.lit(
                            self.market_timing[market_hours.NORMAL][
                                market_hours.MARKET_CLOSE
                            ]
                        )
                    )
                    .otherwise(
                        pl.lit(
                            self.market_timing[market_hours.DAYLIGHT][
                                market_hours.MARKET_CLOSE
                            ]
                        )
                    )
                )
            )
        else:
            filtered_df = df

        return filtered_df.drop(["min_time", "max_time"])

    def prep_time(self) -> None:
        """
        Prepares the date-time structure by computing the minimum and maximum time for each day,
        and joining with the 'daylight' and 'normal' session times.

        This method updates the `self.full_date_time` attribute with the full date-time information.
        """
        self.date_time = {}
        date_time = self.df.group_by("date").agg(
            [
                pl.col("time").min().alias("min_time"),
                pl.col("time").max().alias("max_time"),
            ]
        )

        daylight_days = date_time.filter(
            pl.col("min_time")
            < self.market_timing[market_hours.NORMAL][market_hours.PRE]
        )
        daylight_days = daylight_days.join(
            self.times[market_hours.DAYLIGHT], how="cross"
        )

        normal_days = date_time.filter(
            pl.col("min_time")
            >= self.market_timing[market_hours.NORMAL][market_hours.PRE]
        )
        normal_days = normal_days.join(self.times[market_hours.NORMAL], how="cross")

        self.full_date_time = pl.concat(
            [daylight_days, normal_days], how="vertical"
        ).sort(by=["date", "time"])
