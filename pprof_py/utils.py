import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from zoneinfo import ZoneInfo


def setup_logger(
    name: str, 
    level=logging.INFO, 
    log_to_file: bool = False, 
    log_dir: str = None, 
    time_zone: str = "US/Eastern"
) -> logging.Logger:
    """Setup a logger with the given name and level.

    Parameters:
    - name (str): The name of the logger.
    - level (int): The logging level. Default is logging.INFO.
    - log_to_file (bool): If True, logs will also be written to a file. Default is False.
    - log_dir (str): Directory where the log file will be saved. Default is None (current working directory).
    - time_zone (str): The time zone for the timestamps. Default is "US/Eastern".

    Returns:
    - logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create timezone-aware formatter
    tz = ZoneInfo(time_zone)

    class TimeZoneFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz)
            return dt.strftime(datefmt) if datefmt else dt.isoformat()

    formatter_with_tz = TimeZoneFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add file handler if log_to_file is True
    if log_to_file:
        # Create a log file name with timestamp
        timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        log_file_name = f"{name}_{timestamp}.log"

        print(log_file_name)
        
        # Create directory if log_dir is specified and doesn't exist
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file_name = os.path.join(log_dir, log_file_name)
        
        # Ensure the file handler is added immediately
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(level)
        fh.setFormatter(formatter_with_tz)
        logger.addHandler(fh)

        
    # Create and add console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter_with_tz)
    logger.addHandler(ch)

    # Flush the logger to ensure everything is written immediately
    logger.handlers[0].flush()
    
    return logger

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))


def proc_freq(df: pd.DataFrame, columns: list):
    """Prints the frequency, percentage, and cumulative percentage of each specified column or tuple of columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): The columns to print frequencies for. Each element can be a string (for a single column) or a tuple (for a tuple of columns).
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df parameter must be a pandas DataFrame.")
    
    if not isinstance(columns, list):
        raise TypeError("The columns parameter must be a list.")
    
    total_count = len(df)

    def process_column(col):
        """Process a single column or a tuple of columns to compute frequencies, percentages, and cumulative statistics.

        Parameters:
        col (str or tuple): The column(s) to process.

        Returns:
        None
        """
        if isinstance(col, tuple):
            if not all(isinstance(c, str) for c in col):
                raise TypeError("All elements in the column tuple must be strings.")
            print(f"Cross-tabulation of {col}:")
            result = df.groupby(list(col)).size().reset_index(name='count')
            order_columns = list(col)
        elif isinstance(col, str):
            print(f"Frequency of {col}:")
            result = df.groupby(col).size().reset_index(name='count')
            order_columns = [col]
        else:
            raise TypeError("Each column must be a string or a tuple of strings.")

        result = result.assign(
            Percentage=lambda x: (x['count'] / total_count * 100).round(2),
            Cumulative_Percentage=lambda x: x['Percentage'].cumsum(),
            Cumulative_Count=lambda x: x['count'].cumsum()
        ).sort_values(by=order_columns)

        print(result)

    for column in columns:
        try:
            process_column(column)
        except Exception as e:
            print(f"Error processing column {column}: {e}")

