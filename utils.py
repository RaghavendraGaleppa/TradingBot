import sys
import pandas as pd
import mplfinance as mf
import logging

def get_logger(name, level="DEBUG"):
    logger = logging.getLogger(name)
    _format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    output = sys.stdout
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter(_format))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_data(file_name, reframe=5):
    data = pd.read_csv(file_name, names=['date', 'time', 'open', 'high', 'low', 'close', 'vol'])
    data['datetime'] = data.apply(lambda x: x['date'] + ' ' + x['time'], axis=1)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    data.drop(columns=['date', 'time'], inplace=True)
    data.dropna(inplace=True)
    # Each row represents 1 minute data
    # We should be able to reframe the data to 5 minute data
    if reframe != 1:
        data = data.resample(str(reframe) + 'T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'})
    data['close_prev'] = data['close'].shift(1)
    data.dropna(inplace=True)
    
    return data


def plot_candle_sticks(df):
    mf.plot(df, type='candle')
    
    
def plot_multiple_candle_plots(df, indices):
    """
    :param df: 
    :param indices: 
    :return: 
    """
    n_plots = len(indices)
    rows = n_plots // 2 + 1
    cols = 2
    fig = mf.figure(figsize=(10, 10))
    for i in range(n_plots):
        start_index = indices[i][0]
        end_index = indices[i][1]
        df_slice = df[start_index:end_index]
        ax = fig.add_subplot(rows, cols, i+1)
        mf.plot(df_slice, type='candle', ax=ax)
        
    return fig
