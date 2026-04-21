from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

directory_path = Path("final_clean_data")
files_list = [entry.name for entry in directory_path.iterdir() if entry.is_file()]

def get_normalised_data(file):
    df = pd.read_csv("final_clean_data/"+file, index_col=0)

    df = df.copy()
    
    df['ret_1m'] = np.log(df['price'] / df['price'].shift(1))
    df['ret_3m'] = np.log(df['price'] / df['price'].shift(3))
    
    df['mom_12m'] = df['price'] / df['price'].shift(12) - 1
    
    df['vol_3m'] = df['ret_1m'].rolling(3).std()
    df['vol_6m'] = df['ret_1m'].rolling(6).std()
    
    df['vol_z'] = (
        (df['volume'] - df['volume'].rolling(12).mean()) /
        df['volume'].rolling(12).std()
    )
    
    df['vol_trend'] = np.log(df['volume'] / df['volume'].shift(3))
    
    df['ey'] = 1 / df['pe']
    
    df['ey_z'] = (
        (df['ey'] - df['ey'].rolling(24).mean()) /
        df['ey'].rolling(24).std()
    )
    
    df['bm_z'] = (
        (df['bm'] - df['bm'].rolling(24).mean()) /
        df['bm'].rolling(24).std()
    )
    
    df['log_mc'] = np.log(df['mc'])
    
    df['mc_z'] = (
        (df['log_mc'] - df['log_mc'].rolling(24).mean()) /
        df['log_mc'].rolling(24).std()
    )
    
    df['ma_6m'] = df['price'].rolling(6).mean()
    df['rev_6m'] = (df['price'] - df['ma_6m']) / df['ma_6m']
    
    df['mom_vol_adj'] = df['mom_12m'] / (df['vol_6m'] + 1e-8)
    df['value_mom'] = df['ey'] * df['mom_12m']
    df['liquidity_adj_mom'] = df['mom_12m'] * df['vol_z']
    
    features = [
        'ret_1m',
        'ret_3m',
        'mom_12m',
        'vol_3m',
        'vol_6m',
        'vol_z',
        'vol_trend',
        'ey_z',
        'bm_z',
        'mc_z',
        'rev_6m',
        'mom_vol_adj',
        'value_mom',
        'liquidity_adj_mom'
    ]
    
    rolling_mean = df[features].rolling(24).mean().shift(1)
    rolling_std  = df[features].rolling(24).std().shift(1)
    
    df[features] = (df[features] - rolling_mean) / (rolling_std + 1e-8)
    df['R'] = (df['price'].shift(-1) / df['price'])
    features.append('R')
    
    df = df[features]
    df = df.dropna()
    return df


def get_months():
    most_data_file = -1
    most_data = 0
    for file in files_list:
        data = get_normalised_data(file).shape[0]
        if data > most_data:
            most_data = data
            most_data_file = file
    df = get_normalised_data(most_data_file)
    months = list(df.index)
    return months

features = [
    'ret_1m',
    'ret_3m',
    'mom_12m',
    'vol_3m',
    'vol_6m',
    'vol_z',
    'vol_trend',
    'ey_z',
    'bm_z',
    'mc_z',
    'rev_6m',
    'mom_vol_adj',
    'value_mom',
    'liquidity_adj_mom'
]

months = get_months()


def get_state_reward(time):
    i = time
    state_i = []
    reward_i = []
    month = months[i]
    for file in files_list:
        try:
            stock_data = pd.read_csv('normalised_data/'+file, index_col=0)
            s_i = stock_data.loc[month][features]
            r_i = stock_data.loc[month]['R']
            state_i.append(s_i)
            reward_i.append(r_i)
        except:
            continue
    return state_i, reward_i

