import gym
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
from stockstats import StockDataFrame as Sdf
from finrl.agents.stablebaselines3.models import A2C

from finrl.agents.stablebaselines3.models import DummyVecEnv
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.append('../')
from src.rl_env.stock import SingleStockEnv
from src.models.models import RLModels
from src.data.make_dataset import download_stock_data


#Diable the warnings
import warnings
warnings.filterwarnings('ignore')


data_df = pd.read_csv('../data/snp.csv')[['Date','Close']].rename({'Date':'date','Close':'adjcp'}, axis = 1)

data_df['rsi'] = ta.rsi(data_df['adjcp'])

data_df['macd'] = ta.macd(data_df['adjcp'])['MACD_12_26_9']
data_df.fillna(0, inplace=True)
data_clean = data_df.copy()

train = data_clean[(data_clean.date>='2017-08-17') & (data_clean.date<'2022-03-29')]
# the index needs to start from 0
train=train.reset_index(drop=True)

model_list = ['ddpg','ppo','a2c']

model_dict = {}
for m in model_list:
    print(30*"=", m, 30*"=")
    env_train = DummyVecEnv([lambda: SingleStockEnv(train,feat_list=['macd','rsi'])])
    model = RLModels(m, env_train)
    model.train(total_timesteps=200000)
    model.save(f'{m}stock_only_100k')
    model_dict.update({m:model})
    
    
test = data_clean[(data_clean.date>='2022-03-29') ]
# the index needs to start from 0
test=test.reset_index(drop=True)

def get_DRL_sharpe():
    df_total_value=pd.read_csv('account_value.csv',index_col=0)
    df_total_value.columns = ['account_value']
    df_total_value['daily_return']=df_total_value.pct_change(1)
    sharpe = (252**0.5)*df_total_value['daily_return'].mean()/df_total_value['daily_return'].std()
    
    annual_return = ((df_total_value['daily_return'].mean()+1)**252-1)*100
    print("annual return: ", annual_return)
    print("sharpe ratio: ", sharpe)
    return df_total_value

cum_return = {}
for m in model_list:
    model = model_dict[m]
    env_test = DummyVecEnv([lambda: SingleStockEnv(test,feat_list=['macd','rsi'])])
    obs_test = env_test.reset()
    print("==============Model Prediction===========")
    for i in range(len(test.index.unique())):
        
        action, _states = model.predict(obs_test)
        obs_test, rewards, dones, info = env_test.step(action)
        env_test.render()
    df_total_value=pd.read_csv('account_value.csv',index_col=0)
    df_total_value.columns = ['account_value']
    df_total_value['daily_return']=df_total_value.pct_change(1)
    
    cum_return.update({m: (df_total_value.account_value.pct_change(1)+1).cumprod()-1})
    


fig, ax = plt.subplots(figsize=(12, 8))

plt.plot(test.date, cum_return['ppo'], color='red',label = "PPO")
plt.plot(test.date, cum_return['a2c'], label = "A2C")
plt.plot(test.date, cum_return['ddpg'], color = 'green', label = "DDPG")

plt.title("Cumulative Return for PPO and A2C with Transaction Cost",size= 18)
plt.legend()
plt.rc('legend',fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

plt.figsave("stock-only.png")

bit_cor_df = pd.read_csv('../data/SnP_bit_corr.csv', usecols=['Date','correlation']).rename({'Date':'date', 'correlation':'cor_bit'}, axis = 1)
eth_cor_df = pd.read_csv('../data/SnP_eth_corr.csv', usecols=['Date','correlation']).rename({'Date':'date', 'correlation':'cor_eth'}, axis = 1)
df_corr = pd.concat([data_clean, bit_cor_df['cor_bit'],eth_cor_df['cor_eth']], axis = 1).dropna()

train = df_corr[(data_clean.date>='2017-08-17') & (df_corr.date<'2022-03-29')]
# the index needs to start from 0
train=train.reset_index(drop=True)


model_dict = {}
for m in model_list:
    print(30*"=", m, 30*"=")
    env_train = DummyVecEnv([lambda: SingleStockEnv(train,feat_list=['macd','rsi','cor_bit','cor_eth'])])
    model = RLModels(m, env_train)
    model.train(total_timesteps=200000)
    model.save(f'{m}_corr_100k')
    model_dict.update({m:model})
    
    
test = df_corr[(df_corr.date>='2022-03-29') ]
# the index needs to start from 0
test=test.reset_index(drop=True)

cum_return = {}
for m in model_list:
    model = model_dict[m]
    env_test = DummyVecEnv([lambda: SingleStockEnv(test,feat_list=['macd','rsi'])])
    obs_test = env_test.reset()
    print("==============Model Prediction===========")
    for i in range(len(test.index.unique())):
        
        action, _states = model.predict(obs_test)
        obs_test, rewards, dones, info = env_test.step(action)
        env_test.render()
    df_total_value=pd.read_csv('account_value.csv',index_col=0)
    df_total_value.columns = ['account_value']
    df_total_value['daily_return']=df_total_value.pct_change(1)
    
    cum_return.update({m: (df_total_value.account_value.pct_change(1)+1).cumprod()-1})
    

