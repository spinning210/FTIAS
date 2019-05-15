#%%
import pandas as pd
import dwh.source as source
from Stocker.stocker import Stocker

def init():

    df = pd.read_csv('Stocker/price.csv', index_col='date', parse_dates=['date'])

    stock2016_info = pd.read_csv('data/TWSE2016_en_tt.csv',index_col='date', parse_dates=['date'])
    stock2017_info = pd.read_csv('data/TWSE2017_en_tt.csv',index_col='date', parse_dates=['date'])
    stock2018_info = pd.read_csv('data/TWSE2018_en_tt.csv',index_col='date', parse_dates=['date'])

    stock2016_info = stock2016_info.loc[stock2016_info['stock'].str.contains('|'.join(['鴻海']))]
    stock2017_info = stock2017_info.loc[stock2017_info['stock'].str.contains('|'.join(['鴻海']))]
    stock2018_info = stock2018_info.loc[stock2018_info['stock'].str.contains('|'.join(['鴻海']))]


    stock_info = pd.concat([stock2016_info,stock2017_info,stock2018_info], axis=0)
    stock_info = stock_info.sort_index()
    stock_info = stock_info#[:int(len(stock_info)*0.6)]
    df = stock_info

    df = df.drop(['stock'], axis=1)
    price = df.squeeze()
    #tsmc = Stocker(price)
    return price
#%%
def fb_prophet(predict_days):
    price = init()
    tsmc = Stocker(price)
    model, model_data = tsmc.create_prophet_model(days=predict_days)
    #%%
    tsmc.evaluate_prediction()
    #%%
    tsmc.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
    #%%
    tsmc.predict_future(days=predict_days)
    return model_data

def run():
    predict_days = 90
    model_data = fb_prophet(predict_days)

    first = model_data.iloc[[len(model_data)-predict_days]]
    last = model_data.iloc[[len(model_data)-1]]
    print(last)
    print(first)

    if float(first['trend']) > float(last['trend']):
        return -1, last['ds']
    elif float(first['trend']) <float(last['trend']):
        return 1,last['ds']
    else :
        return 0 ,last['ds']
# #%%
# tsmc.evaluate_prediction()
# #%%
# tsmc.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
# #%%
# tsmc.predict_future(days=100)
run()