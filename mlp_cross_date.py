import dwh.source as source
import pandas as pd
import numpy as np
from datetime import date
import dwh.filter as myfilter
import dwh.jieba_analyse as jieba_analyse
import dwh.export as export
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def run():
    company_news = source.get_company_news()
    
    cross_over_keys, cross_under_keys = source.get_cross_keywords()
    cross_over_keys = cross_over_keys[:128]
    cross_under_keys = cross_under_keys[:128]
    company_news['post_time'] = pd.to_datetime(company_news['post_time']).dt.date
    company_news_train = company_news[int(len(company_news)*0.8):]



    tmp_date = []
    tmp_content = []
    tmp_over_score = []
    tmp_under_score = []
    score = []
    score_date = []

    tmp_content = jieba_analyse.cut_to_list(company_news_train)

    done = []
    row_list = []
    label = []  #-跌 +漲
    for article in tqdm(tmp_content):
        tmp_content_score = pd.DataFrame()
        
        xx = {}
        #for word in article:
        if article[0] not in done:
            done.append(article[0])
            for _, index in cross_over_keys.iterrows():
                if index['key'] in article[1]:
                    sss = {index['key']: float(index['weight']) }
                else:
                    sss = {index['key']: 0}
                xx.update(sss)
            for _, index in cross_under_keys.iterrows():
                if index['key'] in article[1]:
                    sss = {index['key']: float(index['weight']*-1)}
                else:
                    sss = {index['key']: 0}
                xx.update(sss)
            sss = {'date': article[0]}
            xx.update(sss)
            row_list.append(xx)

    tmp_content_score = pd.DataFrame(row_list)
    a = tmp_content_score.columns.values.tolist()

    
    df_score = tmp_content_score.set_index('date')
    print(df_score)
        
    stocks = source.get_stock_info()
    volume_fluctuation = []
    for i in tqdm(range(10, len(stocks))):
        date = stocks.ix[[i]].values[0][1]
        total_volume = list()
        for ii in range(i-10,i):
            total_volume.append(float(stocks.ix[[ii]].values[0][6].replace(',','')))
        volume_std = np.std(total_volume, ddof = 1)
        volume_mean = np.mean(total_volume)
        if float(stocks.ix[[i]].values[0][6].replace(',','')) <= volume_mean + 2*volume_std and float(stocks.ix[[i]].values[0][6].replace(',','')) >= volume_mean - 2*volume_std:
            volume_fluctuation.append(0)
            tmp_date.append(date)
        else:
            volume_fluctuation.append(1)
            tmp_date.append(date)


    tmp_valu_dt = {
        'date' : tmp_date,
        'volume_fluctuation' : volume_fluctuation
    }
    df_volume_fluctuation = pd.DataFrame(tmp_valu_dt)
    df_volume_fluctuation = df_volume_fluctuation.set_index('date')

    cross_data = source.get_cross_index()

    xxxx = []
    count = 0 
    for k,i in cross_data.iterrows():
        xxx = {}
        
        if i['result'] == 0 :
            count += 1
            sss = {'coutinues':count}
        else :
            count = 0
            sss = {'coutinues':count}
            
        xxx.update(sss)
        sss = {'result':i['result']}
        xxx.update(sss)
        sss = {'date':i['date']}
        xxx.update(sss)

        xxxx.append(xxx)

    cross_data = pd.DataFrame(xxxx)
    cross_data = cross_data.set_index('date')
    print(cross_data)
    #組資料 test
    result = pd.concat([df_volume_fluctuation, df_score, cross_data], axis=1, join='inner')
    result = result[result['coutinues']<10]
    print(result)

    a.append('volume_fluctuation')
    a.pop(0)
    Y = result['result'].to_numpy()
    X = result[a].to_numpy()
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y ) 
    model_MLP = MLPClassifier(hidden_layer_sizes=(256, 256,), max_iter=256)
    model_MLP.fit(X_train, Y_train)
    print(model_MLP.score(X_train, Y_train))

    predictions = model_MLP.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))



