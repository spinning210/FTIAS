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
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression


def run():
    company_news = source.get_company_news()

    cross_over_keys, cross_under_keys = source.get_cross_keywords()
    cross_over_keys = cross_over_keys[:128]
    cross_under_keys = cross_under_keys[:128]
    company_news['post_time'] = pd.to_datetime(company_news['post_time']).dt.date
    company_news_train = company_news[int(len(company_news)*0.9):]


    tmp_content = jieba_analyse.cut_to_list(company_news_train)

    done = []
    row_list = []
    label = []  #-跌 +漲

    for article in tqdm(tmp_content):
        xx = {}
        tmp_content_score = pd.DataFrame()
        
        
        
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

    index_b = source.get_company_higher_lower_index()
    index_b = index_b.set_index('date')
    index_b = index_b['result']

    result = pd.concat([df_score, index_b], axis=1, join='inner')

    print(result)
    print(len(result))
        
    a = result.columns.values.tolist()
    a.pop(len(a)-1)


    Y = result['result'].to_numpy()
    X = result[a].to_numpy()
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y ) 
    #===============mlp
    print('mlp===============================================================')
    model_MLP = MLPClassifier(hidden_layer_sizes=(256, 256,), max_iter=256)
    model_MLP.fit(X_train, Y_train)
    print(model_MLP.score(X_train, Y_train))

    

    predictions = model_MLP.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print('==================================================================')

    #===============RandomForest
    print('RandomForest======================================================')
    model_RandomForest = RandomForestClassifier()
    model_RandomForest.fit(X_train, Y_train)
    print(model_RandomForest.score(X_train, Y_train))

    predictions = model_RandomForest.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print('==================================================================')

    #===============XGBClassifier
    print('XGBClassifier=====================================================')
    model_XGBClassifier = XGBClassifier()
    model_XGBClassifier.fit(X_train, Y_train)
    print(model_XGBClassifier.score(X_train, Y_train))

    predictions = model_XGBClassifier.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print('==================================================================')

    #===============LogisticRegression
    print('LogisticRegression================================================')
    model_LogisticRegression = LogisticRegression()
    model_LogisticRegression.fit(X_train, Y_train)
    print(model_LogisticRegression.score(X_train, Y_train))

    predictions = model_LogisticRegression.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print('==================================================================')


