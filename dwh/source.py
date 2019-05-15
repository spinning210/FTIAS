import pandas as pd
import numpy as np
import csv

#路徑要自己改

def get_origin_news():
    """原始新聞資料"""

    bbs = pd.read_csv('data/bbs_utf8.csv') 
    forum = pd.read_csv('data/forum_utf8.csv') 
    news = pd.read_csv('data/news_utf8.csv') 

    return bbs,forum,news

def get_company_news():
    """指定公司的新聞資料"""

    company_news = pd.read_csv('output/category_news/2317_all_news.csv')
    company_news = company_news[company_news['post_time'].notnull()]
    company_news['post_time'] = pd.to_datetime(company_news['post_time']).dt.date

    return company_news

def get_company_higher_lower_index():
    """指定公司漲跌日期資料"""

    company_index = pd.read_csv('data/2317_index.csv') 
    #company_index = pd.read_csv('data/2317_index_mlp.csv') 
    company_index['date'] = pd.to_datetime(company_index['date']).dt.date

    return company_index

def get_category_news():
    """指定公司漲跌新聞"""

    higher_news = pd.read_csv("output/category_news/2317_higher_news.csv") 
    lower_news = pd.read_csv("output/category_news/2317_lower_news.csv") 
    useless_news = pd.read_csv("output/category_news/2317_useless_news.csv") 

    return higher_news, lower_news, useless_news


def get_tf_idf_key():
    """結巴分析後的關鍵字"""

    higher_keys = pd.read_csv('output/jieba/tf_idf_part_higher.csv')
    lower_keys = pd.read_csv('output/jieba/tf_idf_part_lower.csv')

    return higher_keys ,lower_keys

def get_cross_index():
    """取出反轉點index"""
    corss_index = pd.read_csv('data/2318_cross.csv')
    corss_index['date'] = pd.to_datetime(corss_index['date']).dt.date

    return corss_index

def get_kewords():
    """漲跌關鍵字"""
    higher_keys = pd.read_csv('output/jieba/final_higher_keywords.csv')
    lower_keys = pd.read_csv('output/jieba/final_lower_keywords.csv')

    return higher_keys, lower_keys

def get_cross_keywords():
    """取出反轉點關鍵字"""
    cross_over_keys = pd.read_csv('output/jieba/final_over.csv')
    cross_under_keys = pd.read_csv('output/jieba/final_under.csv')

    return cross_over_keys, cross_under_keys

def get_category_cross_index(index):
    """取出反轉點over,under index"""
    higher_mask = index['result'] == 'cross over'
    lower_mask = index['result'] == 'cross under'

    over_index = index[higher_mask]
    under_index = index[lower_mask]

    return over_index, under_index

def get_cross_score():
    cross_score = pd.read_csv('output/jieba/cross_score.csv')
    
    return cross_score

def get_stock_info():
    stock2016_info = pd.read_csv('data/TWSE2016_en.csv')
    stock2017_info = pd.read_csv('data/TWSE2017_en.csv')
    stock2018_info = pd.read_csv('data/TWSE2018_en.csv')

    stock2016_info = stock2016_info.loc[stock2016_info['stock'].str.contains('|'.join(['鴻海']))]
    stock2017_info = stock2017_info.loc[stock2017_info['stock'].str.contains('|'.join(['鴻海']))]
    stock2018_info = stock2018_info.loc[stock2018_info['stock'].str.contains('|'.join(['鴻海']))]


    stock_info = pd.concat([stock2016_info,stock2017_info,stock2018_info], axis=0)

    stock_info['date'] = pd.to_datetime(stock_info['date']).dt.date
    stock_info.sort_values('date', inplace=True)
    stock_info = stock_info.reset_index(drop=True)

    return stock_info