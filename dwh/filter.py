import pandas as pd
import csv
import re

def get_assign_news(bbs,forum,news,company,stock_code):
    """切割出指定公司新聞"""

    search_for = [company,stock_code]

    bbs = bbs[bbs['content'].notnull()]
    bbs = bbs.loc[bbs['content'].str.contains('|'.join(search_for))]

    forum = forum[forum['content'].notnull()]
    forum = forum.loc[forum['content'].str.contains('|'.join(search_for))]

    news = news[news['content'].notnull()]
    news = news.loc[news['content'].str.contains('|'.join(search_for))]

    return bbs,forum,news

def merge_data(bbs,forum,news):
    """將所有新聞結合"""

    results = pd.concat([bbs,forum,news], axis=0, join='inner')
    return results

def get_category_index(index):
    """將漲跌資料分類"""

    higher_mask = index['result'] == 'up'
    lower_mask = index['result'] == 'down'
    useless_mask = index['result'] == 'useless'

    higher_index = index[higher_mask]
    lower_index = index[lower_mask]
    useless_index = index[useless_mask]

    return higher_index,lower_index,useless_index

def match_date(news,cate_index):
    """切割出所有新聞中是 漲\跌\平 的文章"""
    
    print('match date...')
    news_result = pd.DataFrame(columns=['id','p_type','s_name','s_area_name','post_time','title','author','content','page_url'])
    for c_index_index, key in cate_index.iterrows():
        for new_index, new in news.iterrows():
            if new['post_time'] == key['date']:
                news_result = news_result.append(new)
    
    return news_result

def get_content(news):
    """取出新聞中的內文並合併"""

    contents = news['content']
    new_contents = ','.join(str(i) for i in contents)

    contents = clean_str(new_contents)

    return contents

def clean_str(contents):
    """刪除英文數字標點等.."""

    clean = u'[a-zA-Z0-9’!"#$%：；！╰══╯╰═══╯╰═╯╰═╮║「』（）〖」％╭▌˙◆&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    
    contents = re.sub(clean, '', contents) 
    contents = contents.replace(u'\u3000', '') 
    contents = contents.replace('\n','')
    contents = contents.replace('\r','')
    contents = ''.join(contents.split())

    return contents

def same_key_filter(higher_keys, lower_keys):
    """關鍵字交叉比對相同的key"""

    print('keywords filter...')

    higher_results = list()
    lower_results = list()
    
    for h_index, h_row in higher_keys.iterrows():
        for l_index, l_row in lower_keys.iterrows():
            if h_row['key'] == l_row['key']:
                key, status = compare_weight(h_row, l_row)
                if status == 'higher':
                    higher_results.append(key)
                elif status == 'lower' :
                    lower_results.append(key)
                else :
                    higher_results.append(key)
                    lower_results.append(key)

    return higher_results, lower_results

def own_key_filter(higher_keys, lower_keys, higher_results, lower_results):
    """關鍵字交叉比對自己的key"""
    higher_total_keys = list()
    for _, higher in higher_keys.iterrows():
        higher_total_keys.append(higher['key'])

    lower_total_keys = list()
    for _, lower in lower_keys.iterrows():
        lower_total_keys.append(lower['key'])

    higher_owner_keys = list()
    for higher_key in higher_total_keys:
        if higher_key not in lower_total_keys:
            higher_owner_keys.append(higher_key)

    lower_owner_keys = list()
    for lower_key in lower_total_keys:
        if lower_key not in higher_total_keys:
            lower_owner_keys.append(lower_key)

    for _, higher in higher_keys.iterrows():
        for higher_owner_key in higher_owner_keys:
            if higher['key'] == higher_owner_key:
                higher_results.append([higher['key'], higher['weight']])

    for _, lower in lower_keys.iterrows():
        for lower_owner_key in lower_owner_keys:
            if lower['key'] == lower_owner_key:
                lower_results.append([lower['key'], lower['weight']])

    return higher_results, lower_results

def compare_weight(high, low):
    """權重比較"""
    if float(high['weight']) > float(low['weight']):
        key = [ high['key'], high['weight']]
        return key, 'higher'
    elif float(high['weight']) < float(low['weight']):
        key = [low['key'],low['weight']]
        return key, 'lower'
    else:
        key = [high['key'], high['weight']]
        return key, 'both'


