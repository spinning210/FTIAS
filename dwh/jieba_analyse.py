import jieba
import jieba.analyse
import dwh.filter as myfilter
import pandas
from tqdm import tqdm


def init():
    """結巴的initial"""
    #停用詞因為詞性分析，我沒用到，但還是先保留

    jieba.set_dictionary('config/jieba/dict.txt')
    with open('config/jieba/stops_new.txt', 'r', encoding='utf8') as f: 
        stops = f.read().split('\n')

    return jieba, stops

def tf_idf_part(contents, behaviour):
    """結巴tfidf詞性分析"""
    myjieba, stops = init()

    print('tf-idf-' + behaviour + ' part is running...')

    keywords = [t for t in myjieba.analyse.extract_tags(contents, topK = 2000, withWeight=True, allowPOS=('a','ad','an','v','d','vd','vn')) if t not in stops]
    name = 'tf_idf_part_' + behaviour

    print('tf-idf-' + behaviour + ' part has been run succesfully.')
    return keywords, name

def cut_to_list(company_news):
    """結巴切字成陣列"""
    myjieba, stops = init()
    tmp_content = []
    
    with tqdm(total=len(list(company_news.iterrows()))) as pbar:

        for _, data in company_news.iterrows():
            content = myfilter.clean_str(data['content'])
            segments = myjieba.cut(content, cut_all=False)
            words = list(filter(lambda a: a not in stops and a != '\n', segments))
            tmp_content.append([data['post_time'],words])
            pbar.update(1)
    return tmp_content