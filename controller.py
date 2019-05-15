import dwh.source as source
import dwh.export as export
import dwh.filter as filter
import dwh.jieba_analyse as jieba_analyse
import mlp_news
import lstm_stock_price
import mlp_cross_date

def keywords_list():
    bbs , forum, news = source.get_origin_news()
    company = '鴻海'
    stock_code = '2317'
    bbs_contents, forum_contents, news_contents = filter.get_assign_news(bbs, forum, news, company, stock_code)
    assign_news = filter.merge_data(bbs_contents, forum_contents, news_contents)
    
    path = 'output/category_news/'
    name = '2317_all_news'
    
    export.news_to_csv(assign_news, path, name)


    company_news = source.get_company_news()
    company_index = source.get_company_higher_lower_index()
    higher_index, lower_index, useless_index = filter.get_category_index(company_index)

    path = 'output/category_news/'

    topic = '2317_higher_news'
    higher_results = filter.match_date(company_news, higher_index)
    export.news_to_csv(higher_results, path, topic)

    topic = '2317_lower_news'
    lower_results = filter.match_date(company_news,lower_index)
    export.news_to_csv(lower_results, path, topic)

    topic = '2317_useless_news'
    useless_results = filter.match_date(company_news,useless_index)
    export.news_to_csv(useless_results, path, topic)

    higher_news, lower_news, useless_news = source.get_category_news()
    higher_contents = filter.get_content(higher_news)
    lower_contents = filter.get_content(lower_news)
    useless_contents = filter.get_content(useless_news)

    path = 'output/jieba/'
    higher_keys, name = jieba_analyse.tf_idf_part(higher_contents, 'higher')
    export.keywords_to_csv(higher_keys, path, name)

    lower_keys, name = jieba_analyse.tf_idf_part(lower_contents, 'lower')
    export.keywords_to_csv(lower_keys, path, name)


    higher_keys, lower_keys = source.get_tf_idf_key()
    higher_results, lower_results = filter.same_key_filter(higher_keys, lower_keys)
    higher_results, lower_results = filter.own_key_filter(higher_keys, lower_keys, higher_results, lower_results )

    path = 'output/jieba/'
    export.keywords_to_csv(higher_results, path, 'final_higher_keywords')
    export.keywords_to_csv(lower_results, path, 'final_lower_keywords')

def mlp_clasify_news():
    mlp_news.run()

def mlp_cross():
    mlp_cross_date.run()

def lstm_stock_price():
    lstm_stock_price.run()

