import csv

#path 'output/category_news/'
def news_to_csv(assign_news, path, name):
    """輸出新聞格式"""

    print(name + '.csv is exporting...')

    with open(path + name +'.csv', 'w', newline='') as csvFile:
        fieldNames = ['id','p_type','s_name','s_area_name','post_time','title','author','content','page_url']
        writer = csv.DictWriter(csvFile, fieldNames)
        writer.writeheader()

        for index,row in assign_news.iterrows():
            writer.writerow({'id': row['id'],
                            'p_type': row['p_type'],
                            's_name': row['s_name'],
                            's_area_name':row['s_area_name'],
                            #'comment_count':row['comment_count'],  data not same
                            'post_time':row['post_time'],
                            'title':row['title'],
                            'author':row['author'],
                            'content':row['content'],
                            'page_url':row['page_url']})
    
    print(name + '.csv is exporting...Successfully')

def keywords_to_csv(results, path, name):
    """輸出關鍵字格式"""

    print(name + '.csv is exporting...')

    with open(path + name + '.csv', 'w', newline = '') as csvFile:
        fieldNames = ['key','weight']
        writer = csv.DictWriter(csvFile, fieldNames)
        writer.writeheader()

        for item in results:
            writer.writerow({'key': item[0],
                            'weight': item[1]})

    print(name + '.csv is exporting...Successfully')

def cross_score_to_csv(data, path:str, name:str ):
    """輸出新聞格式"""

    print(name + '.csv is exporting...')

    with open(path + name +'.csv', 'w', newline='') as csvFile:
        fieldNames = ['date','content','over_score','under_score']
        writer = csv.DictWriter(csvFile, fieldNames)
        writer.writeheader()

        for index,row in data.iterrows():
            writer.writerow({'date': row['date'],
                            'content': row['content'],
                            'over_score': row['over_score'],
                            'under_score':row['under_score']})
