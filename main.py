import controller as controller

def main(): 
    #req 1 = keywords
    controller.keywords_list()

    # mlp classify artical useless up or down
    controller.mlp_clasify_news()

    #mlp classify cross day
    controller.mlp_cross()

    # lstm predict tomorrow price
    controller.lstm_stock_price()



main()