from operator import truediv
import requests
import json
# from detect_keywords import getkeywords_tfidf
# def find_keywords(keywords, sentences):
#     for keyword in key_words:
#         if sentences.find(keyword) == -1:
#             return False
#     return True
# while True:
#     params = {'keywords':'特朗普', 'size':size, 'startDate':'2020-07-01', 'endDate':'2020-7-02', 'page':page}
#     url = 'https://api2.newsminer.net/svc/news/queryNewsByKeywords?keywords=%E7%89%B9%E6%9C%97%E6%99%AE&size=500&page=1&startDate=2020-07-01&endDate=2020-07-07'
#     r =  requests.get(url, params=params).json()
#     data = r['data']
#     #print("text_data", text_data)
#     text_data = text_data + [ele['title']+ '。'+ele['content'] for ele in data]
#     #print('text',  text_data)
#     raw_data = raw_data + data
#     print(r['total'])
#     if page * size >= r['total']:
#         break
#     page += 1
#     print("循环一次")

# print(len(text_data))
def addDates(string, interval):
    year = int(string[0:4])
    month = int(string[5:7])
    day = int(string[8:10])

    #calculate how many days per year
    day2month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    x = 29 if (year%4==0 and year%100) or (year%100==0 and year%400==0) else 28
    assert interval < 28
    day  = day + interval
    if day > day2month[month - 1]:
        day = day % day2month[month - 1]
        month += 1
        if month > 12:
            year += 1
            month = month % 12

    if month <= 9:
        month = '0' + str(month)
    if day <= 9:
        day = '0' + str(day)
    #print (year, month, day)
    return str(year) + '-' + str(month) + '-' + str(day)

#def date_interval(string1, string2):
#    year = int(string[0:4])
#    month = int(string[5:7])
#    day = int(string[8:10])
#
#    x = 29 if (year%4==0 and year%100) or (year%100==0 and year%400==0) else 28
#    day2month = [31, x, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
startDate = '2020-01-01'
date_interval = 7
prefix = 'https://api2.newsminer.net/svc/news/queryNewsByKeywords?keywords=%E7%89%B9%E6%9C%97%E6%99%AE&size=999'

#pre_data = []
data = []
#i = 36
while  startDate <= '2021-06-30' :
    cur_page = 1
    url = prefix + '&startDate=' + startDate + '&endDate=' + addDates(startDate, date_interval) + '&page=' + str(cur_page)
    r = requests.get(url).json()
    data += r['data']
    cur_page += 1
    page_num = int(r['total'] / r['pageSize']) + 1
    print('page_num', page_num)
    while cur_page <= page_num:
        url = prefix + '&startDate=' + startDate + '&endDate=' + addDates(startDate, date_interval) + '&page=' + str(cur_page)
        r = requests.get(url).json()
        data += r['data']
        print(len(r['data']))
        cur_page += 1
    print(url)

    new_txt_name = 'news/' + startDate + '_' + addDates(startDate, date_interval) + '.txt'
    print("一共的长度", len(data))
    json.dump(data, open(new_txt_name , 'w'), ensure_ascii=False)
    data = []
    startDate = addDates(startDate, date_interval)