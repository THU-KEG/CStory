import json 

a = json.load(open('/data/skj/information_flow/data/2021-03-14-2021-03-20.json'))

for item  in a:
        item['title'] = item['title'].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','').replace('2021阿里云上云采购季：采购补贴、充值返券、爆款抢先购云通信分会场：爆款产品低至7.2折，短信低至0.034元条','')
        item['content'] = item['content'].replace('\n',"").replace(' ','').replace('\t','').replace('\u3000','').replace('2021阿里云上云采购季：采购补贴、充值返券、爆款抢先购云通信分会场：爆款产品低至7.2折，短信低至0.034元条','')


json.dump(a,open('/data/skj/information_flow/data/new_2021-03-14-2021-03-20.json','w'), ensure_ascii=False)