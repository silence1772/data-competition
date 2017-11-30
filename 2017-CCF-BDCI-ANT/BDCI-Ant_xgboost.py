# -*- coding: utf-8 -*-
'''
date: 2017-11-05 14:33
author: silence1772
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import date
import xgboost as xgb
#import lightgbm as lgb    

#read data from directory 
df = pd.read_csv('Data/ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv('Data/ccf_first_round_shop_info.csv')
test = pd.read_csv('Data/evaluation_public.csv')

#add the column 'mall_id' to df
df = pd.merge(df,shop[['shop_id','mall_id']], how='left', on='shop_id')
#concat train set and test set for convenient
train = pd.concat([df,test])
#get mail_id without duplicate
mall_list = list(set(list(shop.mall_id)))
result = pd.DataFrame()

def is_weekend(d):
    if (d == 6 or d == 7):
        return 1
    else:
        return 0

cnt = 1
for mall in mall_list:
    print("extract mall...") 
    train_tmp = train[train.mall_id==mall].reset_index(drop=True) 
    print("extract mall OK")  
    l = []
    wifi_in_test = []
    print("extract feature...")
    for index, row in train_tmp.iterrows():
        r = {}
        average_wifi_power = 0
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        #get wifi in test set without duplicate
        if pd.isnull(row.shop_id):
            for i in wifi_list:
                if i[0] not in wifi_in_test:
                    wifi_in_test.append(i[0])
        for i in wifi_list:
            r[i[0]] = int(i[1])
            average_wifi_power += int(i[1])
        r['average_wifi_power'] = average_wifi_power / len(wifi_list)
        r['day_of_week'] = date(int(row['time_stamp'][0:4]),int(row['time_stamp'][5:7]),int(row['time_stamp'][8:10])).weekday()
        r['hour_of_day'] = int(row['time_stamp'][11:13])
        l.append(r)
    print("extract feature OK")
    print("filter wifi...")
    
    # data leak: filter wifi which not in test set
    m = [] 
    for row in l:
        new = {}
        for n in row.keys():
            if n in wifi_in_test:
                new[n] = row[n]
        new['average_wifi_power'] = r['average_wifi_power']
        new['day_of_week']  = r['day_of_week'] 
        new['is_weekend'] = is_weekend(r['day_of_week'])
        new['hour_of_day'] = r['hour_of_day']
        m.append(new)
    print("filter wifi OK")
    print("process data...")
    #process data
    train_tmp = pd.concat([train_tmp, pd.DataFrame(m)], axis=1)
    df_train = train_tmp[train_tmp.shop_id.notnull()]
    df_test = train_tmp[train_tmp.shop_id.isnull()]
    #label encoder
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class = df_train['label'].max()+1    
    
    params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 10,
            'eval_metric': 'merror',
            'seed': 1024,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }

    #extract feature
    feature=[x for x in train_tmp.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
    
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds = 80
    #start training
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=10)
    #predict
    df_test['label'] = model.predict(xgbtest)
    #label decoder
    df_test['shop_id'] = df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    
    #merge and generate result
    r = df_test[['row_id','shop_id']]
    result = pd.concat([result,r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv('sub4.csv',index=False)

    print("Finish %d/97"%(cnt))
    cnt += 1