import pandas as pd
import numpy as np
from sklearn import  preprocessing
#import xgboost as xgb
import lightgbm as lgb    

df=pd.read_csv('Data/ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv('Data/ccf_first_round_shop_info.csv')
test=pd.read_csv('Data/evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
df['time_stamp']=pd.to_datetime(df['time_stamp'])
train=pd.concat([df,test])
mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
cnt = 1
for mall in mall_list:
    train1=train[train.mall_id==mall].reset_index(drop=True)       
    l=[]
    wifi_dict = {}
    for index,row in train1.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            row[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(row)    
    delate_wifi=[]
    for i in wifi_dict:
        if wifi_dict[i]<20:
            delate_wifi.append(i)
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1=pd.DataFrame(m)
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1  

    params = {
    		'task': 'train',
    		'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_iterations' : 1000, 
            'max_bin' : 100, 
            'num_leaves': 512, 
            'feature_fraction': 0.8,  
            'bagging_fraction': 0.95,
    		'bagging_freq': 5, 
    		'min_data_in_leaf' : 200, 
    		'learning_rate' : 0.05
    }

    ROUNDS = 60
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
    lgbtrain = lgb.Dataset(df_train[feature],label=df_train['label'])
    lgbtest = lgb.Dataset(df_test[feature])
    watchlist=[lgbtrain]

    model = lgb.train(params=params, train_set=lgbtrain, num_boost_round=ROUNDS,valid_sets=watchlist, early_stopping_rounds=10)
    df_test['label'] = model.predict(lgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv('sub.csv',index=False)
    print(cnt + '/97')
    cnt += 1