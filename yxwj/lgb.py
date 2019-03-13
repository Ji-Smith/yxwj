import pandas as pd
import lightgbm as lgb
#from sklearn.cross_validation import train_test_split
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

dic = {'user_id':'int32','infantry_add_value':'int32','infantry_reduce_value':'int32',
       'cavalry_add_value':'int32','cavalry_reduce_value':'int32','shaman_add_value':'int32',
       'shaman_reduce_value':'int32','wound_infantry_add_value':'int32','wound_infantry_reduce_value':'int32',
       'wound_cavalry_add_value':'int32','wound_cavalry_reduce_value':'int32','wound_shaman_add_value':'int32',
       'wound_shaman_reduce_value':'int32','general_acceleration_add_value':'int32','general_acceleration_reduce_value':'int32',
       'building_acceleration_add_value':'int32','building_acceleration_reduce_value':'int32','reaserch_acceleration_add_value':'int32',
       'reaserch_acceleration_reduce_value':'int32','training_acceleration_add_value':'int32','training_acceleration_reduce_value':'int32',
       'treatment_acceleraion_add_value':'int16','treatment_acceleration_reduce_value':'int16',
       'bd_training_hut_level':'int16','bd_healing_lodge_level':'int16','bd_stronghold_level':'int16',
       'bd_outpost_portal_level':'int16','bd_barrack_level':'int16','bd_healing_spring_level':'int16',
       'bd_dolmen_level':'int16','bd_guest_cavern_level':'int16','bd_warehouse_level':'int16','bd_watchtower_level':'int16',
       'bd_magic_coin_tree_level':'int16','bd_hall_of_war_level':'int16','bd_market_level':'int16','bd_hero_gacha_level':'int16',
       'bd_hero_strengthen_level':'int16','bd_hero_pve_level':'int16','sr_scout_level':'int16','sr_training_speed_level':'int16',
       'sr_infantry_tier_2_level':'int16','sr_cavalry_tier_2_level':'int16','sr_shaman_tier_2_level':'int16',
       'sr_infantry_atk_level':'int16','sr_cavalry_atk_level':'int16','sr_shaman_atk_level':'int16','sr_infantry_tier_3_level':'int16',
       'sr_cavalry_tier_3_level':'int16','sr_shaman_tier_3_level':'int16','sr_troop_defense_level':'int16','sr_infantry_def_level':'int16',
       'sr_cavalry_def_level':'int16','sr_shaman_def_level':'int16','sr_infantry_hp_level':'int16','sr_cavalry_hp_level':'int16','sr_shaman_hp_level':'int16',
       'sr_infantry_tier_4_level':'int16','sr_cavalry_tier_4_level':'int16','sr_shaman_tier_4_level':'int16','sr_troop_attack_level':'int16',
       'sr_construction_speed_level':'int16','sr_hide_storage_level':'int16','sr_troop_consumption_level':'int16','sr_rss_a_prod_levell':'int16',
       'sr_rss_b_prod_level':'int16','sr_rss_c_prod_level':'int16','sr_rss_d_prod_level':'int16','sr_rss_a_gather_level':'int16',
       'sr_rss_b_gather_level':'int16','sr_rss_c_gather_level':'int16','sr_rss_d_gather_level':'int16','sr_troop_load_level':'int16','sr_rss_e_gather_level':'int16',
       'sr_rss_e_prod_level':'int16','sr_outpost_durability_level':'int16','sr_outpost_tier_2_level':'int16','sr_healing_space_level':'int16',
       'sr_gathering_hunter_buff_level':'int16','sr_healing_speed_level':'int16','sr_outpost_tier_3_level':'int16','sr_alliance_march_speed_level':'int16',
       'sr_pvp_march_speed_level':'int16','sr_gathering_march_speed_level':'int16','sr_outpost_tier_4_level':'int16',
       'sr_guest_troop_capacity_level':'int16','sr_march_size_level':'int16','sr_rss_help_bonus_level':'int16','pvp_battle_count':'int16',
       'pvp_lanch_count':'int16','pvp_win_count':'int16','pve_battle_count':'int16','pve_lanch_count':'int16','pve_win_count':'int16',#'avg_online_minutes':'float32','pay_price':'float32',
       'pay_count':'int16'}

def lgb_train(train,test,train3):
    Y = train['prediction_pay_price']
    df_test = test[['user_id']]
    train2 = train
    test2 = test
    b = pd.DataFrame()
    b2= pd.DataFrame()
    b['user_id'] = train3['user_id']
    b2['user_id'] = test2['user_id']
    b['real'] = train2['prediction_pay_price']
    train2 = train2.drop(['user_id','prediction_pay_price','register_time','pay_price'],1)
    train3 = train3.drop(['user_id','prediction_pay_price','register_time','pay_price'],1)
    test2 = test2.drop(['user_id','register_time','pay_price','prediction_pay_price'],1)
    k_fold = KFold(n_splits=5,shuffle=False,random_state=14)
    a = pd.DataFrame()
    a['feature'] = train2.columns
    i = 1
    for train_indices,valid_indices in k_fold.split(train2):
        train_features,train_labels = train2.iloc[train_indices,],Y.iloc[train_indices,]
        valid_features,valid_labels = train2.iloc[valid_indices,],Y.iloc[valid_indices,]
        model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=1000000,learning_rate = 0.001,subsample = 0.8)
        model.fit(train_features,train_labels,eval_set=[(valid_features,valid_labels)],eval_metric='l2',early_stopping_rounds = 300,verbose= 100)
        best_iteration = model.best_iteration_
        a['importance_'+str(i)] = model.feature_importances_
        b['pred'+str(i)] = model.predict(train3,num_iteration=best_iteration)
        b2['pred'+str(i)] = model.predict(test2,num_iteration=best_iteration)
        i += 1
    b['prediction_pay_price'] = (b['pred1']+b['pred2']+b['pred3']+b['pred4']+b['pred5'])/5
    b2['prediction_pay_price'] = (b2['pred1']+b2['pred2']+b2['pred3']+b2['pred4']+b2['pred5'])/5
    b3 = b2[['user_id','prediction_pay_price']]
    b3['prediction_pay_price'] += test['pay_price']
    df_test = df_test.merge(b3,on='user_id',how='left')
    return a,b,b2,df_test

def bar_plot_analyse(df,operation,num):
    index = np.arange(num)
    plt.figure()
    for i in df.columns:
        if pd.api.types.is_numeric_dtype(df_train[i]):
            value = []
            for j in range(0,100,100//num):
                down = np.percentile(df[i],j)
                up = np.percentile(df[i],j+100//num)
                #print(i)
                df2 = df[(df[i]>=down)]
                df2 = df2[(df2[i]<=up)]
                value.append(df2['prediction_pay_price'].agg(operation))
            print(value)
            plt.bar(index,value,width=0.3,color='b')
            plt.title(i+'----'+operation)
            plt.xticks(np.arange(0,num,1))
            plt.grid()
            plt.savefig('./picture/'+operation+'/'+i+'--'+operation+'.jpg')
            #plt.show('off')
            #plt.ion()
            #plt.pause(5)
            plt.close()
                
def feature(train,test):
    print("特征构造开始时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    train['prediction_pay_price'] -= train['pay_price']
    train['0_count'] = (train == 0).sum(axis=1)
    b = [time.strftime('%w',time.strptime(a,'%Y-%m-%d %H:%M:%S')) for a in train['register_time']]
    train['weekday'] = b
    train['weekday'] = train['weekday'].astype(np.int8)
    train['wood_change_per'] = train['wood_reduce_value']/train['wood_add_value']
    train['stone_change_per'] = train['stone_reduce_value']/train['stone_add_value']
    train['ivory_change_per'] = train['ivory_reduce_value']/train['ivory_add_value']
    train['meat_change_per'] = train['meat_reduce_value']/train['meat_add_value']
    train['magic_change_per'] = train['magic_reduce_value']/train['magic_add_value']
    train['infantry_change_per'] = train['infantry_reduce_value']/train['infantry_add_value']
    train['cavalry_change_per'] = train['cavalry_reduce_value']/train['cavalry_add_value']
    train['shaman_change_per'] = train['shaman_reduce_value']/train['shaman_add_value']
    train['wound_infantry_change_per'] = train['wound_infantry_reduce_value']/train['wound_infantry_add_value']
    train['wound_cavalry_change_per'] = train['wound_cavalry_reduce_value']/train['wound_cavalry_add_value']
    train['wound_shaman_change_per'] = train['wound_shaman_reduce_value']/train['wound_shaman_add_value']
    train['general_acceleration_change_per'] = train['general_acceleration_reduce_value']/train['general_acceleration_add_value']
    train['building_acceleration_change_per'] = train['building_acceleration_reduce_value']/train['building_acceleration_add_value']
    train['reaserch_acceleration_change_per'] = train['reaserch_acceleration_reduce_value']/train['reaserch_acceleration_add_value']
    train['training_acceleration_change_per'] = train['training_acceleration_reduce_value']/train['training_acceleration_add_value']
    train['treatment_acceleration_change_per'] = train['treatment_acceleration_reduce_value']/train['treatment_acceleraion_add_value']
    train['build_level'] = train['bd_training_hut_level']+train['bd_healing_lodge_level']+train['bd_stronghold_level']+train['bd_outpost_portal_level']+train['bd_barrack_level']+train['bd_healing_spring_level']+train['bd_dolmen_level']+train['bd_guest_cavern_level']+train['bd_warehouse_level']+train['bd_watchtower_level']+train['bd_magic_coin_tree_level']+train['bd_hall_of_war_level']+train['bd_market_level']+train['bd_hero_gacha_level']+train['bd_hero_strengthen_level']+train['bd_hero_pve_level']
    train['sr_attack_level'] = train['sr_scout_level']+train['sr_training_speed_level']+train['sr_infantry_tier_2_level']+train['sr_cavalry_tier_2_level']+train['sr_shaman_tier_2_level']+train['sr_infantry_atk_level']+train['sr_cavalry_atk_level']+train['sr_shaman_atk_level']+train['sr_infantry_tier_3_level']+train['sr_cavalry_tier_3_level']+train['sr_shaman_tier_3_level']+train['sr_troop_defense_level']+train['sr_infantry_def_level']+train['sr_cavalry_def_level']+train['sr_shaman_def_level']+train['sr_infantry_hp_level']+train['sr_cavalry_hp_level']+train['sr_shaman_hp_level']+train['sr_infantry_tier_4_level']+train['sr_cavalry_tier_4_level']+train['sr_shaman_tier_4_level']+train['sr_troop_attack_level']+train['sr_troop_load_level']+train['sr_outpost_durability_level']+train['sr_outpost_tier_2_level']+train['sr_outpost_tier_3_level']+train['sr_alliance_march_speed_level']+train['sr_pvp_march_speed_level']+train['sr_gathering_march_speed_level']+train['sr_outpost_tier_4_level']+train['sr_guest_troop_capacity_level']+train['sr_march_size_level']
    train['sr_source_level'] = train['sr_construction_speed_level']+train['sr_hide_storage_level']+train['sr_troop_consumption_level']+train['sr_rss_a_prod_levell']+train['sr_rss_b_prod_level']+train['sr_rss_c_prod_level']+train['sr_rss_d_prod_level']+train['sr_rss_a_gather_level']+train['sr_rss_b_gather_level']+train['sr_rss_c_gather_level']+train['sr_rss_d_gather_level']+train['sr_rss_e_gather_level']+train['sr_rss_e_prod_level']+train['sr_healing_space_level']+train['sr_gathering_hunter_buff_level']+train['sr_healing_speed_level']+train['sr_rss_help_bonus_level']
    train['sr_level'] = train['sr_scout_level']+train['sr_training_speed_level']+train['sr_infantry_tier_2_level']+train['sr_cavalry_tier_2_level']+train['sr_shaman_tier_2_level']+train['sr_infantry_atk_level']+train['sr_cavalry_atk_level']+train['sr_shaman_atk_level']+train['sr_infantry_tier_3_level']+train['sr_cavalry_tier_3_level']+train['sr_shaman_tier_3_level']+train['sr_troop_defense_level']+train['sr_infantry_def_level']+train['sr_cavalry_def_level']+train['sr_shaman_def_level']+train['sr_infantry_hp_level']+train['sr_cavalry_hp_level']+train['sr_shaman_hp_level']+train['sr_infantry_tier_4_level']+train['sr_cavalry_tier_4_level']+train['sr_shaman_tier_4_level']+train['sr_troop_attack_level']+train['sr_troop_load_level']+train['sr_outpost_durability_level']+train['sr_outpost_tier_2_level']+train['sr_outpost_tier_3_level']+train['sr_alliance_march_speed_level']+train['sr_pvp_march_speed_level']+train['sr_gathering_march_speed_level']+train['sr_outpost_tier_4_level']+train['sr_guest_troop_capacity_level']+train['sr_march_size_level']+train['sr_construction_speed_level']+train['sr_hide_storage_level']+train['sr_troop_consumption_level']+train['sr_rss_a_prod_levell']+train['sr_rss_b_prod_level']+train['sr_rss_c_prod_level']+train['sr_rss_d_prod_level']+train['sr_rss_a_gather_level']+train['sr_rss_b_gather_level']+train['sr_rss_c_gather_level']+train['sr_rss_d_gather_level']+train['sr_rss_e_gather_level']+train['sr_rss_e_prod_level']+train['sr_healing_space_level']+train['sr_gathering_hunter_buff_level']+train['sr_healing_speed_level']+train['sr_rss_help_bonus_level']
    train['battle_count'] = train['pvp_battle_count']+train['pve_battle_count']
    train['pvp_win_per'] = train['pvp_win_count']/train['pvp_battle_count']
    train['pve_win_per'] = train['pve_win_count']/train['pve_battle_count']
    train['pvp_lanch_per'] = train['pvp_lanch_count']/train['pvp_battle_count']
    train['pve_lanch_per'] = train['pve_lanch_count']/train['pve_battle_count']
    #train = train[['user_id','register_time','weekday','wood_change_per','stone_change_per','ivory_change_per','meat_change_per','magic_change_per','infantry_change_per','cavalry_change_per','shaman_change_per','wound_infantry_change_per','wound_cavalry_change_per','wound_shaman_change_per','general_acceleration_change_per','building_acceleration_change_per','reaserch_acceleration_change_per','training_acceleration_change_per','treatment_acceleration_change_per','build_level','sr_attack_level','sr_source_level','sr_level','battle_count','pvp_win_per','pve_win_per','pvp_lanch_per','pve_lanch_per','pay_price','pay_count','avg_online_minutes','prediction_pay_price']]
    #train = train[['user_id','register_time','pay_price','pay_count','avg_online_minutes','prediction_pay_price']]
    
    test['0_count'] = (test == 0).sum(axis=1)
    b = [time.strftime('%w',time.strptime(a,'%Y-%m-%d %H:%M:%S')) for a in test['register_time']]
    test['weekday'] = b
    test['weekday'] = test['weekday'].astype(np.int8)
    test['wood_change_per'] = test['wood_reduce_value']/test['wood_add_value']
    test['stone_change_per'] = test['stone_reduce_value']/test['stone_add_value']
    test['ivory_change_per'] = test['ivory_reduce_value']/test['ivory_add_value']
    test['meat_change_per'] = test['meat_reduce_value']/test['meat_add_value']
    test['magic_change_per'] = test['magic_reduce_value']/test['magic_add_value']
    test['infantry_change_per'] = test['infantry_reduce_value']/test['infantry_add_value']
    test['cavalry_change_per'] = test['cavalry_reduce_value']/test['cavalry_add_value']
    test['shaman_change_per'] = test['shaman_reduce_value']/test['shaman_add_value']
    test['wound_infantry_change_per'] = test['wound_infantry_reduce_value']/test['wound_infantry_add_value']
    test['wound_cavalry_change_per'] = test['wound_cavalry_reduce_value']/test['wound_cavalry_add_value']
    test['wound_shaman_change_per'] = test['wound_shaman_reduce_value']/test['wound_shaman_add_value']
    test['general_acceleration_change_per'] = test['general_acceleration_reduce_value']/test['general_acceleration_add_value']
    test['building_acceleration_change_per'] = test['building_acceleration_reduce_value']/test['building_acceleration_add_value']
    test['reaserch_acceleration_change_per'] = test['reaserch_acceleration_reduce_value']/test['reaserch_acceleration_add_value']
    test['training_acceleration_change_per'] = test['training_acceleration_reduce_value']/test['training_acceleration_add_value']
    test['treatment_acceleration_change_per'] = test['treatment_acceleration_reduce_value']/test['treatment_acceleraion_add_value']
    test['build_level'] = test['bd_training_hut_level']+test['bd_healing_lodge_level']+test['bd_stronghold_level']+test['bd_outpost_portal_level']+test['bd_barrack_level']+test['bd_healing_spring_level']+test['bd_dolmen_level']+test['bd_guest_cavern_level']+test['bd_warehouse_level']+test['bd_watchtower_level']+test['bd_magic_coin_tree_level']+test['bd_hall_of_war_level']+test['bd_market_level']+test['bd_hero_gacha_level']+test['bd_hero_strengthen_level']+test['bd_hero_pve_level']
    test['sr_attack_level'] = test['sr_scout_level']+test['sr_training_speed_level']+test['sr_infantry_tier_2_level']+test['sr_cavalry_tier_2_level']+test['sr_shaman_tier_2_level']+test['sr_infantry_atk_level']+test['sr_cavalry_atk_level']+test['sr_shaman_atk_level']+test['sr_infantry_tier_3_level']+test['sr_cavalry_tier_3_level']+test['sr_shaman_tier_3_level']+test['sr_troop_defense_level']+test['sr_infantry_def_level']+test['sr_cavalry_def_level']+test['sr_shaman_def_level']+test['sr_infantry_hp_level']+test['sr_cavalry_hp_level']+test['sr_shaman_hp_level']+test['sr_infantry_tier_4_level']+test['sr_cavalry_tier_4_level']+test['sr_shaman_tier_4_level']+test['sr_troop_attack_level']+test['sr_troop_load_level']+test['sr_outpost_durability_level']+test['sr_outpost_tier_2_level']+test['sr_outpost_tier_3_level']+test['sr_alliance_march_speed_level']+test['sr_pvp_march_speed_level']+test['sr_gathering_march_speed_level']+test['sr_outpost_tier_4_level']+test['sr_guest_troop_capacity_level']+test['sr_march_size_level']
    test['sr_source_level'] = test['sr_construction_speed_level']+test['sr_hide_storage_level']+test['sr_troop_consumption_level']+test['sr_rss_a_prod_levell']+test['sr_rss_b_prod_level']+test['sr_rss_c_prod_level']+test['sr_rss_d_prod_level']+test['sr_rss_a_gather_level']+test['sr_rss_b_gather_level']+test['sr_rss_c_gather_level']+test['sr_rss_d_gather_level']+test['sr_rss_e_gather_level']+test['sr_rss_e_prod_level']+test['sr_healing_space_level']+test['sr_gathering_hunter_buff_level']+test['sr_healing_speed_level']+test['sr_rss_help_bonus_level']
    test['sr_level'] = test['sr_scout_level']+test['sr_training_speed_level']+test['sr_infantry_tier_2_level']+test['sr_cavalry_tier_2_level']+test['sr_shaman_tier_2_level']+test['sr_infantry_atk_level']+test['sr_cavalry_atk_level']+test['sr_shaman_atk_level']+test['sr_infantry_tier_3_level']+test['sr_cavalry_tier_3_level']+test['sr_shaman_tier_3_level']+test['sr_troop_defense_level']+test['sr_infantry_def_level']+test['sr_cavalry_def_level']+test['sr_shaman_def_level']+test['sr_infantry_hp_level']+test['sr_cavalry_hp_level']+test['sr_shaman_hp_level']+test['sr_infantry_tier_4_level']+test['sr_cavalry_tier_4_level']+test['sr_shaman_tier_4_level']+test['sr_troop_attack_level']+test['sr_troop_load_level']+test['sr_outpost_durability_level']+test['sr_outpost_tier_2_level']+test['sr_outpost_tier_3_level']+test['sr_alliance_march_speed_level']+test['sr_pvp_march_speed_level']+test['sr_gathering_march_speed_level']+test['sr_outpost_tier_4_level']+test['sr_guest_troop_capacity_level']+test['sr_march_size_level']+test['sr_construction_speed_level']+test['sr_hide_storage_level']+test['sr_troop_consumption_level']+test['sr_rss_a_prod_levell']+test['sr_rss_b_prod_level']+test['sr_rss_c_prod_level']+test['sr_rss_d_prod_level']+test['sr_rss_a_gather_level']+test['sr_rss_b_gather_level']+test['sr_rss_c_gather_level']+test['sr_rss_d_gather_level']+test['sr_rss_e_gather_level']+test['sr_rss_e_prod_level']+test['sr_healing_space_level']+test['sr_gathering_hunter_buff_level']+test['sr_healing_speed_level']+test['sr_rss_help_bonus_level']
    test['battle_count'] = test['pvp_battle_count']+test['pve_battle_count']
    test['pvp_win_per'] = test['pvp_win_count']/test['pvp_battle_count']
    test['pve_win_per'] = test['pve_win_count']/test['pve_battle_count']
    test['pvp_lanch_per'] = test['pvp_lanch_count']/test['pvp_battle_count']
    test['pve_lanch_per'] = test['pve_lanch_count']/test['pve_battle_count']
    #test = test[['user_id','register_time','weekday','wood_change_per','stone_change_per','ivory_change_per','meat_change_per','magic_change_per','infantry_change_per','cavalry_change_per','shaman_change_per','wound_infantry_change_per','wound_cavalry_change_per','wound_shaman_change_per','general_acceleration_change_per','building_acceleration_change_per','reaserch_acceleration_change_per','training_acceleration_change_per','treatment_acceleration_change_per','build_level','sr_attack_level','sr_source_level','sr_level','battle_count','pvp_win_per','pve_win_per','pvp_lanch_per','pve_lanch_per','pay_price','pay_count','avg_online_minutes']]
    #test = test[['user_id','register_time','pay_price','pay_count','avg_online_minutes']]
    
    train.replace(np.inf,0)
    test.replace(np.inf,0)
    print("特征构造结束时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    return train,test

def process(data):
    import time
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= 3600
    data['regedit_diff'] = (a - min(a))
    
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    new['date'] = new['date'].apply(lambda x:x.split('-')[2])
    new['date_week'] = new.date.apply(lambda x:1 if x in ['27','28','03', '04','10','11','17','18','24','25'] else 0)
    data = pd.merge(data, new[['date_week', 'user_id']], on='user_id',how='left')
    
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    new['date'] = new['date'].apply(lambda x:x.split('-')[2])
    new['date_holiday'] = new.date.apply(lambda x:1 if x in ['14','15','16'] else 0)
    data = pd.merge(data, new[['date_holiday', 'user_id']], on='user_id',how='left')
    
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[1])
    new['date'] = new['date'].apply(lambda x:int(x.split(':')[0]))
    new['date_h_2'] = new.date.apply(lambda x:1 if ((x >= 4) & (x < 8) ) else 0)
    new['date_h_3'] = new.date.apply(lambda x:1 if ((x >= 8) & (x < 12) ) else 0)
    data = pd.merge(data, new[['date_h_2','date_h_3','user_id']], on='user_id',how='left')
    #del data['register_time']
    
    data['pay_price_ave'] = data['pay_price'] / data['pay_count']
    data['pay_price_ave'] = data['pay_price_ave'].fillna(0)
    data['pve_win_ave'] = data['pve_win_count'] / data['pve_battle_count']
    data['pve_win_ave'] = data['pve_win_ave'].fillna(0)

    data['pve_lanch_ave'] = data['pve_lanch_count'] / data['pve_battle_count']
    data['pve_lanch_ave'] = data['pve_lanch_ave'].fillna(0)
    
    data['pvp_win_ave'] = data['pvp_win_count'] / data['pvp_battle_count']
    data['pvp_win_ave'] = data['pvp_win_ave'].fillna(0)
    
    data['pvp_lanch_ave'] = data['pvp_lanch_count'] / data['pvp_battle_count']
    data['pvp_lanch_ave'] = data['pvp_lanch_ave'].fillna(0)
    
    return data

if __name__ == '__main__':
    df_train = pd.read_csv('data/tap_fun_train.csv',dtype=dic)
    df_test = pd.read_csv('data/tap_fun_test.csv',dtype=dic)
    df_train = process(df_train)
    df_test['prediction_pay_price'] = -1
    df_test = process(df_test)
    df_train,df_test = feature(df_train,df_test)
    #pred,model,a = lgb_train(df_train,df_test)
    #a,b,b2,pred = lgb_train(df_train,df_test)
    #bar_plot_analyse(df_train,'count',20)
    #train_new = df_train[df_train.prediction_pay_price==0]
    #a,b,b2,pred = lgb_train(train_new,df_test,df_train)
    train_new = df_train[df_train.pay_price>0]
    a2,b2,b22,pred2 = lgb_train(train_new,df_test,df_train)
    train_new = df_test[df_test.pay_price==0][['user_id']]
    train_new['new'] = 0
    pred = pred2.merge(train_new,how='left')
    pred = pred.fillna(1)
    pred['prediction_pay_price'] = pred['prediction_pay_price']*pred['new']
    pred = pred[['user_id','prediction_pay_price']]
    pred.to_csv('result.csv',index=False)
    #pred2.to_csv


    """
    df = df_train[df_train.pay_price >0]
    df['month'] = df['register_time'].str.split('-',expand=True)[1]
    df['day'] = df['register_time'].str.split('-',expand=True)[2]
    df['day'] = df['day'].str.split(' ',expand=True)[0]
    df['hour'] = df['register_time'].str.split(':',expand=True)[0]
    df['hour'] = df['hour'].str.split(' ',expand=True)[1]
    df['min'] = df['register_time'].str.split(':',expand=True)[1]
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df['min'] = df['min'].astype(int)

    b = [time.strftime('%A',time.strptime(a,'%Y-%m-%d %H:%M:%S')) for a in df['register_time']]
    df['weekday'] = b

    df3 = df[['weekday','pay_price','pay_count','prediction_pay_price']]
    df_train[df_train.avg_online_minutes < 15]
    """

    """
    #分析相关性
    a = pd.DataFrame()
    b = []
    c = []
    for i in df_train.columns:
	if pd.api.types.is_numeric_dtype(df_train[i]):
		b.append(i)
		c.append(df_train[].corr(df_train[i]))
    a['feature'] = b
    a['corr'] = c
    a2 = pd.DataFrame()
    b = []
    c = []
    for i in df_train.columns:
	if pd.api.types.is_numeric_dtype(df_train[i]):
		b.append(i)
		c.append(df_train['pay_price'].corr(df_train[i]))
    a2['feature'] = b
    a2['corr'] = c
    a = a.merge(a2,on='feature',how='left')
    """
    """
    #分析单变量的取值对目标变量的分布
    a = df_train[['pay_price','prediction_pay_price']].groupby('pay_price',as_index=False).count()
    plt.figure()
    pay_price = a.pay_price
    prediction_pay_price_count = a.prediction_pay_price
    plt.plot(pay_price,prediction_pay_price_count,'-*')
    plt.show()
    #index = a.index
    #plt.plot(index,pay_price,'-*')
    #plt.plot(index,prediction_pay_price_count,'-*')
    #b = ['pay_price','prediction_pay_price']
    #plt.legend(b)
    #plt.show()
    """

    #计算RMSE：mse_test=(np.sum((y_preditc-y_test)**2)/len(y_test))**0.5
    #(np.sum((b['pred4']-df_train['prediction_pay_price'])**2)/b.shape[0])**0.5
    












    

