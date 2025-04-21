import json 
import pandas as pd
import pdb
import torch
def turn_to_json(data):
    json_data = data.to_json(orient='records', force_ascii=False, indent=4)
    with open('data/containers/data.json', 'w') as f:
        f.write(json_data)

def read_json():
    with open('data/containers/data.json', 'r') as f:
        data = json.load(f)
    return data

def read_excel(file_path,sheet_name):
    data = pd.read_excel(file_path,sheet_name=sheet_name)
    return data



def deal_container_data():

    file_path = 'data/containers/DSCH.xlsx'
    #file_path = 'data/containers/data.xlsx'
    data = read_excel(file_path,sheet_name='Sheet3')
    # del 特征
    del_features = ['Time Completed',
                    'Unit Nbr',
                    'From Position',
                    'To Position',
                    'Length',
                    'Height',
                    'Width'
                    ]
    
    # one-hot编码的特征
    one_hot_features = ['Crane CHE Name',
                        'Fetch CHE Name',
                        'Unit IB Actual Visit',
                        'Unit POD', 
                        'Unit Type Length',
                        'WI POW',
                        'Put CHE Name',
                        'yard',
                        "pos1"
                        ]

    # 连续特征
    continuous_features = ['Unit Weight (kg)']


    # 对one-hot编码的特征进行one-hot编码
    for feature in one_hot_features:
        if feature not in data.columns:
            print(f"Warning: {feature} not in data columns")
            continue
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes

    # 对连续特征进行归一化
    for feature in continuous_features:
        data[feature] = data[feature].apply(lambda x: (x - data[feature].min()) / (data[feature].max() - data[feature].min()))


    # 添加label 特征，根据时间升序排序
    data['label'] = data['Time Completed'].rank(method='dense', ascending=True)

    # 删除del_features中的特征
    data = data.drop(del_features, axis=1)
    print(data.columns)
    print(data.head(3))
    data = torch.tensor(data.values)

    return data

deal_container_data()

