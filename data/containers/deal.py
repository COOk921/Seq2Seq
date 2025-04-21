import pandas as pd


# 读取Excel文件
df = pd.read_excel('data/tsp/LOAD.xlsx')

# 按Unit Nbr分组
grouped = df.groupby('Unit Nbr')

# 用于存储结果的列表
result = []

for unit_nbr, group in grouped:
    if len(group) == 1:
        # 如果该Unit Nbr只有一行数据，直接添加到结果中
        result.append(group)
    else:
        # 如果有两行或以上数据，互换To Position的值
        group['From Position'] = group['From Position'].iloc[::-1].values
        result.append(group)

# 合并处理后的分组数据
new_df = pd.concat(result, ignore_index=True)

# 将结果保存到新的Excel文件
new_df.to_excel('data/tsp/LOAD2.xlsx', index=False)