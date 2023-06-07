import pandas as pd
import numpy as np
import pickle

# 从CSV文件中加载输入数据
input_data = pd.read_csv('input.csv')

# 处理空数据，将空数据填充为0
input_data = input_data.fillna(0)

input_data = input_data.drop(['sample_id', 'feature57', 'feature77', 'feature100'], axis=1)

# 加载模型文件
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 使用模型进行预测
predicted = model.predict(input_data)

# 输出预测结果
print(predicted)