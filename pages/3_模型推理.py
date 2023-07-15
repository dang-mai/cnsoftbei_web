import streamlit as st
import os
import pandas as pd
import pickle
from register import DATASET_DIR, MODEL_DIR

st.set_page_config(page_title="模型推理", page_icon="📊", layout="wide")

if 'predict' not in st.session_state:
    st.session_state.predict = None

cols = st.columns(6)
cols[0].caption("By 网安小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# 模型推理")
st.subheader("数据集", anchor="数据集")
_, _, datasets = next(os.walk(DATASET_DIR))
dataset_name = st.selectbox('选择要使用的数据集', datasets)

st.subheader("模型", anchor="模型")
_, _, models = next(os.walk(MODEL_DIR))
model_name = st.selectbox('选择要使用的模型', models)


def inference():
    with process:
        with st.spinner("正在推理..."):
            # 从CSV文件中加载输入数据
            input_data = pd.read_csv(os.path.join(DATASET_DIR, dataset_name))
            # 处理空数据，将空数据填充为众数
            input_data = input_data.fillna(input_data.mode().iloc[0])
            input_data = input_data.drop(['sample_id', 'feature57', 'feature77', 'feature100'], axis=1)
            # 加载模型文件
            with open(os.path.join(MODEL_DIR, model_name), 'rb') as f:
                model = pickle.load(f)
            # 使用模型进行预测
            st.session_state.predict = model.predict(input_data)


st.button("开始推理", on_click=inference)
process = st.empty()
if st.session_state.predict is not None:
    dataset_name_out = dataset_name[0:-5] + ".out"
    st.write(st.session_state.predict)
    st.session_state.predict_df = pd.DataFrame(st.session_state.predict)
    st.session_state.predict_df.to_csv(os.path.join(DATASET_DIR, dataset_name_out))
