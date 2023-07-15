import time

import streamlit as st
import os
from register import DATASET_DIR, MODEL_DIR


st.set_page_config(page_title="模型训练", page_icon="🌍", layout="wide")

cols = st.columns(6)
cols[0].caption("By NCC小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# 模型训练")

st.subheader("数据集", anchor="数据集")
_, _, datasets = next(os.walk(DATASET_DIR))
dataset_name = st.selectbox('选择要使用的数据集', datasets)

st.subheader("模型", anchor="模型")
models = ['knn', 'lightgbm', 'svm', 'xgboost']
model_type = st.selectbox('选择要使用的模型', models)

st.write(model_type)

def train():
    with process:
        with st.spinner("正在训练..."):
            time.sleep(5)


st.button("开始训练", on_click=train)
process = st.empty()
