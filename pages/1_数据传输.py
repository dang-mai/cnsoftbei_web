import streamlit as st
import os
from register import DATASET_DIR, MODEL_DIR

st.set_page_config(page_title="数据传输", page_icon="📈", layout="wide")

cols = st.columns(6)
cols[0].caption("By NCC小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# 数据传输")
st.write("""**上传/下载 数据集以及模型文件**""")

st.subheader("数据集", anchor="数据集")
_, _, datasets = next(os.walk(DATASET_DIR))
dataset_name = st.selectbox('选择要下载的数据集', datasets)
st.download_button(
    label="Download data as CSV",
    data=open(os.path.join(DATASET_DIR, dataset_name), 'rb').read(),
    file_name=dataset_name,
    mime='text/csv',
)
st.divider()
st.text("选择要上传的数据集")
dataset_file = st.file_uploader("Choose a CSV file")
if dataset_file is not None:
    # To read file as bytes:
    bytes_data = dataset_file.read()

    with open(os.path.join(DATASET_DIR, dataset_file.name), "wb") as f:
        f.write(bytes_data)

st.subheader("模型", anchor="模型")
_, _, models = next(os.walk(MODEL_DIR))
model_name = st.selectbox('选择要下载的模型', models)
st.download_button(
    label="Download Model File",
    data=open(os.path.join(MODEL_DIR, model_name), 'rb').read(),
    file_name=model_name,
    mime='text/csv',
)
st.divider()
st.text("选择要上传的模型")
model_file = st.file_uploader("Choose a Model file")
if model_file is not None:
    # To read file as bytes:
    bytes_data = model_file.read()
    with open(os.path.join(DATASET_DIR, model_file.name), "wb") as f:
        f.write(bytes_data)
