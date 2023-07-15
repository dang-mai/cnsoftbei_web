import time

import streamlit as st
import os

from lightgbm import LGBMClassifier
from register import DATASET_DIR, MODEL_DIR

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="模型训练", page_icon="🌍", layout="wide")

cols = st.columns(6)
cols[0].caption("By NCC小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# 模型训练")

st.subheader("数据集", anchor="数据集")
_, _, datasets = next(os.walk(DATASET_DIR))
dataset_name = st.selectbox('选择要使用的数据集', datasets)

st.subheader("模型", anchor="模型")
models = ['GBDT', 'lightgbm', 'RandomForest', 'stacking']
model_type = st.selectbox('选择要使用的模型', models)

if 'predict' not in st.session_state:
    st.session_state.predict = None


def train():
    with process:
        with st.spinner("正在训练..."):
            # 从CSV文件中加载输入数据
            data = pd.read_csv(os.path.join(DATASET_DIR, dataset_name))
            # 处理空数据，将空数据填充为众数
            data = data.fillna(data.mode().iloc[0])
            data = data.drop(['sample_id', 'feature57', 'feature77', 'feature100'], axis=1)
            data = data.sample(frac=1, random_state=42)
            X = data.drop('label', axis=1)
            y = data['label']

            # 将数据拆分为训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == 'GBDT':
                gbc = GradientBoostingClassifier(learning_rate=0.05, max_depth=7, max_features=None, min_samples_leaf=7,
                                                 min_samples_split=20, n_estimators=200, subsample=0.75)
                # 训练模型
                gbc.fit(X_train, y_train)
                # 在测试集上评估模型
                y_pred = gbc.predict(X_test)
                st.session_state.predict = classification_report(y_test, y_pred)
            elif model_type == 'lightgbm':
                lgc = LGBMClassifier(lambda_l1=0, lambda_l2=0, learning_rate=0.1, max_depth=3, n_estimators=200,
                                     subsample=0.8)
                # 训练模型
                lgc.fit(X_train, y_train)
                # 在测试集上评估模型
                y_pred = lgc.predict(X_test)
                st.session_state.predict = classification_report(y_test, y_pred)
            elif model_type == 'RandomForest':
                rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                # 训练模型
                rf.fit(X_train, y_train)
                # 在测试集上评估模型
                y_pred = rf.predict(X_test)
                st.session_state.predict = classification_report(y_test, y_pred)
            elif model_type == 'stacking':
                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
                    ('dt', DecisionTreeClassifier()),
                    ('ligh', LGBMClassifier(learning_rate=0.1, n_estimators=200, max_depth=3, lambda_l1=0, lambda_l2=0,
                                            subsample=0.8)),
                    ('gbdt', GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=7,
                                                        max_features=None, min_samples_leaf=7, min_samples_split=20,
                                                        subsample=0.75))
                ]
                stc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=1, penalty='l1', 
                                                                                                   solver='liblinear'),
                                         stack_method='predict_proba')
                # 训练模型
                stc.fit(X_train, y_train)
                # 在测试集上评估模型
                y_pred = stc.predict(X_test)
                st.session_state.predict = classification_report(y_test, y_pred)


st.button("开始训练", on_click=train)
process = st.empty()
if st.session_state.predict is not None:
    st.text(st.session_state.predict)
