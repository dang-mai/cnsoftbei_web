import streamlit as st

st.set_page_config(page_title="System Fault Diagnosis System", page_icon="👋", layout="wide")

cols = st.columns(6)
cols[0].caption("By 网安小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# System Fault Diagnosis System: 基于机器学习的分布式系统故障诊断系统")

st.sidebar.success("Welcome!")

st.markdown(
    """
    **System Fault Diagnosis System 是一个基于机器学习的分布式系统故障诊断系统，系统分为三个模块——xxx。**
    - xxx: xxx。
    - xxx: xxx。  
    **👈 从侧边栏选择一个模块** 来查看系统的具体示例。
    ### 动机
    - xxx
    - xxx
    - xxx
    
    因此，我们设计并实现了基于机器学习的分布式系统故障诊断系统。
    ### 代码
    System Fault Diagnosis System的Python参考实现在GitHub(https://github.com/dang-mai/xxx) 上可获得。
    ### 数据集
    我们公开System Fault Diagnosis System的数据集:
    - xxx数据集(https://xxx/dataset.zip)
    ### 贡献者
    NCC小萝卜团队
    ### 参考
    基于机器学习的分布式系统故障诊断系统报告. NCC小萝卜团队.
""")
