import streamlit as st

st.set_page_config(page_title="System Fault Diagnosis System", page_icon="👋", layout="wide")

cols = st.columns(6)
cols[0].caption("By 网安小萝卜")
cols[-1].caption("第十二届软件杯大赛")

st.write("# System Fault Diagnosis System")

st.sidebar.success("Welcome!")

st.markdown(
    """
    **我们实现了一个基于机器学习的分布式系统故障诊断系统，本系统一共分为四个模块:)**
    - 数据传输: 我们支持数据集（.csv）和模型文件（.pkl）的上传和下载。
    - 模型训练: 我们支持使用云端数据训练模型，提供了多种模型选择方案。
    - 模型推理：我们支持指定云端模型和数据集进行分类工作，分类结果（.csv）支持下载。
    - 智能对话：我们提供了基于ChatGPT3的智能数据分析服务，选择数据集，just ask anything you want to know!
    
    **👈 从侧边栏选择一个模块** 来查看系统的具体示例。
    ### 动机
    
    - 大数据时代，分布式系统成为信息存储和处理的主流系统。
    - 如何对分布式系统进行高效、准确的运维，成为保障信息系统高效、可靠运行的关键问题。
    - 对分布式系统的故障数据进行分析，高效地分析并识别故障类别可以大大降低分布式系统运维工作的难度。
    
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
