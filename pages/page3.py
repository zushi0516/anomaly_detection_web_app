import numpy as np
import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt

#データ分析関連
df = pd.read_csv('./data/example.csv',index_col = 0)
st.dataframe(df)
# st.table(df)
# st.line_chart(df)

fig, ax = plt.subplots()
ax.hist(df['0'], bins=20)
ax.set_title('matplot graph')
st.pyplot(fig)