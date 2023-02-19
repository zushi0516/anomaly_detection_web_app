import numpy as np
import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt

with st.form(key = 'profile_form'):
    #テキストボックス
    name = st.text_input('名前')
    adress = st.text_input('住所')
    
    #セレクトボックス
    plot_dim = st.radio(
        '可視化方法',
        ('二次元','三次元','両方')
    )
    #複数選択
    hobby = st.multiselect(
        '趣味',
        ('スポーツ','読書','プログラミング','アニメ・映画')
    )
    
    #チェックボックス 
    mail_subscribe = st.checkbox('メルマガ購読')
    
    #日付
    start_date = st.date_input(
        '開始日',
        datetime.date(2023,2,1)
    )
    
    #カラーピッカー
    color = st.color_picker('デフォルトカラー','#00f900')
    #スライダー
    height = st.slider('訓練データ範囲',min_value = 110, max_value = 220)
    
    #ボタン
    submit_btn = st.form_submit_button('送信')
    cancel_btn = st.form_submit_button('キャンセル')

    if submit_btn:
        st.text(f'ようこそ、{name}さん！')
        st.text(f'{plot_dim}で可視化します')
        st.text(f'趣味:{", ".join(hobby)}')