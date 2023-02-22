# ライブラリの読み込み
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# グラフ描画設定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.grid"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.right"] = "True"
plt.rcParams["xtick.top"] = "True"
plt.rcParams["figure.subplot.left"] = 0.14  # 余白
plt.rcParams["figure.subplot.bottom"] = 0.14  # 余白
plt.rcParams["figure.subplot.right"] = 0.80  # 余白
plt.rcParams["figure.subplot.top"] = 0.81  # 余白

# タイトル
st.title("異常検知用可視化アプリ")
st.caption("csvファイルに含まれる多次元データをPCA(主成分分析)を使って二次元or三次元上にプロットします")

# ファイルアップロード
uploaded_files = st.file_uploader("ファイルアップロード", type="csv")

# ファイルがアップロードされたら以下が実行される
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    # データフレームを表示
    st.markdown("### 入力データ")
    st.dataframe(df)
    # matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### 可視化 単変量")
    # データフレームのカラムを選択オプションに設定する
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    # 選択した変数を用いてmtplotlibで可視化
    fig = plt.figure(figsize=(12, 8))
    plt.plot(df[x], df[y])
    plt.xlabel(x, fontsize=18)
    plt.ylabel(y, fontsize=18)
    st.pyplot(fig)

    st.markdown("### データ範囲")
    # スライダー
    train_min_value, train_max_value = st.slider(
        "学習データ範囲",
        min_value=0,
        max_value=len(df),
        value=(0, len(df)),
    )
    st.write(f"学習データとして: {train_min_value} ~ {train_max_value}を使用します")

    test_min_value, test_max_value = st.slider(
        "可視化データ範囲",
        min_value=0,
        max_value=len(df),
        value=(train_max_value, len(df)),
    )
    st.write(f"可視化データとして: {test_min_value} ~ {test_max_value}を使用します")

    # フロー制御
    if train_min_value == train_max_value or test_min_value == test_max_value:
        st.warning("少なくとも一つのデータを選択してください")
        # 条件を満たないときは処理を停止する
        st.stop()

    st.markdown("### 変数選択")
    # 全ての変数を使うか、選択した変数のみを使うか
    column_option = st.selectbox("全ての変数を使いますか？", ["全て", "指定"])

    if column_option == "全て":
        df_train = df.iloc[train_min_value:train_max_value, :]
        df_test = df.iloc[test_min_value:test_max_value, :]
        column_flag = True
    if column_option == "指定":
        column_flag = False
        column_option_select = st.multiselect("インプットする変数を選択してください（複数選択可）", df_columns)
        if column_option_select:
            st.write(f"以下の変数をインプットします: {column_option_select}")
            df_train = df[column_option_select].iloc[train_min_value:train_max_value, :]
            df_test = df[column_option_select].iloc[test_min_value:test_max_value, :]
            column_flag = True

    if column_flag:

        # フロー制御
        category = df_train.select_dtypes(include=["object"]).columns
        if category.shape[0] != 0:
            st.warning(f"カテゴリ変数{category.tolist()}が含まれています。カテゴリ変数を除いてください")
            # 条件を満たないときは処理を停止する
            st.stop()

        # 圧縮次元選択
        st.markdown("### 圧縮次元選択")
        plot_dim = st.selectbox(
            "圧縮する次元を選択してください ※'三次元'はデータ数が多い場合、表示に時間がかかります", ["二次元", "三次元", "両方"]
        )

        # フロー制御
        if (plot_dim == "二次元") & (df_train.shape[1] < 3):
            st.warning("二次元に圧縮するには、変数が3つ以上必要です")
            # 条件を満たないときは処理を停止する
            st.stop()

        # フロー制御
        if (plot_dim == "三次元" or plot_dim == "両方") & (df_train.shape[1] < 4):
            st.warning("三次元に圧縮するには、変数が4つ以上必要です")
            # 条件を満たないときは処理を停止する
            st.stop()

        st.markdown("#### 可視化を実行します")
        execute = st.button("実行")
        pca = PCA()
        ss = preprocessing.StandardScaler()
        X_train = pd.DataFrame(ss.fit_transform(df_train))
        X_test = pd.DataFrame(ss.fit_transform(df_test))
        # 実行ボタンを押したら下記が進む
        if execute:
            pca.fit(X_train)
            test_pca = pca.transform(X_test)

            # 累積寄与率
            df_pc = pd.DataFrame(
                test_pca,
                columns=["PC{}".format(x + 1) for x in range(len(X_test.columns))],
            )
            cumsum_list = pd.Series(
                [0] + list(np.cumsum(pca.explained_variance_ratio_))
            )

            text = df_pc.index

            fig = plt.figure(figsize=(8, 8))
            plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
            plt.hlines(cumsum_list[2], 0, 2, color="r", linestyles="dotted")
            plt.vlines(2, 0, cumsum_list[2], color="r", linestyles="dotted")
            plt.hlines(cumsum_list[3], 0, 3, color="g", linestyles="dotted")
            plt.vlines(3, 0, cumsum_list[3], color="g", linestyles="dotted")
            plt.xlabel("Number of principal components")
            plt.ylabel("Cumulative contribution rate")
            plt.gca().set_xlim(left=0)
            plt.gca().set_ylim(bottom=0)
            plt.grid()
            st.pyplot(fig)

            if plot_dim == "二次元":
                st.write(f"累積寄与率：{ cumsum_list[2]}")
                fig = px.scatter(
                    df_pc,
                    x="PC1",
                    y="PC2",
                    hover_name=text,
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            elif plot_dim == "三次元":
                st.write(f"累積寄与率：{ cumsum_list[3]}")
                fig = px.scatter_3d(
                    df_pc,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    hover_name=text,
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            elif plot_dim == "両方":
                st.write(f"二次元累積寄与率：{ cumsum_list[2]}")
                text = df_pc.index
                fig = px.scatter(
                    df_pc,
                    x="PC1",
                    y="PC2",
                    hover_name=text,
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                st.write(f"三次元累積寄与率：{ cumsum_list[3]}")
                fig = px.scatter_3d(
                    df_pc,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    hover_name=text,
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
