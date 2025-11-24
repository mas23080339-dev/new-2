import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1) Load data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("group6.csv")

    # Chuẩn hóa từ khóa
    df["Từ khóa"] = df["Từ khóa"].fillna("").str.replace(";", " ")

    # Gộp nội dung để TF-IDF hiểu
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("")
    )

    return df

df = load_data()

# ==== DEBUG: XEM CỘT & LINK ẢNH ====
st.write("CÁC CỘT:", df.columns.tolist())
st.write("5 LINK ẢNH:", df["link ảnh"].head())

# =========================
# 2) TF-IDF
# =========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["FullText"])

# =========================
# 3) Streamlit UI
# =========================
st.title("🎯 Hệ thống gợi ý sản phẩm (CBF) có hình ảnh")

user_query = st.text_input("Nhập mô tả hoặc tên sản phẩm bạn muốn tìm:")

if user_query:

    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranking = scores.argsort()[::-1]

    best_idx = ranking[0]

    # ============ 4) SẢN PHẨM GIỐNG NHẤT ============
    st.subheader("🎯 SẢN PHẨM GIỐNG NHẤT")

    # Hiển thị hình ảnh
    if "link ảnh" in df.columns and pd.notna(df.loc[best_idx, "link ảnh"]):
        st.image(df.loc[best_idx, "link ảnh"], width=250)

    st.write(f"**Tên:** {df.loc[best_idx, 'Tên sản phẩm']}")
    st.write(f"**Thương hiệu:** {df.loc[best_idx, 'Thương hiệu']}")
    st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
    st.write(f"**Giá:** {df.loc[best_idx, 'Giá']}")
    st.write(f"Similarity:** `{scores[best_idx]:.3f}`")


    # ============ 5) GỢI Ý TƯƠNG TỰ ============
    st.subheader("🔍 GỢI Ý SẢN PHẨM TƯƠNG TỰ")

    for idx in ranking[1:6]:

        if "link ảnh" in df.columns and pd.notna(df.loc[idx, "link ảnh"]):
            st.image(df.loc[idx, "link ảnh"], width=180)

        st.write(f"**Tên:** {df.loc[idx, 'Tên sản phẩm']}")
        st.write(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
        st.write(f"Giá: {df.loc[idx, 'Giá']}")
        st.write(f"Similarity: `{scores[idx]:.3f}`")
        st.write("---")
