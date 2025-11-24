import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1) Load data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # Chuẩn hóa từ khóa (thay ; bằng space)
    df["Từ khóa"] = df["Từ khóa"].fillna("").str.replace(";", " ")

    # Gộp nội dung để TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("")
    )

    # Chuẩn hóa cột Link ảnh
    df["Link ảnh"] = df["Link ảnh"].fillna("").str.strip()

    return df

df = load_data()

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

    # ============ SẢN PHẨM GIỐNG NHẤT ============
    st.subheader("🎯 SẢN PHẨM GIỐNG NHẤT")

    # Hiển thị ảnh
    if pd.notna(df.loc[best_idx, "Link ảnh"]) and df.loc[best_idx, "Link ảnh"] != "":
        st.image(df.loc[best_idx, "Link ảnh"], width=250)

    st.write(f"**Tên:** {df.loc[best_idx, 'Tên sản phẩm']}")
    st.write(f"**Thương hiệu:** {df.loc[best_idx, 'Thương hiệu']}")
    st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
    st.write(f"**Giá:** {df.loc[best_idx, 'Giá']}")
    st.write(f"**Similarity:** `{scores[best_idx]:.3f}`")

    # ============ GỢI Ý TƯƠNG TỰ ============
    st.subheader("🔍 GỢI Ý SẢN PHẨM TƯƠNG TỰ")

    for idx in ranking[1:6]:
        if pd.notna(df.loc[idx, "Link ảnh"]) and df.loc[idx, "Link ảnh"] != "":
            st.image(df.loc[idx, "Link ảnh"], width=180)

        st.write(f"**Tên:** {df.loc[idx, 'Tên sản phẩm']}")
        st.write(f"Thương hiệu: {df.loc[idx, 'Thương hiệu']}")
        st.write(f"Giá: {df.loc[idx, 'Giá']}")
        st.write(f"Similarity: `{scores[idx]:.3f}`")
        st.write("---")
