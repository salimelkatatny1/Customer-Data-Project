import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="🛒 Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")



# =========================
# Load model & data
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("best_regressor.pkl")
    preprocess = joblib.load("preprocess.pkl")
    mae = joblib.load("regression_mae.pkl")
    return model, preprocess, mae

@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

model, preprocess, mae = load_models()
df = load_data()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# =========================
# Title
# =========================
st.markdown("""
# 🛒 Customer Purchase Analytics Dashboard
### 🤖 Prediction • 📊 EDA • 🧹 Cleaning • 📈 Visualization • 💬 Chatbot
---
""")

# =========================
# Tabs
# =========================
tabE, tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧠 Project Explanation",
    "📋 Data Summary",
    "🧹 Data Cleaning",
    "📊 EDA",
    "📊 Dashboard",
    "📈 Manual Plot Builder",
    "🛒 Prediction",
    "🤖 Chatbot"
])

# =====================================================
# 🧠 TAB E: PROJECT EXPLANATION
# =====================================================
with tabE:
    st.header("🧠 Project Explanation")
    st.markdown("""
    ### 🛒 Customer Purchase Analytics Project

    Welcome to the **Customer Purchase Analytics Dashboard**!  
    This project is designed to help businesses understand customer behavior and
    predict how much a customer is likely to spend.

    ---
    ### 🎯 Goal of the Project
    The main objectives are:
    - 📊 Analyze customer data to discover patterns and insights.
    - 🧠 Understand how factors like **age, income, loyalty, promotions, and satisfaction**
      affect purchasing behavior.
    - 🤖 Build a machine learning model to **predict the purchase amount** of a customer.
    - 📈 Provide an interactive dashboard for decision makers.

    ---
    ### 🗂️ About the Data
    The dataset contains information about customers such as:
    - 🎂 Age  
    - 🚻 Gender  
    - 💰 Income  
    - 🎓 Education  
    - 🌍 Region  
    - 🏷️ Loyalty status  
    - 🔁 Purchase frequency  
    - 📦 Product category  
    - 🎁 Promotion usage  
    - ⭐ Satisfaction score  

    Each row represents **one customer**, and the target variable is:
    👉 **`purchase_amount`** — how much the customer spent.

    ---
    ### 🤖 What Are We Predicting?
    We predict:
    > 💵 **The expected purchase amount of a customer**

    This helps businesses:
    - 🎯 Target high-value customers  
    - 🎁 Design better promotions  
    - 📦 Optimize product offerings  
    - 💼 Improve revenue forecasting  

    ---
    ### 🧠 Why This Prediction?
    Knowing how much a customer may spend allows companies to:
    - Increase profitability  
    - Improve customer satisfaction  
    - Personalize marketing strategies  
    - Make data-driven decisions  

    ---
    ### ⚙️ How It Works
    - 🧹 Data is cleaned and preprocessed.
    - 🔄 Categorical features are encoded, numeric features scaled.
    - 🤖 A trained regression model (saved as `best_regressor.pkl`) learns patterns.
    - 📈 The model predicts purchase amount for new customers.
    - 📉 Model performance is evaluated using **MAE (Mean Absolute Error)**.

    ---
    ### 📊 What You Can Do in This App
    - 📋 View data summary and statistics  
    - 🧹 Check missing values and outliers  
    - 📊 Explore data with EDA charts  
    - 📈 Build your own plots  
    - 🤖 Predict purchase amount interactively  
    - 💬 Ask the smart chatbot questions  

    ---
    ### 🚀 Outcome
    This dashboard transforms raw customer data into:
    - 📊 Actionable insights  
    - 🤖 Smart predictions  
    - 💡 Better business understanding  

    ---
    👨‍💻 Developed by **Salim Elkatatny**  
    📚 Machine Learning & Data Analytics Project
    """)

# =====================================================
# 📊 TAB 3: DASHBOARD
# =====================================================
with tab3:
    st.header("📊 Dataset Dashboard Overview")

    # ===== KPIs =====
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Customers", df.shape[0])
    k2.metric("🧾 Features", df.shape[1])
    k3.metric("💰 Avg Purchase", f"{df['purchase_amount'].mean():.2f}")
    k4.metric("⭐ Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}")

    st.markdown("---")

    # ===== Row 1 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("💰 Purchase Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["purchase_amount"], kde=True, ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("🏷️ Purchase by Loyalty")
        fig, ax = plt.subplots()
        sns.boxplot(x="loyalty_status", y="purchase_amount", data=df, ax=ax)
        st.pyplot(fig)

    with c3:
        st.subheader("🔁 Purchase Frequency")
        fig, ax = plt.subplots()
        sns.countplot(x="purchase_frequency", data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ===== Row 2 =====
    c4, c5, c6 = st.columns(3)
    with c4:
        st.subheader("🚻 Avg Purchase by Gender")
        fig, ax = plt.subplots()
        sns.barplot(x="gender", y="purchase_amount", data=df, estimator=np.mean, ax=ax)
        st.pyplot(fig)

    with c5:
        st.subheader("🎁 Promo vs Purchase")
        fig, ax = plt.subplots()
        sns.boxplot(x="promotion_usage", y="purchase_amount", data=df, ax=ax)
        st.pyplot(fig)

    with c6:
        st.subheader("📦 Category vs Purchase")
        fig, ax = plt.subplots()
        sns.violinplot(x="product_category", y="purchase_amount", data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ===== Row 3 =====
    c7, c8, c9 = st.columns(3)

    with c7:
        st.subheader("🎂 Age vs Purchase")
        fig, ax = plt.subplots()
        sns.scatterplot(x="age", y="purchase_amount", data=df, ax=ax)
        st.pyplot(fig)

    with c8:
        st.subheader("⭐ Satisfaction vs Purchase")
        fig, ax = plt.subplots()
        sns.scatterplot(x="satisfaction_score", y="purchase_amount", data=df, ax=ax)
        st.pyplot(fig)

    with c9:
        st.subheader("📊 Correlation Heatmap")
        corr_df = df[num_cols].dropna().corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)


# =====================================================
# 🛒 TAB 5: PREDICTION
# =====================================================
with tab5:
    st.header("🤖 Predict Purchase Amount")

    st.sidebar.header("⚙️ Customer Info")

    age = st.sidebar.slider("🎂 Age", int(df["age"].min()), int(df["age"].max()), int(df["age"].mean()))
    gender = st.sidebar.selectbox("🚻 Gender", df["gender"].unique())
    income = st.sidebar.number_input("💰 Income", min_value=0.0, value=float(df["income"].median()),step=500.0)
    education = st.sidebar.selectbox("🎓 Education", df["education"].unique())
    region = st.sidebar.selectbox("🌍 Region", df["region"].unique())
    loyalty_status = st.sidebar.selectbox("🏷️ Loyalty Status", df["loyalty_status"].unique())
    purchase_frequency = st.sidebar.selectbox("🔁 Purchase Frequency", df["purchase_frequency"].unique())
    product_category = st.sidebar.selectbox("📦 Product Category", df["product_category"].unique())
    promotion_usage = st.sidebar.slider("🎁 Promotions Used", 0, int(df["promotion_usage"].max()), int(df["promotion_usage"].median()))
    satisfaction_score = st.sidebar.slider(
        "⭐ Satisfaction Score",
        float(df["satisfaction_score"].min()),
        float(df["satisfaction_score"].max()),
        float(df["satisfaction_score"].mean())
    )

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "income": income,
        "education": education,
        "region": region,
        "loyalty_status": loyalty_status,
        "purchase_frequency": purchase_frequency,
        "product_category": product_category,
        "promotion_usage": promotion_usage,
        "satisfaction_score": satisfaction_score
    }])

    st.subheader("📋 Customer Snapshot")
    st.dataframe(input_data)

    if st.button("🚀 Predict"):
        X_prep = preprocess.transform(input_data)
        pred = float(model.predict(X_prep)[0])
        confidence = max(0.0, 1 - (mae / (abs(pred) + 1e-6)))
        confidence = min(confidence, 1.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("💵 Predicted Amount", f"${pred:,.2f}")
        c2.metric("📉 Expected Error", f"± ${mae:.2f}")
        c3.metric("📊 Confidence", f"{confidence*100:.1f}%")
        st.progress(int(confidence * 100))

# =====================================================
# 📊 TAB 2: EDA
# =====================================================
with tab2:
    st.header("📊 Exploratory Data Analysis")

    col = st.selectbox("🔢 Numeric column", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    cat = st.selectbox("📦 Category for boxplot", cat_cols)
    fig, ax = plt.subplots()
    sns.boxplot(x=df[cat], y=df[col], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    cat2 = st.selectbox("📊 Count plot column", cat_cols)
    fig, ax = plt.subplots()
    sns.countplot(x=df[cat2], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =====================================================
# 🧹 TAB 1: DATA CLEANING
# =====================================================
with tab1:
    st.header("🧹 Data Cleaning")

    st.subheader("❓ Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("🔁 Duplicates")
    st.write("Duplicate rows:", df.duplicated().sum())

    col_out = st.selectbox("🚨 Outlier column", num_cols)
    Q1, Q3 = df[col_out].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col_out] < Q1 - 1.5*IQR) | (df[col_out] > Q3 + 1.5*IQR)]
    st.write("Outliers detected:", outliers.shape[0])
    st.dataframe(outliers.head())

# =====================================================
# 📈 TAB 4: MANUAL PLOT
# =====================================================
with tab4:
    st.header("📈 Manual Plot Builder")

    plot_type = st.selectbox(
        "📊 Plot Type",
        ["Histogram", "Boxplot", "Violin", "Scatter", "Line", "Bar (mean)"]
    )

    if plot_type in ["Histogram", "Boxplot", "Violin"]:
        x = st.selectbox("Numeric column", num_cols)
        y = None
    else:
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis (numeric)", num_cols)

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(df[x], kde=True, ax=ax)
    elif plot_type == "Boxplot":
        sns.boxplot(y=df[x], ax=ax)
    elif plot_type == "Violin":
        sns.violinplot(y=df[x], ax=ax)
    elif plot_type == "Scatter":
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
    elif plot_type == "Line":
        sns.lineplot(x=df[x], y=df[y], ax=ax)
    elif plot_type == "Bar (mean)":
        sns.barplot(x=df[x], y=df[y], estimator=np.mean, ax=ax)
        plt.xticks(rotation=45)

    ax.set_title(plot_type)
    st.pyplot(fig)

# =====================================================
# 📋 TAB 0: SUMMARY
# =====================================================
with tab0:
    st.header("📋 Dataset Summary")

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.subheader("🧮 Statistics")
    st.dataframe(df.describe())
    st.subheader("🏷️ Data Types")
    st.dataframe(df.dtypes.astype(str))
    st.subheader("👀 Preview")
    st.dataframe(df.head(10))

# =====================================================
# 🤖 TAB 6: Offline Smart Chatbot (No API)
# =====================================================
with tab6:
    st.header("🤖 Smart Data Assistant")
    st.markdown("Select a question below and I’ll analyze the dataset for you 📊✨")

    questions = [
        "📊 What is the average purchase amount?",
        "👥 How many customers are in the dataset?",
        "💰 What is the maximum purchase amount?",
        "💵 What is the minimum purchase amount?",
        "⭐ What is the average satisfaction score?",
        "🏷️ Which loyalty status spends the most on average?",
        "🚻 Do males or females spend more?",
        "🌍 Which region has the highest average purchase?",
        "📦 Which product category is most popular?",
        "🔁 What is the most common purchase frequency?",
        "🎁 Do promotions increase purchase amount?",
        "📈 What is the correlation between income and purchase?",
        "🎂 What is the average age of customers?",
        "📊 Show summary statistics of numeric features",
        "🧠 Which numeric feature correlates most with purchase amount?"
    ]

    q = st.selectbox("❓ Choose a question:", questions)

    if st.button("💬 Get Answer"):
        with st.spinner("Analyzing... 🤔"):

            if q == questions[0]:
                ans = f"📊 The average purchase amount is **${df['purchase_amount'].mean():.2f}**."

            elif q == questions[1]:
                ans = f"👥 There are **{df.shape[0]} customers** in the dataset."

            elif q == questions[2]:
                ans = f"💰 The maximum purchase amount is **${df['purchase_amount'].max():.2f}**."

            elif q == questions[3]:
                ans = f"💵 The minimum purchase amount is **${df['purchase_amount'].min():.2f}**."

            elif q == questions[4]:
                ans = f"⭐ The average satisfaction score is **{df['satisfaction_score'].mean():.2f}**."

            elif q == questions[5]:
                top = df.groupby("loyalty_status")["purchase_amount"].mean().idxmax()
                val = df.groupby("loyalty_status")["purchase_amount"].mean().max()
                ans = f"🏷️ **{top}** loyalty customers spend the most on average: **${val:.2f}**."

            elif q == questions[6]:
                g = df.groupby("gender")["purchase_amount"].mean()
                ans = f"🚻 Average spend:\n- Male: ${g.get('Male',0):.2f}\n- Female: ${g.get('Female',0):.2f}"

            elif q == questions[7]:
                r = df.groupby("region")["purchase_amount"].mean()
                ans = f"🌍 **{r.idxmax()}** region has the highest average purchase: **${r.max():.2f}**."

            elif q == questions[8]:
                top_cat = df["product_category"].value_counts().idxmax()
                ans = f"📦 The most popular product category is **{top_cat}**."

            elif q == questions[9]:
                freq = df["purchase_frequency"].value_counts().idxmax()
                ans = f"🔁 The most common purchase frequency is **{freq}**."

            elif q == questions[10]:
                promo = df.groupby("promotion_usage")["purchase_amount"].mean()
                ans = f"🎁 Avg without promo: ${promo.get(0,0):.2f}\n🎁 Avg with promo: ${promo.get(1,0):.2f}"

            elif q == questions[11]:
                corr = df["income"].corr(df["purchase_amount"])
                ans = f"📈 Correlation between income and purchase amount is **{corr:.3f}**."

            elif q == questions[12]:
                ans = f"🎂 The average age of customers is **{df['age'].mean():.1f} years**."

            elif q == questions[13]:
                ans = "📊 Summary Statistics:\n" + df.describe().round(2).to_string()

            elif q == questions[14]:
                corrs = df.select_dtypes(np.number).corr()["purchase_amount"].drop("purchase_amount")
                top_feat = corrs.abs().idxmax()
                ans = f"🧠 **{top_feat}** has the strongest correlation with purchase amount: **{corrs[top_feat]:.3f}**."

            st.success("✅ Answer:")
            st.markdown(ans)



# =========================
# Footer
# =========================
st.markdown("""
---
👨‍💻 Developed by **Salim Elkatatny**  
📚 ML Project | Customer Analytics Dashboard  
✨ Streamlit • XGBoost • Data Science
""")
