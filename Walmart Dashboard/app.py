import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title(" Market Trending Dashboard - Walmart Sales")

@st.cache_data
def load_data():
    df = pd.read_csv("Walmart.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

df = load_data()

st.sidebar.header("Filter Market")

store_option = st.sidebar.selectbox(
    "Select Store",
    ["All"] + list(df['Store'].unique())
)

if store_option != "All":
    df = df[df['Store'] == store_option]

total_sales = df['Weekly_Sales'].sum()
avg_sales = df['Weekly_Sales'].mean()
best_store = df.groupby('Store')['Weekly_Sales'].sum().idxmax()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Average Weekly Sales", f"${avg_sales:,.0f}")
col3.metric("Top Store", best_store)

st.subheader(" Weekly Sales Trend")

trend = df.groupby('Date')['Weekly_Sales'].sum()

fig1 = plt.figure()
trend.plot()
plt.title("Sales Over Time")
st.pyplot(fig1)

st.subheader(" Store Performance Comparison")

store_sales = df.groupby('Store')['Weekly_Sales'].sum()

fig2 = plt.figure()
store_sales.sort_values().plot(kind='barh')
st.pyplot(fig2)

st.subheader("🎄 Holiday Impact Analysis")

fig3 = plt.figure()
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
st.pyplot(fig3)

st.subheader(" Market Driver Correlation")

fig4 = plt.figure()
sns.heatmap(df.corr(), annot=True)
st.pyplot(fig4)

st.success("Market Trend Dashboard Loaded Successfully ")