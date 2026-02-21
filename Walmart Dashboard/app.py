import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Walmart Market Trend Dashboard", layout="wide")

st.title(" Walmart Sales Trend & Forecast Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Walmart.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

df = load_data()

st.sidebar.header("Filter Options")

store_option = st.sidebar.selectbox(
    "Select Store",
    options=["All"] + list(df['Store'].unique())
)

if store_option != "All":
    df = df[df['Store'] == store_option]

total_sales = df['Weekly_Sales'].sum()
avg_sales = df['Weekly_Sales'].mean()
max_store = df.groupby('Store')['Weekly_Sales'].sum().idxmax()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Average Weekly Sales", f"${avg_sales:,.0f}")
col3.metric("Best Performing Store", max_store)

st.subheader("Sales Trend Over Time")

trend_data = df.groupby('Date')['Weekly_Sales'].sum()

fig1 = plt.figure()
trend_data.plot()
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Weekly Sales Trend")
st.pyplot(fig1)

st.subheader("Holiday vs Non-Holiday Sales")

fig2 = plt.figure()
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
st.pyplot(fig2)

st.subheader("Correlation Heatmap")

fig3 = plt.figure()
sns.heatmap(df.corr(), annot=True)
st.pyplot(fig3)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

df['Lag_1'] = df['Weekly_Sales'].shift(1)
df['Rolling_Mean_4'] = df['Weekly_Sales'].rolling(4).mean()

df = df.dropna()

st.subheader("XGBoost Forecast")

features = ['Store', 'Holiday_Flag', 'Temperature',
            'Fuel_Price', 'CPI', 'Unemployment',
            'Year', 'Month', 'Week',
            'Lag_1', 'Rolling_Mean_4']

X = df[features]
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
st.write(f"Model MAE: {mae:,.2f}")

fig4 = plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("XGBoost Forecast vs Actual")
st.pyplot(fig4)

st.subheader("Prophet Future Forecast (Next 12 Weeks)")

prophet_df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
prophet_df.columns = ['ds', 'y']

model_prophet = Prophet()
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=12, freq='W')
forecast = model_prophet.predict(future)

fig5 = model_prophet.plot(forecast)
st.pyplot(fig5)

st.success("Dashboard Successfully Loaded with Walmart Sales Analysis and Forecasting!")