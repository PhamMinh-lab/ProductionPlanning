import streamlit as st
import pandas as pd
import numpy as np
import pulp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# === LOAD & CLEAN DATA ===
@st.cache_data
def load_data():
    df = pd.read_excel('B202.xlsx', sheet_name='Demand')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').asfreq('MS').interpolate(method='linear')
    return df

# === REMOVE OUTLIERS ===
def remove_outliers(df, col='Sản lượng', lower=0.25, upper=0.75, threshold=1.7):
    Q1 = df[col].quantile(lower)
    Q3 = df[col].quantile(upper)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)]

# === ADD FEATURES ===
def add_features(df):
    df['lag_1'] = df['Sản lượng'].shift(1)
    df['lag_2'] = df['Sản lượng'].shift(2)
    df['lag_12'] = df['Sản lượng'].shift(12)
    df['rolling_mean_3'] = df['Sản lượng'].rolling(3).mean().shift(1)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['Rain'] = df.index.strftime('%Y-%m').isin(rain_months).astype(int)
    df['Tet'] = df.index.strftime('%Y-%m').isin(tet_months).astype(int)
    return df.dropna()

# === TRAIN MODEL ===
def train_model(df, features):
    X = df[features]
    y = df[['Sản lượng']]
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(df[['Sản lượng', 'lag_1', 'lag_2', 'lag_12', 'rolling_mean_3']])
    y_scaled = pd.DataFrame(y_scaled, columns=['Sản lượng','lag_1','lag_2','lag_12','rolling_mean_3'], index=df.index)
    X_scaled = pd.concat([y_scaled[['lag_1','lag_2','lag_12','rolling_mean_3']], df[['Rain','Tet','month','quarter']]], axis=1)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_scaled, y_scaled['Sản lượng'])
    X_scaled_columns = X_scaled.columns.tolist()  # Save the exact column order
    return model, scaler, X_scaled, y_scaled, X_scaled_columns

# === FORECAST ===
def forecast_future(current_date, model, scaler,X_scaled_columns, history, months=3):
    preds = []
    start_date = pd.to_datetime(current_date + "-01") + pd.DateOffset(months=1)
    future_dates = pd.date_range(start=start_date, periods=months, freq='MS')
    for date in future_dates:
        lag_1 = history.iloc[-1]['Sản lượng']
        lag_2 = history.iloc[-2]['Sản lượng']
        lag_12 = history.iloc[-12]['Sản lượng'] if len(history) >= 12 else lag_1
        rolling_mean_3 = history['Sản lượng'].iloc[-3:].mean()
        month, quarter = date.month, (date.month - 1) // 3 + 1
        rain = int(date.strftime('%Y-%m') in rain_months)
        tet = int(date.strftime('%Y-%m') in tet_months)
        X_future = pd.DataFrame([{
        'Rain': rain,
        'Tet': tet,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_12': lag_12,
        'rolling_mean_3': rolling_mean_3,
        'month': month,
        'quarter': quarter
    }])

# ✅ Reindex to match training feature names exactly (fill missing with 0 if needed)
        X_columns = ['Rain', 'Tet', 'lag_1', 'lag_2', 'lag_12', 'rolling_mean_3', 'month', 'quarter']
        X_future = X_future.reindex(columns=X_scaled_columns, fill_value=0)
        pred_scaled = model.predict(X_future)[0]
        y_pred = scaler.inverse_transform(np.array([[pred_scaled, lag_1, lag_2, lag_12, rolling_mean_3]]))[0,0]
        preds.append((date.strftime('%Y-%m'), y_pred))
        history.loc[date] = {
            'Sản lượng': pred_scaled, 'lag_1': lag_1, 'lag_2': lag_2,
            'lag_12': lag_12, 'rolling_mean_3': rolling_mean_3,
            'Rain': rain, 'Tet': tet, 'month': month, 'quarter': quarter
        }
    return pd.DataFrame(preds, columns=['Month', 'Forecasted_Demand'])

# === OPTIMIZATION ===
def optimize_plan(forecast_df, inv_start, safety_stock=5000, warehouse_cap=5000):
    months = forecast_df['Month'].tolist()
    demand = forecast_df['Forecasted_Demand'].tolist()
    capacity = [150000 if pd.to_datetime(m).month in [5,6,7,8,9,10] else 200000 for m in months]
    model = pulp.LpProblem("ProductionPlanning", pulp.LpMinimize)
    Prod = pulp.LpVariable.dicts("Prod", months, lowBound=0, cat='Continuous')
    Inv = pulp.LpVariable.dicts("Inv", months, lowBound=0, cat='Continuous')
    model += pulp.lpSum(Prod[m] for m in months), "Total_Production"
    for i, m in enumerate(months):
        model += Inv[m] == (inv_start if i == 0 else Inv[months[i-1]]) + Prod[m] - demand[i]
        model += Prod[m] <= capacity[i]
        model += Inv[m] >= safety_stock
        model += Inv[m] <= warehouse_cap
    model.solve()
    return pd.DataFrame({
        'Month': months,
        'Production': [Prod[m].varValue for m in months],
        'Inventory': [Inv[m].varValue for m in months]
    })

# === SIDE INPUTS ===
rain_months = [
    '2012-06', '2012-07', '2012-08', '2012-09',
    '2013-06', '2013-07', '2013-08', '2013-09',
    '2014-06', '2014-07', '2014-08', '2014-09',
    '2015-06', '2015-07', '2015-08', '2015-09',
    '2016-06', '2016-07', '2016-08', '2016-09',
    '2017-06', '2017-07', '2017-08', '2017-09',
    '2018-06', '2018-07', '2018-08', '2018-09',
    '2019-06', '2019-07', '2019-08', '2019-09',
    '2020-06', '2020-07', '2020-08', '2020-09',
    '2021-06', '2021-07', '2021-08', '2021-09',
    '2022-06', '2022-07', '2022-08', '2022-09',
    '2023-06', '2023-07', '2023-08', '2023-09',
    '2024-06', '2024-07', '2024-08', '2024-09',
    '2025-06', '2025-07', '2025-08', '2025-09'
]


tet_months = [
    '2012-02', '2013-03', '2014-02', '2015-03', '2016-03', '2017-02',
    '2018-03', '2019-03', '2020-02', '2021-02', '2022-03', '2023-03',
    '2024-01','2025-02'
]  
features = ['Rain','Tet','lag_1','lag_2','lag_12','rolling_mean_3','month','quarter']

# === STREAMLIT APP ===
# st.image('dudaco.png', use_container_width=True, width=50)
st.markdown(
    "<h1 style='text-align: center;'>📊 Demand Forecasting & Production Planning</h1>",
    unsafe_allow_html=True
)

df = load_data()
df = remove_outliers(df)
df = add_features(df)
model, scaler, X_scaled, y_scaled, X_scaled_columns = train_model(df, features)
X_scaled_columns = X_scaled.columns.tolist()  # ✅ Get column names
history = df.copy()                           # ✅ Make sure it's a DataFrame

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>B202 Sprayer</h2>", unsafe_allow_html=True)
    current_date = st.text_input("📅 Current Date (YYYY-MM)", value='2025-07')
    inventory = st.number_input("🏭 Current Inventory", min_value=0, value=5000)
    forecast_horizon = st.slider("🔮 Forecast Horizon (months)", 1, 6, 3)
    
    # Add vertical space to push the image to the bottom
    st.markdown("<div style='flex: 1; height: 480px;'></div>", unsafe_allow_html=True)
    


if current_date:
    forecast_df = forecast_future(current_date, model, scaler, X_scaled_columns, history, months=forecast_horizon)

    st.subheader("📈 Forecasted Demand")
    st.dataframe(forecast_df)

    st.subheader("📦 Optimized Production Plan")
    plan_df = optimize_plan(forecast_df, inv_start=inventory)
    st.dataframe(plan_df)

    st.line_chart(plan_df.set_index('Month')[['Production', 'Inventory']])
