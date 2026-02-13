import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Program Started")


df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully")
print(df.head())

df['date'] = pd.to_datetime(df['date'])


daily_sales = df.groupby('date')['sales'].sum().reset_index()


daily_sales = daily_sales.sort_values('date')


daily_sales['Time'] = np.arange(len(daily_sales))

train = daily_sales[:-90]
test = daily_sales[-90:]

X_train = train[['Time']]
y_train = train['sales']

X_test = test[['Time']]
y_test = test['sales']


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)


plt.figure(figsize=(12,6))
plt.plot(train['date'], train['sales'], label="Training Data")
plt.plot(test['date'], y_test, label="Actual Sales")
plt.plot(test['date'], predictions, linestyle="--", label="Predicted Sales")
plt.legend()
plt.title("Sales Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()


future_time = np.arange(len(daily_sales), len(daily_sales)+30)
future_predictions = model.predict(future_time.reshape(-1,1))

future_dates = pd.date_range(
    start=daily_sales['date'].iloc[-1],
    periods=31,
    freq='D'
)[1:]

plt.figure(figsize=(12,6))
plt.plot(daily_sales['date'], daily_sales['sales'], label="Historical Sales")
plt.plot(future_dates, future_predictions, linestyle="--", label="Future Forecast")
plt.legend()
plt.title("Future 30-Day Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

input("Press Enter to exit...")
