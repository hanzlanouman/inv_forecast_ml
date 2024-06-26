import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import xgboost
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('data.csv')

data['key'] = data['week'].astype(str)+'_'+data['store_id'].astype(str)

data = data.drop(['record_ID', 'week', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)
data = data.dropna()
data = data.groupby('key').sum()
#Plot this data
# Set Figure size to 12,8
data['day_1'] = data['units_sold'].shift(-1)
data['day_2'] = data['units_sold'].shift(-2)
data['day_3'] = data['units_sold'].shift(-3)
data['day_4'] = data['units_sold'].shift(-4)


df=data

x1, x2, x3, x4, y = df['day_1'], df['day_2'], df['day_3'], df['day_4'], df['units_sold']
x1, x2, x3, x4, y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1, x2, x3, x4, y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)

split_percentage = 15
test_split = int(len(df)*(split_percentage/100))
x = np.concatenate((x1, x2, x3, x4), axis=1)
X_train,X_test,y_train,y_test = x[:-test_split],x[-test_split:],y[:-test_split],y[-test_split:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


rf_regressor = RandomForestRegressor()
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)
print("R Sq. Score for Random Forest Regression :", rf_regressor.score(X_test, y_test))
plt.rcParams["figure.figsize"] = (12,8)
plt.plot(y_pred[-50:], label='Predictions')
plt.plot(y_test[-50:], label='Actual Sales')
plt.legend(loc="upper left")
plt.show()
plt.savefig('plot.png')

xgb_regressor = xgboost.XGBRegressor()
xgb_regressor.fit(X_train, y_train)

y_pred = xgb_regressor.predict(X_test)
print("R Sq. Score for XGBoost :", xgb_regressor.score(X_test, y_test))

print(data.head())

