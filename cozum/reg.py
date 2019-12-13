import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data_set = pd.read_csv("kc_house_data.csv")

#sqft_living = data_set["sqft_living15"]
#price = data_set["price"]

sqft_living = data_set.iloc[:, -2:-1]
price = data_set.iloc[:, 2:3]

sqft_train, sqft_test, price_train, price_test = tts(sqft_living, price, test_size = 1/3, random_state = 0)

regressor = lr()
regressor.fit(sqft_train, price_train)

price_prediction = regressor.predict(sqft_test)

f = plt.figure(1)
plt.scatter(sqft_train, price_train, color = "red")
plt.plot(sqft_train, regressor.predict(sqft_train), color = "blue")
plt.title("First Regression")
plt.xlabel("SQFT")
plt.ylabel("Price")
f.show()

g = plt.figure(2)
plt.scatter(sqft_test, price_test, color ="red")
plt.plot(sqft_train, regressor.predict(sqft_train), color = "blue")
plt.title("Test")
plt.xlabel("SQFT")
plt.ylabel("Price")
g.show()