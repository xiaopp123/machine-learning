from numpy import genfromtxt
from sklearn import linear_model

data_path = "Delivery.csv"
delivery_data = genfromtxt(data_path, delimiter=',')

print("data")
print(delivery_data)

x = delivery_data[:, :-1]
y = delivery_data[:,-1]

print(x)
print(y)

lr = linear_model.LinearRegression()
lr.fit(x, y)

print(lr)

print("coefficients:")
print(lr.coef_)

print("intercept:")
print(lr.intercept_)
#这里有错误:
xPredict = [102,6]
yPredict = lr.predict(xPredict)
print("predict:")
print(yPredict)
