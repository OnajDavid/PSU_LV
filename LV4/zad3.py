import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    return 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622*np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    return y + 0.1*varNoise*np.random.normal(0, 1, len(y))

x = np.linspace(1, 10, 50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

np.random.seed(12)
indeksi = np.random.permutation(len(x))
train_size = int(np.floor(0.7 * len(x)))
indeksi_train = indeksi[:train_size]
indeksi_test = indeksi[train_size:]

x_train = x[indeksi_train]
y_train = y_measured[indeksi_train]
x_test = x[indeksi_test]
y_test = y_measured[indeksi_test]

degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []

plt.figure(figsize=(12, 6))

for i, deg in enumerate(degrees):
    poly = PolynomialFeatures(degree=deg)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    x_full_poly = poly.transform(x)

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    MSEtrain.append(mse_train)
    MSEtest.append(mse_test)

    plt.subplot(1, 3, i+1)
    plt.plot(x, y_true, 'b-', label='f(x)')
    plt.plot(x, model.predict(x_full_poly), 'r--', label='model')
    plt.scatter(x_train, y_train, c='black', s=20, label='train')
    plt.title(f'Degree {deg}\nMSE train={mse_train:.3f}, test={mse_test:.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()

print("MSEtrain =", MSEtrain)
print("MSEtest  =", MSEtest)
