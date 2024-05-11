import numpy as np
import matplotlib.pyplot as plt

# Data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Matriks desain
X_b = np.c_[np.ones((100, 1)), X]

# Menghitung parameter regresi menggunakan rumus normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Membuat prediksi
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

# Membuat plot data dan regresi linier
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Matrix")
plt.legend()
plt.show()