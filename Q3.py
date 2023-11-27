import numpy as np
import matplotlib.pyplot as plt

c1 = np.genfromtxt("logisticX.csv", delimiter=",")
col1 = np.array([np.array([el[0], el[1], 1]) for el in c1])
col2 = np.genfromtxt("logisticY.csv", delimiter=",")
n = len(col1)

X = col1
Y = col2

def norm(X):
    # This function normalizes the X and Y datasets.
    for j in range(2):
        mean_x = 0
        var_x = 0
        for i in range(n):
            mean_x += X[i][j]/n
        for i in range(n):
            var_x += ((X[i][j] - mean_x)**2)/n
        for i in range(n):
            X[i][j] = (X[i][j] - mean_x)/((var_x)**0.5)

### Normalize the X and Y datasets
norm(X)

# red points are those whose y value is 0 and blue points are those whose y value is 1. 
colours = np.array(["red" if val == 0 else "blue" for val in col2])
plt.scatter(np.array([x[0] for x in X]), np.array([x[1] for x in X]), color = colours)

##################### Part(a) #########################

def sigmoid1(q, x):
    return 1 / (1 + np.exp(-np.dot(x, q)))

def hessian(x, q = np.zeros(3)):
    # P1 is the value of probability got from sigmoid function
    P1 = sigmoid1(q, x)
    mat = (1 / n) * np.dot(x.T, np.dot(np.diag((P1 * (1 - P1)).flatten()), x))
    return mat
    
def Newton(x, y, n, h_inv, q = np.ones(3), eps = 10**(-5)):
    # dldq is the matrix containing the first order derivatives
    dldq = (1 / n) * (x.T @ (y - sigmoid1(q, x)))
    q1 = q - np.linalg.inv(hessian(x, q)) @ dldq
    sum = 0
    for j in range(3):
        sum += (q1[j])**2
    q1 = q1/((sum) ** (0.5))
    l2= np.array([abs(q1[0]-q[0]) < eps, abs(q1[1]-q[1]) < eps, abs(q1[2]-q[2]) < eps])
    if l2.all():
        return q1
    else:
        h_inv = np.linalg.inv(hessian(x, q1))
        return Newton(x, y, n, h_inv, q1, eps)

h_inv = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

q = Newton(X, Y, n, h_inv)
print(q)


#########################  Part-(b) #########################

x = np.linspace(-2.5, 2.5, 100)
y = (-q[0]/q[1]) * x - (q[2]/q[1])
plt.plot(x, y)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

