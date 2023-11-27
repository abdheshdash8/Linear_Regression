import numpy as np
import matplotlib.pyplot as plt
import copy
f1_path = 'q4x.dat'
f2_path = 'q4y.dat'

x = np.zeros((100, 2))
y = np.zeros(0)
j = 0
with open(f1_path, 'r') as file:
    for line in file:
        l1 = line.split()
        x[j] = np.array([int(l1[0]), int(l1[1])])
        j += 1

with open(f2_path, 'r') as file:
    for line in file:
        y = np.append(y, line.strip())
n = len(y)

def str1(y):
    if y == "Alaska":
        return 0 
    else:
        return 1

########################## Part (a) #############################

def indicator_fun(y, k):
    if y == k:
        return 1
    else:
        return 0
    
def mean_vector(x):
    # This function will evaluate the mean vector that corresponds tot he maximum likelihood.
    # The class y = 0 represents Alaska and the class y = 1 represents Canada.
    vec_0 = np.zeros(2)
    vec_1 = np.zeros(2)
    sum_0 = 0
    sum_1 = 0
    for j in range(n):
        sum_0 += indicator_fun(y[j], "Alaska")
        sum_1 += indicator_fun(y[j], "Canada")
    for j in range(n):
        vec_0 += (indicator_fun(y[j], "Alaska") * x[j])/sum_0
    for j in range(n):
        vec_1 += (indicator_fun(y[j], "Canada") * x[j])/sum_1
    return np.array([vec_0, vec_1])
    
def lda_sigma_mat(x,  mu):
    # This function will evaluate the covariance matrix by taking x and mu (mean vector) as input.
    x1 = copy.deepcopy(x)
    return np.cov(x1, rowvar= False)

mu_vec = mean_vector(x)
sigma = lda_sigma_mat(x, mu_vec)
print("The value of mu0 is: ", mu_vec[0])
print("The value of mu1 is: ", mu_vec[1])
print("The value of sigma matrix is: ", sigma)

######################## Part(b) ###########################

z = np.array([str1(z) for z in y])
colours = np.array(['red' if val == 0 else 'blue' for val in z])
fig = plt.figure()
plt.scatter(x[:, 0], x[:, 1], c = colours, marker = "o")

######################### Part(c) ###########################

# Here we have sum_0 = sum_1 = 50 i.e. the no. elements in y = 1 class = no .of elements in y = 0 class.
# So, both fi and (1- fi) values are 0.5.
fi = 0.5
x1 = np.zeros((n, 2))
x2 = np.zeros((n, 2))
for j in range(n):
    x1[j] = x[j] - mu_vec[0]
    x2[j] = x[j] - mu_vec[1]

sigma_inv = np.linalg.inv(sigma)

def f(x1, x2):
    x = np.array([x1, x2])
    return (((x-mu_vec[0]) @ sigma_inv) @ (x-mu_vec[0]).T) - (((x-mu_vec[1]) @ sigma_inv) @ (x-mu_vec[1]).T)

# Now the equation of the boundary is a*x1 + b*x2 + c = 0 where a, b, c coefficients are determined
# by using f(0, 1), f(1, 0), f(0, 0) values.
# f(0, 0) = c
# f(1, 0) = a + c
# f(0, 1) = b + c
# So, c = f(0, 0), b = f(0, 1) - f(0, 0) and a = f(1, 0) - f(0, 0).

a = f(1, 0) - f(0, 0)
b = f(0, 1) - f(0, 0)
c = f(0, 0)
x1 = np.array([a[0] for a in x])

def f(x1, x2):
    return a*x1 + b*x2 + c
x2 = (-a/b)*x1 + (-c/b)
plt.plot(x1, x2)

########################## Part(d) ##########################

# The value of mu0 and mu1 will remain the same as they are defined in the same way here also.

# Calculation of sigma_0 and sigma_1 is done below:
sigma_0 = np.zeros((2, 2))
sigma_1 = np.zeros((2, 2))
for j in range(2):
    for k in range(2):
        for l in range(n):
            sigma_0[j][k] += (x[l][j] - mu_vec[0][j]) * (x[l][k] - mu_vec[0][k])*indicator_fun(str1(y[l]), 0)
            sigma_1[j][k] += (x[l][j] - mu_vec[1][j]) * (x[l][k] - mu_vec[1][k])*indicator_fun(str1(y[l]), 1)
# sigma_0 and sigma_1 are divided by 50 i.e. the no. of times y[l] == 1 and 
# that is also equal to no. of times y[l] == 0 in this case. 
sigma_0 /= 50
sigma_1 /= 50
print("The value of mu0: ", mu_vec[0])
print("The value of mu1: ", mu_vec[1])
print("The value of sigma0 matrix is : ", sigma_0)
print("The value of sigma1 matrix is : ", sigma_1)

######################## Part(e) ##########################

sigma_0_inv = np.linalg.inv(sigma_0)
sigma_1_inv = np.linalg.inv(sigma_1)

def f1(x1, x2):
    x = np.array([x1, x2])
    return (((x-mu_vec[0]) @ sigma_0_inv) @ (x-mu_vec[0]).T) - (((x-mu_vec[1]) @ sigma_1_inv) @ (x-mu_vec[1]).T) - 2*np.log(np.linalg.det(sigma_0)/np.linalg.det(sigma_1))

# Now the equation of the boundary is z = a1*x1^2 + a2*x2^2 + b1*x1 + b2*x2 + c where a1, a2, b1, b2, c coefficients are determined
# by using f(2, 0), f(-1, -1), f(0, 1), f(1, 0), f(0, 0) values. 
# f(0, 0) = c
# f(1, 0) = a + c
# f(0, 1) = b + c
# So, c = f(0, 0), b = f(0, 1) - f(0, 0) and a = f(1, 0) - f(0, 0).

a1 = 0.5*f1(0, 0) + 0.5*f1(2, 0) - f1(1, 0)
a2 = 2.5*f1(1, 0) - 0.5*f1(2, 0) - 3*f1(0, 0) + 0.5*f1(0, 1) + 0.5*f1(-1, -1) 
b1 = 2*f1(1, 0) - 1.5*f1(0, 0) - 0.5*f1(2,0)
b2 = 0.5*f1(0, 1) + 2*f1(0, 0) - 2.5*f1(1, 0) + 0.5*f1(2, 0) - 0.5*f1(-1, -1)
c = f1(0, 0)

def implicit_eq(x):
        # Let the equation of the contour be z = a1*x1^2 + a2*x2^2 + b1*x1 + b2*x2 + c
        a1 = sigma_1_inv - sigma_0_inv
        b1 = (-2) * (mu_vec[1] @ sigma_1_inv - mu_vec[0] @ sigma_0_inv)
        c = ((mu_vec[1] @ sigma_1_inv) @ mu_vec[1].T - (mu_vec[0] @ sigma_0_inv) @ mu_vec[0].T) - (np.log(np.linalg.det(sigma_0)/np.linalg.det(sigma_1)))
        return x @ a1 @ x.T + b1 @ x.T + c
            
x1 = np.linspace(40, 180, 100)
x2 = np.linspace(250, 600, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack((X1.flatten(), X2.flatten()))

# Evaluate the equation for each point in the grid
Z = np.array([implicit_eq(point) for point in X_grid])
Z = Z.reshape(X1.shape)

plt.contour(X1, X2, Z, levels=[0], colors = 'r')
plt.xlabel("x1 : Growth ring diameter in fresh water")
plt.ylabel("x2 : Growth ring diameter in marine water")
plt.show()

