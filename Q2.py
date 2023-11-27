import numpy as np
import matplotlib.pyplot as plt

###############  2(a) part - Generating data points  ################

np.random.seed(0)
# The value of theta as given in the question.
q = np.array([3, 1, 2])
# The number of data points to be generated.
n = 1000000
# Generating Normal distributed data for x1, x2 and noise.
x1 = np.random.normal(loc=3, scale=2, size = n)
x2 = np.random.normal(loc=-1, scale=2, size = n)
noise = np.random.normal(loc=0, scale=np.sqrt(2), size=n)

# Generating values of y using values of q, x1, x2, and q.
Y = q[0] + q[1] * x1 + q[2] * x2 + noise

############### 2(b) part - Applying stochastic gradient descent ################ 

X =  np.zeros((n, 3))
for j in range(n):
    X[j] = np.array([1, x1[j], x2[j]])

def shuffle(X, Y):
    ind = np.random.permutation(len(X))
    return X[ind], Y[ind]

def J(X, Y, q):
    n = len(Y)
    sum = 0
    for j in range(n):
        sum += (1/2)*(1/n)*((Y[j] - np.dot(q, X[j]))**2)
    return sum

def mini_batch_gradient_descent(X, Y, nt, r, k, n, eps = 10**(-7)):
    # X is an array containing 3 dimensional vector [x2, x1, x0] where x0 = 1.
    # Y is an array containing the y values coresponding to the x values.
    # Batch size = r.
    # nt is the learning rate.
    # eps is the tolerance value for convergence of Jq.
    # initial value of theta = 0 which is represented by a variable q here.
    q = np.array([0, 0, 0])
    # t denotes the iteration number
    t = 0
    # q_l stores the values of theta as it changes through iterations
    q_l = np.array([[0, 0, 0]])
    # Jq_l stores the J_q in each iteration
    Jq_l = np.array([])
    # k is the parameter corresponding to moving average of Jq value for last k iterations and its also the
    # gamma that is required to check convergence of Jq over last gamma values and last to last gamma values of Jq stored in Jq_B. 
    # itl is the iteration limit i.e. the upper bound on number of epochs
    itl = 10*(n/r)
    # sum1_k stores the sum of values of Jq from 1st to 't'th iteration where t is the cureent iteration number.
    sum1_k = np.array([])
    # curr_k and prev_k are variables that will store average value of Jq over last k iterations
    # and last to last k iterations respectively.
    curr_k = 0
    prev_k = 0
    if r != n:
        for i in range(2*k):
            t += 1
            X1 = X[i*r : (i+1)*r]
            Y1 = Y[i*r : (i+1)*r]
            # Jq is the cost function
            Jq = 0
            # delJq is the first order derivative of Jq
            delJq = np.zeros(3)
            for j in range(r):
                Jq += (1/2)*(1/r)*((Y1[j] - np.dot(q, X1[j]))**2)
                delJq += (1/r)*(-1)*(Y1[j] - np.dot(q, X1[j]))*X1[j]
            Jq_l = np.append(Jq_l, Jq)
            if i == 0:
                sum1_k = np.append(sum1_k, Jq)
            else:
                sum1_k = np.append(sum1_k, sum1_k[i-1] + Jq)
            q1 = q - nt * delJq
            q = q1
            sum = 0
            for j in range(3):
                sum += (q1[j])**2
            q1 = q1/((sum) ** (0.5))
            q_l = np.vstack((q_l, q))
    else:
        q = np.array([0, 0, 0])
        for i in range(2*k):
            t += 1
            X, Y = shuffle(X, Y)
            X1 = X[0 : r]
            Y1 = Y[0 : r]
            # Jq is the cost function
            Jq = 0
            # delJq is the first order derivative of Jq
            delJq = np.zeros(3)
            for j in range(r):
                Jq += (1/2)*(1/r)*((Y1[j] - np.dot(q, X1[j]))**2)
                delJq += (1/r)*(-1)*(Y1[j] - np.dot(q, X1[j]))*X1[j]
            Jq_l = np.append(Jq_l, Jq)
            if i == 0:
                sum1_k = np.append(sum1_k, Jq)
            else:
                sum1_k = np.append(sum1_k, sum1_k[i-1] + Jq)
            q1 = q - nt * delJq
            q = q1
            q_l = np.vstack((q_l, q))
    # checker is the value that will check the termination of the method
    checker = True
    while(checker):
        X, Y = shuffle(X, Y)
        for i in range(int(n/r)):
            t += 1
            X1 = X[i*(r):(i+1)*(r)]
            Y1 = Y[i*(r):(i+1)*(r)]
            # Jq is the cost function
            Jq = 0
            # delJq is the first order derivative of Jq
            delJq = np.zeros(3)    
            for j in range(r):
                Jq += (1/2)*(1/r)*((Y1[j] - np.dot(q, X1[j]))**2)
                delJq += (1/r)*(-1)*(Y1[j] - np.dot(q, X1[j]))*X1[j]
            Jq_l = np.append(Jq_l, Jq)
            sum1_k = np.append(sum1_k, sum1_k[t-2] + Jq)
            q1 = q - nt * delJq
            q = q1
            sum = 0
            for j in range(3):
                sum += (q1[j])**2
            q1 = q1/((sum) ** (0.5))
            q_l = np.append(q_l, q)
            curr_k = (sum1_k[t-1] - sum1_k[t-k-1])/k
            prev_k = (sum1_k[t-k-1] - sum1_k[t-2*k-1])/k
    
            if t > itl:
                checker = False
                break
            if abs(prev_k - curr_k) < eps:
                checker = False
                break
    return [q, q_l, Jq_l, t]

################### Part(d) ########################

def scatter_plot(q_l):
    n = len(q_l)
    q0 = np.array([])
    q1 = np.array([])
    q2 = np.array([])
    for j in range(int(n/3)):
        q0 = np.append(q0, q_l[3*j])
        q1 = np.append(q1, q_l[3*j+1])
        q2 = np.append(q2, q_l[3*j+2])  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection= "3d")
    ax.scatter(q0, q1, q2, c = "blue", marker = ".", s = 50)
    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('Theta_2')
    plt.show()

l1 = mini_batch_gradient_descent(X, Y, 0.001, 1, 10000, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
t = l1[3]
print(t, l1[0], Jq)

l1 = mini_batch_gradient_descent(X, Y, 0.001, 100, 1000, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
t = l1[3]
print(t, l1[0], Jq)

l1 = mini_batch_gradient_descent(X, Y, 0.001, 10000, 10, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
t = l1[3]
print(t, l1[0], Jq)

l1 = mini_batch_gradient_descent(X, Y, 0.001, 1000000, 1, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
t = l1[3]
print(t, l1[0], Jq)

########################### Part(c) #############################

filename = "q2test.csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(0, 1, 2), dtype=None)
column1 = data[:, 0]  # Replace 'f0' with the actual field name for the first column
column2 = data[:, 1]
Y = data[:, 2]
n = 10000

X =  np.zeros((n, 3))
for j in range(n):
    X[j] = np.array([1, column1[j], column2[j]])

# Jq1_l is array containing the test error data w.r.t. the learned hypothesis.
Jq1_l = np.array([])

### Batch size r = 10
l1 = mini_batch_gradient_descent(X, Y, 0.01, 10, 100, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
# Jq1_l is array containing the test error data w.r.t. the learned hypothesis.
Jq1_l = np.append(Jq1_l, J(X, Y, q))
t = l1[3]
print(t, l1[0], Jq)
scatter_plot(q_l)

### Batch size r = 100
l1 = mini_batch_gradient_descent(X, Y, 0.01, 100, 10, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
# Jq1_l is array containing the test error data w.r.t. the learned hypothesis.
Jq1_l = np.append(Jq1_l, J(X, Y, q))
t = l1[3]
print(t, l1[0], Jq)
scatter_plot(q_l)

### Batch size r = 10000
l1 = mini_batch_gradient_descent(X, Y, 0.005, 10000, 1, n)
q = l1[0]
q_l = l1[1]
# Jq is the final value of the error function
Jq = l1[2][-1]
# Jq1_l is array containing the test error data w.r.t. the learned hypothesis.
Jq1_l = np.append(Jq1_l, J(X, Y, q))
t = l1[3]
print(t, l1[0], Jq)


scatter_plot(q_l) 

### Now obtaining the test error data w.r.t. the original hypothesis.
Jq_ori = J(X, Y, np.array([3, 1, 2]))
print("the test error data w.r.t. the original hypothesis and learned hypothesis : ", Jq_ori, Jq1_l)