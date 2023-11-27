import numpy as np
import matplotlib.pyplot as plt

c1 = np.genfromtxt("linearX.csv", delimiter=",")
print(c1)
col1 = np.array([np.array([el, 1]) for el in c1])
col2 = np.genfromtxt("linearY.csv", delimiter=",")
# n is the size of the dataset 
n = col2.size

def norm(X):
    # This function normalizes the X and Y datasets.
    mean_x = 0
    var_x = 0
    for i in range(n):
        mean_x += X[i][0]/n
    for i in range(n):
        var_x += ((X[i][0] - mean_x)**2)/n
    for i in range(n):
        X[i][0] = (X[i][0] - mean_x)/((var_x)**0.5)

### Normalize the X and Y datasets
norm(col1)
plt.scatter(np.array([x[0] for x in col1]), col2)
#### nt is the learning rate.
X = col1
Y = col2

################### Part(a) ########################

def J1(q, X, Y):
    Jq = 0
    for j in range(n):
        Jq += (1/2)*(1/n)*((Y[j] - np.dot(q, X[j]))**2)
    return Jq

def batch_gradient_descent(X, Y, nt, eps = 0.000001):
    nt = 0.1
    ##### eps is the tolerance value for convergence of Jq 
    eps = 0.000000001
    ### initial value of theta = 0 which is represented by a variable q here.
    q = np.ones(2)
    # t denotes the iteration number
    t = 0
    # Jq_l stores the J_q in each iteration
    Jq_l = np.array([])
    # q_l_1 and q_l_0 stores the value of q_1 and q_0 in each iteration
    q_l_1 = np.array([])
    q_l_0 = np.array([])
    # Jq is the cost function 
    Jq = 0
    # delJq is the derivative of the cost function 
    delJq = np.array([0, 0])
    checker = True
    while(checker):
        t += 1
        # Jq is the cost function
        Jq = 0
        # delJq is the first order derivative of Jq
        delJq = np.array([0, 0])
        for j in range(n):
            Jq += (1/2)*(1/n)*((Y[j] - np.dot(q, X[j]))**2)
            delJq = delJq + (1/n)*(-1)*(Y[j] - np.dot(q, X[j]))*X[j]
        Jq_l = np.append(Jq_l, Jq)
        q[0] -= nt * delJq[0]
        q[1] -= nt * delJq[1]
        q_l_1 = np.append(q_l_1, q[0])
        q_l_0 = np.append(q_l_0, q[1])
        if  len(Jq_l) > 1 and abs(Jq_l[-1] - Jq_l[-2]) <= eps:
            checker = False
    return [q, t, q_l_0, q_l_1, Jq_l]

l = batch_gradient_descent(X, Y, 0.1, 0.000000001)
q = l[0]
t = l[1]
q_l_0 = l[2]
q_l_1 = l[3]
Jq_l = l[4]

print("The total number of itertations: ", l[1])
print("The final value of theta: ", [q[1], q[0]])

################### Part(b) ########################

plt.plot(np.array([x[0] for x in col1]), np.array([np.dot(q, x) for x in col1]), color = "red")
plt.show()

################### Part(c) ########################

def mesh_plot(q_l_0, q_l_1, X, Y):
    # Here q_0 and q_1 are some randommly
    q_1 = q_l_0
    q_0 = q_l_1
    Q_0, Q_1 = np.meshgrid(q_0, q_1)
    # filling the cost values for each combination of q0 and q1 in J
    J = np.zeros_like(Q_0)
    for i in range(len(q_0)):
        for j in range(len(q_1)):
            q = np.array([q_0[i], q_1[j]])
            J[i, j] = J1(q, X, Y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection= "3d")
    surface = ax.plot_surface(Q_0, Q_1, J, cmap = "plasma", alpha = 0.5)
    ax.set_xlabel(r'theta_{1}')
    ax.set_ylabel(r'theta_{0}')
    ax.set_title("mesh grid plot of the error function")
    scatter = ax.scatter([], [], [], color='green', s=50)
    # Animate the scatter plot over iterations
    for i in range(t):
        scatter._offsets3d = (q_l_1[:i+1], q_l_0[:i+1], Jq_l[:i+1])
        plt.pause(0.02)
    plt.show()

################### Part(d) ########################

def contour_plot(q_l_0, q_l_1, X, Y):
    q_1 = q_l_0
    q_0 = q_l_1
    Q_0, Q_1 = np.meshgrid(q_0, q_1)
    # filling the cost values for each combination of q0 and q1 in J
    J = np.zeros_like(Q_0)
    for i in range(len(q_0)):
        for j in range(len(q_1)):
            q = np.array([q_0[i], q_1[j]])
            J[i, j] = J1(q, X, Y)
    
    fig, ax = plt.subplots()
    contour1 = ax.contour(Q_0, Q_1, J, levels = 20, cmap = "plasma", alpha = 0.9)
    plt.clabel(contour1, inline=True, fontsize=10)
    ax.set_xlabel(r'theta_{1}')
    ax.set_ylabel(r'theta_{0}')
    ax.set_title("contour plot of the error function")
    scatter = ax.scatter([], [], color='green', s=50)

    # Animate the scatter plot over iterations
    for i in range(t):
        scatter.set_offsets(np.column_stack((q_l_1[:i+1], q_l_0[:i+1])))
        plt.pause(0.02)  # Pause for 0.2 seconds for visualization
    plt.show()


mesh_plot(q_l_0, q_l_1, X, Y)
contour_plot(q_l_0, q_l_1, X, Y)

############################## Part (e) ###############################

nt = 0.001
l1 = batch_gradient_descent(X, Y, nt)
q_l_0 = l1[2]
q_l_1 = l1[3]
contour_plot(q_l_0, q_l_1, X, Y)

nt = 0.025
l1 = batch_gradient_descent(X, Y, nt)
q_l_0 = l1[2]
q_l_1 = l1[3]
contour_plot(q_l_0, q_l_1, X, Y)

nt = 0.1
l1 = batch_gradient_descent(X, Y, nt)
q_l_0 = l1[2]
q_l_1 = l1[3]
contour_plot(q_l_0, q_l_1, X, Y)