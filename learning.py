import cv2
import numpy as np
import os

def grad_descent(init_theta, alpha, X, Y, acc):

    prev_theta = init_theta
    current_theta = np.zeros(init_theta.shape)

    prev_J = 0

    while True:
        H = np.dot(X, prev_theta)
        H = 1 / (1 + np.exp(-H))
        Xt = np.transpose(X)
        current_theta = prev_theta - (alpha/m) * np.dot(Xt, H - Y)
        J = (1/m) * np.sum(-np.multiply(Y, np.log(H)) - np.multiply((1 - Y), np.log(1 - H)), 0)
        if np.abs(J - prev_J) < acc:
            break
        prev_theta = current_theta
        prev_J = J
        print(J)
        print(current_theta)

    return current_theta


# images paths
p = '..\\linear classifier_3features\\'
train_file = open(p + "train_feat.txt", 'r')

# parameters
n = 4
m = 0

# ---------------------------reading training features ------------------------------
X_lin = []
Y_lin = []
X_dia = []
Y_dia = []
X_ell = []
Y_ell = []

while True:
    s = train_file.readline()
    if s == "":
        break
    l = s.split(" ")

    temp = list()
    temp.append(np.float64(l[1]))
    temp.append(np.float64(l[2]))
    temp.append(np.float64(l[3]))
    temp.append(np.float64(l[4]))
    clas = np.float64(l[5])

    if clas == 1:
        X_lin.append(temp)
        Y_lin.append(clas)
    elif clas == 2:
        X_dia.append(temp)
        Y_dia.append(clas)
    else:
        X_ell.append(temp)
        Y_ell.append(clas)

X_lin = np.array(X_lin)
Y_lin = np.array(Y_lin)
X_dia = np.array(X_dia)
Y_dia = np.array(Y_dia)
X_ell = np.array(X_ell)
Y_ell = np.array(Y_ell)

Y_lin = Y_lin.reshape((len(Y_lin), 1))
Y_dia = Y_dia.reshape((len(Y_dia), 1))
Y_ell = Y_ell.reshape((len(Y_ell), 1))


# --------------------------------- learn line -----------------------------------------
X = np.concatenate((X_lin, X_dia, X_ell), 0)
Y = np.concatenate((Y_lin, Y_dia, Y_ell), 0)
m = X.shape[0]

init_theta = np.array([[4], [-1], [0], [0]])
alpha = 0.003
acc = 0.01
Y_L = Y.copy()
Y_L[Y_L != 1] = 0
Y_L[Y_L == 1] = 1
line_theta = grad_descent(init_theta, alpha, X, Y_L, acc)
print(line_theta)


# ------------------------------- learn diamond -----------------------------------------

X = np.concatenate((X_dia, X_ell), 0)
Y = np.concatenate((Y_dia, Y_ell), 0)
m = X.shape[0]

init_theta = np.array([[25], [0], [0], [0]])
alpha = 0.003
acc = 0.0000001
Y_L = Y.copy()
Y_L[Y_L != 2] = 0
Y_L[Y_L == 2] = 1
diamond_theta = grad_descent(init_theta, alpha, X, Y_L, acc)
print(diamond_theta)


# ------------------------------- learn ellipse -----------------------------------------

# init_theta = np.zeros((n, 1))
# alpha = 0.003
# Y_L = Y.copy()
# Y_L[Y_L != 3] = 0
# Y_L[Y_L == 3] = 1
# ellipse_theta = grad_descent(init_theta, alpha, X, Y_L)
# print(ellipse_theta)


# ------------------------------------- saving learned parameters ---------------------------------
theta_file = open(p + 'theta.txt', 'w')

theta_file.write(str(line_theta[0][0]) + " " + str(line_theta[1][0]) + " " + str(line_theta[2][0]) + " " + str(line_theta[3][0]) + "\n")
theta_file.write(str(diamond_theta[0][0]) + " " + str(diamond_theta[1][0]) + " " + str(diamond_theta[2][0]) + " " + str(diamond_theta[3][0]) + "\n")
# theta_file.write(str(ellipse_theta[0][0]) + " " + str(ellipse_theta[1][0]) + " " + str(ellipse_theta[2][0]) + " " + str(ellipse_theta[3][0]) + "\n")

theta_file.close()
