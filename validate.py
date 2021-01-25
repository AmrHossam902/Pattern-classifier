import cv2
import numpy as np
import os

p = '..\\linear classifier_3features\\'
names = []


# --------------------------------read learnt theta -------------------------------------
theta_file = open(p + 'theta.txt', 'r')
line_theta = theta_file.readline()
diamond_theta = theta_file.readline()
# ellipse_theta = theta_file.readline()

l = line_theta.split(' ')
line_theta = np.array([[float(l[0])], [float(l[1])], [float(l[2])], [float(l[3])]], np.float64)

d = diamond_theta.split(' ')
diamond_theta = np.array([[float(d[0])], [float(d[1])], [float(d[2])], [float(d[3])]], np.float64)

#e = ellipse_theta.split(' ')
#ellipse_theta = np.array([[float(e[0])], [float(e[1])], [float(e[2])], [float(e[3])]], np.float64)

theta_file.close()

# ------------------------------------read test set-----------------------------------------------
test_file = open(p + 'test_feat.txt', 'r')
X = []
Y = []
while True:
    s = test_file.readline()
    if s == '':
        break
    l = s.split(' ')
    names.append(l[0])

    t = list()
    t.append(np.float64(l[1]))
    t.append(np.float64(l[2]))
    t.append(np.float64(l[3]))
    t.append(np.float64(l[4]))
    X.append(t)
    Y.append(np.float64(l[5]))
    print("reading " + l[0])
test_file.close()

# ------------------------------------ validate ------------------------------------------------

X = np.array(X)
Y = np.array(Y)
Errors = []
count = 0
for i in range(len(X)):
    # test against line classifier
    r = np.dot(X[i], line_theta)
    print(names[i] + " F: " + str(X[i]))
    if r > 0:
        print("classified as line")
        if Y[i] == 1:
            count += 1
        else:
            Errors.append(names[i])
        print("------------------------------------------------")
        continue

    r = np.dot(X[i], diamond_theta)
    if r > 0:
        print("classified as diamond")
        if Y[i] == 2:
            count += 1
        else:
            Errors.append(names[i])
        print("------------------------------------------------")
        continue

    print("classified as ellipse")
    if Y[i] == 3:
        count += 1
    else:
        Errors.append(names[i])
    print("------------------------------------------------")
    # r = np.dot(X[i], ellipse_theta)
    # if r > 0:
    #     print("classified as ellipse")
    #     if Y[i] == 3:
    #         count += 1
    #     print("------------------------------------------------")
    #     continue


print("total = " + str(Y.shape[0]) + ",     faults = " + str(Y.shape[0] - count))
print("accuracy = " + str(count/Y.shape[0] * 100))
print("\n\n--- Errors ---")
for E in Errors:
    print(E)