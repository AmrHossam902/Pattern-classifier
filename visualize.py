import matplotlib.pyplot as plt
import matplotlib.lines as mplines
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


p = '..\\linear classifier_3features\\'

# -------------------------------------- read learnt theta ------------------------------------
theta_file = open(p + 'theta.txt', 'r')
line_theta = theta_file.readline()
diamond_theta = theta_file.readline()
#ellipse_theta = theta_file.readline()

l = line_theta.split(' ')
line_theta = np.array([[float(l[0])], [float(l[1])], [float(l[2])], [float(l[3])]], np.float64)

d = diamond_theta.split(' ')
diamond_theta = np.array([[float(d[0])], [float(d[1])], [float(d[2])], [float(d[3])]], np.float64)

#e = ellipse_theta.split(' ')
#ellipse_theta = np.array([[float(e[0])], [float(e[1])], [float(e[2])], [float(e[3])]], np.float64)

theta_file.close()


# ----------------------------------------- read training features -------------------------------------
train_file = open(p + 'train_feat.txt', 'r')

X = []
Y = []
Feat_lin = []
Feat_dia = []
Feat_ell = []
names = []
while True:
    s = train_file.readline()
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

    if np.float64(l[5]) == 1:
        Feat_lin.append(t)
    elif np.float64(l[5]) == 2:
        Feat_dia.append(t)
    else:
        Feat_ell.append(t)
    print("reading " + l[0])

train_file.close()

X_lin = []
X_dia = []
X_ell = []

Y_lin = []
Y_dia = []
Y_ell = []

Z_lin = []
Z_dia = []
Z_ell = []



for i in range(len(X)):
    if Y[i] == 1:
        X_lin.append(X[i][1])
        Y_lin.append(X[i][2])
        Z_lin.append(X[i][3])
    elif Y[i] == 2:
        X_dia.append(X[i][1])
        Y_dia.append(X[i][2])
        Z_dia.append(X[i][3])
    else:
        X_ell.append(X[i][1])
        Y_ell.append(X[i][2])
        Z_ell.append(X[i][3])

# plotting 3d view
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel('ratio', fontsize=20)
ax.set_ylabel('# hull count', fontsize=20)
ax.set_zlabel('size', fontsize=20)
Axes3D.scatter(ax, X_lin, Y_lin, Z_lin, c=['r'], marker='x')
Axes3D.scatter(ax, X_dia, Y_dia, Z_dia, c=['k'], marker='o')
Axes3D.scatter(ax, X_ell, Y_ell, Z_ell, c=['b'], marker='v')
scatter1_proxy = mplines.Line2D([0],[0], linestyle="none", c='k', marker='o')
scatter2_proxy = mplines.Line2D([0],[0], linestyle="none", c='b', marker='v')
scatter3_proxy = mplines.Line2D([0],[0], linestyle="none", c='r', marker='x')
ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['Diamond', 'Ellipse', 'Line'], numpoints = 1)


# plotting histogram of R for line
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.hist(x=np.array(Feat_lin)[:, 1], bins=range(11), rwidth=0.25, align='left')
plt.title("line patterns distribution on Feature1 \"R\"")
plt.xlabel("R * 10")
plt.ylabel("Frequency")
plt.xticks(range(12))
plt.yticks(range(0, 300, 50))


#plotting histogram of R for ellipse
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
plt.hist(x=np.array(Feat_ell)[:, 1], bins=range(11), rwidth=0.25, align='left')
plt.title("Ellipse patterns distribution on Feature1 \"R\"")
plt.xlabel("R * 10")
plt.ylabel("Frequency")
plt.xticks(range(12))
plt.yticks(range(0, 150, 50))


# plotting histogram of hull count for ellipse
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
plt.hist(x=np.array(Feat_ell)[:, 2], bins=range(50), rwidth=0.5, align='left')
plt.title("Ellipse patterns distribution on Feature2 \"# Hull vertices\"")
plt.xlabel("# vertices")
plt.ylabel("Frequency")
plt.xticks(range(0, 51, 5))
plt.yticks(range(0,25,5))



# plotting histogram of hull count for diamond
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
plt.hist(x=np.array(Feat_dia)[:, 2], bins=range(50), rwidth=0.5, align='left')
plt.title("Diamond patterns distribution on Feature2 \"# Hull vertices\"")
plt.xlabel("# vertices")
plt.ylabel("Frequency")
plt.xticks(range(0, 51, 5))
plt.yticks(range(0,30,5))
plt.show()



# #plot line boundary
# x1 = 0
# y1 = line_theta[0]/-line_theta[2]
# x2 = 1
# y2 = (line_theta[0] + line_theta[1]) / -line_theta[2]
# plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
#
# #plot diamond boundary
# x1 = 0
# y1 = diamond_theta[0]/-diamond_theta[2]
# x2 = 1
# y2 = (diamond_theta[0] + diamond_theta[1]) / -diamond_theta[2]
# plt.plot([x1, x2], [y1, y2], color='g', linestyle='-', linewidth=2)
#
# #plot diamond boundary
# x1 = 0
# y1 = ellipse_theta[0]/-ellipse_theta[2]
# x2 = 1
# y2 = (ellipse_theta[0] + ellipse_theta[1]) / -ellipse_theta[2]
# plt.plot([x1, x2], [y1, y2], color='b', linestyle='-', linewidth=2)


