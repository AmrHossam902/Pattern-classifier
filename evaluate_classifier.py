import numpy as np
import cv2
import os

def extract_feature2(gray_image):
    _, contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = np.array([])
    for cont in contours:
        l1 = list(cont)
        l2 = list(hull)
        l2.extend(l1)
        hull = np.array(l2)

    #hull2 = cv2.convexHull(hull, None, returnPoints=False)
    hull = cv2.convexHull(hull, None, returnPoints=True)

    # try:
    #     ellipse = cv2.fitEllipse(hull)
    # except Exception as e:
    #     ellipse = ((1,1),(1,1),(0,0))
    # c_a = cv2.contourArea(hull)
    # e_a = ellipse[1][0] * ellipse[1][1] * np.pi / 4

    #defects = cv2.convexityDefects(contours, hull2)
    return len(hull)


def extract_feature3_1(gray_image):
    total_rows = 0
    total_cols = 0
    rowsd = 0
    colsd = 0

    size =gray_image.shape[0]
    for i in range(size):
        white_pieces = 0
        last_state = 0
        j = 0
        while j < size:
            while j < size and gray_image[i][j] == last_state:
                j += 1

            if j == size:
                break
            elif gray_image[i][j] != last_state:
                if last_state == 0:
                    white_pieces += 1
                last_state = 255 - last_state
                continue

        if white_pieces > 0:
            total_rows += 1
        if white_pieces > 1:
            rowsd += 1

    for i in range(size):
        white_pieces = 0
        last_state = 0
        j = 0
        while j < size:
            while j < size and gray_image[j][i] == last_state:
                j += 1

            if j == size:
                break
            elif gray_image[j][i] != last_state:
                if last_state == 0:
                    white_pieces += 1
                last_state = 255 - last_state
                continue

        if white_pieces > 0:
            total_cols += 1
        if white_pieces > 1:
            colsd += 1

    r = rowsd / total_rows
    c = colsd / total_cols
    return max([total_cols, total_rows]), min([r, c]) * 10


p = "..\linear classifier_3features\\"

# ----------------------------------- reading learning parameters --------------------------
theta_file = open(p + 'theta.txt', 'r')
line_theta = theta_file.readline()
diamond_theta = theta_file.readline()

l = line_theta.split(' ')
line_theta = np.array([[float(l[0])], [float(l[1])], [float(l[2])], [float(l[3])]], np.float64)

d = diamond_theta.split(' ')
diamond_theta = np.array([[float(d[0])], [float(d[1])], [float(d[2])], [float(d[3])]], np.float64)

theta_file.close()
# -------------------------------------- reading test set --------------------------------------
lines = os.listdir(p + "test set\Line_TS\\")
diamonds = os.listdir(p + "test set\Diamond_TS\\")
ellipses = os.listdir(p + "test set\Ellipse_TS\\")
report_file = open(p + "test set\\report.txt", "w")


# ------------------------------------- evaluating Line --------------------------------------
correct = 0
Error = []
for line in lines:
    im = cv2.imread(p + "test set\Line_TS\\" + line)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    f3, f1 = extract_feature3_1(gray)
    f2 = extract_feature2(gray)
    print("evaluating " + line)
    f_vector = np.array([[1], [f1], [f2], [f3]])

    r = np.dot(np.transpose(f_vector), line_theta)
    if r[0][0] > 0:
        correct += 1
        continue;

    r = np.dot(np.transpose(f_vector), diamond_theta)
    if r[0][0] > 0:
        Error.append(line + " classified as diamond")
    else:
        Error.append(line + " classified as ellipse")

report_file.write("------------------ Line Folder -----------------\n")
report_file.write("#Images = " + str(len(lines)) + "     Faults = " + str(len(lines) - correct) + "\n\n")
report_file.write("Accuracy = " + str(correct / len(lines) * 100) + "\n")
report_file.write("Errors:\n")
for e in Error:
    report_file.write(e + "\n")
report_file.write("\n\n\n")



# ------------------------------------ evaluating Diamond -----------------------------------
correct = 0
Error = []
for diamond in diamonds:
    im = cv2.imread(p + "test set\Diamond_TS\\" + diamond)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    f3, f1 = extract_feature3_1(gray)
    f2 = extract_feature2(gray)
    print("evaluating " + diamond)
    f_vector = np.array([[1], [f1], [f2], [f3]])

    r = np.dot(np.transpose(f_vector), line_theta)
    if r[0][0] > 0:
        Error.append(diamond + " classified as line")
        continue

    r = np.dot(np.transpose(f_vector), diamond_theta)
    if r[0][0] > 0:
        correct += 1
    else:
        Error.append(diamond + " classified as ellipse")

report_file.write("------------------ Diamond Folder -----------------\n")
report_file.write("#Images = " + str(len(diamonds)) + "     Faults = " + str(len(diamonds) - correct) + "\n\n")
report_file.write("Accuracy = " + str(correct / len(diamonds) * 100) + "\n")
report_file.write("Errors:\n")
for e in Error:
    report_file.write(e + "\n")
report_file.write("\n\n\n")



# -------------------------------------- evaluating ellipses ---------------------------------
correct = 0
Error = []
for ellipse in ellipses:
    im = cv2.imread(p + "test set\Ellipse_TS\\" + ellipse)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    f3, f1 = extract_feature3_1(gray)
    f2 = extract_feature2(gray)
    print("evaluating " + ellipse)
    f_vector = np.array([[1], [f1], [f2], [f3]])

    r = np.dot(np.transpose(f_vector), line_theta)
    if r[0][0] > 0:
        Error.append(ellipse + " classified as line")
        continue

    r = np.dot(np.transpose(f_vector), diamond_theta)
    if r[0][0] > 0:
        Error.append(ellipse + " classified as diamond")
    else:
        correct += 1


report_file.write("------------------ Ellipse Folder -----------------\n")
report_file.write("#Images = " + str(len(ellipses)) + "     Faults = " + str(len(ellipses) - correct) + "\n\n")
report_file.write("Accuracy = " + str(correct / len(ellipses) * 100) + "\n")
report_file.write("Errors:\n")
for e in Error:
    report_file.write(e + "\n")
report_file.write("\n\n\n")


report_file.close()
