import cv2
import numpy as np
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


# images paths
n = 6
p = '..\\linear classifier_3features\\'
sub = ['1.0', '0.8', '0.6', '0.4']

# images paths
train_file = open(p + "train_feat.txt", 'w')
test_file = open(p + "test_feat.txt", 'w')


# ------------------------- extracting features -------------------------------

# ############################# extracting line #################################3
for scale in sub:
    lines = os.listdir(p + 'Line_TS\\' + scale + '\\')
    c = 0
    X = np.zeros((len(lines), n))
    for line in lines:
        image = cv2.imread(p + 'Line_TS\\' + scale + '\\' + line)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f3, f1 = extract_feature3_1(gray_image)
        f2 = extract_feature2(gray_image)
        # gray_image = np.float32(gray_image
        X[c][0] = 1
        X[c][1] = f1
        X[c][2] = f2
        X[c][3] = f3
        X[c][4] = 1   # Y = line
        X[c][5] = c   # index of line in the array
        print("line #" + str(c) + " processed")
        c += 1

    np.random.shuffle(X)
    lim = int(len(lines) * 0.75)
    # save training data
    for c in range(lim):
        train_file.write(lines[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("training line #" + str(c) + " written")
        c += 1

    # save test data
    for c in range(lim, len(lines), 1):
        test_file.write(lines[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("testing line #" + str(c) + " written")
        c += 1

# ############################# extracting diamond #################################3
for scale in sub:
    diamonds = os.listdir(p + 'Diamond_TS\\' + scale + '\\')
    c = 0
    X = np.zeros((len(diamonds), n))
    for diamond in diamonds:
        image = cv2.imread(p + 'Diamond_TS\\' + scale + '\\' + diamond)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f3, f1 = extract_feature3_1(gray_image)
        f2 = extract_feature2(gray_image)
        # gray_image = np.float32(gray_image
        X[c][0] = 1
        X[c][1] = f1
        X[c][2] = f2
        X[c][3] = f3
        X[c][4] = 2   # Y = diamond
        X[c][5] = c   # index of diamond in the array
        print("diamond #" + str(c) + " processed")
        c += 1

    np.random.shuffle(X)
    lim = int(len(diamonds) * 0.75)
    # save training data
    for c in range(lim):
        train_file.write(diamonds[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("training diamond #" + str(c) + " written")
        c += 1

    # save test data
    for c in range(lim, len(diamonds), 1):
        test_file.write(diamonds[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("testing diamond #" + str(c) + " written")
        c += 1

# ############################# extracting ellipse #################################3
for scale in sub:
    ellipses = os.listdir(p + 'Ellipse_TS\\' + scale + '\\')
    c = 0
    X = np.zeros((len(ellipses), n))
    for ellipse in ellipses:
        image = cv2.imread(p + 'Ellipse_TS\\' + scale + '\\' + ellipse)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f3, f1 = extract_feature3_1(gray_image)
        f2 = extract_feature2(gray_image)
        # gray_image = np.float32(gray_image
        X[c][0] = 1
        X[c][1] = f1
        X[c][2] = f2
        X[c][3] = f3
        X[c][4] = 3   # Y = ellipse
        X[c][5] = c   # index of ellipse in the array
        print("ellipse #" + str(c) + " processed")
        c += 1

    np.random.shuffle(X)
    lim = int(len(ellipses) * 0.75)
    # save training data
    for c in range(lim):
        train_file.write(ellipses[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("training ellipse #" + str(c) + " written")
        c += 1

    # save test data
    for c in range(lim, len(ellipses), 1):
        test_file.write(ellipses[int(X[c][5])] + " " + str(X[c][0]) + " " + str(X[c][1]) + " " + str(X[c][2]) + " " + str(X[c][3]) + " " + str(X[c][4]) + "\n")
        print("testing ellipse #" + str(c) + " written")
        c += 1

train_file.close()
test_file.close()
