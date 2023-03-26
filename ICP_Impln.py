import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


with open(r"./pclX.txt", "r") as point_file:
    points_cloudX = point_file.readlines()
    points_cloudX = [x.split() for x in points_cloudX]
    points_cloudX = np.array([[float(x[0]), float(x[1]), float(x[2])] for x in points_cloudX])

# fig = plt.figure()
# ax = plt.subplot(111, projection="3d")
#
# ax.scatter(points_cloudX[:, 0],  # x
#            points_cloudX[:, 1],  # y
#            points_cloudX[:, 2],  # z
#            )


with open(r"./pclY.txt", "r") as point_file:
    points_cloudY = point_file.readlines()
    points_cloudY = [x.split() for x in points_cloudY]
    points_cloudY = np.array([[float(x[0]), float(x[1]), float(x[2])] for x in points_cloudY])

fig = plt.figure()
ax = plt.subplot(111, projection="3d")

ax.scatter(points_cloudY[:, 0],  # x
           points_cloudY[:, 1],  # y
           points_cloudY[:, 2],  # z
           )


def compute_optimal_ridge_registration(X, Y, C):
    C = np.array(C)
    x_cap = C[:, 0]-C[:, 0].mean(axis=0)
    y_cap = C[:, 1]-C[:, 1].mean(axis=0)

    x_cap = x_cap/x_cap.shape[0]
    y_cap = y_cap/y_cap.shape[0]
    # W = y_cap.T.dot(x_cap)
    # W = W.sum()
    W = [np.matmul(np.expand_dims(y, axis=0).T, np.expand_dims(x, axis=0)) for x,y in zip(x_cap, y_cap)]
    W = np.array(W)
    W = W.sum(axis=0)
    W = W/x_cap.shape[0]

    U, S, VT = np.linalg.svd(W)
    R = U.dot(VT)

    t = Y.mean(axis=0) - X.mean(axis=0).dot(R)
    return R, t


def EstimateCorrespondences(x, Y, t, R, dmax):
    global points_cloudY
    temp_y = Y-x
    # temp_y = temp_y**2
    # temp_y = np.sum(temp_y,  axis=1)**0.5
    temp_y = np.linalg.norm(temp_y, axis=1)
    temp_y_i = np.argmin(temp_y)
    if temp_y[temp_y_i]<dmax:
        return points_cloudY[temp_y_i]
    else:
        return None


def ICP(X, Y, t0, R0, dmax, iterations):
    t = t0
    R = R0

    while iterations:
        C = []
        new_X = X.dot(R) + t

        for x in new_X:
            set_y = EstimateCorrespondences(x, Y, t, R, dmax)
            if set_y is not None:
                C.append((x, set_y))

        R, t = compute_optimal_ridge_registration(X, Y, C)
        print(R, t)
        iterations -= 1
    return R, t


new_R, new_t = ICP(points_cloudX, points_cloudY, np.array([0,0,0]), np.identity(3), 0.25, iterations=30)


points_cloudX_transform = points_cloudX.dot(new_R) + new_t
ax = plt.subplot(111, projection="3d")

ax.scatter(points_cloudX_transform[:, 0],  # x
           points_cloudX_transform[:, 1],  # y
           points_cloudX_transform[:, 2],  # z
           )

plt.show()