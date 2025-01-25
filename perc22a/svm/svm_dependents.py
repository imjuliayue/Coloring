import numpy as np

def cones_to_midline(cones):
    "cones is a Cones type"
    blue_cones, yellow_cones, _ = cones.to_numpy()
    if len(blue_cones) == 0 and len(yellow_cones) == 0:
        return []

    # augment dataset to make it better for SVM training
    self.supplement_cones(cones)
    aug_cones = self.augment_cones_circle(cones, deg=10, radius=1.2)

    X, y = self.cones_to_xy(aug_cones)

    model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
    model.fit(X, y)
    self.prev_svm_model = model

    if DEBUG_SVM:
        self.debug_svm(aug_cones, X, y, model)

    # TODO: prediction takes 20-30+ ms, need to figure out how to optimize
    step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    svm_input = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(svm_input)
    Z = Z.reshape(xx.shape)

    if DEBUG_PRED:
        self.debug_pred(Z)

    # get top-left corner (TL) and bottom-right (BR) corner of Z
    Z_TL = Z[:-1, :-1]
    Z_BR = Z[1:, :-1]
    Z_C = Z[1:, 1:]
    XX_C = xx[1:, 1:]
    YY_C = yy[1:, 1:]
    idxs = np.where(np.logical_or(Z_C != Z_TL, Z_C != Z_BR))
    boundary_xx = XX_C[idxs].reshape((-1, 1))
    boundary_yy = YY_C[idxs].reshape((-1, 1))
    boundary_points = np.concatenate([boundary_xx, boundary_yy], axis=1)

    # sort the points in the order of a spline
    boundary_points = self.sort_boundary_points(boundary_points)

    # downsample the points
    downsampled = []
    accumulated_dist = 0
    for i in range(1, len(boundary_points)):
        p1 = boundary_points[i]
        p0 = boundary_points[i - 1]
        curr_dist = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        accumulated_dist += curr_dist
        if np.abs(accumulated_dist - 0.5) < 0.1:  # TODO: make this 50cm
            downsampled.append(p1)
            accumulated_dist = 0

        if accumulated_dist > 0.55:
            accumulated_dist = 0

    if DEBUG_POINTS:
        self.debug_points(boundary_points)

    curr_timestep_spline = np.array(list(downsampled))
    # print(downsampled)

    # curr_timestep_spline = self.outlier_rejection(curr_timestep_spline)
    self.prev_spline = curr_timestep_spline

    return curr_timestep_spline