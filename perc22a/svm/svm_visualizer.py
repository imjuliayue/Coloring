import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import Axes3D
# from perc22a.predictors.utils.cones import Cones
from svm_coloring_simple import *
from perc22a.svm.cones import Cones

OFFSET = 2
BEND_FACTOR = 3
NUM_CONES = 10

'''
    Generate Spline
'''


def generate_spline(flatness):
    X = np.linspace(0, 20, 40)
    Y = flatness * np.sin(X) * np.exp(-X / 20)
    Y = np.sort(Y)
    Y += np.abs(np.min(Y))

    # Create a cubic spline interpolation for smoothness
    X_smooth = np.linspace(X.min(), X.max(), 10)
    spline = make_interp_spline(X, Y)
    Y_smooth = spline(X_smooth)

    return spline, X_smooth, Y_smooth, X, Y


# Generate parallel tracks
def generate_parallel_splines(origin_spline, offset=5, bend_factor=5, num_points=500):
    y_values = np.linspace(0, 20, num_points)
    central_spline_z = origin_spline(y_values)

    # Create a bend
    bend_curve = np.sin(y_values / bend_factor) * offset

    spline_left = np.array([bend_curve - offset, y_values, central_spline_z]).T
    spline_right = np.array([bend_curve + offset, y_values, central_spline_z]).T

    return spline_left, spline_right


# Simulate cones on points in the track
def generate_points_above_splines(spline_left, spline_right, height=2, num_points=5):
    # Select `num_points` evenly spaced indices along the splines
    indices = np.linspace(0, len(spline_left) - 1, num_points).astype(int)

    points_above = []

    for idx in indices:
        point_left = spline_left[idx]
        point_right = spline_right[idx]

        point_left_above = point_left + np.array([0, 0, height])
        point_right_above = point_right + np.array([0, 0, height])

        points_above.append(point_left_above)
        points_above.append(point_right_above)

    return np.array(points_above)


def generate_clusters_around_points(points, cluster_size=50, base_radius_x=1.0, height=0.5):
    clusters = []

    for point in points:
        for _ in range(cluster_size):
            x_offset = np.random.uniform(-base_radius_x, base_radius_x)
            y_offset = np.random.uniform(-base_radius_x, base_radius_x)

            z_offset = np.random.uniform(-height / 2, height / 2)

            cluster_point = point + np.array([x_offset, y_offset, z_offset])
            clusters.append(cluster_point)

    return np.array(clusters)


# Randomness
flatness = random.uniform(0.1, 0.4)

# Generate spline
spline, X_smooth, Y_smooth, X, Y = generate_spline(flatness)
spline_left, spline_right = generate_parallel_splines(spline, offset=OFFSET, bend_factor=BEND_FACTOR, num_points=1000)
points_above = generate_points_above_splines(spline_left, spline_right, height=0.5, num_points=NUM_CONES)
clusters = generate_clusters_around_points(points_above, cluster_size=80, base_radius_x=1.0, height=0.8)

print(points_above)

# Plot the splines and clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot left and right splines
ax.plot(spline_left[:, 0], spline_left[:, 1], spline_left[:, 2], label='Left Spline', linewidth=2)
ax.plot(spline_right[:, 0], spline_right[:, 1], spline_right[:, 2], label='Right Spline', linewidth=2)

# Plot points above splines
ax.scatter(points_above[:, 0], points_above[:, 1], points_above[:, 2], color='red', label='Points Above Splines', s=50)

# Plot clusters
ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], color='blue', alpha=0.5, label='Clusters', s=10)

# Set labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title('3D Splines and Clusters')

plt.show()

colCones = Cones()
colCones.add_blue_cone(-2, -2, 0)
colCones.add_yellow_cone(2, -2, 0)
points_above[:,2] = 0
result = SVM_update([], colCones, points_above)
print(result)
# testing with the simple coloring algorithm
# line = midlineToAvgLine([], colCones, [])

# perpSlope = None 
# perpIntercept = None

# classed = []

# for cone in points_above:
#     classification, perpLine = classify(line, cone)

#     classed.append(classification)

print("----------------------------------------------------------------")


# print ("Cones: ")
# print (points_above)

# print()

# print("Midline_to_line result:")
# print("slopeVec: " + str(slopeVec))
# print("slopeIntercept: " + str(intercept))

# print("\nClassification result:")
# print("perpSlope: " + str(perpSlope))
# print("perpIntercept: " + str(perpIntercept))
# print(classed)

# print("---------------------------------------------------------------------") 

# print("\nTest2 with y = 3x + 4 slope")
# slopeVec = (1,-3,0)
# intercept = 3
# points = [(0,0,0), (-2,1,0), (2,4,0), (-4,7,0)]

# classed = []
# slopes = []
# intercepts = []

# for cone in points:
#     classification, perpSlope, perpIntercept = classify(slopeVec, intercept, cone)

#     classed.append(classification)
#     slopes.append(perpSlope)
#     intercepts.append(perpIntercept)

# print("\nClassification result:")
# print("perpSlope: " + str(perpSlope))
# print("perpIntercept: " + str(perpIntercept))
# print(classed)
# print(slopes)
# print(intercepts)

# print("----------------------------------------------------------------")
# print("\nTesting conesBeforeLine")

# perpSlope2 = -2
# perpIntercept2 = 4

# newList = conesBeforeLine(points, perpSlope2, perpIntercept2)

# print("\n Class result:")
# print(newList)