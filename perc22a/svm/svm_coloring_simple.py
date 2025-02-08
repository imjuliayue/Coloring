import numpy as np
from svm_dependents import *
from SVM import *
from cones import *
from numpy.typing import NDArray
import matplotlib.pyplot as plt
# from svm_dependents import *

# TODO: need to import SVM class and Cones class

def midlineToLine(midline: NDArray):
    """
    Find line through end of midline
    Args:
        midline: set of 3d points representing the midline for points; sorted from start to end of sorted track
    Returns:
        tuple(tuple(num, num, num), num): 
            the slope and intercept of the line (slope, intercept) which are a 3d vector and a scalar (x-intercept)
    """
    if len(midline) == 0 or len(midline) == 1:
        return (0,1,0), 0

    # take last two points
    lastPoint = midline[-1]
    secondLP = midline[-2]

    # # slope
    slopeVec = lastPoint - secondLP

    # # vertical line
    if slopeVec[0] == 0:
        return (0,1,0), slopeVec[0]     # return x-intercept instead

    slope = slopeVec[1]/slopeVec[0]     # = y/x (y is forward direction)

    # intercept (use y = mx + b --> b = mx - y)
    intercept = slope * lastPoint[0] - lastPoint[1]

    sv, i = midlineToAvgLine(midline)
    print(f"normal: slopevec-{slopeVec}   intercept-{intercept}")
    print(f"avg: slopevec-{sv}   intercept-{i}")
    return slopeVec, intercept

def midlineToAvgLine(midline: NDArray, cones: Cones):
    """
    Find line through end of midline
    Args:
        midline: set of 3d points representing the midline for points; sorted from start to end of sorted track
    Returns:
        tuple(tuple(num, num, num), num): 
            the slope and intercept of the line (slope, intercept) which are a 3d vector and a scalar (x-intercept)
    """
    if len(midline) == 0 or len(midline) == 1:
        return (0,1,0), 0

    # Take last two points
    lastPoint = midline[-1]

    # Get slopes between first & third, second & fourth furthest points
    tailPts = midline[-1:-5:-1]
    print(tailPts)
    slopeVec1 = tailPts[0] - tailPts[2]  # Slopevec through 1 and 3
    slopeVec2 = tailPts[2] - tailPts[3]  # Slopevec through 2 and 4
    midSlopeVec = avgSlope(slopeVec1, slopeVec2)

    # Weight with coneSlopeVec
    if (len(cones.blue_cones) >= 2 and len(cones.yellow_cones)):
        coneSlopeVec = coneSlope(cones)
        slopeVec = (midSlopeVec + .5*coneSlopeVec)/2
    else:
        slopeVec = midSlopeVec

    # Vertical line
    if slopeVec[0] == 0:
        return (0,1,0), slopeVec[0]     # return x-intercept instead

    slope = slopeVec[1]/slopeVec[0]     # = y/x (y is forward direction)
    intercept = slope * lastPoint[0] - lastPoint[1]  # y = mx + b --> b = mx - y

    return slopeVec, intercept

def coneSlope(cones: Cones):
    """
    Args:
        cones: all colored cones
    Returns:
        furthest two blue and yellow cone slopeVecs, averaged
    """
    # Grab furthest two blue and yellow cones
    farB1 = np.array(cones.blue_cones[-1])
    farB2 = np.array(cones.blue_cones[-2])
    farY1 = np.array(cones.yellow_cones[-1])
    farY2 = np.array(cones.yellow_cones[-2])

    slopeBVec = farB1 - farB2
    slopeYVec = farY1 - farY2
    coneSlopeVec = avgSlope(slopeBVec, slopeYVec)
    return coneSlopeVec[:2]

def avgSlope(slopeVec1, slopeVec2):
    # Check if either line is vertical
    if slopeVec2[0] == 0:
        if slopeVec1[0] == 0:               # Both vertical
            return (0, 1, 0)
        else:                               # Only 2 vertical
            slopeVec1[0] /= 2
            avgSlopeVec = slopeVec1
    else:
        if slopeVec1[0] == 0:               # Only 1 vertical
            slopeVec2[0] /= 2
            avgSlopeVec = slopeVec2
        else:                               # Both not vertical
            avgSlopeVec = (slopeVec1 + slopeVec2)/2

    return avgSlopeVec

def classify(slopeVec, intercept, point):
    """
    Args:
        slopeVec:     3d vector representing slope. X is forward direction, Y is left/right direction
        intercept:    a numerical value (x-intercept)
        point:        the point to be classified
    Returns:
        classification:   0 if blue and 1 if yellow
        perpSlope:        Slope of perpendicular line including point (SCALAR)
        perpIntercept:    intercept of perpendicular line including point (SCALAR)
    TODO: there's a chance point ENDS UP on the line. Have a tie breaker.
    TODO: there's a chance slope could be 0. Possible solution: midlineToLine also outputs boolean to indicate y-intercept instead
    """

    # CORNER CASES
    # Case 1: slope is vertical line

    # print()
    # print(str(slopeVec) + " " + str(intercept) + " " + str(point))
    if slopeVec[0] == 0:
        res = (1,0,0)
        if slopeVec[1] < 0:
            res = (-1,0,0)
        return slopeVec[1] < 0 and point[0] < intercept or slopeVec[1] >= 0 and point[0] >= intercept, res, point[1]

    # Case 2: slope is horizontal line
    if slopeVec[1] == 0:
        res = (0,1,0)
        if(slopeVec[0] > 0):
            res = (0,-1,0)
        return slopeVec[0] < 0 and point[1] > intercept or slopeVec[0] >= 0 and point[1] <= intercept, res, point[0]


    # find perpendicular line with 'point' in it

    slope = slopeVec[1] / slopeVec[0]
    perpSlope = slopeVec[0] * -1 / slopeVec[1]      # -y/x (y is forward direction)

    perpIntercept =  point[1] - perpSlope * point[0] # b = y - mx

    # find the pt on the line to compare 'point' to: m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
    x = (intercept - perpIntercept) / (perpSlope - slope)

    y = perpSlope * x + perpIntercept

    # print("x,y: " + str((x,y)))

    # classify y by taking difference between point and (x,y)s' x components
    print(f"pt 0: {point[0]}    x: {x}")
    direction = point[0] - x

    classification = 0

    print(direction)
    if direction > 0 and (slopeVec[1] > 0) or direction < 0 and (slopeVec[1] < 0):
        classification = 1

    return classification, (0,perpSlope,0), perpIntercept


def get_closest_point_idx(points: NDArray, curr_point: NDArray) -> int:
    """
    Return index of coord in points closest to curr_point

    TODO: for testing, create "colored cones" so that points isn't empty list
    """
    assert(points.size > 0)
    # gets index of point in points farthest from curr_point and returns the dist
    assert(points.shape[1] == curr_point.shape[0])

    sq_dists = np.sum((points - curr_point) ** 2, axis=1)
    idx = np.argmin(sq_dists)
    return idx
    # return idx, np.sqrt(sq_dists[idx])

def conesBeforeLine(points, perpSlopeVec, perpIntercept):
    # finds all cones closer than farthest cone
    
    # OUTPUT: List of True/False at corresponding indices; True --> is closer.

    newList = []                    # holds True/False values

    for i in range(len(points)):               # classify function kind of does it lol
        print(perpSlopeVec)
        c, _, _ = classify(perpSlopeVec, perpIntercept, points[i])
        newList.append(c)
    
    return newList

def initClassification(points: NDArray, cones: Cones, SVM: SVM) -> tuple[NDArray, NDArray]:                                          # WIP
    """
    Classify cone pair nearest to car.

    Args:
        points: all cone points
        cones: current colored cones
        SVM: current svm
    Returns:
        NDArray: midline points
        NDArray: unclassified cone points
    """

    # Get one of the closest points and add to Cones
    idx = get_closest_point_idx(points, np.array((0,0,0)))       # function is in SVM.py
    pt1 = points[idx]
    points = np.delete(points, idx, axis=0)                                        # remove classified point

    # Get second closest point and add to Cones
    idx2 = get_closest_point_idx(points, np.array((0,0,0)))
    pt2 = points[idx2]
    points = np.delete(points, idx2, axis=0)                                            # remove classified point

    # Classify these cones as left or right
    y1 = pt1[1]
    y2 = pt2[1]
    blue = pt1
    yellow = pt2

    if y1 > y2:
        blue = pt2
        yellow = pt1
    
    cones.add_blue_cone(blue[0], blue[1], blue[2])
    cones.add_yellow_cone(yellow[0], yellow[1], yellow[2])

    print(cones)
    return SVM.cones_to_midline(cones), points

def filter_out_cones(cones_in: NDArray, cones_notin: NDArray) -> NDArray:
    """Returns coordinates of cones_in \ cones_notin"""
    combined = np.concatenate((cones_in, cones_notin))
    return np.unique(combined, axis=0)

def SVM_update(midline: NDArray, coloredCones: Cones, points: NDArray) -> Cones:
    """
    Classify all cones in current frame.

    Args:
        midline: midline points (MUST have ordering like a spline)
        coloredCones: previously colored cones
        points: cone coords in frame
    Returns:
        Cones: classified cones (including input)

    TODO: only output classifications of new cones!!!
    """

    # Create an SVM class
    svm = SVM()

    # Corner case: coloredCones is initially empty
    if len(coloredCones) == 0:
        print("Colored cones empty")
        midline, points = initClassification(points, coloredCones, svm)
    

    # points set minus colorCones
    print(coloredCones)
    npColored = np.array(np.concatenate((coloredCones.blue_cones, coloredCones.yellow_cones)))
    npPoints = np.array(points) 

    print("npColored:\n" + str(npColored))      
    print("npPoints:\n" + str(npPoints))      
    # points = np.diff(np.array([npPoints, npColored]), axis=1)     # Note: now points is coords of UNCLASSIFIED cones
    npPoints = np.array([row for row in npPoints if not any(np.array_equal(row, r) for r in npColored)])
    print("npPoints after:\n" + str(npPoints))      
    points = npPoints.tolist()

    # Make a new cones class and pass these colored cones into it
    cones = Cones()
    cones.add_cones(coloredCones)

    # Find farthest blue and yellow cones (should be closest to last midline point)
    idxBlue = get_closest_point_idx(np.array(cones.blue_cones), np.append(np.array(midline[-1]),0))
    idxYellow = get_closest_point_idx(np.array(cones.yellow_cones), np.append(np.array(midline[-1]),0))

    farBlue = np.array(cones.blue_cones[idxBlue])
    farYellow = np.array(cones.yellow_cones[idxYellow])

    loopCt = 0
    # Iteratively classify all the points
    while(len(points) > 0):
        print("======================================")
        print(f"Num points: {len(points)}")
        # Find line extending end of midlline
        # slopeVec0 = midlineToLine(midline)
        slopeVec, intercept = midlineToAvgLine(midline, cones)
        # print(f"slope0: {slopeVec0[1]/slopeVec0[0]}  slope1: {slopeVec[1]/slopeVec[0]} ")
        
        # print("farBlue:\n" + str(farBlue))
        # print("farYellow:\n" + str(farYellow))

        # Find the two points closest to farBlue and farYellow to classsify
        idx1 = get_closest_point_idx(np.array(points), farBlue)
        point1 = points[idx1]
        points.remove(point1)

        idx2 = get_closest_point_idx(np.array(points), farYellow)
        point2 = points[idx2]
        points.remove(point2)
        print()
        # print("pointsAfterRemove:\n" + str(points))
        print("point1: \n" + str(point1))
        print("point2: \n" + str(point2))
        print()

        # classify them
        # TODO: improve modularity
        class1, pSlope1, pIntercept1 = classify(slopeVec, intercept, point1)
        class2, pSlope2, pIntercept2 = classify(slopeVec, intercept, point2)

        print("class1: \n" + str(class1))
        print("class2: \n" + str(class2))
        print()

        if class1:
            cones.add_yellow_cone(point1[0],point1[1],point1[2])
        else:
            cones.add_blue_cone(point1[0],point1[1],point1[2])
        
        if class2:
            cones.add_yellow_cone(point2[0],point2[1],point2[2])
        else:
            cones.add_blue_cone(point2[0],point2[1],point2[2])

        # print("\nCones classified:\n" + str(cones.yellow_cones) + "\n" + str(cones.blue_cones))

        # determine farthest cone
        farthest = (point2, pSlope2, pIntercept2)
        # print("FARTHEST:\n" + str(farthest)) 

        if(slopeVec[1] >= 0 and pIntercept1 > pIntercept2 or slopeVec[1] < 0 and pIntercept1 < pIntercept2):
            farthest = (point1, pSlope1, pIntercept1)

        # print("farthest: (point, slope, intercept\n" + str(farthest) + ")")

        # if no more points, we are done!
        if(len(points) == 0):
            return cones

        farBlue = np.array(cones.blue_cones[-1])
        farYellow = np.array(cones.yellow_cones[-1])

        print("farBlue: " + str(farBlue))
        print("farYellow: " + str(farYellow))
        # plug into SVM
        midline = svm.cones_to_midline(cones)
        # print("midline: ")
        # print(midline)
        print("cones:\n" + str(cones))

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        bcones = np.array(cones.blue_cones)
        ycones = np.array(cones.yellow_cones)
        p = np.array(points)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color='black', label='pts', s=50)
        ax.scatter(bcones[:, 0], bcones[:, 1], bcones[:, 2], color='blue', label='blue', s=50)
        ax.scatter(ycones[:, 0], ycones[:, 1], ycones[:, 2], color='orange', label='orange', s=50)
        ax.scatter(midline[:, 0], midline[:, 1], 0, color='red', label='midline', s=50)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        plt.title('3D Splines and Clusters')

        plt.show()

        # print(midline)

    # return the cones
    return cones
        
        




