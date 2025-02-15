import numpy as np
from svm_dependents import *
from SVM import *
from cones import *
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import math
# from svm_dependents import *

class Slope:
    """
    Fields
        mx, my: scalar slope vector components
        slope: scalar slope value
        isVert, isHoriz: indicates vertical/horizontal slope
        headPos: direction of slope if vertical/horizontal
    Notes
        (mx, my) = None if (isVert or isHoriz) else (float, float)
        slope = my/mx if (not (isVert or isHoriz)) else None
        headPos = None if (not (isVert or isHoriz)) else (y > 0 and isVert) or (x > 0 and isHoriz)
    """
    def __init__(self, headPos: bool, mx: float = None, my: float = None, slope = None,
                 vert: bool = False, horiz: bool = False):
        assert ((mx is not None and my is not None) or slope or vert or horiz)
        assert (not (mx == 0 and my == 0))
        self.isVert = self.isHoriz = False
        self.headPos = headPos
        if slope is not None:
            self.slope = slope
        elif vert or mx == 0:
            assert not mx and not horiz
            self.isVert = True
            self.slope = None
        elif horiz or my == 0:
            assert not my and not vert
            self.isHoriz = True
            self.slope = 0
        else:
            self.slope = my/mx

class Line(Slope):
    def __init__(self, headPos: bool, slope_x: float = None, slope_y: float = None, slope = None,
                 intercept: float = 0, vert: bool = False, horiz: bool = False,):
        Slope.__init__(self, headPos, slope_x, slope_y, slope, vert, horiz)
        self.intercept = intercept

def getAvgSlope(slope1: Slope, slope2: Slope, w1: int = 1, w2: int = 1) -> Slope:
        """Returns average of the two slopes as a Slope class"""
        if slope2.isVert:
            if slope1.isVert:           # Both vertical
                return Slope(headPos=slope1.headPos, vert=True)
            else:                       # Avg slope 1 with vertical
                theta = math.atan(slope1.slope)
                if theta >= 0:
                    vert = math.radians(90)
                else:
                    vert = math.radians(-90)
                return Slope(headPos=slope1.headPos, slope=math.tan((vert+theta)/2))
        else:
            if slope1.isVert:           # Avg slope 2 with vertical
                angle = math.atan(slope2.slope)
                if angle >= 0:
                    vert = math.radians(90)
                else:
                    vert = math.radians(-90)
                return Slope(headPos=slope2.headPos, slope=math.tan((vert+angle)/2))
            else:                       # Both not vertical
                theta1 = math.atan(slope1.slope)
                theta2 = math.atan(slope2.slope)
                if (theta1 >= 0 and not slope1.headPos or
                    theta1 < 0 and slope1.headPos):
                    theta1 += math.radians(180)
                if (theta2 >= 0 and not slope2.headPos or
                    theta2 < 0 and slope2.headPos):
                    theta2 += math.radians(180)

                thetaAvg = (w1*theta1 + w2*theta2)/(w1+w2)
                headPos = thetaAvg >=0 and thetaAvg < math.radians(180)
                return Slope(headPos=headPos, slope=math.tan(thetaAvg))

def getConeSlope(cones: Cones):
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

    # Headed up if y1 > y2 or if same y, then x1 > x2
    headPosB = farB1[1] > farB2[1] or farB1[1] == farB2[1] and farB1[0] > farB2[0] # Cone points sohuldn't be identical
    headPosY = farY1[1] > farY2[1] or farY1[1] == farY2[1] and farB1[0] > farB2[0]
    slopeB = Slope(headPosB, farB1[0] - farB2[0], farB1[1] - farB2[1])
    slopeY = Slope(headPosY, farY1[0] - farY2[0], farY1[1] - farY2[1])
    return getAvgSlope(slopeB, slopeY)



def midlineToAvgLine(midline: NDArray, cones: Cones) -> Line:
    """
    Find line through end of midline
    Args:
        midline: set of 3d points representing the midline for points; sorted from start to end of sorted track
    Returns:
        Line: line extending midline
    """
    if len(midline) == 0 or len(midline) == 1:
        return Line(headPos=True, vert=True, intercept=0)
    
    lastPoint = midline[-1]

    # Get slopes between first & third, second & fourth furthest points
    tailPts = midline[-1:-5:-1]  # Points ordered in descending distance

    # Slopevec through 1 and 3
    headPos13 = (tailPts[0][1] > tailPts[2][1] or 
                 tailPts[0][1] == tailPts[2][1] and tailPts[0][0] > tailPts[2][0])
    slope13 = Slope(headPos13, tailPts[0][0] - tailPts[2][0], tailPts[0][1] - tailPts[2][1])
    # Slopevec through 2 and 4
    headPos24 = (tailPts[1][1] > tailPts[3][1] or
                 tailPts[1][1] == tailPts[3][1] and tailPts[1][0] > tailPts[3][0])
    slope24 = Slope(headPos24, tailPts[1][0] - tailPts[3][0], tailPts[1][1] - tailPts[3][1])
    midlineSlope = getAvgSlope(slope13, slope24)

    # Average midline slope with cone slope
    if (len(cones.blue_cones) >= 2 and len(cones.yellow_cones) >= 2):
        coneSlope = getConeSlope(cones)
        extnSlope = getAvgSlope(midlineSlope, coneSlope)
    else:
        extnSlope = midlineSlope

    # Vertical line
    if extnSlope.isVert:
        return Line(headPos=extnSlope.headPos, vert=True, intercept=lastPoint[0])  # return x-intercept instead 

    intercept = lastPoint[1] - extnSlope.slope * lastPoint[0] # y = mx + b --> b = y - mx
    print(f"Slope: {extnSlope}, intercept: {intercept}")
    return Line(headPos=extnSlope.headPos, slope=extnSlope.slope, intercept=intercept)

def classify(midline: Line, point: NDArray):
    """
    Args:
        slope:        scalar slope value
        intercept:    a numerical value (x-intercept)
        point:        the point to be classified (3D NDArray)
    Returns:
        classification:   0 if blue and 1 if yellow
        perpSlope:        Slope of perpendicular line including point (SCALAR)
        perpIntercept:    intercept of perpendicular line including point (SCALAR)
    TODO: there's a chance point ENDS UP on the line. Have a tie breaker.
    TODO: there's a chance slope could be 0. Possible solution: midlineToLine also outputs boolean to indicate y-intercept instead
    """

    if midline.isVert:
        classification = ((midline.headPos and point[0] >= midline.intercept) or
                         ((not midline.headPos) and point[0] < midline.intercept))
        perpLine = Line(headPos=midline.headPos, horiz=True, intercept=point[1])
        return classification, perpLine

    if midline.isHoriz:
        classification = ((midline.headPos and point[1] <= midline.intercept) or
                          ((not midline.headPos) and point[1] > midline.intercept))
        perpLine = Line(headPos=not midline.headPos, vert=True, intercept=point[0])
        return classification, perpLine

    # find perpendicular line with 'point' in it
    # slope = slopeVec[1] / slopeVec[0]
    # perpSlope = slopeVec[0] * -1 / slopeVec[1]      # -y/x (y is forward direction)
    perpSlope = -1/midline.slope
    perpIntercept =  point[1] - perpSlope * point[0] # b = y - mx

    # find the pt on the line to compare 'point' to: m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
    x = (midline.intercept - perpIntercept) / (perpSlope - midline.slope)
    y = perpSlope * x + perpIntercept

    # classify y by taking difference between point and (x,y)s' x components
    print(f"pt 0: {point[0]}    x: {x}")
    direction = point[0] - x

    classification = 0

    print(direction)
    if direction > 0 and midline.headPos or direction < 0 and not midline.headPos:
        classification = 1


    if math.tan(midline.slope) > 0:
        if midline.headPos:
            perpHead = False
        else:
            perpHead = True
    else:
        perpHead = midline.headPos
    return classification, Line(perpHead, slope=perpSlope)


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

    coneSlopes = []
    # Iteratively classify all the points
    while(len(points) > 0):
        print("======================================")
        print(f"Num points: {len(points)}")
        # Find line extending end of midlline
        # slopeVec0 = midlineToLine(midline)
        extnMidline = midlineToAvgLine(midline, cones)
        coneSlopes.append(extnMidline)
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
        class1, perpLine = classify(extnMidline, point1)
        class2, perpLine = classify(extnMidline, point2)

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
        # farthest = (point2, pSlope2, pIntercept2)
        # # print("FARTHEST:\n" + str(farthest)) 

        # if(slopeVec[1] >= 0 and pIntercept1 > pIntercept2 or slopeVec[1] < 0 and pIntercept1 < pIntercept2):
        #     farthest = (point1, pSlope1, pIntercept1)

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
        
        




