import numpy as np
from svm_dependents import *
from perc22a.svm.cones import Cones
from perc22a.svm.SVM import SVM
# from svm_dependents import *

# TODO: need to import SVM class and Cones class

def midlineToLine(midline):
    # TODO: there's a chance slope could be 0. Possible solution: midlineToLine also outputs boolean to indicate y-intercept instead
    # midline:      set of 3d points representing the midline for points; sorted from start to end of sorted track

    # OUTPUT:       the slope and intercept of the line (slope, intercept) which are a 3d vector and a scalar (x-intercept)

    if len(midline) == 0 or len(midline) == 1:
        return (0,1,0), 0

    # take last two points
    lastPoint = midline[-1]
    secondLP = midline[-2]

    # slope
    slopeVec = lastPoint - secondLP

    # vertical line
    if slopeVec[0] == 0:
        return (0,1,0), slopeVec[0]     # return x-intercept instead

    slope = slopeVec[1]/slopeVec[0]     # = y/x (y is forward direction)

    # intercept (use y = mx + b --> b = mx - y)
    intercept = slope * lastPoint[0] - lastPoint[1]

    return slopeVec, intercept


def classify(slopeVec, intercept, point):
    # TODO: there's a chance point ENDS UP on the line. Have a tie breaker.
    # TODO: there's a chance slope could be 0. Possible solution: midlineToLine also outputs boolean to indicate y-intercept instead

    # slopeVec:     3d vector representing slope. X is forward direction, Y is left/right direction
    # intercept:    a numerical value (x-intercept)
    # point:        the point to be classified

    # OUTPUTS: 
    # classification:   0 if blue and 1 if yellow
    # perpSlope:        Slope of perpendicular line including point (SCALAR)
    # perpIntercept:    intercept of perpendicular line including point (SCALAR)

    # CORNER CASES
    # Case 1: slope is vertical line
    if slopeVec[0] == 0:
        return slopeVec[1] < 0 and point[0] < intercept or slopeVec[1] >= 0 and point[0] >= intercept, 0, point[1]

    # Case 2: slope is horizontal line
    if slopeVec[1] == 0:
        return slopeVec[0] < 0 and point[1] > intercept or slopeVec[0] >= 0 and point[1] <= intercept, None, point[0]


    # find perpendicular line with 'point' in it

    slope = slopeVec[1] / slopeVec[0]
    perpSlope = slopeVec[0] * -1 / slopeVec[1]      # -y/x (y is forward direction)

    perpIntercept =  point[1] - perpSlope * point[0] # b = y - mx

    # find the pt on the line to compare 'point' to: m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
    x = (intercept - perpIntercept) / (perpSlope - slope)

    y = perpSlope * x + perpIntercept

    print("x,y: " + str((x,y)))

    # classify y by taking difference between point and (x,y)s' x components
    direction = point[0] - x

    classification = 0

    if direction > 0 and slopeVec[1] > 0 or direction < 0 and slopeVec[1] < 0:
        classification = 1

    return classification, perpSlope, perpIntercept


def get_closest_point_idx(points, curr_point):
    # TODO: for testing, create "colored cones" so that points isn't empty list
    assert(points.size > 0)
    # gets index of point in points farthest from curr_point and returns the dist
    assert(points.shape[1] == curr_point.shape[0])

    sq_dists = np.sum((points - curr_point) ** 2, axis=1)
    idx = np.argmin(sq_dists)
    return idx, np.sqrt(sq_dists[idx])

def conesBeforeLine(points, perpSlope, perpIntercept):
    # finds all cones closer than farthest cone
    
    # OUTPUT: List of True/False at corresponding indices; True --> is closer.

    newList = []                    # holds True/False values

    for xy in points:               # classify function kind of does it lol
        c, _, _ = classify((1,perpSlope,0), perpIntercept, xy)
        newList.append(c)
    
    return newList

def initClassification(points, cones, SVM):                                          # WIP
    # returns: VOID

    # Get one of the closest points and add to Cones
    idx, _ = get_closest_point_idx(points, np.array((0,0,0)))       # function is in SVM.py
    pt1 = points[idx]
    points = np.delete(points, idx, axis=0)                                        # remove classified point



    # Get second closest point and add to Cones
    idx2, _ = get_closest_point_idx(points, np.array((0,0,0)))
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

    midline = SVM.cones_to_midline(cones)

    print(cones)

    return SVM.cones_to_midline(cones), points




def SVM_update(midline, coloredCones, points):# SHOULD BE CALLED AT BEGINNING OF FRAME
    # midline:      path-planning given midline     (MUST having ordering like a spline)
    # coloredCones: path-planning given coloredCones (CORRECT coordinates) : cones object
    # points:       all 3d coords of all cones in current frame

    # OUTPUT: colored cones (INCLUDING input)
    #   TODO: only output classifications of new cones!!!

    # Create an SVM class
    svm = SVM()

    # corner case: coloredCones is empty.
    if len(coloredCones) == 0:
        midline, points = initClassification(points, coloredCones, svm)
    

    # rid of coloredCones in points using setminus
    print(coloredCones)
    npColored = np.array(np.concatenate((coloredCones.blue_cones, coloredCones.yellow_cones)))
    npPoints = np.array(points) 

    print("npColored:\n" + str(npColored))      
    print("npPoints:\n" + str(npPoints))      



    # points = np.diff(np.array([npPoints, npColored]), axis=1)     # Note: now points is coords of UNCLASSIFIED cones
    npPoints = np.array([row for row in npPoints if not any(np.array_equal(row, r) for r in npColored)])
    print("npPoints after:\n" + str(npPoints))      

    points = npPoints.tolist()

    # make a new cones class and pass these colored cones into it
    cones = Cones()
    cones.add_cones(coloredCones)           # function in cones.py

    # find farthest blue and yellow cones (should be closest to last midline point)
    idxBlue, _ = get_closest_point_idx(np.array(cones.blue_cones), np.append(np.array(midline[-1]),0))
    idxYellow, _ = get_closest_point_idx(np.array(cones.yellow_cones), np.append(np.array(midline[-1]),0))


    farBlue = np.array(cones.blue_cones[idxBlue])
    farYellow = np.array(cones.blue_cones[idxYellow])

    print("farBlue:\n" + str(farBlue))
    print("farYellow:\n" + str(farYellow))

    # classify all the points
    while(len(points) > 0):
        # extend the line
        slopeVec, intercept = midlineToLine(midline)
        slope = slopeVec[0] / slopeVec[1]

        # choose the points to classify
        point1,_ = get_closest_point_idx(np.array(points), farBlue)
        print()
        print("point1:\n" + str(points[point1]))
        points.remove(points[point1])
        point2,_ = get_closest_point_idx(np.array(points), farYellow)
        print("point2:\n" + str(points[point2]))
        points.remove(points[point2])
        print("pointsAfterRemove:\n" + str(points))

        # classify them
        # TODO: improve modularity
        class1, pSlope1, pIntercept1 = classify(slopeVec, intercept, point1)
        class2, pSlope2, pIntercept2 = classify(slopeVec, intercept, point2)

        if class1:
            cones.add_yellow_cone(point1)
        else:
            cones.add_blue_cone
        
        if class2:
            cones.add_yellow_cone(point2)
        else:
            cones.add_blue_cone(point2)

        # determine farthest cone
        farthest = (point2, pSlope2, pIntercept2)

        if(slope >= 0 and pIntercept1 > pIntercept2 or slope < 0 and pIntercept1 < pIntercept2):
            farthest = (point1, pSlope1, pIntercept1)

        # Find all cones within distance from line
        nearCones = conesBeforeLine(farthest[1], farthest[2])

        nearPoints = points[nearCones]                                                      # filter out False cones
        points = np.setdiff1d(np.array(points), np.array(nearPoints)).tolist()              # TODO: really please optimize this

        # classify all these near points
        for xy in nearPoints:
            resXY, _, _ = classify(slopeVec, intercept, xy)
            
            if resXY:
                cones.add_yellow_cone(xy)
            else:
                cones.add_blue_cone(xy)

        # update farBlue and farYellow (many cases; if same class, depends on farthest cone)
        if class1 == class2:
            if class1:
                farYellow = farthest[0]
            else:
                farBlue = farthest[0]
        else:
            if class1:
                farYellow = point1
                farBlue = point2
            else:
                farBlue = point1
                farYellow = point2

        # plug into SVM
        midline = SVM.cones_to_midline(cones)

    # return the cones
    return cones
        
        




