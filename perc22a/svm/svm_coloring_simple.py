import numpy as np
from perc22a.predictors.utils.cones import Cones

# TODO: need to import SVM class and Cones class

def midlineToLine(midline):
    # TODO: there's a chance slope could be 0. Possible solution: midlineToLine also outputs boolean to indicate y-intercept instead
    # midline:      set of 3d points representing the midline for points; sorted from start to end of sorted track

    # OUTPUT:       the slope and intercept of the line (slope, intercept) which are a 3d vector and a scalar (x-intercept)

    if midline.size == 0 or midline.size == 1:
        return (0,1,0), 0

    # take last two points
    lastPoint = midline[-1]
    secondLP = midline[-2]

    # slope
    slopeVec = lastPoint - secondLP

    if slopeVec[0] == 0:
        return (0,1,0), slopeVec[1]

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

    # find perpendicular line with 'point' in it
    slope = slopeVec[0] / slopeVec[1]
    perpSlope = slopeVec[1] * -1 / slopeVec[0]      # -y/x (x is forward direction)

    perpIntercept = perpSlope * point[1] - point[0] # b = my - x

    # find the pt on the line to compare 'point' to: m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
    x = (perpIntercept - intercept) / (perpSlope - slope)

    y = perpSlope * x + perpIntercept

    # classify y by taking difference between point and (x,y)s' y components
    direction = point[1] - y

    classification = 0

    if direction > 0 and slopeVec[1] > 0 or direction < 0 and slopeVec[1] < 0:
        classification = 1

    return classification, perpSlope, perpIntercept


def get_closest_point_idx(points, curr_point):
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
        c, _, _ = classification(perpSlope, perpIntercept, xy)
        newList.append(c)
    
    return newList




def SVM_update(midline, coloredCones, points):
    # midline:      path-planning given midline     (MUST having ordering like a spline)
    # coloredCones: path-planning given coloredCones (CORRECT coordinates) : cones object
    # points:       all 3d coords of all cones in current frame

    # OUTPUT: colored cones (INCLUDING input)
    #   TODO: only output classifications of new cones!!!

    # Create an SVM class
    SVM = SVM()

    # rid of coloredCones in points using setminus
    npColored = np.array(coloredCones)
    npPoints = np.array(points)             

    points = np.setdiff1d(npPoints, npColored).tolist()     # Note: now points is coords of UNCLASSIFIED cones

    # make a new cones class and pass these colored cones into it
    cones = Cones()
    cones.add_cones(coloredCones)           # function in cones.py

    # find farthest blue and yellow cones (should be closest to last midline point)
    idxBlue = get_closest_point_idx(cones.blue_cones, (0,0))
    idxYellow = get_closest_point_idx(cones.yellow_cones, (0,0))
    
    farBlue = cones.blue_cones[idxBlue]
    farYellow = cones.blue_cones[idxYellow]

    # classify all the points
    while(points.size > 0):
        # extend the line
        slopeVec, intercept = midlineToLine(midline)
        slope = slopeVec[0] / slopeVec[1]

        # choose the points to classify
        point1 = get_closest_point_idx(points, farBlue)
        points.remove(point1)
        point2 = get_closest_point_idx(points, farYellow)
        points.remove(point2)

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
        midline = cones_to_midline(cones)

    # return the cones
    return cones
        
        




