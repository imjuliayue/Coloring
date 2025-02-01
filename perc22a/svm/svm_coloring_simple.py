import numpy as np
from svm_dependents import *
from SVM import *
from cones import *
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
    direction = point[0] - x

    classification = 0

    if direction > 0 and slopeVec[1] > 0 or direction < 0 and slopeVec[1] < 0:
        classification = 1

    return classification, (0,perpSlope,0), perpIntercept


def get_closest_point_idx(points, curr_point):
    # TODO: for testing, create "colored cones" so that points isn't empty list
    assert(points.size > 0)
    # gets index of point in points farthest from curr_point and returns the dist
    assert(points.shape[1] == curr_point.shape[0])

    sq_dists = np.sum((points - curr_point) ** 2, axis=1)
    idx = np.argmin(sq_dists)
    return idx, np.sqrt(sq_dists[idx])

def conesBeforeLine(points, perpSlopeVec, perpIntercept):
    # finds all cones closer than farthest cone
    
    # OUTPUT: List of True/False at corresponding indices; True --> is closer.

    newList = []                    # holds True/False values

    for i in range(len(points)):               # classify function kind of does it lol
        print(perpSlopeVec)
        c, _, _ = classify(perpSlopeVec, perpIntercept, points[i])
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


    # classify all the points
    while(len(points) > 0):

        print("======================================")
        print(len(points))
        # extend the line
        slopeVec, intercept = midlineToLine(midline)
        # slope = slopeVec[1] / slopeVec[0]

        
        # print("farBlue:\n" + str(farBlue))
        # print("farYellow:\n" + str(farYellow))

        # choose the points to classify
        idx1,_ = get_closest_point_idx(np.array(points), farBlue)
        # print()
        # print("point1:\n" + str(points[idx1]))
        point1 = points[idx1]
        points.remove(point1)

        idx2,_ = get_closest_point_idx(np.array(points), farYellow)
        # print("point2:\n" + str(points[idx2]))
        point2 = points[idx2]
        points.remove(point2)
        # print("pointsAfterRemove:\n" + str(points))

        # classify them
        # TODO: improve modularity
        class1, pSlope1, pIntercept1 = classify(slopeVec, intercept, point1)
        class2, pSlope2, pIntercept2 = classify(slopeVec, intercept, point2)

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

        print("farthest: (point, slope, intercept\n" + str(farthest))

        # if no more points, we are done!
        if(len(points) == 0):
            return cones
        
        # Find all cones within distance from line
        # nearCones = np.array(conesBeforeLine(points, farthest[1], farthest[2]))

        # print(nearCones)

        # # print("nearcones:\n" + str(nearCones))

        # npPoints = np.array(points)

        # nearPoints = npPoints[nearCones]                                                      # filter out False cones
        # npPoints = np.array([row for row in npPoints if not any(np.array_equal(row, r) for r in nearPoints)])

        # points = npPoints.tolist()

        # points = np.setdiff1d(np.array(points), nearPoints).tolist()              # TODO: really please optimize this
        # print("fdsklafjlkdsajfkldsajflksajflkdajsfsad\n" + str(points))

        # classify all these near points
        # for xy in nearPoints:
        #     resXY, _, _ = classify(slopeVec, intercept, xy)

        #     (xy0, xy1, xy2) = xy
            
        #     if resXY:
        #         cones.add_yellow_cone(xy0, xy1, xy2)
        #     else:
        #         cones.add_blue_cone(xy0, xy1, xy2)

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


        farBlue = np.array(farBlue)
        farYellow = np.array(farYellow)

        print("farBlue: " + str(farBlue))
        print("farYellow: " + str(farYellow))
        # plug into SVM
        midline = svm.cones_to_midline(cones)
        print("cones:\n" + str(cones))

        # print(midline)

    # return the cones
    return cones
        
        




