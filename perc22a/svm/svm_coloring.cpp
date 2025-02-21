#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

#define MID_WEIGHT .5
#define CONE_WEIGHT 1
#define DERIV2_WEIGHT 1

struct Point {
    float x;
    float y;
};

// class Cones {
//     public:
//         std::vector<Point> blueCones;
//         std::vector<Point> yellowCones;
//         std::vector<Point> orangeCones;
//         int len() {
//             return blueCones.size() + yellowCones.size() + orangeCones.size();
//         }
//         void addBlueCone(float x, float y) {
//             blueCones.push_back({x, y});
//         }
//         void addYellowCone(float x, float y) {
//             yellowCones.push_back({x, y});
//         }
//         void addOrangeCones(float x, float y) {
//             orangeCones.push_back({x, y});
//         }
//         std::string conesToString(const std::vector<Point> &cones) {
//             std::stringstream ss;
//             ss << std::fixed << std::setprecision(2);
//             for (int i = 0; i < cones.size(); ++i) {
//                 ss << "  " << i << ": (" << cones[i].x << ", " << cones[i].y << ")\n";
//             }
//             return ss.str();
//         }
//         std::string toString() {
//             std::stringstream ss;
//             ss << std::fixed << std::setprecision(2);
//             ss << "-------Cones--------\n";
//             ss << "Blue (" << blueCones.size() << " cones)\n";
//             ss << conesToString(blueCones);
//             ss << "Yellow (" << yellowCones.size() << " cones)\n";
//             ss << conesToString(yellowCones);
//             ss << "Orange (" << orangeCones.size() << " cones)\n";
//             ss << conesToString(orangeCones);
//             return ss.str();
//         }
// };

struct Cones {
    std::vector<Point> blueCones;
    std::vector<Point> yellowCones;
    std::vector<Point> orangeCones;
};

struct Slope {
    bool isVert = false;
    bool isHoriz = false;
    bool headPos = true;
    float slope;
};

struct Line {
    Slope slope;
    float intercept;
};

class SVM {
    public:
        std::vector<Point> conesToMidline(Cones cones) {
            std::vector<Point> midline;
            return midline;
        }
};

/**
 * @brief Return coord in points closest to currPoint and optionally remove from points
 */
Point getClosestPt(std::vector<Point> &points, Point currPoint, bool removePoint = false) {
    float minDist;
    float sqDist;
    int idx = 0;
    for (int i = 0; i < points.size(); ++i) {
        sqDist = pow(points[i].x - currPoint.x, 2) + pow(points[i].y - currPoint.y, 2);
        if (i > 0 && sqDist < minDist) {
            idx = i;
            minDist = sqDist;
        }
    }
    Point closestPt = points[idx];
    if (removePoint) {
        points.erase(points.begin()+idx);
    }
    return closestPt;
}

/**
 * @brief Classify cone pair nearest to car.
 * 
 * Removes the two closest cone coords to the car from points, classifies them,
 *  adds them to coloredCones, and returns a midline.
 * 
 * @param points All cone points.
 * @param coloredCones Current colored cones.
 * @param svm Current svm.
 * 
 * @return Midline points ordered by distance (like a spline).
 */
std::vector<Point> initClassification(std::vector<Point> &points, Cones &coloredCones, SVM &svm) {
    // Get the closest two points to origin
    Point origin = {.x = 0, .y = 0};
    Point pt1 = getClosestPt(points, origin, true);
    Point pt2 = getClosestPt(points, origin, true);

    // Classify points as left or right
    if (pt1.x < pt2.x) {
        coloredCones.blueCones.push_back(pt1);
        coloredCones.yellowCones.push_back(pt2);
    }
    else{
        coloredCones.blueCones.push_back(pt2);
        coloredCones.yellowCones.push_back(pt1);
    }

    return svm.conesToMidline(coloredCones);
}

bool inCones(Point point, const std::vector<Point> &conePoints) {
    for (int i = 0; i < conePoints.size(); ++i) {
        if (point.x == conePoints[i].x && point.y == conePoints[i].y) {
            return true;
        }
    }
    return false;
}

void rmClassifiedCones(std::vector<Point> &points, const Cones &cones) {
    for (int i = 0; i < points.size(); ++i) {
        if (inCones(points[i], cones.blueCones) || inCones(points[i], cones.yellowCones)
            || inCones(points[i], cones.orangeCones)) {
                points.erase(points.begin() + i);
                --i;
        }
    }
}

Slope toSlope(bool headPos, float my, float mx = 1) {
    Slope slope = {.headPos = headPos, .isVert = mx==0, .isHoriz = my==0};
    if (mx == 0) return;
    slope.slope = my / mx;
    return slope;
}

bool heading(Point further, Point closer) {
    return further.y > closer.y || further.y == closer.y && further.x > closer.x;
}

float slopeToAngle(Slope slope) {
    if (slope.isVert) {
        return slope.headPos ? M_PI_2 : -M_PI_2;
    }
    else {
        float theta = atan(slope.slope);
        if (theta >= 0 && !slope.headPos ||
            theta < 0 && slope.headPos) 
        {
            return theta + M_PI;
        }
        return theta;
    }
}

Slope getAvgSlope(Slope slope1, Slope slope2, int w1 = 1, int w2 = 1) {
    // Both slopes vertical
    if (slope1.isVert && slope2.isVert) {
        return toSlope(slope1.headPos, 1, 0);
    }
    // Average non vertical slope with vertical
    else if (slope1.isVert && !slope2.isVert ||
            !slope1.isVert && slope2.isVert) 
    {
        Slope slantSlope = slope1.isVert ? slope2 : slope1;
        float theta = atan(slantSlope.slope);
        float vert = (theta >= 0) ? M_PI_2 : -M_PI_2;
        return toSlope(slantSlope.headPos, tan((vert+theta)/2));
    }
    // Both not vertical
    else {
        float theta1 = slopeToAngle(slope1);
        float theta2 = slopeToAngle(slope2);
        float thetaAvg = (w1*theta1 + w2*theta2)/(w1 + w2);
        bool headPos = thetaAvg >=0 && thetaAvg < M_PI;
        return toSlope(headPos, tan(thetaAvg));
    }
}

Slope getConeSlope(const Cones &cones) {
    // Get farthest two blue and yellow cones
    size_t sizeb = cones.blueCones.size();
    size_t sizey = cones.yellowCones.size();
    Point b1 = cones.blueCones[sizeb-1];
    Point b2 = cones.blueCones[sizeb-2];
    Point y1 = cones.yellowCones[sizey-1];
    Point y2 = cones.yellowCones[sizey-2];

    Slope slopeB = toSlope(heading(b1, b2), b1.y - b2.y, b1.x - b2.x);
    Slope slopeY = toSlope(heading(y1, y2), y1.y - y2.y, y1.x - y2.x);
    return getAvgSlope(slopeB, slopeY);
}

Slope applyDeriv2(const std::vector<Slope> &coneSlopes, Slope slope, float weight) {
    size_t sizeCS = coneSlopes.size();
    float thetaS = slopeToAngle(slope);
    float thetaC1 = slopeToAngle(coneSlopes[sizeCS-1]);
    float thetaC2 = slopeToAngle(coneSlopes[sizeCS-2]);
    float ratio = (thetaC1-thetaC2)/thetaC2;
    thetaS += weight*ratio*thetaS;
    bool headPos = thetaS >= 0 && thetaS < M_PI;
    return toSlope(headPos, tan(thetaS));
}

Line midlineToAvgLine(const std::vector<Point> midline, const Cones &coloredCones, std::vector<Slope> &coneSlopes) {
    // Midline too short
    if (midline.size() < 2) {
        return Line{.slope = toSlope(true, 1, 0), .intercept = 0};
    }

    size_t sizeMline = midline.size();
    Point lastPt1 = midline.back();

    // Find average slope of end of midline
    std::vector<Point> lastPoints = {lastPt1}; 
    Slope midSlope;
    // Slope btwn last two points
    if (midline.size() == 2) {
        Point lastPt2 = midline[sizeMline-2];
        midSlope = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1.y - lastPt2.y, 
            lastPt1.x - lastPt2.x
        );
    }
    // Slope btwn 1st and 2nd last + 2nd and 3rd last
    else if (midline.size() == 3) {
        Point lastPt2 = midline[sizeMline-2];
        Point lastPt3 = midline[sizeMline-3];
        Slope slope12 = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1.y - lastPt2.y, 
            lastPt1.x - lastPt2.x
        );
        Slope slope23 = toSlope(
            heading(lastPt2, lastPt3), 
            lastPt2.y - lastPt3.y, 
            lastPt2.x - lastPt3.x
        );
        midSlope = getAvgSlope(slope12, slope23);
    }
    // Slope btwn 1st and 3rd last + 2nd and 4th last
    else{
        Point lastPt2 = midline[sizeMline-2];
        Point lastPt3 = midline[sizeMline-3];
        Point lastPt4 = midline[sizeMline-4];
        Slope slope13 = toSlope(
            heading(lastPt1, lastPt3), 
            lastPt1.y - lastPt3.y, 
            lastPt1.x - lastPt3.x
        );
        Slope slope24 = toSlope(
            heading(lastPt2, lastPt4), 
            lastPt2.y - lastPt4.y, 
            lastPt2.x - lastPt4.x
        );
        midSlope = getAvgSlope(slope13, slope24);
    }

    // Average midline slope with cone slope
    Slope extnSlope;
    if (coloredCones.blueCones.size() >= 2 
        && coloredCones.yellowCones.size() >=2)
    {
        Slope coneSlope = getConeSlope(coloredCones);
        coneSlopes.push_back(coneSlope);
        extnSlope = getAvgSlope(midSlope, coneSlope, MID_WEIGHT, CONE_WEIGHT);
        if (coneSlopes.size() >= 2) {
            extnSlope = applyDeriv2(coneSlopes, extnSlope, DERIV2_WEIGHT);
        }
    }
    else {
        extnSlope = midSlope;
    }

    // Vertical extended line
    if (extnSlope.isVert) {
        return Line{.slope = toSlope(extnSlope.headPos, 1, 0), .intercept=lastPt1.x};
    }

    float intercept = lastPt1.y - extnSlope.slope * lastPt1.x; // y = mx + b --> b = y - mx
    return Line{.slope = extnSlope, .intercept = intercept};
}

void classify(Line midline, Point point, Cones &coloredCones) {
    bool isYellow;
    if (midline.slope.isVert) {
        isYellow = ((midline.slope.headPos && point.x >= midline.intercept) ||
                   (!midline.slope.headPos && point.x < midline.intercept));
    }
    else if (midline.slope.isHoriz) {
        isYellow = ((midline.slope.headPos && point.y <= midline.intercept) ||
                   (!midline.slope.headPos && point.y > midline.intercept));
    }
    else {
        // Find perpendicular line with point in it
        float perpSlope = -1/midline.slope.slope;
        float perpIntercept = point.y - perpSlope * point.x; // b = y - mx

        // Find the point on the line to compare "point" to
        // m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
        float midLineX = (midline.intercept - perpIntercept)/(perpSlope - midline.slope.slope);
        float direction = point.x - midLineX;
        isYellow = direction > 0 && midline.slope.headPos || 
                   direction < 0 && !midline.slope.headPos;
    }

    if (isYellow) {
        coloredCones.yellowCones.push_back(point);
    }
    else{
        coloredCones.blueCones.push_back(point);
    }
}

/**
 * @brief Classify all cone points in current frame.
 * 
 * @param coloredCones Previously colored cones.
 * @param points All cone coordinates in the current frame.
 * 
 * @return All cones in frame classified as blue, yellow, or orange. 
 */
Cones SVM_update(std::vector<Point> points, Cones coloredCones) {
    SVM svm;
    // Add blue and yellow cones behind car
    coloredCones.blueCones.push_back({-2.0f, -2.0f});
    coloredCones.yellowCones.push_back({2.0f, -2.0f});

    // Remove classified cones from points list
    rmClassifiedCones(points, coloredCones);

    // Initialize midline
    std::vector<Point> midline = svm.conesToMidline(coloredCones);

    // Find farthest colored cones (should be closest to last midline point)
    Point farBlue = getClosestPt(coloredCones.blueCones, midline.back());
    Point farYellow = getClosestPt(coloredCones.yellowCones, midline.back());

    // Iteratively classify all points
    std::vector<Slope> coneSlopes;
    Line extnMidline;
    Point point1;
    Point point2;
    while (points.size() > 0) {
        // Find line extending the end of the midline
        extnMidline = midlineToAvgLine(midline, coloredCones, coneSlopes);

        // Find the two points closest to farBlue and farYellow and classify them
        point1 = getClosestPt(points, farBlue, true);
        point2 = getClosestPt(points, farYellow, true);
        classify(extnMidline, point1, coloredCones);
        classify(extnMidline, point2, coloredCones);

        // Update farthest blue and yellow cones
        farBlue = coloredCones.blueCones.back();
        farYellow = coloredCones.yellowCones.back();

        // Update midline
        midline = svm.conesToMidline(coloredCones);
    }
    return coloredCones;
}