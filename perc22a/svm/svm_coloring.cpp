#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "cones.h"
#include "svm_conv.h"

#define MID_WEIGHT 0.5
#define CONE_WEIGHT 1
#define DERIV2_WEIGHT 1

// struct Point {
//     double x;
//     double y;
// };
typedef std::pair<double, double> Point;

struct Slope {
    bool isVert {false};
    bool isHoriz {false};
    bool headPos {true};
    double slope;
};

struct Line {
    Slope slope;
    double intercept;
};

class SVM {
    public:
    conesList conesToMidline(Cones &cones) {
            conesList midline = {
                { 0.1, -3.6},
                { 0.1, -2.1},
                { 0.1, -3.1},
                { 0.1, -2.6},
                { 0.1, -2.1},
                { 0.1, -1.6},
                { 0.1, -1.1},
                { 0.1, -0.6},
                { 0.1, -0.1},
                { 0.1,  0.4},
                { 0.1,  0.9},
                { 0.1,  1.4},
                { 0.1,  1.9}
            };
            return midline;
        }
};

/**
 * @brief Return coord in points closest to currPoint and optionally remove from points
 */
Point getClosestPt(conesList &points, Point currPoint, bool removePoint = false) {
    float minDist;
    float sqDist;
    int idx = 0;
    for (int i = 0; i < points.size(); ++i) {
        sqDist = pow(points[i].first - currPoint.first, 2) + pow(points[i].second - currPoint.second, 2);
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
 * adds them to coloredCones, and returns a midline.
 * 
 * @param points All cone points.
 * @param coloredCones Current colored cones.
 * @param svm Current svm.
 * 
 * @return Midline points ordered by distance (like a spline).
 */
conesList initClassification(conesList &points, Cones &coloredCones, SVM &svm) {
    // Get the closest two points to origin
    Point origin = {0, 0};
    Point pt1 = getClosestPt(points, origin, true);
    Point pt2 = getClosestPt(points, origin, true);

    // Classify points as left or right
    if (pt1.first < pt2.first) {
        coloredCones.addBlueCone(pt1.first, pt1.second, 0);
        coloredCones.addYellowCone(pt2.first, pt2.second, 0);
    }
    else{
        coloredCones.addBlueCone(pt2.first, pt2.second, 0);
        coloredCones.addYellowCone(pt1.first, pt2.second, 0);
    }

    return svm.conesToMidline(coloredCones);
}

bool inCones(Point point, const std::vector<std::vector<double>> &conePoints) {
    for (int i = 0; i < conePoints.size(); ++i) {
        if (point.first == conePoints[i][0] && point.second == conePoints[i][1]) {
            return true;
        }
    }
    return false;
}

void rmClassifiedCones(conesList &points, const Cones &cones) {
    for (int i = 0; i < points.size(); ++i) {
        // Doesn't account for orange cones
        if (inCones(points[i], cones.getBlueCones()) || 
            inCones(points[i], cones.getYellowCones())) {
                points.erase(points.begin() + i);
                --i;
        }
    }
}

Slope toSlope(bool headPos, float my, float mx = 1) {
    Slope slope {.headPos = headPos, .isVert = mx==0, .isHoriz = my==0};
    if (mx == 0) {
        return slope;
    }
    slope.slope = my / mx;
    return slope;
}

bool heading(Point further, Point closer) {
    return further.second > closer.second || further.second == closer.second && further.first > closer.first;
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

Slope getAvgSlope(Slope slope1, Slope slope2, float w1 = 1, float w2 = 1) {
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
    size_t sizeb = cones.getBlueCones().size();
    size_t sizey = cones.getYellowCones().size();
    Point b1 = {cones.getBlueCones()[sizeb-1][0], cones.getBlueCones()[sizeb-1][1]};
    Point b2 = {cones.getBlueCones()[sizeb-2][0], cones.getBlueCones()[sizeb-2][1]};
    Point y1 = {cones.getYellowCones()[sizey-1][0], cones.getYellowCones()[sizey-1][1]};
    Point y2 = {cones.getYellowCones()[sizey-2][0], cones.getYellowCones()[sizey-2][1]};

    Slope slopeB = toSlope(heading(b1, b2), b1.second - b2.second, b1.first - b2.first);
    Slope slopeY = toSlope(heading(y1, y2), y1.second - y2.second, y1.first - y2.first);
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

Line midlineToAvgLine(const conesList midline, const Cones &coloredCones, std::vector<Slope> &coneSlopes) {
    // Midline too short
    if (midline.size() < 2) {
        return Line{.slope = toSlope(true, 1, 0), .intercept = 0};
    }

    size_t sizeMline = midline.size();
    Point lastPt1 = midline.back();

    // Find average slope of end of midline
    conesList lastPoints = {lastPt1}; 
    Slope midSlope;
    // Slope btwn last two points
    if (midline.size() == 2) {
        Point lastPt2 = midline[sizeMline-2];
        midSlope = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1.second - lastPt2.second, 
            lastPt1.first - lastPt2.first
        );
    }
    // Slope btwn 1st and 2nd last + 2nd and 3rd last
    else if (midline.size() == 3) {
        Point lastPt2 = midline[sizeMline-2];
        Point lastPt3 = midline[sizeMline-3];
        Slope slope12 = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1.second - lastPt2.second, 
            lastPt1.first - lastPt2.first
        );
        Slope slope23 = toSlope(
            heading(lastPt2, lastPt3), 
            lastPt2.second - lastPt3.second, 
            lastPt2.first - lastPt3.first
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
            lastPt1.second - lastPt3.second, 
            lastPt1.first - lastPt3.first
        );
        Slope slope24 = toSlope(
            heading(lastPt2, lastPt4), 
            lastPt2.second - lastPt4.second, 
            lastPt2.first - lastPt4.first
        );
        midSlope = getAvgSlope(slope13, slope24);
    }

    // Average midline slope with cone slope
    Slope extnSlope;
    if (coloredCones.size() >= 2 
        && coloredCones.getYellowCones().size() >=2)
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
        return Line{.slope = toSlope(extnSlope.headPos, 1, 0), .intercept=lastPt1.first};
    }

    float intercept = lastPt1.second - extnSlope.slope * lastPt1.first; // y = mx + b --> b = y - mx
    return Line{.slope = extnSlope, .intercept = intercept};
}

void classify(Line midline, Point point, Cones &coloredCones) {
    bool isYellow;
    if (midline.slope.isVert) {
        isYellow = ((midline.slope.headPos && point.first >= midline.intercept) ||
                   (!midline.slope.headPos && point.first < midline.intercept));
    }
    else if (midline.slope.isHoriz) {
        isYellow = ((midline.slope.headPos && point.second <= midline.intercept) ||
                   (!midline.slope.headPos && point.second > midline.intercept));
    }
    else {
        // Find perpendicular line with point in it
        float perpSlope = -1/midline.slope.slope;
        float perpIntercept = point.second - perpSlope * point.first; // b = y - mx

        // Find the point on the line to compare "point" to
        // m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
        float midLineX = (midline.intercept - perpIntercept)/(perpSlope - midline.slope.slope);
        float direction = point.first - midLineX;
        isYellow = direction > 0 && midline.slope.headPos || 
                   direction < 0 && !midline.slope.headPos;
    }

    if (isYellow) {
        coloredCones.addYellowCone(point.first, point.second, 0);
    }
    else{
        coloredCones.addBlueCone(point.first, point.second, 0);
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
Cones SVM_update(conesList points, Cones coloredCones) {
    SVM svm;
    // Add blue and yellow cones behind car
    coloredCones.addBlueCone(-2.0, -2.0, 0);
    coloredCones.addYellowCone(2.0, -2.0, 0);

    // Remove classified cones from points list
    rmClassifiedCones(points, coloredCones);

    // Initialize midline
    conesList midline = svm.conesToMidline(coloredCones);

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

int main () {
    Cones coloredCones;
    conesList points = {
        {-2, 0}, 
        {2, 0}, 
        {-6.5, 2.2},
        {3.35, 2.2},
        {-8, 4.4},
        {4, 4.4},
        {-4, 6.7},
        {3.6, 6.7},
        {-1.6, 8.9},
        {2.4, 8.9},
        {-3, 1.1},
        { 9.3, 1.1},
        {-4, 1.3},
        { 7.1, 1.3},
        {-3.8, 1.5},
        { 2.2, 1.5},
        {-2.7, 1.8},
        { 1.3, 1.8},
        {-1.3, 2},
        {2.7, 2}};
    Cones result = SVM_update(points, coloredCones);
    std::cout << result.toString();
    return 0;
}