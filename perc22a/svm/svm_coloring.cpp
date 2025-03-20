#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "cones.hpp"
#include "svm_conv.hpp"
#include "svm.hpp"

#define MID_WEIGHT 0.5
#define CONE_WEIGHT 1
#define DERIV2_WEIGHT 1

typedef std::vector<double> Point;
typedef std::vector<std::vector<double>> pointsList;

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

pointsList cones_to_pointsList(const controls::midline::conesList &clist) {
    std::vector<std::vector<double>> plist;
    for (int i = 0; i < clist.size(); ++i) {
        plist.push_back({clist[i].first, clist[i].second, 0});
    }
    return plist;
}

/**
 * @brief Return coord in points closest to currPoint and optionally remove from points
 */
std::vector<double> getClosestPt(std::vector<std::vector<double>> &points, std::vector<double> currPoint, bool removePoint = false) {
    double minDist;
    double sqDist;
    int idx = 0;
    for (int i = 0; i < points.size(); ++i) {
        sqDist = pow(points[i][0] - currPoint[0], 2) + pow(points[i][1] - currPoint[1], 2);
        if (i > 0 && sqDist < minDist) {
            idx = i;
            minDist = sqDist;
        }
    }
    std::vector<double> closestPt = points[idx];
    if (removePoint) {
        points.erase(points.begin()+idx);
    }
    return closestPt;
}

/**
 * @brief Return coord in points closest to currPoint and optionally remove from points
 */
std::vector<double> getClosestPt(const std::vector<std::vector<double>> &points, std::vector<double> currPoint) {
    double minDist;
    double sqDist;
    int idx = 0;
    for (int i = 0; i < points.size(); ++i) {
        sqDist = pow(points[i][0] - currPoint[0], 2) + pow(points[i][1] - currPoint[1], 2);
        if (i > 0 && sqDist < minDist) {
            idx = i;
            minDist = sqDist;
        }
    }
    return points[idx];
}

// /**
//  * @brief Classify cone pair nearest to car.
//  * 
//  * Removes the two closest cone coords to the car from points, classifies them,
//  * adds them to coloredCones, and returns a midline.
//  * 
//  * @param points All cone points.
//  * @param coloredCones Current colored cones.
//  * @param svm Current svm.
//  * 
//  * @return coneslist type - Midline points ordered by distance (like a spline).
//  */
// conesList initClassification(pointsList &points, controls::midline::Cones&coloredCones, svm_model &svm) {
//     // Get the closest two points to origin
//     std::vector<double> origin = {0, 0};
//     std::vector<double> pt1 = getClosestPt(points, origin, true);
//     std::vector<double> pt2 = getClosestPt(points, origin, true);

//     // Classify points as left or right
//     if (pt1[0] < pt2[0]) {
//         coloredCones.addBlueCone(pt1[0], pt1[1], 0);
//         coloredCones.addYellowCone(pt2[0], pt2[1], 0);
//     }
//     else{
//         coloredCones.addBlueCone(pt2[0], pt2[1], 0);
//         coloredCones.addYellowCone(pt1[0], pt2[1], 0);
//     }

//     return controls::midline::cones_to_midline(coloredCones);
// }

// GOOD
bool inCones(std::vector<double> point, const std::vector<std::vector<double>> &conePoints) {
    for (int i = 0; i < conePoints.size(); ++i) {
        if (point[0] == conePoints[i][0] && point[1] == conePoints[i][1]) {
            return true;
        }
    }
    return false;
}

// GOOD
void rmClassifiedCones(pointsList &points, const controls::midline::Cones&cones) {
    for (int i = 0; i < points.size(); ++i) {
        // Doesn't account for orange cones
        if (inCones(points[i], cones.getBlueCones()) || 
            inCones(points[i], cones.getYellowCones())) {
                points.erase(points.begin() + i);
                --i;
        }
    }
}

Slope toSlope(bool headPos, double my, double mx = 1) {
    Slope slope {.isVert = mx==0, .isHoriz = my==0, .headPos = headPos};
    if (mx == 0) {
        return slope;
    }
    slope.slope = my / mx;
    return slope;
}

bool heading(std::vector<double> further, std::vector<double> closer) {
    return further[1] > closer[1] || further[1] == closer[1] && further[0] > closer[0];
}

double slopeToAngle(Slope slope) {
    if (slope.isVert) {
        return slope.headPos ? M_PI_2 : -M_PI_2;
    }
    else {
        double theta = atan(slope.slope);
        if (theta >= 0 && !slope.headPos ||
            theta < 0 && slope.headPos) 
        {
            return theta + M_PI;
        }
        return theta;
    }
}

Slope getAvgSlope(Slope slope1, Slope slope2, double w1 = 1, double w2 = 1) {
    // Both slopes vertical
    if (slope1.isVert && slope2.isVert) {
        return toSlope(slope1.headPos, 1, 0);
    }
    // Average non vertical slope with vertical
    else if (slope1.isVert && !slope2.isVert ||
            !slope1.isVert && slope2.isVert) 
    {
        Slope slantSlope = slope1.isVert ? slope2 : slope1;
        double theta = atan(slantSlope.slope);
        double vert = (theta >= 0) ? M_PI_2 : -M_PI_2;
        return toSlope(slantSlope.headPos, tan((vert+theta)/2));
    }
    // Both not vertical
    else {
        double theta1 = slopeToAngle(slope1);
        double theta2 = slopeToAngle(slope2);
        double thetaAvg = (w1*theta1 + w2*theta2)/(w1 + w2);
        bool headPos = thetaAvg >=0 && thetaAvg < M_PI;
        return toSlope(headPos, tan(thetaAvg));
    }
}

Slope getConeSlope(const controls::midline::Cones&cones) {
    // Get farthest two blue and yellow cones
    size_t sizeb = cones.getBlueCones().size();
    size_t sizey = cones.getYellowCones().size();
    std::vector<double> b1 = {cones.getBlueCones()[sizeb-1][0], cones.getBlueCones()[sizeb-1][1]};
    std::vector<double> b2 = {cones.getBlueCones()[sizeb-2][0], cones.getBlueCones()[sizeb-2][1]};
    std::vector<double> y1 = {cones.getYellowCones()[sizey-1][0], cones.getYellowCones()[sizey-1][1]};
    std::vector<double> y2 = {cones.getYellowCones()[sizey-2][0], cones.getYellowCones()[sizey-2][1]};

    Slope slopeB = toSlope(heading(b1, b2), b1[1] - b2[1], b1[0] - b2[0]);
    Slope slopeY = toSlope(heading(y1, y2), y1[1] - y2[1], y1[0] - y2[0]);
    return getAvgSlope(slopeB, slopeY);
}

Slope applyDeriv2(const std::vector<Slope> &coneSlopes, Slope slope, double weight) {
    size_t sizeCS = coneSlopes.size();
    double thetaS = slopeToAngle(slope);
    double thetaC1 = slopeToAngle(coneSlopes[sizeCS-1]);
    double thetaC2 = slopeToAngle(coneSlopes[sizeCS-2]);
    double ratio = (thetaC1-thetaC2)/thetaC2;
    thetaS += weight*ratio*thetaS;
    bool headPos = thetaS >= 0 && thetaS < M_PI;
    return toSlope(headPos, tan(thetaS));
}

Line midlineToAvgLine(const pointsList midline, const controls::midline::Cones&coloredCones, std::vector<Slope> &coneSlopes) {
    // Midline too short
    if (midline.size() < 2) {
        return Line{.slope = toSlope(true, 1, 0), .intercept = 0};
    }

    size_t sizeMline = midline.size();
    std::vector<double> lastPt1 = midline.back();

    // Find average slope of end of midline
    pointsList lastPoints = {lastPt1}; 
    Slope midSlope;
    // Slope btwn last two points
    if (midline.size() == 2) {
        std::vector<double> lastPt2 = midline[sizeMline-2];
        midSlope = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1[1] - lastPt2[1], 
            lastPt1[0] - lastPt2[0]
        );
    }
    // Slope btwn 1st and 2nd last + 2nd and 3rd last
    else if (midline.size() == 3) {
        std::vector<double> lastPt2 = midline[sizeMline-2];
        std::vector<double> lastPt3 = midline[sizeMline-3];
        Slope slope12 = toSlope(
            heading(lastPt1, lastPt2), 
            lastPt1[1] - lastPt2[1], 
            lastPt1[0] - lastPt2[0]
        );
        Slope slope23 = toSlope(
            heading(lastPt2, lastPt3), 
            lastPt2[1] - lastPt3[1], 
            lastPt2[0] - lastPt3[0]
        );
        midSlope = getAvgSlope(slope12, slope23);
    }
    // Slope btwn 1st and 3rd last + 2nd and 4th last
    else{
        std::vector<double> lastPt2 = midline[sizeMline-2];
        std::vector<double> lastPt3 = midline[sizeMline-3];
        std::vector<double> lastPt4 = midline[sizeMline-4];
        Slope slope13 = toSlope(
            heading(lastPt1, lastPt3), 
            lastPt1[1] - lastPt3[1], 
            lastPt1[0] - lastPt3[0]
        );
        Slope slope24 = toSlope(
            heading(lastPt2, lastPt4), 
            lastPt2[1] - lastPt4[1], 
            lastPt2[0] - lastPt4[0]
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
        return Line{.slope = toSlope(extnSlope.headPos, 1, 0), .intercept=lastPt1[0]};
    }

    double intercept = lastPt1[1] - extnSlope.slope * lastPt1[0]; // y = mx + b --> b = y - mx
    return Line{.slope = extnSlope, .intercept = intercept};
}

void classify(Line midline, std::vector<double> point, controls::midline::Cones &coloredCones) {
    bool isYellow;
    if (midline.slope.isVert) {
        isYellow = ((midline.slope.headPos && point[0] >= midline.intercept) ||
                (!midline.slope.headPos && point[0] < midline.intercept));
    }
    else if (midline.slope.isHoriz) {
        isYellow = ((midline.slope.headPos && point[1] <= midline.intercept) ||
                (!midline.slope.headPos && point[1] > midline.intercept));
    }
    else {
        // Find perpendicular line with point in it
        double perpSlope = -1/midline.slope.slope;
        double perpIntercept = point[1] - perpSlope * point[0]; // b = y - mx

        // Find the point on the line to compare "point" to
        // m1x + b1 = m2x + b2 --> x = (b2 - b1)/(m1 - m2)
        double midLineX = (midline.intercept - perpIntercept)/(perpSlope - midline.slope.slope);
        double direction = point[0] - midLineX;
        isYellow = direction > 0 && midline.slope.headPos || 
                direction < 0 && !midline.slope.headPos;
    }

    if (isYellow) {
        coloredCones.addYellowCone(point[0], point[1], 0);
    }
    else{
        coloredCones.addBlueCone(point[0], point[1], 0);
    }
}

/**
 * @brief Classify all cone points in current frame.
 * 
 * @param coloredCones Previously colored cones.
 * @param points All cone coordinates in the current frame.
 * 
 * @return All controls::midline::Conesin frame classified as blue, yellow, or orange. 
 */
controls::midline::Cones SVM_update(pointsList points, controls::midline::Cones coloredCones) {
    controls::midline::svm_model svm;
    // Add blue and yellow controls::midline::Conesbehind car
    coloredCones.addBlueCone(-2.0, -2.0, 0);
    coloredCones.addYellowCone(2.0, -2.0, 0);

    // Remove classified controls::midline::Conesfrom points list
    rmClassifiedCones(points, coloredCones);

    // Initialize midline
    pointsList midline = cones_to_pointsList(controls::midline::cones_to_midline(coloredCones));

    // Find farthest colored controls::midline::Cones(should be closest to last midline point)
    std::vector<double> farBlue = getClosestPt(coloredCones.getBlueCones(), midline.back());
    std::vector<double> farYellow = getClosestPt(coloredCones.getYellowCones(), midline.back());
    // std::cout << "back midline: " << midline.back << "\n"
    // std::cout << "farBlue: " << farBlue << "\n"
    // std::cout << "farYellow: " << farYellow << "\n"

    // Iteratively classify all points
    std::vector<Slope> coneSlopes;
    Line extnMidline;
    std::vector<double> point1;
    std::vector<double> point2;
    while (points.size() > 0) {
        // Find line extending the end of the midline
        extnMidline = midlineToAvgLine(midline, coloredCones, coneSlopes);

        // Find the two points closest to farBlue and farYellow and classify them
        point1 = getClosestPt(points, farBlue, true);
        point2 = getClosestPt(points, farYellow, true);
        classify(extnMidline, point1, coloredCones);
        classify(extnMidline, point2, coloredCones);

        // Update farthest blue and yellow cones
        farBlue = coloredCones.getBlueCones().back();
        farYellow = coloredCones.getYellowCones().back();

        // Update midline
        midline = cones_to_pointsList(controls::midline::cones_to_midline(coloredCones));
        std::cout << "\nMidline\n";
        for (int i = 0; i < midline.size(); ++i) {
            std::cout << midline[i][0] << "," << midline[i][1] << "\n";
        }
    }
    return coloredCones;
}

int main () {
    controls::midline::Cones coloredCones;
    pointsList points;
    for (double i = 0; i < 10; i++) {
        points.push_back({-2+i, i, 0});
        points.push_back({2+i, i, 0});
        std::cout << -2+i << "," << i << "\n";
        std::cout << 2+i << "," << i << "\n";
    }
    // points = {
    //     {-2.00, -2.00},
    //     {-2.00,  0.00},
    //     {-0.65,  2.22},
    //     {-0.01,  4.44},
    //     {-0.41,  6.67},
    //     {-1.64,  8.89},
    //     {-3.07, 11.11},
    //     {-3.93, 13.33},
    //     {-3.78, 15.56},
    //     {-2.70, 17.78},
    //     { 1.30, 17.78},
    //     { 2.75, 20.00},
    //     {-1.25, 20.00},
    //     { 2.00, -2.00},
    //     { 2.00,  0.00},
    //     { 3.35,  2.22},
    //     { 3.99,  4.44},
    //     { 3.59,  6.67},
    //     { 2.36,  8.89},
    //     { 0.93, 11.11},
    //     { 0.07, 13.33},
    //     { 0.22, 15.56}
    // };
    controls::midline::Cones result = SVM_update(points, coloredCones);
    // std::cout << result.toString();
    std::cout << "Blue Cones: \n";
    for (double i = 0; i < result.getBlueCones().size(); i++) {
        std::cout << result.getBlueCones()[i][0] << "," << result.getBlueCones()[i][1] << "\n";
    }

    std::cout << "Yellow Cones: \n";
    for (double i = 0; i < result.getYellowCones().size(); i++) {
        std::cout << result.getYellowCones()[i][0] << "," << result.getYellowCones()[i][1] << "\n";
    }

    return 0;
}