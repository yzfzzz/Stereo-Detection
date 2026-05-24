#ifndef TYPES_H
#define TYPES_H

#include <array>

struct Detection {
    // x1, y1, x2, y2
    std::array<float, 4> bbox;
    float                conf;
    int                  classId;
};

#endif  // TYPES_H
