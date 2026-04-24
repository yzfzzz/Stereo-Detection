#ifndef TYPES_H
#define TYPES_H

#include <string>


struct Detection
{
    // x1, y1, x2, y2
    float bbox[4];
    float conf;
    int classId;
};

#endif  // TYPES_H
