#pragma once
#include "config.h"
#include "STrack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

enum MotionState {
    INVAILD   = 0,
    UNKNOWN   = 1,
    STABLE    = 2,
    APPROACH  = 3,
    MOVE_AWAY = 4,
    ACCELE    = 5,
    DECELE    = 6,
    CONSTANT  = 7
};

const std::map<std::pair<MotionState, MotionState>, std::string> MOTION_STR_MAP = {
    { { STABLE, CONSTANT },    "Stable"               },
    { { APPROACH, ACCELE },    "Approach (Accele)"    },
    { { APPROACH, DECELE },    "Approach (Decele)"    },
    { { APPROACH, CONSTANT },  "Approach (Constant)"  },
    { { MOVE_AWAY, ACCELE },   "Move Away (Accele)"   },
    { { MOVE_AWAY, DECELE },   "Move Away (Decele)"   },
    { { MOVE_AWAY, CONSTANT }, "Move Away (Constant)" },
};

class MotionStateEngine {
  public:
    MotionStateEngine(int    sma_window_size      = 5,
                      float  velocity_threshold   = 5.0f,
                      float  acceleration_threshold = 1.5f,
                      float  jump_threshold       = 0.3f,
                      int    consistency_frames   = 3);

    std::pair<MotionState, MotionState> computeMotionState(int track_id, float raw_depth, double timestamp);

    float getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size);

    float computeMeanDepth(cv::Mat depth, const std::vector<float> & tlwh, int num_samples = 25) const;

  private:
    struct ObjectState {
        float  smoothed_depth;
        double timestamp;
        float  velocity;
    };

    bool isOutlier(int track_id, float raw_depth) const;

    float computeMedian(const std::deque<float> & queue) const;

    bool checkConsistency(int track_id, MotionState direction) const;

    std::unordered_map<int, std::list<ObjectState>>   history_states_;
    std::unordered_map<int, std::deque<float>>         depth_history_queues_;
    std::unordered_map<int, std::deque<MotionState>>   direction_history_;
    std::unordered_map<int, int>                        outlier_count_;

    int   sma_window_size_;
    float velocity_threshold_;
    float acceleration_threshold_;
    float jump_threshold_;
    int   consistency_frames_;
};
