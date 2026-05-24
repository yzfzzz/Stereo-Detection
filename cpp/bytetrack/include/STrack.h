#pragma once

#include "kalmanFilter.h"

#include <opencv2/opencv.hpp>

enum TrackState { NEW = 0, TRACKED, LOST, REMOVED };

class STrack {
  public:
    STrack(std::vector<float> tlwh, float score, int class_id, float distance = 0.0f);
    ~STrack();

    static std::vector<float> tlbrToTlwh(std::vector<float> & tlbr);
    static void               multiPredict(std::vector<STrack *> &     stracks,
                                           byte_kalman::KalmanFilter & kalman_filter);
    void                      staticTlwh();
    void                      staticTlbr();
    std::vector<float>        tlwhToXyah(std::vector<float> tlwh_tmp);
    std::vector<float>        toXyah();
    void                      markLost();
    void                      markRemoved();
    int                       nextId();
    int                       endFrame();

    void activate(byte_kalman::KalmanFilter & kalman_filter, int frame_id);
    void re_activate(STrack & new_track, int frame_id, bool new_id = false);
    void update(STrack & new_track, int frame_id);

  public:
    bool is_activated_;
    int  track_id_;
    int  state_;

    std::vector<float> tlwh_raw_;
    std::vector<float> tlwh_;
    std::vector<float> tlbr_;
    int                frame_id_;
    int                tracklet_len_;
    int                start_frame_;

    KAL_MEAN mean_;
    KAL_COVA covariance_;
    float    score_;
    int      class_id_;
    float    distance_;

  private:
    byte_kalman::KalmanFilter kalman_filter_;
};
