#include "STrack.h"

STrack::STrack(std::vector<float> tlwh, float score, int class_id, float distance) {
    tlwh_raw_.resize(4);
    tlwh_raw_.assign(tlwh.begin(), tlwh.end());

    is_activated_ = false;
    class_id_     = class_id;
    distance_     = distance;
    track_id_     = 0;
    state_        = TrackState::NEW;

    tlwh_.resize(4);
    tlbr_.resize(4);

    staticTlwh();
    staticTlbr();
    frame_id_     = 0;
    tracklet_len_ = 0;
    score_        = score;
    start_frame_  = 0;
}

STrack::~STrack() {}

void STrack::activate(byte_kalman::KalmanFilter & kalman_filter, int frame_id) {
    kalman_filter_ = kalman_filter;
    track_id_      = nextId();

    std::vector<float> tlwh_tmp(4);
    tlwh_tmp[0]             = tlwh_raw_[0];
    tlwh_tmp[1]             = tlwh_raw_[1];
    tlwh_tmp[2]             = tlwh_raw_[2];
    tlwh_tmp[3]             = tlwh_raw_[3];
    std::vector<float> xyah = tlwhToXyah(tlwh_tmp);
    DETECTBOX          xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc     = kalman_filter_.initiate(xyah_box);
    mean_       = mc.first;
    covariance_ = mc.second;

    staticTlwh();
    staticTlbr();

    tracklet_len_ = 0;
    state_        = TrackState::TRACKED;
    if (frame_id == 1) {
        is_activated_ = true;
    }
    frame_id_    = frame_id;
    start_frame_ = frame_id;
}

void STrack::re_activate(STrack & new_track, int frame_id, bool new_id) {
    std::vector<float> xyah = tlwhToXyah(new_track.tlwh_);
    DETECTBOX          xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc     = kalman_filter_.update(mean_, covariance_, xyah_box);
    mean_       = mc.first;
    covariance_ = mc.second;

    staticTlwh();
    staticTlbr();

    tracklet_len_ = 0;
    state_        = TrackState::TRACKED;
    is_activated_ = true;
    frame_id_     = frame_id;
    score_        = new_track.score_;
    if (new_id) {
        track_id_ = nextId();
    }
}

void STrack::update(STrack & new_track, int frame_id) {
    frame_id_ = frame_id;
    tracklet_len_++;

    std::vector<float> xyah = tlwhToXyah(new_track.tlwh_);
    DETECTBOX          xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];

    auto mc     = kalman_filter_.update(mean_, covariance_, xyah_box);
    mean_       = mc.first;
    covariance_ = mc.second;

    staticTlwh();
    staticTlbr();

    state_        = TrackState::TRACKED;
    is_activated_ = true;

    score_ = new_track.score_;
}

void STrack::staticTlwh() {
    if (state_ == TrackState::NEW) {
        tlwh_[0] = tlwh_raw_[0];
        tlwh_[1] = tlwh_raw_[1];
        tlwh_[2] = tlwh_raw_[2];
        tlwh_[3] = tlwh_raw_[3];
        return;
    }

    tlwh_[0] = mean_[0];
    tlwh_[1] = mean_[1];
    tlwh_[2] = mean_[2];
    tlwh_[3] = mean_[3];

    tlwh_[2] *= tlwh_[3];
    tlwh_[0] -= tlwh_[2] / 2;
    tlwh_[1] -= tlwh_[3] / 2;
}

void STrack::staticTlbr() {
    tlbr_.clear();
    tlbr_.assign(tlwh_.begin(), tlwh_.end());
    tlbr_[2] += tlbr_[0];
    tlbr_[3] += tlbr_[1];
}

std::vector<float> STrack::tlwhToXyah(std::vector<float> tlwh_tmp) {
    std::vector<float> tlwh_output = tlwh_tmp;
    tlwh_output[0] += tlwh_output[2] / 2;
    tlwh_output[1] += tlwh_output[3] / 2;
    tlwh_output[2] /= tlwh_output[3];
    return tlwh_output;
}

std::vector<float> STrack::toXyah() {
    return tlwhToXyah(tlwh_);
}

std::vector<float> STrack::tlbrToTlwh(std::vector<float> & tlbr) {
    tlbr[2] -= tlbr[0];
    tlbr[3] -= tlbr[1];
    return tlbr;
}

void STrack::markLost() {
    state_ = TrackState::LOST;
}

void STrack::markRemoved() {
    state_ = TrackState::REMOVED;
}

int STrack::nextId() {
    static int _count = 0;
    _count++;
    return _count;
}

int STrack::endFrame() {
    return frame_id_;
}

void STrack::multiPredict(std::vector<STrack *> & stracks, byte_kalman::KalmanFilter & kalman_filter) {
    for (int i = 0; i < stracks.size(); i++) {
        if (stracks[i]->state_ != TrackState::TRACKED) {
            stracks[i]->mean_[7] = 0;
        }
        kalman_filter.predict(stracks[i]->mean_, stracks[i]->covariance_);
        stracks[i]->staticTlwh();
        stracks[i]->staticTlbr();
    }
}
