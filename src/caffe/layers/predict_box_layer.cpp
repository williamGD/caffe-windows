#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/predict_box_layer.hpp"

namespace caffe {

template <typename Dtype>
void PredictBoxLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PredictBoxParameter predict_box_param = this->layer_param_.predict_box_param();
  stride_ = predict_box_param.stride();
  receptive_field_ = predict_box_param.receptive_field();
  nms_ = predict_box_param.nms();
  output_vector_ = predict_box_param.output_vector();
  positive_thresh_ = predict_box_param.positive_thresh();
}

template <typename Dtype>
void PredictBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), 4);
  CHECK_EQ(bottom[1]->channels(), 2);
  if (output_vector_) {
    CHECK_EQ(top.size(), 2);
  }

  top[0]->Reshape({ bottom[0]->num(), 5, bottom[0]->height(), bottom[0]->width() });
  if (output_vector_) {
    top[1]->Reshape({ bottom[0]->num(), 1, 5 });//will be modified on the fly.
  }
}

template <typename Dtype>
void PredictBoxLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bbr_data = bottom[0]->cpu_data();
  const Dtype* score_data = bottom[1]->cpu_data();
  Dtype* bb_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int output_height = bottom[0]->height();
  int output_width = bottom[1]->width();
  int count = 0;

  for (int n = 0; n < num; n++) {
    for (int x = 0; x < output_width; x ++) {
      for (int y = 0; y < output_height; y ++) {
        if (score_data[bottom[1]->offset(n, 1, y, x)] > positive_thresh_) {
          bb_data[top[0]->offset(n, 0, y, x)] = x * stride_ + bbr_data[bottom[0]->offset(n, 0, y, x)] * receptive_field_;
          bb_data[top[0]->offset(n, 1, y, x)] = y * stride_ + bbr_data[bottom[0]->offset(n, 1, y, x)] * receptive_field_;
          bb_data[top[0]->offset(n, 2, y, x)] = receptive_field_ * exp(bbr_data[bottom[0]->offset(n, 2, y, x)]);
          bb_data[top[0]->offset(n, 3, y, x)] = receptive_field_ * exp(bbr_data[bottom[0]->offset(n, 3, y, x)]);
          bb_data[top[0]->offset(n, 4, y, x)] = score_data[bottom[1]->offset(n, 1, y, x)];
          count++;
        }
      }
    }
  }

  if (output_vector_) {
    if (num == 1 && count > 0) {
      top[1]->Reshape({ bottom[0]->num(), count, 5 });
      int i = 0;
      for (int x = 0; x < output_width; x++) {
        for (int y = 0; y < output_height; y++) {
          if (score_data[bottom[1]->offset(0, 1, y, x)] > positive_thresh_) {
            top[1]->mutable_cpu_data()[i * 5] = bb_data[top[0]->offset(0, 0, y, x)];
            top[1]->mutable_cpu_data()[i * 5 + 1] = bb_data[top[0]->offset(0, 1, y, x)];
            top[1]->mutable_cpu_data()[i * 5 + 2] = bb_data[top[0]->offset(0, 2, y, x)];
            top[1]->mutable_cpu_data()[i * 5 + 3] = bb_data[top[0]->offset(0, 3, y, x)];
            top[1]->mutable_cpu_data()[i * 5 + 4] = score_data[bottom[1]->offset(0, 1, y, x)];
            i++;
          }
        }
      }
    }
    else {
      caffe_set<Dtype>(top[1]->count(), 0, top[1]->mutable_cpu_data());
    }
  }
}

INSTANTIATE_CLASS(PredictBoxLayer);
REGISTER_LAYER_CLASS(PredictBox);

}  // namespace caffe
