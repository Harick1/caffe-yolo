#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    if (this->box_label_) {
      for (int i = 0; i < top.size() - 1; ++i) {
        top[i+1]->ReshapeLike(*(batch->multi_label_[i]));
        caffe_copy(batch->multi_label_[i]->count(), batch->multi_label_[i]->gpu_data(),
            top[i+1]->mutable_gpu_data());
      }
    } else {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(batch->label_);
      // Copy the labels.
      caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
          top[1]->mutable_gpu_data());
    }
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
