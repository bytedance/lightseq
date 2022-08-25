#include "util.h"

namespace lightseq {
namespace cuda {

template <typename T>
void print_vec(const thrust::device_vector<T>& outv, std::string outn,
               int num_output_ele) {
  std::cout << outn << ": ";
  if (num_output_ele > 0) {
    num_output_ele = std::min(size_t(num_output_ele), outv.size());
    thrust::copy(outv.begin(), outv.begin() + num_output_ele,
                 std::ostream_iterator<T>(std::cout, " "));
    std::cout << " ...";
  } else {
    thrust::copy(outv.begin(), outv.end(),
                 std::ostream_iterator<T>(std::cout, " "));
  }
  std::cout << std::endl;
}

template void print_vec<float>(const thrust::device_vector<float>& outv,
                               std::string outn, int num_output_ele);

template void print_vec<int>(const thrust::device_vector<int>& outv,
                             std::string outn, int num_output_ele);

template void print_vec<int8_t>(const thrust::device_vector<int8_t>& outv,
                                std::string outn, int num_output_ele);

template <typename T>
void print_vec(thrust::device_ptr<T> outv, std::string outn,
               int num_output_ele) {
  std::cout << outn << ": ";
  thrust::copy(outv, outv + num_output_ele,
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template void print_vec<float>(thrust::device_ptr<float> outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(thrust::device_ptr<int> outv, std::string outn,
                             int num_output_ele);

template void print_vec<int8_t>(thrust::device_ptr<int8_t> outv,
                                std::string outn, int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int num_output_ele) {
  std::cout << outn << ": ";
  std::vector<T> hout(num_output_ele, (T)0);
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(T),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << hout[i] << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half* outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << ": ";
  std::vector<__half> hout(num_output_ele, (__half)0.f);
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(__half),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<int8_t>(const int8_t* outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << ": ";
  std::vector<int8_t> hout(num_output_ele, (int8_t)0);
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(int8_t),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << static_cast<int>(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float* outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(const int* outv, std::string outn,
                             int num_output_ele);

template void print_vec<int8_t>(const int8_t* outv, std::string outn,
                                int num_output_ele);

template void print_vec<__half>(const __half* outv, std::string outn,
                                int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int start, int end) {
  std::cout << outn << ": ";
  thrust::copy(thrust::device_pointer_cast(outv + start),
               thrust::device_pointer_cast(outv + end),
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half* outv, std::string outn, int start,
                       int end) {
  std::cout << outn << ": ";
  int num_elements = end - start;
  std::vector<__half> hout(num_elements, (__half)0.f);
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv + start,
                             num_elements * sizeof(__half),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_elements; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float* outv, std::string outn, int start,
                               int end);

template void print_vec<int>(const int* outv, std::string outn, int start,
                             int end);
void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name, cudaStream_t stream) {
  CHECK_GPU_ERROR(cudaStreamSynchronize(stream));
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << duration_name
            << " duration time is: " << (elapsed).count() * 1000 << " ms"
            << std::endl;
  return;
}

struct prg_uniform {
  float a, b;

  __host__ __device__ prg_uniform(float _a = 0.f, float _b = 1.f)
      : a(_a), b(_b){};

  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

struct prg_norm {
  float a, b;

  __host__ __device__ prg_norm(float _a = 0.f, float _b = 1.f) : a(_a), b(_b){};

  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::random::normal_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

void generate_distribution(thrust::device_vector<float>& input_output,
                           std::string mode, float a, float b) {
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);
  if (mode == "uniform")
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + input_output.size(),
                      input_output.begin(), prg_uniform(a, b));
  if (mode == "norm")
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + input_output.size(),
                      input_output.begin(), prg_norm(a, b));
}

void read_batch_tokenids_from_file(std::string file_name, int& batch_size,
                                   int& batch_seq_len,
                                   std::vector<int>& input_ids) {
  std::ifstream fin(file_name);
  fin >> batch_size >> batch_seq_len;
  input_ids = std::vector<int>(batch_size * batch_seq_len, 0);
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < batch_seq_len; j++) {
      int idx = i * batch_seq_len + j;
      fin >> input_ids[idx];
    }
  }
}

bool endswith(std::string const& full, std::string const& end) {
  if (full.length() >= end.length()) {
    return (0 == full.compare(full.length() - end.length(), end.length(), end));
  }
  return false;
}

int get_hdf5_dataset_size(hid_t dataset) {
  hid_t dataspace = H5Dget_space(dataset); /* dataspace handle */
  int n_dims = H5Sget_simple_extent_ndims(dataspace);
  // return 1 for scalar
  if (n_dims < 1) {
    return 1;
  }
  // get dimensions for N-Dimension vector
  hsize_t dims[n_dims];
  int status = H5Sget_simple_extent_dims(dataspace, dims, NULL);
  if (status != n_dims || status < 0) {
    // return negative number on error
    return -1;
  }
  // accumulate size from every dimension
  int vec_size = 1;
  for (int i = 0; i < n_dims; ++i) {
    vec_size *= dims[i];
  }
  return vec_size;
}

int get_hdf5_dataset_size(hid_t hdf5_file, std::string dataset_name) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  // parse dataset size
  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);
  if (ds_size < 0) {
    throw std::runtime_error("HDF5 parsing error: " + dataset_name);
  }
  H5Dclose(ds);
  return ds_size;
}

int read_hdf5_dataset_data(hid_t hdf5_file, std::string dataset_name,
                           hid_t output_type, void* output_buf,
                           std::function<bool(int)> size_predicate,
                           std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  herr_t status =
      H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, output_buf);

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return ds_size;
}

std::vector<float> read_hdf5_dataset_data_float(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate, std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  std::vector<float> output_vec(ds_size);
  herr_t status = H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          output_vec.data());

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return output_vec;  // return with copy elision
}

std::vector<int> read_hdf5_dataset_data_int(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate, std::string extra_msg) {
  // check if dataset exists or not
  if (!H5Lexists(hdf5_file, dataset_name.c_str(), H5P_DEFAULT)) {
    throw HDF5DatasetNotFoundError(
        (dataset_name + " Not Found in HDF5 File").c_str());
  }

  hid_t ds = H5Dopen2(hdf5_file, dataset_name.c_str(), H5P_DEFAULT);
  if (ds < 0) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + dataset_name);
  }
  int ds_size = get_hdf5_dataset_size(ds);

  // sanity (custom) check for size with extra message.
  if (size_predicate(ds_size)) {
    throw std::runtime_error("Invalid shape " + std::to_string(ds_size) + ". " +
                             extra_msg);
  }

  std::vector<int> output_vec(ds_size);
  herr_t status = H5Dread(ds, output_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                          output_vec.data());

  if (status < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset: " + dataset_name);
  }
  H5Dclose(ds);
  return output_vec;  // return with copy elision
}

int read_hdf5_dataset_scalar(hid_t hdf5_file, std::string dataset_name,
                             hid_t output_type, void* output_buf) {
  return read_hdf5_dataset_data(
      hdf5_file, dataset_name, output_type, output_buf,
      [](int size) { return size != 1; }, "Expect scalar with shape of 1.");
}

float dequantize(unsigned char i, float scale, float clip_max) {
  return (float(i) - scale) * clip_max / scale;
}

void dequantize_array(std::vector<unsigned char>& i8, std::vector<float>& f,
                      float clip_max, float quant_range, int start, int num) {
  for (int i = start; i < start + num; ++i) {
    f[i] = dequantize(i8[i], quant_range, clip_max);
  }
}

}  // namespace cuda
}  // namespace lightseq
