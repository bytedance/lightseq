#include "util.h"

namespace lightseq {

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

void transform_param_shape(float* origin, float* buffer, int row_size,
                           int col_size) {
  int idx = 0;
  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      *(buffer + j * row_size + i) = *(origin + idx);
      idx++;
    }
  }
  for (int i = 0; i < row_size * col_size; i++) {
    *(origin + i) = *(buffer + i);
  }
}

}  // namespace lightseq
