#pragma once
#include "proto_headers.h"

namespace lightseq {

/*
Utility function for initializing
*/
bool endswith(std::string const& full, std::string const& end);

/*
Helper function of HDF5.

Return the 1D size of given hdf5 dataset if dataset is already open.
*/
int get_hdf5_dataset_size(hid_t dataset);

/*
Helper function of HDF5.

Return the 1D size of given hdf5 dataset in the given file.
*/
int get_hdf5_dataset_size(hid_t hdf5_file, std::string dataset_name);

/*
Helper function of HDF5.

Read the data of specified type `output_type` into `output_buf`.
return: the size of output data.
*/
int read_hdf5_dataset_data(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    void* output_buf,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

/*
Helper function of HDF5.

Read the data of specified type `output_type` into a vector<T>,
and the vector will be returned.
*/
// TODO: merge these two _float _int function together to improve readability
std::vector<float> read_hdf5_dataset_data_float(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

std::vector<int> read_hdf5_dataset_data_int(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

/*
Helper function of HDF5.

Read a scalar of specified type `output_type` into `output_buf`.

return: the size of output data.
*/
int read_hdf5_dataset_scalar(hid_t hdf5_file, std::string dataset_name,
                             hid_t output_type, void* output_buf);

class HDF5DatasetNotFoundError : public std::runtime_error {
 public:
  HDF5DatasetNotFoundError(const char* what) : runtime_error(what) {}
};

void transform_param_shape(float* origin, float* buffer, int row_size,
                           int col_size);
}  // namespace lightseq
