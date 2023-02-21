#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "hdf5.h"

#include "declaration.h"
