/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../../stdexec/execution.hpp"

#include "config.cuh"

#include <string>
#include <string_view>
#include <exception>
#include <iostream>

// TODO: We should probably just use `source_location` here.

#if (defined(DEBUG) || defined(_DEBUG)) && !defined(STDEXEC_THROW_ON_CUDA_ERROR) && !defined(STDEXEC_PRINT_ON_CUDA_EROR)
  #define STDEXEC_THROW_ON_CUDA_ERROR
#endif

namespace nvexec {
namespace STDEXEC_STREAM_DETAIL_NS {

  inline std::string format_cuda_error_message(
    cudaError_t error,
    std::string_view info = "") {
    std::string message("CUDA error ");
    message += cudaGetErrorName(error);
    message += ": ";
    message += cudaGetErrorString(error);
    if (!info.empty()) {
      message += ": ";
      message += info;
    }
    return message;
  }

  inline std::string format_cuda_error_message(
    cudaError_t error,
    std::string_view file_name,
    int line,
    std::string_view info = "") {
    std::string message(format_cuda_error_message(error, file_name, line, info));
    message += " [";
    message += file_name;
    message += ":";
    message += std::to_string(line);
    message += "]: ";
    return message;
  }

} // namespace STDEXEC_STREAM_DETAIL_NS

  struct cuda_exception : std::exception {
    cuda_exception(
      cudaError_t error_,
      std::string_view file_name_,
      int line_,
      std::string_view info_ = "")
      : error(error_),
        message(
          STDEXEC_STREAM_DETAIL_NS::format_cuda_error_message(error_,
                                                              file_name_,
                                                              line_,
                                                              info_))
    {}

    cuda_exception(
      cudaError_t error_,
      std::string_view info_ = "")
      : error(error_),
        message(
          STDEXEC_STREAM_DETAIL_NS::format_cuda_error_message(error_,
                                                              info_))
    {}

    cudaError_t code() const noexcept {
      return error;
    }

    virtual char const* what() const noexcept override {
      return message.c_str();
    }

  private:
    const cudaError_t error;
    const std::string message;
  };

namespace STDEXEC_STREAM_DETAIL_NS {

  // Clears the global CUDA error state and returns the `cudaError_t` code.
  // If the `cudaError_t` parameter represents an error:
  // * If `STDEXEC_THROW_ON_CUDA_ERROR` is defined, an exception is thrown.
  // * If `STDEXEC_PRINT_ON_CUDA_ERROR` is defined, a diagnostic is printed to stdout.
  // * Otherwise, nothing happens.
  inline cudaError_t check_cuda_error(
    [[maybe_unused]] std::string_view file_name,
    [[maybe_unused]] int line,
    cudaError_t error,
    [[maybe_unused]] std::string_view info = "") {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    cudaGetLastError();

    #if defined(STDEXEC_THROW_ON_CUDA_ERROR)
      if (error != cudaSuccess)
        throw cuda_exception(error, file_name, line, info);
    #elif defined(STDEXEC_PRINT_ON_CUDA_ERROR)
      if (error != cudaSuccess)
        std::cout << format_cuda_error_message(error, file_name, line, info)
                  << std::endl;
    #endif

    return error;
  }

  #define STDEXEC_CHECK_CUDA_ERROR(...) \
    ::nvexec::STDEXEC_STREAM_DETAIL_NS::check_cuda_error(__FILE__, __LINE__, __VA_ARGS__) /**/

} // namespace STDEXEC_STREAM_DETAIL_NS
} // namespace nvexec
