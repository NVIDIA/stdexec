/*
 * Copyright (c) NVIDIA
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
#define CATCH_CONFIG_RUNNER

#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <iostream>

int device_guard(int device_id)
{
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
  {
    std::cerr << "Can't query devices number." << std::endl;
    std::exit(-1);
  }

  if (device_id >= device_count || device_id < 0)
  {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    std::exit(-1);
  }

  return device_id;
}

int main(int argc, char *argv[])
{
  Catch::Session session;

  int device_id{};

  // Build a new parser on top of Catch's
  using namespace Catch::clara;
  auto cli = session.cli() |
             Opt(device_id, "device")["-d"]["--device"]("device id to use");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
  {
    return returnCode;
  }

  cudaSetDevice(device_guard(device_id));

  return session.run(argc, argv);
}
