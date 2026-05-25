/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <exec/asio/asio_config.hpp>
#include <stdexec/__detail/__config.hpp>

/**
 * This is a boost::throw_exception implementation provided for fno-exceptions builds
 */
#if STDEXEC_ASIO_USES_BOOST && STDEXEC_NO_STDCPP_EXCEPTIONS()

#  include <boost/asio/thread_pool.hpp>
#  include <boost/throw_exception.hpp>

#  include <cstdlib>
#  include <exception>

namespace boost
{
  void throw_exception(std::exception const &)
  {
    std::abort();
  }

  void throw_exception(std::exception const &, boost::source_location const &)
  {
    std::abort();
  }
}  // namespace boost
#endif
