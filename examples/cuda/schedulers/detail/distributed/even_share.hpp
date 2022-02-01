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
#pragma once

#include <utility>

namespace example::cuda::distributed
{

template <class T>
std::pair<T, T> even_share(T n, unsigned int rank, unsigned int size)
{
  const auto avg_per_device = n / size;
  const auto n_big_share = avg_per_device + 1;
  const auto big_shares = n % size;
  const auto is_big_share = rank < big_shares;
  const auto begin = is_big_share ? n_big_share * rank
                                  : n_big_share * big_shares +
                                      (rank - big_shares) * avg_per_device;
  const auto end = begin + (is_big_share ? n_big_share : avg_per_device);

  return std::make_pair(begin, end);
}

}
