/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

/*
 * General context:
 * - server application that processes images
 * - execution contexts:
 *    - 1 dedicated thread for network I/O
 *    - N worker threads used for CPU-intensive work
 *    - M threads for auxiliary I/O
 *    - optional GPU context that may be used on some types of servers
 *
 * Specific problem description:
 * - reading data from the socket before processing the request
 * - reading of the data is done on the I/O context
 * - no processing of the data needs to be done on the I/O context
 *
 * Example goals:
 * - show how one can change the execution context
 * - exemplify the use of `starts_on` and `continues_on` algorithms
 */

#include <iostream>
#include <array>
#include <string_view>
#include <cstring>
#include <mutex>

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>
#include <exec/async_scope.hpp>
// Use a thread pool
#include "exec/static_thread_pool.hpp"

namespace ex = stdexec;

struct sync_stream {
 private:
  static std::mutex s_mtx_;

 public:
  std::ostream& sout_;
  std::unique_lock<std::mutex> lock_{s_mtx_};

  template <class T>
  friend auto operator<<(sync_stream&& self, const T& value) -> sync_stream&& {
    self.sout_ << value;
    return std::move(self);
  }

  friend auto
    operator<<(sync_stream&& self, std::ostream& (*manip)(std::ostream&) ) -> sync_stream&& {
    self.sout_ << manip;
    return std::move(self);
  }
};

std::mutex sync_stream::s_mtx_{};

auto legacy_read_from_socket(int, char* buffer, size_t buffer_len) -> size_t {
  const char fake_data[] = "Hello, world!";
  size_t sz = sizeof(fake_data) - 1;
  size_t count = std::min(sz, buffer_len);
  std::memcpy(buffer, fake_data, count);
  return count;
}

void process_read_data(const char* read_data, size_t read_len) {
  sync_stream{.sout_ = std::cout} << "Processing '" << std::string_view{read_data, read_len}
                                  << "'\n";
}

auto main() -> int {
  // Create a thread pool and get a scheduler from it
  exec::static_thread_pool work_pool{8};
  ex::scheduler auto work_sched = work_pool.get_scheduler();

  exec::static_thread_pool io_pool{1};
  ex::scheduler auto io_sched = io_pool.get_scheduler();

  std::array<std::byte, 16 * 1024> buffer;

  exec::async_scope scope;

  // Fake a couple of requests
  for (int i = 0; i < 10; i++) {
    int sock = i;
    auto buf = reinterpret_cast<char*>(&buffer[0]);

    // A sender that just calls the legacy read function
    auto snd_read = ex::just(sock, buf, buffer.size()) | ex::then(legacy_read_from_socket);
    // The entire flow
    auto snd =
      // start by reading data on the I/O thread
      ex::starts_on(io_sched, std::move(snd_read))
      // do the processing on the worker threads pool
      | ex::continues_on(work_sched)
      // process the incoming data (on worker threads)
      | ex::then([buf](size_t read_len) { process_read_data(buf, read_len); })
      // done
      ;

    // execute the whole flow asynchronously
    scope.spawn(std::move(snd));
  }

  (void) stdexec::sync_wait(scope.on_empty());

  return 0;
}
