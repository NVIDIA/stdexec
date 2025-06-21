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
 * - we are looking at the flow of processing an HTTP request and sending back
 * the response
 * - show how one can break the (slightly complex) flow into steps with let_*
 * functions
 * - different phases of processing HTTP requests are broken down into separate
 * concerns
 * - each part of the processing might use different execution contexts (details
 * not shown in this example)
 * - error handling is generic, regardless which component fails; we always send
 * the right response to the clients
 *
 * Example goals:
 * - show how one can break more complex flows into steps with let_* functions
 * - exemplify the use of let_value, let_error, let_stopped, transfer_just and just
 * algorithms
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>
// Use a thread pool
#include "exec/static_thread_pool.hpp"

#if !STDEXEC_STD_NO_EXCEPTIONS()
namespace ex = stdexec;

struct http_request {
  std::string url_;
  std::vector<std::pair<std::string, std::string>> headers_;
  std::string body_;
};

struct http_response {
  int status_code_;
  std::string body_;
};

// Returns a sender that yields an http_request object for an incoming request
template <ex::scheduler S>
auto schedule_request_start(S sched, int idx) -> ex::sender auto {
  // app-specific-details: building of the http_request object
  auto url = std::string("/query?image_idx=") + std::to_string(idx);
  if (idx == 7)
    url.clear(); // fake invalid request
  http_request req{.url_ = std::move(url), .headers_ = {}, .body_ = {}};
  std::cout << "HTTP request " << idx << " arrived\n";

  // Return a sender for the incoming http_request
  return ex::transfer_just(std::forward<S>(sched), std::move(req));
}

// Sends a response back to the client; yields a void signal on success
auto send_response(const http_response& resp) -> ex::sender auto {
  std::cout << "Sending back response: " << resp.status_code_ << "\n";
  // Signal that we are done successfully
  return ex::just();
}

// Validate that the HTTP request is well-formed
auto validate_request(const http_request& req) -> ex::sender auto {
  std::cout << "validating request " << req.url_ << "\n";
  if (req.url_.empty())
    throw std::invalid_argument("No URL");
  return ex::just(req);
}

// Handle the request; main application logic
auto handle_request(const http_request& req) -> ex::sender auto {
  std::cout << "handling request " << req.url_ << "\n";
  //...
  return ex::just(http_response{.status_code_ = 200, .body_ = "image details"});
}

// Transforms server errors into responses to be sent to the client
auto error_to_response(std::exception_ptr err) -> ex::sender auto {
  try {
    std::rethrow_exception(err);
  } catch (const std::invalid_argument& e) {
    return ex::just(http_response{.status_code_ = 404, .body_ = e.what()});
  } catch (const std::exception& e) {
    return ex::just(http_response{.status_code_ = 500, .body_ = e.what()});
  } catch (...) {
    return ex::just(http_response{.status_code_ = 500, .body_ = "Unknown server error"});
  }
}

// Transforms cancellation of the server into responses to be sent to the client
auto stopped_to_response() -> ex::sender auto {
  return ex::just(http_response{.status_code_ = 503, .body_ = "Service temporarily unavailable"});
}

auto main() -> int {
  // Create a thread pool and get a scheduler from it
  exec::static_thread_pool pool{8};
  ex::scheduler auto sched = pool.get_scheduler();

  // Fake a couple of requests
  for (int i = 0; i < 10; i++) {
    // The whole flow for transforming incoming requests into responses
    ex::sender auto snd =
      // get a sender when a new request comes
      schedule_request_start(sched, i)
      // make sure the request is valid; throw if not
      | ex::let_value(validate_request)
      // process the request in a function that may be using a different execution context
      | ex::let_value(handle_request)
      // If there are errors transform them into proper responses
      | ex::let_error(error_to_response)
      // If the flow is cancelled, send back a proper response
      | ex::let_stopped(stopped_to_response)
      // write the result back to the client
      | ex::let_value(send_response)
      // done
      ;

    // execute the whole flow asynchronously
    ex::start_detached(std::move(snd));
  }

  pool.request_stop();

  return 0;
}

#else

int main() {
  std::cout << "This example requires C++ exceptions to be enabled.\n";
  return 0;
}
#endif // !STDEXEC_STD_NO_EXCEPTIONS()
