/*
 * Copyright (c) Lucian Radu Teodorescu
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
 * - we are looking at the flow of handling a "classify" request type
 * - show how a single-threaded process can be broken into multiple steps with `execution::then` and
 * `execution::upon_*` functions
 * - handle errors and cancellations that might occur during the processing
 * - at the end of the flow, we always end up with an HTTP response
 *
 * Example goals:
 * - show how one can compose single-threaded steps to form a more complex flow
 * - exemplify the use of `then`, `upon_error` and `upon_done` algorithms
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

// Pull in the reference implementation of P2300:
#include <execution.hpp>
// Keep track of spawned work in an async_scope:
#include <async_scope.hpp>
// Use a thread pool
#include "../schedulers/static_thread_pool.hpp"

namespace ex = std::execution;

struct http_request {
  std::string url_;
  std::vector<std::pair<std::string, std::string>> headers_;
  std::string body_;
};

struct http_response {
  int status_code_;
  std::string body_;
};

enum class obj_type {
  human,
  dog,
  cat,
  bird,
  unknown,
  general_error,
  cancelled,
};

const char* as_string(obj_type t) {
  switch (t) {
  case obj_type::human:
    return "human";
  case obj_type::dog:
    return "dog";
  case obj_type::cat:
    return "cat";
  case obj_type::bird:
    return "bird";
  case obj_type::unknown:
    return "unknown";
  case obj_type::general_error:
    return "general error";
  case obj_type::cancelled:
    return "cancelled";
  }
  return "general error";
}

struct classification_result {
  obj_type type_;
  int accuracy_;
  std::string details_;
};

struct image {
  std::string image_data_;
};

// Extract the image from the HTTP request
image extract_image(http_request req) {
  // TODO: make upon_error work before enabling this
  // if (req.body_.empty())
  //   throw std::invalid_argument("no image found");
  return {req.body_};
}

// Classify the image received
classification_result do_classify(image img) {
  if (img.image_data_ == "human")
    return {obj_type::human, 93};
  else if (img.image_data_ == "cat")
    return {obj_type::cat, 97};
  if (img.image_data_ == "dog")
    return {obj_type::dog, 92};
  if (img.image_data_ == "bird")
    return {obj_type::bird, 96};
  return {obj_type::unknown, 0};
}

// Check for errors and transform them into classification result
classification_result on_classification_error(std::exception_ptr ex) {
  return {obj_type::general_error, 100, {}};
}

// Check for cancellation and transform it into classification result
classification_result on_classification_cancelled() { return {obj_type::cancelled, 100}; }

// Convert the classification result into an HTTP response
http_response to_response(classification_result res) {
  if (res.type_ == obj_type::general_error)
    // Send a 500 response back if we have a general error
    return {500, res.details_};
  else if (res.type_ == obj_type::cancelled) {
    // Send a 503 response back if the computation is cancelled
    return {503, "cancelled"};
  } else {
    // Send a success response back, with the object type, accuracy and details
    std::ostringstream oss;
    oss << as_string(res.type_) << " (" << res.accuracy_ << ")\n" << res.details_;
    return {200, oss.str()};
  }
}

// Handler for the "classify" request type
ex::sender auto handle_classify_request(const http_request& req) {
  return
      // start with the input buffer
      ex::just(req)
      // extract the image from the input request
      | ex::then(extract_image)
      // analyze the content of the image and classify it
      // we are doing the processing on the same thread
      | ex::then(do_classify)
      // handle errors
      | ex::upon_error(on_classification_error)
      // handle cancellation
      | ex::upon_stopped(on_classification_cancelled)
      // transform this into a response
      | ex::then(to_response)
      // done
      ;
}

int main() {
  // Create a thread pool and get a scheduler from it
  _P2519::execution::async_scope scope;
  example::static_thread_pool pool{8};
  ex::scheduler auto sched = pool.get_scheduler();

  // Fake a couple of requests
  for (int i = 0; i < 12; i++) {
    // Create a test request
    const char* body = "";
    if (i % 2 == 0)
      body = "human";
    else if (i % 3 == 0)
      body = "cat";
    else if (i % 5 == 0)
      body = "dog";
    else if (i % 7 == 0)
      body = "bird";
    http_request req{"/classify", {}, body};

    // The handler for the "classify" requests
    ex::sender auto snd = handle_classify_request(req);

    // Pack this into a simplified flow and execute it asynchronously
    ex::sender auto action = std::move(snd) | ex::then([](http_response resp) {
      std::ostringstream oss;
      oss << "Sending response: " << resp.status_code_ << " / " << resp.body_ << "\n";
      std::cout << oss.str();
    });
    scope.spawn(ex::on(sched, std::move(action)));
  }

  std::this_thread::sync_wait(scope.on_empty());
  pool.request_stop();
}
