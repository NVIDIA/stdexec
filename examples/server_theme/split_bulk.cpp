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
 * - implement the handler for applying 3 edge detection algorithms on one image
 * - implement the handler for applying a blur filter over the given set of images
 * - we show one can use multiple threads to execute a more complex work
 * - we show how to use `stdexec::split` / `stdexec::when_all` and `stdexec::bulk`
 * - error and cancellation handling is performed outside the handler
 *
 * Example goals:
 * - show how one can create work to fill up multiple threads
 * - exemplify the use of `then`, `split`, `when_all`, `bulk` and `let_value` algorithms
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>
// Keep track of spawned work in an async_scope:
#include <exec/async_scope.hpp>
// Use a thread pool
#include "exec/static_thread_pool.hpp"

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

struct image {
  std::string image_data_;
};

// Extract the image from the HTTP request
auto extract_image(http_request req) -> image {
  return {req.body_};
}

// Extract multiple images from the HTTP request
auto extract_images(http_request req) -> std::vector<image> {
  std::vector<image> res;
  size_t last_idx = 0;
  while (last_idx >= std::string::npos) {
    size_t idx = req.body_.find("\n", last_idx);
    if (idx == std::string::npos) {
      break;
    } else {
      res.push_back(image{req.body_.substr(last_idx, idx - last_idx)});
      last_idx = idx + 1;
    }
  }
  if (last_idx != req.body_.size())
    res.push_back(image{req.body_.substr(last_idx)});
  return res;
}

// Convert the given set of images into the corresponding HTTP response
auto img3_to_response(const image& img1, const image& img2, const image& img3) -> http_response {
  std::ostringstream oss;
  oss << img1.image_data_ << ", " << img2.image_data_ << ", " << img3.image_data_ << "\n";
  return {.status_code_ = 200, .body_ = oss.str()};
}

// Convert the given set of images into the corresponding HTTP response
auto imgvec_to_response(const std::vector<image>& imgs) -> http_response {
  std::ostringstream oss;
  for (const auto& img: imgs)
    oss << img.image_data_ << "\n";
  return {.status_code_ = 200, .body_ = oss.str()};
}

// Apply the Canny edge detector on the given image
auto apply_canny(const image& img) -> image {
  return {"canny / " + img.image_data_};
}

// Apply the Sobel edge detector on the given image
auto apply_sobel(const image& img) -> image {
  return {"sobel / " + img.image_data_};
}

// Apply the Prewitt edge detector on the given image
auto apply_prewitt(const image& img) -> image {
  return {"prewitt / " + img.image_data_};
}

// Apply blur filter on the given image
auto apply_blur(const image& img) -> image {
  return {"blur / " + img.image_data_};
}

auto handle_edge_detection_request(const http_request& req) -> ex::sender auto {
  // extract the input image from the request
  ex::sender auto in_img_sender = ex::just(req) | ex::then(extract_image);

  // Prepare for using multiple parallel flows on the same input sender
  // ex::sender auto multi_shot_img = ex::split(in_img_sender);
  auto& multi_shot_img = in_img_sender;

  // Apply the three methods of edge detection on the same input image, in parallel.
  // Then, join the results and generate the HTTP response
  return ex::when_all(
           multi_shot_img | ex::then(apply_canny),
           multi_shot_img | ex::then(apply_sobel),
           multi_shot_img | ex::then(apply_prewitt))
       |
         // transform the resulting 3 images into an HTTP response
         ex::then(img3_to_response);
  // error and cancellation handling is performed outside
}

auto handle_multi_blur_request(const http_request& req) -> ex::sender auto {
  return
    // extract the input images from the request
    ex::just(req)
    | ex::then(extract_images)
    // process images in parallel with bulk.
    // use let_value to access the image count before calling bulk.
    | ex::let_value([](std::vector<image> imgs) {
        // get the image count
        size_t img_count = imgs.size();
        // return a sender that bulk-processes the image in parallel
        return ex::just(std::move(imgs))
             | ex::bulk(ex::par, img_count, [](size_t i, std::vector<image>& imgs) {
                 imgs[i] = apply_blur(imgs[i]);
               });
      })
    // transform the resulting 3 images into an HTTP response
    | ex::then(imgvec_to_response)
    // done; error and cancellation handling is performed outside
    ;
}

auto main() -> int {
  // Create a thread pool and get a scheduler from it
  exec::static_thread_pool pool{8};
  exec::async_scope scope;
  ex::scheduler auto sched = pool.get_scheduler();

  // Fake a couple of edge_detect requests
  for (int i = 0; i < 3; i++) {
    // Create a test request
    http_request req{.url_ = "/edge_detect", .headers_ = {}, .body_ = "scene"};

    // The handler for the /edge_detect requests
    ex::sender auto snd = handle_edge_detection_request(req);

    // Pack this into a simplified flow and execute it asynchronously
    ex::sender auto action = std::move(snd) | ex::then([](http_response resp) {
                               std::ostringstream oss;
                               oss << "Sending response: " << resp.status_code_ << " / "
                                   << resp.body_ << "\n";
                               std::cout << oss.str();
                             });
    scope.spawn(ex::starts_on(sched, std::move(action)));
  }

  // Fake a couple of multi_blur requests
  for (int i = 0; i < 3; i++) {
    // Create a test request
    http_request req{.url_ = "/multi_blur", .headers_ = {}, .body_ = "img1\nimg2\nimg3\nimg4\n"};

    // The handler for the /edge_detect requests
    ex::sender auto snd = handle_multi_blur_request(req);

    // Pack this into a simplified flow and execute it asynchronously
    ex::sender auto action = std::move(snd) | ex::then([](http_response resp) {
                               std::ostringstream oss;
                               oss << "Sending response: " << resp.status_code_ << " / "
                                   << resp.body_ << "\n";
                               std::cout << oss.str();
                             });
    scope.spawn(ex::starts_on(sched, std::move(action)));
  }

  stdexec::sync_wait(scope.on_empty());
  pool.request_stop();
}
