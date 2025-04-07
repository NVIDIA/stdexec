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

// clang-format Language: Cpp

#pragma once

#include "../stdexec/execution.hpp"

#include "detail/config.cuh"              // IWYU pragma: export
#include "detail/memory.cuh"              // IWYU pragma: export
#include "stream/sync_wait.cuh"           // IWYU pragma: export
#include "stream/bulk.cuh"                // IWYU pragma: export
#include "stream/let_xxx.cuh"             // IWYU pragma: export
#include "stream/schedule_from.cuh"       // IWYU pragma: export
#include "stream/start_detached.cuh"      // IWYU pragma: export
#include "stream/split.cuh"               // IWYU pragma: export
#include "stream/then.cuh"                // IWYU pragma: export
#include "stream/continues_on.cuh"        // IWYU pragma: export
#include "stream/launch.cuh"              // IWYU pragma: export
#include "stream/upon_error.cuh"          // IWYU pragma: export
#include "stream/upon_stopped.cuh"        // IWYU pragma: export
#include "stream/when_all.cuh"            // IWYU pragma: export
#include "stream/reduce.cuh"              // IWYU pragma: export
#include "stream/ensure_started.cuh"      // IWYU pragma: export
#include "stream/common.cuh"              // IWYU pragma: export
#include "detail/queue.cuh"               // IWYU pragma: export
#include "detail/throw_on_cuda_error.cuh" // IWYU pragma: export

namespace nvexec {
  // The stream_domain is how the stream scheduler customizes the sender algorithms. All of the
  // algorithms use the current scheduler's domain to transform senders before starting them.
  struct stream_domain : stdexec::default_domain {
    template <stdexec::sender_expr Sender, class Tag = stdexec::tag_of_t<Sender>, class... Env>
      requires stdexec::
        __callable<stdexec::__sexpr_apply_t, Sender, _strm::transform_sender_for<Tag, Env...>>
      static auto transform_sender(Sender&& sndr, const Env&... env) {
      return stdexec::__sexpr_apply(
        static_cast<Sender&&>(sndr), _strm::transform_sender_for<Tag, Env...>{env...});
    }

    template <class Tag, stdexec::sender Sender, class... Args>
      requires stdexec::__callable<_strm::apply_sender_for<Tag>, Sender, Args...>
    static auto apply_sender(Tag, Sender&& sndr, Args&&... args) {
      return _strm::apply_sender_for<Tag>{}(
        static_cast<Sender&&>(sndr), static_cast<Args&&>(args)...);
    }
  };

  namespace _strm {
    struct stream_scheduler_env {
      [[nodiscard]]
      static auto query(get_forward_progress_guarantee_t) noexcept -> forward_progress_guarantee {
        return forward_progress_guarantee::weakly_parallel;
      }

      [[nodiscard]]
      static auto query(get_domain_t) noexcept -> stream_domain {
        return {};
      }
    };

    struct stream_scheduler : private stream_scheduler_env {
      using __t = stream_scheduler;
      using __id = stream_scheduler;

      explicit stream_scheduler(context_state_t context_state) noexcept
        : context_state_(context_state) {
      }

      auto operator==(const stream_scheduler& other) const noexcept -> bool {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      [[nodiscard]]
      STDEXEC_ATTRIBUTE((host, device)) auto schedule() const noexcept {
        return sender_t{context_state_};
      }

      using stream_scheduler_env::query;

     private:
      template <class ReceiverId>
      struct operation_state_ {
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t : operation_state_base_t<ReceiverId> {
          using __id = operation_state_;

          explicit __t(Receiver rcvr, context_state_t context_state)
            : operation_state_base_t<ReceiverId>(static_cast<Receiver&&>(rcvr), context_state) {
          }

          void start() & noexcept {
            this->propagate_completion_signal(set_value);
          }

         private:
          friend stream_context;
          cudaStream_t stream_{};
          cudaError_t status_{cudaSuccess};
        };
      };

      template <class ReceiverId>
      using operation_state_t = stdexec::__t<operation_state_<ReceiverId>>;

      struct sender_t : stream_sender_base {
        using __t = sender_t;
        using __id = sender_t;

        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_error_t(cudaError_t)>;

        STDEXEC_ATTRIBUTE((host, device))

          explicit sender_t(context_state_t context_state) noexcept
          : env_{context_state} {
        }

        template <class Receiver>
        auto connect(Receiver rcvr) const & noexcept(__nothrow_move_constructible<Receiver>)
          -> operation_state_t<stdexec::__id<Receiver>> {
          return operation_state_t<stdexec::__id<Receiver>>(
            static_cast<Receiver&&>(rcvr), env_.context_state_);
        }

        [[nodiscard]]
        auto get_env() const noexcept -> decltype(auto) {
          return (env_);
        }

       private:
        struct env {
          using __t = env;
          using __id = env;
          context_state_t context_state_;

          [[nodiscard]]
          auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> stream_scheduler {
            return stream_scheduler{context_state_};
          }
        };

        env env_;
      };

     public:
      // private: TODO
      context_state_t context_state_;
    };
  } // namespace _strm

  using _strm::stream_scheduler;

  struct stream_context {
    stream_context()
      : dev_id_(_get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
    }

    auto get_scheduler(stream_priority priority = stream_priority::normal) -> stream_scheduler {
      return stream_scheduler{_strm::context_state_t(
        pinned_resource_.get(), managed_resource_.get(), &stream_pools_, &hub_, priority)};
    }

   private:
    static auto _get_device() -> int {
      int dev_id{};
      cudaGetDevice(&dev_id);
      return dev_id;
    }

    _strm::resource_storage<_strm::pinned_resource> pinned_resource_{};
    _strm::resource_storage<_strm::managed_resource> managed_resource_{};
    _strm::stream_pools_t stream_pools_{};

    int dev_id_{};
    _strm::queue::task_hub_t hub_;
  };
} // namespace nvexec
