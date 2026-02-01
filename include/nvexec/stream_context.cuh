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
#include "detail/queue.cuh"               // IWYU pragma: export
#include "detail/throw_on_cuda_error.cuh" // IWYU pragma: export
#include "stream/bulk.cuh"                // IWYU pragma: export
#include "stream/common.cuh"              // IWYU pragma: export
#include "stream/continues_on.cuh"        // IWYU pragma: export
#include "stream/ensure_started.cuh"      // IWYU pragma: export
#include "stream/launch.cuh"              // IWYU pragma: export
#include "stream/let_xxx.cuh"             // IWYU pragma: export
#include "stream/reduce.cuh"              // IWYU pragma: export
#include "stream/repeat_n.cuh"            // IWYU pragma: export
#include "stream/schedule_from.cuh"       // IWYU pragma: export
#include "stream/split.cuh"               // IWYU pragma: export
#include "stream/start_detached.cuh"      // IWYU pragma: export
#include "stream/sync_wait.cuh"           // IWYU pragma: export
#include "stream/then.cuh"                // IWYU pragma: export
#include "stream/upon_error.cuh"          // IWYU pragma: export
#include "stream/upon_stopped.cuh"        // IWYU pragma: export
#include "stream/when_all.cuh"            // IWYU pragma: export

namespace nvexec {
  namespace _strm {
    struct stream_scheduler;

    template <class StreamScheduler>
    struct stream_scheduler_env { // NOLINT(bugprone-crtp-constructor-accessibility)
      STDEXEC_ATTRIBUTE(nodiscard)
      static auto query(get_forward_progress_guarantee_t) noexcept -> forward_progress_guarantee {
        return forward_progress_guarantee::weakly_parallel;
      }

      STDEXEC_ATTRIBUTE(nodiscard)
      auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> StreamScheduler;

      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept -> stream_domain {
        return {};
      }
    };

    struct stream_scheduler : private stream_scheduler_env<stream_scheduler> {
      explicit stream_scheduler(context ctx) noexcept
        : ctx_(ctx) {
      }

      auto operator==(const stream_scheduler& other) const noexcept -> bool {
        return ctx_.hub_ == other.ctx_.hub_;
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device) auto schedule() const noexcept {
        return sender{ctx_};
      }

      using stream_scheduler_env::query;

     private:
      template <class Receiver>
      struct opstate : _strm::opstate_base<Receiver> {
        explicit opstate(Receiver rcvr, context ctx)
          : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), ctx) {
        }

        void start() & noexcept {
          this->propagate_completion_signal(set_value);
        }

       private:
        friend stream_context;
        cudaStream_t stream_{};
        cudaError_t status_{cudaSuccess};
      };

      struct sender : stream_sender_base {
       private:
        struct attrs {
          context ctx_;

          STDEXEC_ATTRIBUTE(nodiscard)
          auto query(get_completion_scheduler_t<set_value_t>, __ignore = {}) const noexcept
            -> stream_scheduler {
            return stream_scheduler{ctx_};
          }

          STDEXEC_ATTRIBUTE(nodiscard)
          constexpr auto
            query(get_completion_domain_t<set_value_t>) const noexcept -> stream_domain {
            return {};
          }
        };

        attrs env_;

       public:
        using completion_signatures =
          STDEXEC::completion_signatures<set_value_t(), set_error_t(cudaError_t)>;

        STDEXEC_ATTRIBUTE(host, device)
        explicit sender(context ctx) noexcept
          : env_{ctx} {
        }

        template <class Receiver>
        auto connect(Receiver rcvr) const & noexcept(__nothrow_move_constructible<Receiver>)
          -> opstate<Receiver> {
          return opstate<Receiver>(static_cast<Receiver&&>(rcvr), env_.ctx_);
        }

        STDEXEC_ATTRIBUTE(nodiscard)
        auto get_env() const noexcept -> const attrs& {
          return (env_);
        }
      };

     public:
      // private: TODO
      context ctx_;
    };

    template <>
    STDEXEC_ATTRIBUTE(nodiscard)
    inline auto stream_scheduler_env<stream_scheduler>::query(
      get_completion_scheduler_t<set_value_t>) const noexcept -> stream_scheduler {
      return STDEXEC::__c_downcast<stream_scheduler>(*this);
    }
  } // namespace _strm

  using _strm::stream_scheduler;

  struct stream_context {
    stream_context()
      : dev_id_(_get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
    }

    auto get_scheduler(stream_priority priority = stream_priority::normal) -> stream_scheduler {
      return stream_scheduler{_strm::context(
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
    _strm::queue::task_hub hub_;
  };
} // namespace nvexec
