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

#include "common.cuh"
#include "stdexec/execution.hpp"
#include "exec/on.hpp"


#ifdef _NVHPC_CUDA
#include "nvexec/detail/throw_on_cuda_error.cuh"
#include <nvexec/stream_context.cuh>
#include <nvexec/multi_gpu_context.cuh>
#else
namespace nvexec {
  struct stream_receiver_base{};
  struct stream_sender_base{};

  namespace detail {
    struct stream_op_state_base{};
  }

  inline bool is_on_gpu() {
    return false;
  }
}
#endif 

#include <optional>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = std::execution;

#ifdef _NVHPC_CUDA
namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace repeat_n {
    template <class OpT>
      class receiver_t : public stream_receiver_base {
        using Sender = typename OpT::Sender;
        using Receiver = typename OpT::Receiver;

        OpT &op_state_;

      public:
        template <stdexec::__one_of<ex::set_error_t, ex::set_stopped_t> _Tag, class... _Args _NVCXX_CAPTURE_PACK(_Args)>
          friend void tag_invoke(_Tag __tag, receiver_t&& __self, _Args&&... __args) noexcept {
            _NVCXX_EXPAND_PACK(_Args, __args,
              OpT &op_state = __self.op_state_;
              op_state.propagate_completion_signal(_Tag{}, (_Args&&)__args...);
            )
          }

        friend void tag_invoke(ex::set_value_t, receiver_t&& __self) noexcept {
          using inner_op_state_t = ex::connect_result_t<Sender, receiver_t>;

          OpT &op_state = __self.op_state_;
          op_state.i_++;

          if (op_state.i_ == op_state.n_) {
            op_state.propagate_completion_signal(std::execution::set_value);
            return;
          }

          inner_op_state_t& inner_op_state = op_state.inner_op_state_.emplace(
              stdexec::__conv{[&]() noexcept {
                return ex::connect((Sender&&)op_state.sender_, receiver_t{op_state});
              }});

          ex::start(inner_op_state);
        }

        friend auto tag_invoke(ex::get_env_t, const receiver_t& self) noexcept
          -> make_stream_env_t<stdexec::env_of_t<Receiver>> {
          return make_stream_env(
              stdexec::get_env(self.op_state_.receiver_), 
              std::optional<cudaStream_t>{self.op_state_.stream_});
        }

        explicit receiver_t(OpT& op_state)
          : op_state_(op_state)
        {}
      };

    template <class SenderId, class ReceiverId>
      struct operation_state_t : operation_state_base_t<ReceiverId>  {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        using inner_op_state_t = ex::connect_result_t<Sender, receiver_t<operation_state_t>>;

        Sender sender_;
        std::optional<inner_op_state_t> inner_op_state_;
        std::size_t n_{};
        std::size_t i_{};

        cudaStream_t get_stream() {
          return this->stream_;
        }

        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
          op.stream_ = op.allocate();

          if (op.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            op.propagate_completion_signal(std::execution::set_error, std::move(op.status_));
          } else {
            if (op.n_) {
              std::execution::start(*op.inner_op_state_);
            } else {
              op.propagate_completion_signal(std::execution::set_value);
            }
          }
        }

        operation_state_t(Sender&& sender, Receiver&& receiver, std::size_t n)
          : operation_state_base_t<ReceiverId>((Receiver&&)receiver)
          , sender_{(Sender&&)sender}
          , n_(n) {
          inner_op_state_.emplace(
            stdexec::__conv{[&]() noexcept {
              return ex::connect((Sender&&)sender_, receiver_t{*this});
            }});
        }
      };
  }

  template <class SenderId>
    struct repeat_n_sender_t : stream_sender_base {
      using Sender = stdexec::__t<SenderId>;

      using completion_signatures = std::execution::completion_signatures<
        std::execution::set_value_t(),
        std::execution::set_error_t(std::exception_ptr)>;

      Sender sender_;
      std::size_t n_{};

      template <stdexec::__decays_to<repeat_n_sender_t> Self, class Receiver>
        requires std::tag_invocable<std::execution::connect_t, Sender, Receiver> friend auto
      tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r) 
        -> repeat_n::operation_state_t<SenderId, stdexec::__x<Receiver>> {
        return repeat_n::operation_state_t<SenderId, stdexec::__x<Receiver>>(
          (Sender&&)self.sender_,
          (Receiver&&)r,
          self.n_);
      }

      template <stdexec::tag_category<std::execution::forwarding_sender_query> Tag, class... Ts>
        requires std::tag_invocable<Tag, Sender, Ts...> friend decltype(auto)
      tag_invoke(Tag tag, const repeat_n_sender_t &s, Ts &&...ts) noexcept {
        return tag(s.sender_, std::forward<Ts>(ts)...);
      }
    };
} 
#endif

namespace repeat_n_detail {
  template <class SenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      Sender sender_;
      Receiver receiver_;
      std::size_t n_{};

      friend void
      tag_invoke(std::execution::start_t, operation_state_t &self) noexcept {
        for (std::size_t i = 0; i < self.n_; i++) {
          std::this_thread::sync_wait((Sender&&)self.sender_);
        }
        ex::set_value((Receiver&&)self.receiver_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver, std::size_t n)
        : sender_{(Sender&&)sender}
        , receiver_{(Receiver&&)receiver}
        , n_(n)
      {}
    };

  template <class SenderId>
    struct repeat_n_sender_t {
      using Sender = stdexec::__t<SenderId>;

      using completion_signatures = std::execution::completion_signatures<
        std::execution::set_value_t(),
        std::execution::set_stopped_t(),
        std::execution::set_error_t(std::exception_ptr)>;

      Sender sender_;
      std::size_t n_{};

      template <stdexec::__decays_to<repeat_n_sender_t> Self, class Receiver>
        requires std::tag_invocable<std::execution::connect_t, Sender, Receiver> friend auto
      tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r)
        -> operation_state_t<SenderId, stdexec::__x<Receiver>> {
        return operation_state_t<SenderId, stdexec::__x<Receiver>>(
          (Sender&&)self.sender_,
          (Receiver&&)r,
          self.n_);
      }

      template <stdexec::__none_of<std::execution::get_completion_scheduler_t<std::execution::set_value_t>> Tag, class... Ts>
        requires std::tag_invocable<Tag, Sender, Ts...> friend decltype(auto)
      tag_invoke(Tag tag, const repeat_n_sender_t &s, Ts &&...ts) noexcept {
        return tag(s.sender_, std::forward<Ts>(ts)...);
      }
    };
}

struct repeat_n_t {
#ifdef _NVHPC_CUDA
  template <nvexec::STDEXEC_STREAM_DETAIL_NS::stream_completing_sender Sender>
    auto operator()(Sender &&__sndr, std::size_t n) const noexcept
      -> nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n_sender_t<stdexec::__x<Sender>> {
      return nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n_sender_t<stdexec::__x<Sender>>{
        {}, std::forward<Sender>(__sndr), n};
    }
#endif

  template <class Sender>
    auto operator()(Sender &&__sndr, std::size_t n) const noexcept
      -> repeat_n_detail::repeat_n_sender_t<stdexec::__x<Sender>> {
      return repeat_n_detail::repeat_n_sender_t<stdexec::__x<Sender>>{
        std::forward<Sender>(__sndr), n};
    }

  auto operator()(std::size_t n) const noexcept
    -> stdexec::__binder_back<repeat_n_t, std::size_t> {
    return {{}, {}, n};
  }
};

inline constexpr repeat_n_t repeat_n{};

template <class SchedulerT>
[[nodiscard]] bool is_gpu_scheduler(SchedulerT &&scheduler) {
  auto snd = ex::just() | exec::on(scheduler, ex::then([] { return nvexec::is_on_gpu(); }));
  auto [on_gpu] = std::this_thread::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(float dt,
                     float *time,
                     bool write_results,
                     std::size_t &report_step,
                     std::size_t n_inner_iterations,
                     std::size_t n_outer_iterations,
                     fields_accessor accessor,
                     std::execution::scheduler auto &&computer) {
  return exec::on(exec::inline_scheduler{},
                  ex::just()
                | exec::on(computer,
                           ex::bulk(accessor.cells, update_h(accessor))
                         | ex::bulk(accessor.cells, update_e(time, dt, accessor))
                         | repeat_n(n_inner_iterations))
                | ex::then(dump_vtk(write_results, report_step, accessor))
                | repeat_n(n_outer_iterations));
}

void run_snr(float dt,
             bool write_vtk,
             std::size_t n_inner_iterations,
             std::size_t n_outer_iterations,
             grid_t &grid,
             std::string_view scheduler_name,
             std::execution::scheduler auto &&computer) {
  time_storage_t time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  auto init = ex::just() 
            | exec::on(computer, ex::bulk(grid.cells, grid_initializer(dt, accessor)));
  std::this_thread::sync_wait(init);

  std::size_t report_step = 0;
  auto snd = maxwell_eqs_snr(dt,
                             time.get(),
                             write_vtk,
                             report_step,
                             n_inner_iterations,
                             n_outer_iterations,
                             accessor,
                             computer);

  report_performance(grid.cells,
                     n_inner_iterations * n_outer_iterations,
                     scheduler_name,
                     [&snd] { std::this_thread::sync_wait(std::move(snd)); });
}

