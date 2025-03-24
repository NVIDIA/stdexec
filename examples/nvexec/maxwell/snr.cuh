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

#include "common.cuh"
#include "stdexec/execution.hpp"
#include "exec/on.hpp"


#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
#  include "nvexec/detail/throw_on_cuda_error.cuh"
#  include <nvexec/stream_context.cuh>
#  include <nvexec/multi_gpu_context.cuh>
#else
namespace nvexec {
  struct stream_receiver_base {
    using receiver_concept = stdexec::receiver_t;
  };

  struct stream_sender_base {
    using sender_concept = stdexec::sender_t;
  };

  namespace detail {
    struct stream_op_state_base { };
  } // namespace detail

  inline bool is_on_gpu() {
    return false;
  }
} // namespace nvexec
#endif

#include <optional>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace ex = stdexec;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)

namespace nvexec::_strm::repeat_n {
  template <class OpT>
  class receiver_2_t : public stream_receiver_base {
    using Sender = typename OpT::PredSender;
    using Receiver = typename OpT::Receiver;

    OpT& op_state_;

   public:
    void set_value() noexcept {
      using inner_op_state_t = typename OpT::inner_op_state_t;

      op_state_.i_++;

      if (op_state_.i_ == op_state_.n_) {
        op_state_.propagate_completion_signal(stdexec::set_value);
        return;
      }

      auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
      inner_op_state_t& inner_op_state =
        op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
          return ex::connect(ex::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
        }});

      ex::start(inner_op_state);
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept {
      op_state_.propagate_completion_signal(set_stopped_t());
    }

    auto get_env() const noexcept -> typename OpT::env_t {
      return op_state_.make_env();
    }

    explicit receiver_2_t(OpT& op_state)
      : op_state_(op_state) {
    }
  };

  template <class OpT>
  class receiver_1_t : public stream_receiver_base {
    using Receiver = typename OpT::Receiver;

    OpT& op_state_;

   public:
    void set_value() noexcept {
      using inner_op_state_t = typename OpT::inner_op_state_t;

      if (op_state_.n_) {
        auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
        inner_op_state_t& inner_op_state =
          op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
            return ex::connect(ex::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
          }});

        ex::start(inner_op_state);
      } else {
        op_state_.propagate_completion_signal(set_value_t());
      }
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept {
      op_state_.propagate_completion_signal(set_stopped_t());
    }

    auto get_env() const noexcept -> typename OpT::env_t {
      return op_state_.make_env();
    }

    explicit receiver_1_t(OpT& op_state)
      : op_state_(op_state) {
    }
  };

  template <class PredecessorSenderId, class Closure, class ReceiverId>
  struct operation_state_t : operation_state_base_t<ReceiverId> {
    using PredSender = stdexec::__t<PredecessorSenderId>;
    using Receiver = stdexec::__t<ReceiverId>;
    using Scheduler = std::invoke_result_t<stdexec::get_scheduler_t, stdexec::env_of_t<Receiver>>;
    using InnerSender = std::invoke_result_t<Closure, stdexec::schedule_result_t<Scheduler>>;

    using predecessor_op_state_t =
      ex::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
    using inner_op_state_t = ex::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

    PredSender pred_sender_;
    Closure closure_;
    std::optional<predecessor_op_state_t> pred_op_state_;
    std::optional<inner_op_state_t> inner_op_state_;
    std::size_t n_{};
    std::size_t i_{};

    void start() & noexcept {
      if (this->stream_provider_.status_ != cudaSuccess) {
        // Couldn't allocate memory for operation state, complete with error
        this->propagate_completion_signal(stdexec::set_error, std::move(this->stream_provider_.status_));
      } else {
        if (n_) {
          stdexec::start(*pred_op_state_);
        } else {
          this->propagate_completion_signal(stdexec::set_value);
        }
      }
    }

    operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
      : operation_state_base_t<ReceiverId>(
          static_cast<Receiver&&>(rcvr),
          stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(pred_sender))
            .context_state_)
      , pred_sender_{static_cast<PredSender&&>(pred_sender)}
      , closure_(closure)
      , n_(n) {
      pred_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
        return ex::connect(static_cast<PredSender&&>(pred_sender_), receiver_1_t{*this});
      }});
    }
  };
} // namespace nvexec::_strm::repeat_n
#endif

namespace repeat_n_detail {

  template <class OpT>
  class receiver_2_t {
    using Sender = typename OpT::PredSender;
    using Receiver = typename OpT::Receiver;

    OpT& op_state_;

   public:
    using receiver_concept = stdexec::receiver_t;

    void set_value() noexcept {
      using inner_op_state_t = typename OpT::inner_op_state_t;

      op_state_.i_++;

      if (op_state_.i_ == op_state_.n_) {
        stdexec::set_value(std::move(op_state_.rcvr_));
        return;
      }

      auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
      inner_op_state_t& inner_op_state =
        op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
          return ex::connect(ex::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
        }});

      ex::start(inner_op_state);
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      stdexec::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept {
      stdexec::set_stopped(std::move(op_state_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
      return stdexec::get_env(op_state_.rcvr_);
    }

    explicit receiver_2_t(OpT& op_state)
      : op_state_(op_state) {
    }
  };

  template <class OpT>
  class receiver_1_t {
    using Receiver = typename OpT::Receiver;

    OpT& op_state_;

   public:
    using receiver_concept = stdexec::receiver_t;

    void set_value() noexcept {
      using inner_op_state_t = typename OpT::inner_op_state_t;

      if (op_state_.n_) {
        auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
        inner_op_state_t& inner_op_state =
          op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
            return ex::connect(ex::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
          }});

        ex::start(inner_op_state);
      } else {
        stdexec::set_value(std::move(op_state_.rcvr_));
      }
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      stdexec::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept {
      stdexec::set_stopped(std::move(op_state_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
      return stdexec::get_env(op_state_.rcvr_);
    }

    explicit receiver_1_t(OpT& op_state)
      : op_state_(op_state) {
    }
  };

  template <class PredecessorSenderId, class Closure, class ReceiverId>
  struct operation_state_t {
    using PredSender = stdexec::__t<PredecessorSenderId>;
    using Receiver = stdexec::__t<ReceiverId>;
    using Scheduler = std::invoke_result_t<stdexec::get_scheduler_t, stdexec::env_of_t<Receiver>>;
    using InnerSender = std::invoke_result_t<Closure, stdexec::schedule_result_t<Scheduler>>;

    using predecessor_op_state_t =
      ex::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
    using inner_op_state_t = ex::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

    PredSender pred_sender_;
    Closure closure_;
    Receiver rcvr_;
    std::optional<predecessor_op_state_t> pred_op_state_;
    std::optional<inner_op_state_t> inner_op_state_;
    std::size_t n_{};
    std::size_t i_{};

    void start() & noexcept {
      if (n_) {
        stdexec::start(*pred_op_state_);
      } else {
        stdexec::set_value(std::move(rcvr_));
      }
    }

    operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
      : pred_sender_{static_cast<PredSender&&>(pred_sender)}
      , closure_(closure)
      , rcvr_(rcvr)
      , n_(n) {
      pred_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
        return ex::connect(static_cast<PredSender&&>(pred_sender_), receiver_1_t{*this});
      }});
    }
  };

  template <class SenderId, class Closure>
  struct repeat_n_sender_t {
    using __t = repeat_n_sender_t;
    using __id = repeat_n_sender_t;
    using Sender = stdexec::__t<SenderId>;
    using sender_concept = stdexec::sender_t;

    using completion_signatures = //
      stdexec::completion_signatures<
        stdexec::set_value_t(),
        stdexec::set_stopped_t(),
        stdexec::set_error_t(std::exception_ptr)
#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
          ,
        stdexec::set_error_t(cudaError_t)
#endif
        >;

    Sender sender_;
    Closure closure_;
    std::size_t n_{};

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
    template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
      requires(stdexec::sender_to<Sender, Receiver>)
           && (!nvexec::_strm::receiver_with_stream_env<Receiver>)
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver r)
      -> repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>> {
      return repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>>(
        static_cast<Sender&&>(self.sender_), self.closure_, static_cast<Receiver&&>(r), self.n_);
    }

    template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
      requires(stdexec::sender_to<Sender, Receiver>)
           && (nvexec::_strm::receiver_with_stream_env<Receiver>)
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver r)
      -> nvexec::_strm::repeat_n::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>> {
      return nvexec::_strm::repeat_n::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>>(
        static_cast<Sender&&>(self.sender_), self.closure_, static_cast<Receiver&&>(r), self.n_);
    }
#else
    template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
      requires stdexec::sender_to<Sender, Receiver>
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver r)
      -> repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>> {
      return repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>>(
        static_cast<Sender&&>(self.sender_), self.closure_, static_cast<Receiver&&>(r), self.n_);
    }
#endif

    auto get_env() const noexcept -> stdexec::env_of_t<const Sender&> {
      return stdexec::get_env(sender_);
    }
  };
} // namespace repeat_n_detail

struct repeat_n_t {
  template <stdexec::sender Sender, stdexec::__sender_adaptor_closure Closure>
  auto operator()(Sender&& __sndr, std::size_t n, Closure closure) const noexcept
    -> repeat_n_detail::repeat_n_sender_t<stdexec::__id<Sender>, Closure> {
    return repeat_n_detail::repeat_n_sender_t<stdexec::__id<Sender>, Closure>{
      std::forward<Sender>(__sndr), closure, n};
  }

  template <stdexec::__sender_adaptor_closure Closure>
  auto operator()(std::size_t n, Closure closure) const
    -> stdexec::__binder_back<repeat_n_t, std::size_t, Closure> {
    return {
      {n, static_cast<Closure&&>(closure)},
      {},
      {}
    };
  }
};

inline constexpr repeat_n_t repeat_n{};

template <class SchedulerT>
[[nodiscard]]
auto is_gpu_scheduler(SchedulerT&& scheduler) -> bool {
  auto snd = ex::just() | exec::on(scheduler, ex::then([] { return nvexec::is_on_gpu(); }));
  auto [on_gpu] = stdexec::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(
  float dt,
  float* time,
  bool write_results,
  std::size_t n_iterations,
  fields_accessor accessor,
  stdexec::scheduler auto&& computer) {
  return ex::just()
       | exec::on(
           computer,
           repeat_n(
             n_iterations,
             ex::bulk(accessor.cells, update_h(accessor))
               | ex::bulk(accessor.cells, update_e(time, dt, accessor))))
       | ex::then(dump_vtk(write_results, accessor));
}

void run_snr(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t& grid,
  std::string_view scheduler_name,
  stdexec::scheduler auto&& computer) {
  time_storage_t time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  auto init = ex::just() | exec::on(computer, ex::bulk(grid.cells, grid_initializer(dt, accessor)));
  stdexec::sync_wait(init);

  auto snd = maxwell_eqs_snr(dt, time.get(), write_vtk, n_iterations, accessor, computer);

  report_performance(grid.cells, n_iterations, scheduler_name, [&snd] {
    stdexec::sync_wait(std::move(snd));
  });
}

STDEXEC_PRAGMA_POP()
