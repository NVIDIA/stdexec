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


#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
#  include "nvexec/detail/throw_on_cuda_error.cuh"
#  include <nvexec/stream_context.cuh>
#  include <nvexec/multi_gpu_context.cuh>
#else
namespace nvexec {
  struct stream_receiver_base { };

  struct stream_sender_base { };

  namespace detail {
    struct stream_op_state_base { };
  }

  inline bool is_on_gpu() {
    return false;
  }
}
#endif

#include <optional>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
namespace nvexec::STDEXEC_STREAM_DETAIL_NS { namespace repeat_n {
    template <class OpT>
    class receiver_2_t : public stream_receiver_base {
      using Sender = typename OpT::PredSender;
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

     public:
      template <ex::same_as<ex::set_value_t> _Tag>
      STDEXEC_DEFINE_CUSTOM(void set_value)(this receiver_2_t&& __self, _Tag) noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        OpT& op_state = __self.op_state_;
        op_state.i_++;

        if (op_state.i_ == op_state.n_) {
          op_state.propagate_completion_signal(ex::set_value);
          return;
        }

        auto sch = ex::get_scheduler(ex::get_env(op_state.receiver_));
        inner_op_state_t& inner_op_state = op_state.inner_op_state_.emplace(
          ex::__conv{[&]() noexcept {
            return ex::connect(ex::schedule(sch) | op_state.closure_, receiver_2_t<OpT>{op_state});
          }});

        ex::start(inner_op_state);
      }

      template <ex::same_as<ex::set_error_t> _Tag, class _Error>
      STDEXEC_DEFINE_CUSTOM(void set_error)(this receiver_2_t&& __self, _Tag, _Error&& __err) noexcept {
        OpT& op_state = __self.op_state_;
        op_state.propagate_completion_signal(_Tag{}, (_Error&&) __err);
      }

      template <ex::same_as<ex::set_stopped_t> _Tag>
      STDEXEC_DEFINE_CUSTOM(void set_stopped)(this receiver_2_t&& __self, _Tag) noexcept {
        OpT& op_state = __self.op_state_;
        op_state.propagate_completion_signal(_Tag{});
      }

      STDEXEC_DEFINE_CUSTOM(typename OpT::env_t get_env)(
        this const receiver_2_t& self,
        ex::get_env_t) noexcept {
        return self.op_state_.make_env();
      }

      explicit receiver_2_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class OpT>
    class receiver_1_t : public stream_receiver_base {
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

      STDEXEC_CPO_ACCESS(ex::set_value_t);
      STDEXEC_CPO_ACCESS(ex::set_error_t);
      STDEXEC_CPO_ACCESS(ex::set_stopped_t);
      STDEXEC_CPO_ACCESS(ex::get_env_t);

      // BUGBUG necessary because of nvc++ strangeness:
     public:
      template <ex::same_as<ex::set_value_t> _Tag>
      STDEXEC_DEFINE_CUSTOM(void set_value)(this receiver_1_t&& __self, _Tag) noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        OpT& op_state = __self.op_state_;

        if (op_state.n_) {
          auto sch = ex::get_scheduler(ex::get_env(op_state.receiver_));
          inner_op_state_t& inner_op_state = op_state.inner_op_state_.emplace(
            ex::__conv{[&]() noexcept {
              return ex::connect(
                ex::schedule(sch) | op_state.closure_, receiver_2_t<OpT>{op_state});
            }});

          ex::start(inner_op_state);
        } else {
          op_state.propagate_completion_signal(ex::set_value);
        }
      }

      template <ex::same_as<ex::set_error_t> _Tag, class _Error>
      STDEXEC_DEFINE_CUSTOM(void set_error)(this receiver_1_t&& __self, _Tag, _Error&& __err) noexcept {
        OpT& op_state = __self.op_state_;
        op_state.propagate_completion_signal(_Tag{}, (_Error&&) __err);
      }

      template <ex::same_as<ex::set_stopped_t> _Tag>
      STDEXEC_DEFINE_CUSTOM(void set_stopped)(this receiver_1_t&& __self, _Tag) noexcept {
        OpT& op_state = __self.op_state_;
        op_state.propagate_completion_signal(_Tag{});
      }

      STDEXEC_DEFINE_CUSTOM(typename OpT::env_t get_env)(
        this const receiver_1_t& self,
        ex::get_env_t) noexcept {
        return self.op_state_.make_env();
      }

     public:
      explicit receiver_1_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class PredecessorSenderId, class ClosureId, class ReceiverId>
    struct operation_state_t : operation_state_base_t<ReceiverId> {
      using PredSender = ex::__t<PredecessorSenderId>;
      using Closure = ex::__t<ClosureId>;
      using Receiver = ex::__t<ReceiverId>;
      using Scheduler =
        ex::tag_invoke_result_t<ex::get_scheduler_t, ex::env_of_t<Receiver>>;
      using InnerSender =
        std::invoke_result_t<Closure, ex::tag_invoke_result_t<ex::schedule_t, Scheduler>>;

      using predecessor_op_state_t =
        ex::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
      using inner_op_state_t = ex::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

      PredSender pred_sender_;
      Closure closure_;
      std::optional<predecessor_op_state_t> pred_op_state_;
      std::optional<inner_op_state_t> inner_op_state_;
      std::size_t n_{};
      std::size_t i_{};

      STDEXEC_DEFINE_CUSTOM(void start)(this operation_state_t& op, ex::start_t) noexcept {
        if (op.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          op.propagate_completion_signal(ex::set_error, std::move(op.status_));
        } else {
          if (op.n_) {
            ex::start(*op.pred_op_state_);
          } else {
            op.propagate_completion_signal(ex::set_value);
          }
        }
      }

      operation_state_t(
        PredSender&& pred_sender,
        Closure closure,
        Receiver&& receiver,
        std::size_t n)
        : operation_state_base_t<ReceiverId>(
          (Receiver&&) receiver,
          ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred_sender))
            .context_state_,
          false)
        , pred_sender_{(PredSender&&) pred_sender}
        , closure_(closure)
        , n_(n) {
        pred_op_state_.emplace(ex::__conv{[&]() noexcept {
          return ex::connect((PredSender&&) pred_sender_, receiver_1_t{*this});
        }});
      }
    };
}}
#endif

namespace repeat_n_detail {
  template <class OpT>
  class receiver_t {
    using Receiver = typename OpT::Receiver;

    OpT& op_state_;

   public:
    template <ex::same_as<ex::set_error_t> _Tag, class _Error>
    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
    STDEXEC_DEFINE_CUSTOM(void set_error)(this receiver_t&& __self, _Tag __tag, _Error&& __err) noexcept {
      __tag(std::move(__self.op_state_.receiver_), (_Error&&) __err);
    }

    template <ex::same_as<ex::set_stopped_t> _Tag>
    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
    STDEXEC_DEFINE_CUSTOM(void set_stopped)(this receiver_t&& __self, _Tag __tag) noexcept {
      __tag(std::move(__self.op_state_.receiver_));
    }

    STDEXEC_DEFINE_CUSTOM(void set_value)(this receiver_t&& __self, ex::set_value_t) noexcept {
      OpT& op_state = __self.op_state_;

      for (std::size_t i = 0; i < op_state.n_; i++) {
        ex::sync_wait(ex::schedule(exec::inline_scheduler{}) | op_state.closure_);
      }

      ex::set_value(std::move(op_state.receiver_));
    }

    STDEXEC_DEFINE_CUSTOM(auto get_env)(this const receiver_t& self, ex::get_env_t) noexcept
      -> ex::env_of_t<Receiver> {
      return ex::get_env(self.op_state_.receiver_);
    }

    explicit receiver_t(OpT& op_state)
      : op_state_(op_state) {
    }
  };

  template <class SenderId, class ClosureId, class ReceiverId>
  struct operation_state_t {
    using Sender = ex::__t<SenderId>;
    using Closure = ex::__t<ClosureId>;
    using Receiver = ex::__t<ReceiverId>;

    using inner_op_state_t = ex::connect_result_t<Sender, receiver_t<operation_state_t>>;

    inner_op_state_t op_state_;
    Closure closure_;
    Receiver receiver_;
    std::size_t n_{};

    STDEXEC_DEFINE_CUSTOM(void start)(this operation_state_t& self, ex::start_t) noexcept {
      ex::start(self.op_state_);
    }

    operation_state_t(Sender&& sender, Closure closure, Receiver&& receiver, std::size_t n)
      : op_state_{ex::connect((Sender&&) sender, receiver_t<operation_state_t>{*this})}
      , closure_{closure}
      , receiver_{(Receiver&&) receiver}
      , n_(n) {
    }
  };
}

struct repeat_n_t {
  template <class SenderId, class ClosureId>
  struct repeat_n_sender_t {
    using Sender = stdexec::__t<SenderId>;
    using Closure = stdexec::__t<ClosureId>;
    using is_sender = void;

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
      requires(stdexec::tag_invocable<stdexec::connect_t, Sender, Receiver>)
           && (!nvexec::STDEXEC_STREAM_DETAIL_NS::receiver_with_stream_env<Receiver>)
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& r)
      -> repeat_n_detail::operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>> {
      return repeat_n_detail::operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>>(
        (Sender&&) self.sender_, self.closure_, (Receiver&&) r, self.n_);
    }

    template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
      requires(stdexec::tag_invocable<stdexec::connect_t, Sender, Receiver>)
           && (nvexec::STDEXEC_STREAM_DETAIL_NS::receiver_with_stream_env<Receiver>)
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& r)
      -> nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n::
        operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>> {
      return nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n::
        operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>>(
          (Sender&&) self.sender_, self.closure_, (Receiver&&) r, self.n_);
    }
#else
    template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
      requires stdexec::tag_invocable<stdexec::connect_t, Sender, Receiver>
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& r)
      -> repeat_n_detail::operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>> {
      return repeat_n_detail::operation_state_t<SenderId, ClosureId, stdexec::__id<Receiver>>(
        (Sender&&) self.sender_, self.closure_, (Receiver&&) r, self.n_);
    }
#endif

    STDEXEC_DEFINE_CUSTOM(auto get_env)(this const repeat_n_sender_t& s, stdexec::get_env_t) //
      noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
        -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
      return stdexec::get_env(s.sender_);
    }
  };

  template <stdexec::sender Sender, stdexec::__sender_adaptor_closure Closure>
  auto operator()(Sender&& __sndr, std::size_t n, Closure closure) const noexcept
    -> repeat_n_sender_t<stdexec::__x<Sender>, stdexec::__x<Closure>> {
    return repeat_n_sender_t<stdexec::__x<Sender>, stdexec::__x<Closure>>{
      std::forward<Sender>(__sndr), closure, n};
  }

  template <stdexec::__sender_adaptor_closure Closure>
  auto operator()(std::size_t n, Closure closure) const
    -> stdexec::__binder_back<repeat_n_t, std::size_t, Closure> {
    return {
      {},
      {},
      {n, (Closure&&) closure}
    };
  }
};

inline constexpr repeat_n_t repeat_n{};

template <class SchedulerT>
[[nodiscard]] bool is_gpu_scheduler(SchedulerT&& scheduler) {
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
