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
#include "stdexec/execution.hpp"  // IWYU pragma: export

namespace ex = stdexec;

#if STDEXEC_CUDA_COMPILATION()
#  include <nvexec/multi_gpu_context.cuh>  // IWYU pragma: export
#  include <nvexec/stream_context.cuh>     // IWYU pragma: export
#else
namespace nvexec
{
  struct stream_receiver_base
  {
    using receiver_concept = ex::receiver_t;
  };

  struct stream_sender_base
  {
    using sender_concept = ex::sender_t;
  };

  namespace detail
  {
    struct stream_op_state_base
    {};
  }  // namespace detail

  inline auto is_on_gpu() -> bool
  {
    return false;
  }
}  // namespace nvexec
#endif

#include <exec/inline_scheduler.hpp>
#include <optional>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace ex = stdexec;

namespace repeat_n_detail
{
  template <class Sender, class Closure>
  struct repeat_n_sender_t;
}  // namespace repeat_n_detail

struct repeat_n_t
{
  template <ex::sender Sender, ex::__sender_adaptor_closure Closure>
  auto operator()(Sender __sndr, std::size_t n, Closure closure) const noexcept
    -> repeat_n_detail::repeat_n_sender_t<Sender, Closure>
  {
    return repeat_n_detail::repeat_n_sender_t<Sender, Closure>{
      {},
      {closure, n},
      std::move(__sndr)
    };
  }

  template <ex::__sender_adaptor_closure Closure>
  auto operator()(std::size_t n, Closure closure) const
  {
    return ex::__closure(*this, n, closure);
  }
};

inline constexpr repeat_n_t repeat_n{};

namespace repeat_n_detail
{

  template <class OpT>
  class receiver_2_t
  {
    using Sender   = OpT::child_t;
    using Receiver = OpT::receiver_t;

    OpT& op_state_;

   public:
    using receiver_concept = ex::receiver_t;

    void set_value() noexcept
    {
      using inner_op_state_t = OpT::inner_op_state_t;

      op_state_.i_++;

      if (op_state_.i_ == op_state_.n_)
      {
        ex::set_value(std::move(op_state_.rcvr_));
        return;
      }

      auto              sch            = ex::get_scheduler(ex::get_env(op_state_.rcvr_));
      inner_op_state_t& inner_op_state = op_state_.inner_op_state_.emplace(ex::__emplace_from{
        [&]() noexcept
        {
          return ex::connect(ex::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
        }});

      ex::start(inner_op_state);
    }

    template <class Error>
    void set_error(Error&& err) noexcept
    {
      ex::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept
    {
      ex::set_stopped(std::move(op_state_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Receiver>
    {
      return ex::get_env(op_state_.rcvr_);
    }

    explicit receiver_2_t(OpT& op_state)
      : op_state_(op_state)
    {}
  };

  template <class OpT>
  class receiver_1_t
  {
    using Receiver = OpT::receiver_t;

    OpT& op_state_;

   public:
    using receiver_concept = ex::receiver_t;

    void set_value() noexcept
    {
      using inner_op_state_t = OpT::inner_op_state_t;

      if (op_state_.n_)
      {
        inner_op_state_t& inner_op_state = op_state_.inner_op_state_.emplace(
          ex::__emplace_from{[this]() noexcept
                             {
                               auto sch = ex::get_scheduler(ex::get_env(op_state_.rcvr_));
                               return ex::connect(ex::schedule(sch) | op_state_.closure_,
                                                  receiver_2_t<OpT>{op_state_});
                             }});

        ex::start(inner_op_state);
      }
      else
      {
        ex::set_value(std::move(op_state_.rcvr_));
      }
    }

    template <class Error>
    void set_error(Error&& err) noexcept
    {
      ex::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept
    {
      ex::set_stopped(std::move(op_state_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Receiver>
    {
      return ex::get_env(op_state_.rcvr_);
    }

    explicit receiver_1_t(OpT& op_state)
      : op_state_(op_state)
    {}
  };

  template <class PredSender, class Closure, class Receiver>
  struct operation_state_t
  {
    using receiver_t  = Receiver;
    using child_t     = PredSender;
    using Scheduler   = std::invoke_result_t<ex::get_scheduler_t, ex::env_of_t<Receiver>>;
    using InnerSender = std::invoke_result_t<Closure, ex::schedule_result_t<Scheduler>>;

    using predecessor_op_state_t =
      ex::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
    using inner_op_state_t = ex::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

    PredSender                            pred_sender_;
    Closure                               closure_;
    Receiver                              rcvr_;
    std::optional<predecessor_op_state_t> pred_op_state_;
    std::optional<inner_op_state_t>       inner_op_state_;
    std::size_t                           n_{};
    std::size_t                           i_{};

    void start() & noexcept
    {
      if (n_)
      {
        ex::start(*pred_op_state_);
      }
      else
      {
        ex::set_value(std::move(rcvr_));
      }
    }

    operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
      : pred_sender_{static_cast<PredSender&&>(pred_sender)}
      , closure_(closure)
      , rcvr_(rcvr)
      , n_(n)
    {
      pred_op_state_.emplace(ex::__emplace_from{
        [&]() noexcept
        { return ex::connect(static_cast<PredSender&&>(pred_sender_), receiver_1_t{*this}); }});
    }
  };

  template <class Sender, class Closure>
  struct repeat_n_sender_t
  {
    using sender_concept = ex::sender_t;

    template <class>
    static consteval auto get_completion_signatures() noexcept
    {
      return ex::completion_signatures<
        ex::set_value_t(),
        ex::set_stopped_t(),
        ex::set_error_t(std::exception_ptr)
        // STDEXEC_WHEN(STDEXEC_CUDA_COMPILATION(), , ex::set_error_t(cudaError_t))
        >();
    }

    template <ex::__decays_to<repeat_n_sender_t> Self, ex::receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver r)
      -> repeat_n_detail::operation_state_t<Sender, Closure, Receiver>
    {
      return repeat_n_detail::operation_state_t<Sender, Closure, Receiver>(
        static_cast<Self&&>(self).sender_,
        static_cast<Self&&>(self).data_.first,
        static_cast<Receiver&&>(r),
        self.data_.second);
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Sender const &>
    {
      return ex::get_env(sender_);
    }

    STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused)
    repeat_n_t                      tag_;
    std::pair<Closure, std::size_t> data_;
    Sender                          sender_;
  };
}  // namespace repeat_n_detail

namespace STDEXEC
{
  template <class Sender, class Closure>
  inline constexpr std::size_t
    __structured_binding_size_v<repeat_n_detail::repeat_n_sender_t<Sender, Closure>> = 3;
}  // namespace STDEXEC

#if STDEXEC_CUDA_COMPILATION()
// A CUDA stream implementation of repeat_n
namespace nv::execution::_strm
{
  namespace repeat_n
  {
    template <class OpT>
    class receiver_2_t : public stream_receiver_base
    {
      using Sender   = OpT::PredSender;
      using Receiver = OpT::Receiver;

      OpT& op_state_;

     public:
      void set_value() noexcept
      {
        using inner_op_state_t = OpT::inner_op_state_t;

        op_state_.i_++;

        if (op_state_.i_ == op_state_.n_)
        {
          op_state_.propagate_completion_signal(ex::set_value);
          return;
        }

        inner_op_state_t& inner_op_state = op_state_.inner_op_state_.emplace(ex::__emplace_from{
          [&]() noexcept
          {
            return ex::connect(op_state_.closure_(ex::schedule(op_state_.scheduler_)),
                               receiver_2_t<OpT>{op_state_});
          }});

        ex::start(inner_op_state);
      }

      template <class Error>
      void set_error(Error&& err) noexcept
      {
        op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
      {
        op_state_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> OpT::env_t
      {
        return op_state_.make_env();
      }

      explicit receiver_2_t(OpT& op_state)
        : op_state_(op_state)
      {}
    };

    template <class OpT>
    class receiver_1_t : public stream_receiver_base
    {
      using Receiver = OpT::Receiver;

      OpT& op_state_;

     public:
      void set_value() noexcept
      {
        using inner_op_state_t = OpT::inner_op_state_t;

        if (op_state_.n_)
        {
          inner_op_state_t& inner_op_state = op_state_.inner_op_state_.emplace(ex::__emplace_from{
            [&]() noexcept
            {
              return ex::connect(op_state_.closure_(ex::schedule(op_state_.scheduler_)),
                                 receiver_2_t<OpT>{op_state_});
            }});

          ex::start(inner_op_state);
        }
        else
        {
          op_state_.propagate_completion_signal(set_value_t());
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept
      {
        op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
      {
        op_state_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> OpT::env_t
      {
        return op_state_.make_env();
      }

      explicit receiver_1_t(OpT& op_state)
        : op_state_(op_state)
      {}
    };

    template <class PredSender, class Closure, class Receiver>
    struct operation_state_t : _strm::opstate_base<Receiver>
    {
      using Scheduler   = std::invoke_result_t<ex::get_completion_scheduler_t<ex::set_value_t>,
                                               ex::env_of_t<PredSender>,
                                               ex::env_of_t<Receiver>>;
      using InnerSender = std::invoke_result_t<Closure, ex::schedule_result_t<Scheduler>>;

      using predecessor_op_state_t =
        ex::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
      using inner_op_state_t = ex::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

      Scheduler                             scheduler_;
      Closure                               closure_;
      std::optional<predecessor_op_state_t> pred_op_state_;
      std::optional<inner_op_state_t>       inner_op_state_;
      std::size_t                           n_{};
      std::size_t                           i_{};

      void start() & noexcept
      {
        if (this->stream_provider_.status_ != cudaSuccess)
        {
          // Couldn't allocate memory for operation state, complete with error
          this->propagate_completion_signal(ex::set_error,
                                            std::move(this->stream_provider_.status_));
        }
        else
        {
          if (n_)
          {
            ex::start(*pred_op_state_);
          }
          else
          {
            this->propagate_completion_signal(ex::set_value);
          }
        }
      }

      operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
        : _strm::opstate_base<Receiver>(
            static_cast<Receiver&&>(rcvr),
            ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred_sender),
                                                          ex::get_env(rcvr))
              .context_state_)
        , scheduler_(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred_sender),
                                                                   ex::get_env(rcvr)))
        , closure_(closure)
        , n_(n)
      {
        pred_op_state_.emplace(ex::__emplace_from{
          [&]() noexcept
          { return ex::connect(static_cast<PredSender&&>(pred_sender), receiver_1_t{*this}); }});
      }
    };

    template <class Sender, class Closure>
    struct sender_t
    {
      using sender_concept = ex::sender_t;

      using completion_signatures = ex::completion_signatures<ex::set_value_t(),
                                                              ex::set_stopped_t(),
                                                              ex::set_error_t(std::exception_ptr),
                                                              ex::set_error_t(cudaError_t)>;

      template <ex::__decays_to<sender_t> Self, ex::receiver Receiver>
        requires(ex::sender_to<Sender, Receiver>)
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver r)
        -> nvexec::_strm::repeat_n::operation_state_t<Sender, Closure, Receiver>
      {
        return nvexec::_strm::repeat_n::operation_state_t<Sender, Closure, Receiver>(
          static_cast<Self&&>(self).sender_,
          static_cast<Self&&>(self).closure_,
          static_cast<Receiver&&>(r),
          self.n_);
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      [[nodiscard]]
      auto get_env() const noexcept -> ex::env_of_t<Sender const &>
      {
        return ex::get_env(sender_);
      }

      Sender      sender_;
      Closure     closure_;
      std::size_t n_{};
    };
  }  // namespace repeat_n

  template <>
  struct transform_sender_for<repeat_n_t>
  {
    template <class Env, class Data, class Sender>
    auto operator()(Env const &, ex::__ignore, Data&& data, Sender sndr) const
    {
      static_assert(sizeof(Env) == 0);
      auto& [closure, n] = data;
      using closure_t    = decltype(closure);

      return repeat_n::sender_t<Sender, closure_t>(static_cast<Sender&&>(sndr),
                                                   ex::__forward_like<Data>(closure),
                                                   n);
    }
  };
}  // namespace nv::execution::_strm

#endif  // STDEXEC_CUDA_COMPILATION()

template <class SchedulerT>
[[nodiscard]]
auto is_gpu_scheduler([[maybe_unused]] SchedulerT&& scheduler) -> bool
{
  auto snd      = ex::just() | ex::on(scheduler, ex::then([] { return nvexec::is_on_gpu(); }));
  auto [on_gpu] = ex::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(float                dt,
                     float*               time,
                     bool                 write_results,
                     std::size_t          n_iterations,
                     fields_accessor      accessor,
                     ex::scheduler auto&& computer)
{
  return ex::just()
       | repeat_n(n_iterations,
                  ex::on(computer,
                         ex::bulk(ex::par, accessor.cells, update_h(accessor))
                           | ex::bulk(ex::par, accessor.cells, update_e(time, dt, accessor))))
       | ex::then(dump_vtk(write_results, accessor));
}

void run_snr(float                dt,
             bool                 write_vtk,
             std::size_t          n_iterations,
             grid_t&              grid,
             std::string_view     scheduler_name,
             ex::scheduler auto&& computer)
{
  time_storage_t  time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  auto init = ex::just()
            | ex::on(computer, ex::bulk(ex::par, grid.cells, grid_initializer(dt, accessor)));
  ex::sync_wait(init);

  auto snd = maxwell_eqs_snr(dt, time.get(), write_vtk, n_iterations, accessor, computer);

  report_performance(grid.cells,
                     n_iterations,
                     scheduler_name,
                     [&snd] { ex::sync_wait(std::move(snd)); });
}

STDEXEC_PRAGMA_POP()
