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

#include "exec/repeat_n.hpp"  // IWYU pragma: export

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

namespace _repeat_n
{
  template <class Sender, class Closure>
  struct sender;
}  // namespace _repeat_n

struct repeat_n_t
{
  template <ex::sender Sender, ex::__sender_adaptor_closure Closure>
  auto operator()(Sender __sndr, std::size_t count, Closure closure) const noexcept
    -> _repeat_n::sender<Sender, Closure>
  {
    return _repeat_n::sender<Sender, Closure>{
      {},
      {closure, count},
      std::move(__sndr)
    };
  }

  template <ex::__sender_adaptor_closure Closure>
  auto operator()(std::size_t count, Closure closure) const
  {
    return ex::__closure(*this, count, closure);
  }
};

inline constexpr repeat_n_t repeat_n{};

namespace _repeat_n
{
  template <class OpState>
  class receiver
  {
    using receiver_t = OpState::receiver_t;
   public:
    using receiver_concept = ex::receiver_t;

    explicit receiver(OpState& op_state)
      : opstate_(op_state)
    {}

    void set_value() noexcept
    {
      if (opstate_.count_ == 0)
      {
        ex::set_value(std::move(opstate_.rcvr_));
      }
      else
      {
        --opstate_.count_;
        ex::start(opstate_._connect());
      }
    }

    template <class Error>
    void set_error(Error&& err) noexcept
    {
      ex::set_error(std::move(opstate_.rcvr_), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept
    {
      ex::set_stopped(std::move(opstate_.rcvr_));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<receiver_t>
    {
      return ex::get_env(opstate_.rcvr_);
    }

   private:
    OpState& opstate_;
  };

  template <class CvSender, class Closure, class Receiver>
  struct opstate
  {
    opstate(CvSender&& sndr, Closure closure, Receiver&& rcvr, std::size_t count)
      : rcvr_(static_cast<Receiver&&>(rcvr))
      , count_(count)
      , closure_(std::move(closure))
      , sched_(_get_scheduler(sndr))
    {
      pred_opstate_.emplace(ex::__emplace_from{
        [&]() noexcept
        { return ex::connect(static_cast<CvSender&&>(sndr), receiver<opstate>{*this}); }});
    }

    void start() & noexcept
    {
      if (count_ > 0)
      {
        ex::start(*pred_opstate_);
      }
      else
      {
        ex::set_value(std::move(rcvr_));
      }
    }

   private:
    friend receiver<opstate>;

    using receiver_t      = Receiver;
    using scheduler_t     = std::invoke_result_t<ex::get_completion_scheduler_t<ex::set_value_t>,
                                                 ex::env_of_t<CvSender>,
                                                 ex::env_of_t<Receiver>>;
    using inner_sender_t  = std::invoke_result_t<Closure, ex::schedule_result_t<scheduler_t>>;
    using pred_opstate_t  = ex::connect_result_t<CvSender, receiver<opstate>>;
    using inner_opstate_t = ex::connect_result_t<inner_sender_t, receiver<opstate>>;

    auto& _connect()
    {
      return inner_opstate_.emplace(ex::__emplace_from{
        [&]() noexcept
        { return ex::connect(closure_(ex::schedule(sched_)), receiver<opstate>{*this}); }});
    }

    scheduler_t _get_scheduler(CvSender const & sndr) noexcept
    {
      return ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr),
                                                           ex::get_env(this->rcvr_));
    }

    Receiver                       rcvr_;
    std::size_t                    count_;
    Closure                        closure_;
    scheduler_t                    sched_;
    std::optional<pred_opstate_t>  pred_opstate_;
    std::optional<inner_opstate_t> inner_opstate_;
  };

  template <class Sender, class Closure>
  struct sender
  {
    using sender_concept = ex::sender_t;

    template <class, class Env>
    static consteval auto get_completion_signatures() noexcept
    {
      return ex::completion_signatures<
        ex::set_value_t(),
        ex::set_stopped_t(),
        ex::set_error_t(std::exception_ptr)
        // STDEXEC_WHEN(STDEXEC_CUDA_COMPILATION(), , ex::set_error_t(cudaError_t))
        >();
    }

    template <ex::__decays_to<sender> Self, ex::receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver r)
      -> _repeat_n::opstate<Sender, Closure, Receiver>
    {
      return _repeat_n::opstate<Sender, Closure, Receiver>(static_cast<Self&&>(self).sndr_,
                                                           static_cast<Self&&>(self).data_.first,
                                                           static_cast<Receiver&&>(r),
                                                           self.data_.second);
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env_of_t<Sender const &>
    {
      return ex::get_env(sndr_);
    }

    STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused)
    repeat_n_t                      tag_;
    std::pair<Closure, std::size_t> data_;
    Sender                          sndr_;
  };
}  // namespace _repeat_n

namespace STDEXEC
{
  template <class Sender, class Closure>
  inline constexpr std::size_t __structured_binding_size_v<_repeat_n::sender<Sender, Closure>> = 3;
}  // namespace STDEXEC

#if STDEXEC_CUDA_COMPILATION()
// A CUDA stream implementation of repeat_n
namespace nv::execution::_strm
{
  namespace _repeat_n
  {
    template <class OpState>
    class receiver : public stream_receiver_base
    {
     public:
      explicit receiver(OpState& op_state)
        : opstate_(op_state)
      {}

      void set_value() noexcept
      {
        if (opstate_.count_ == 0)
        {
          opstate_.propagate_completion_signal(ex::set_value);
        }
        else
        {
          --opstate_.count_;
          ex::start(opstate_._connect());
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept
      {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
      {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> OpState::env_t
      {
        return opstate_.make_env();
      }

     private:
      OpState& opstate_;
    };

    template <class CvSender, class Closure, class Receiver>
    struct opstate : _strm::opstate_base<Receiver>
    {
      explicit opstate(CvSender&& sndr, Closure closure, Receiver&& rcvr, std::size_t count)
        : _strm::opstate_base<Receiver>(std::move(rcvr), _get_scheduler(sndr).ctx_)
        , count_(count)
        , closure_(std::move(closure))
        , sched_(_get_scheduler(sndr))
      {
        pred_opstate_.emplace(ex::__emplace_from{
          [&]() noexcept { return ex::connect(static_cast<CvSender&&>(sndr), receiver{*this}); }});
      }

      void start() & noexcept
      {
        if (this->stream_provider_.status_ != cudaSuccess)
        {
          // Couldn't allocate memory for operation state, complete with error
          this->propagate_completion_signal(ex::set_error,
                                            cudaError_t(this->stream_provider_.status_));
        }
        else if (count_ > 0)
        {
          ex::start(*pred_opstate_);
        }
        else
        {
          this->propagate_completion_signal(ex::set_value);
        }
      }

     private:
      friend receiver<opstate>;

      using scheduler_t     = std::invoke_result_t<ex::get_completion_scheduler_t<ex::set_value_t>,
                                                   ex::env_of_t<CvSender>,
                                                   ex::env_of_t<Receiver>>;
      using inner_sender_t  = std::invoke_result_t<Closure, ex::schedule_result_t<scheduler_t>>;
      using pred_opstate_t  = ex::connect_result_t<CvSender, receiver<opstate>>;
      using inner_opstate_t = ex::connect_result_t<inner_sender_t, receiver<opstate>>;

      auto& _connect()
      {
        return inner_opstate_.emplace(ex::__emplace_from{
          [&]() noexcept
          { return ex::connect(closure_(ex::schedule(sched_)), receiver<opstate>{*this}); }});
      }

      scheduler_t _get_scheduler(CvSender const & sndr) noexcept
      {
        return ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr),
                                                             ex::get_env(this->rcvr_));
      }

      std::size_t                    count_;
      Closure                        closure_;
      scheduler_t                    sched_;
      std::optional<pred_opstate_t>  pred_opstate_;
      std::optional<inner_opstate_t> inner_opstate_;
    };

    template <class Sender, class Closure>
    struct sender
    {
      using sender_concept = ex::sender_t;

      using completion_signatures = ex::completion_signatures<ex::set_value_t(),
                                                              ex::set_stopped_t(),
                                                              ex::set_error_t(std::exception_ptr),
                                                              ex::set_error_t(cudaError_t)>;

      template <ex::__decays_to<sender> Self, ex::receiver Receiver>
        requires(ex::sender_to<Sender, Receiver>)
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver r)
        -> nvexec::_strm::_repeat_n::opstate<Sender, Closure, Receiver>
      {
        return nvexec::_strm::_repeat_n::opstate<Sender, Closure, Receiver>(
          static_cast<Self&&>(self).sndr_,
          static_cast<Self&&>(self).closure_,
          static_cast<Receiver&&>(r),
          self.count_);
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      [[nodiscard]]
      auto get_env() const noexcept -> ex::env_of_t<Sender const &>
      {
        return ex::get_env(sndr_);
      }

      Sender      sndr_;
      Closure     closure_;
      std::size_t count_{};
    };
  }  // namespace _repeat_n

  template <>
  struct transform_sender_for<::repeat_n_t>
  {
    template <class Env, class Data, class Sender>
    auto operator()(Env const &, ::repeat_n_t, Data&& data, Sender sndr) const
    {
      auto& [closure, count] = data;
      using closure_t        = decltype(closure);

      return _strm::_repeat_n::sender<Sender, closure_t>(static_cast<Sender&&>(sndr),
                                                         ex::__forward_like<Data>(closure),
                                                         count);
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
#if 0
  return ex::on(computer,
                exec::repeat_n(ex::just()  //
                                 | ex::bulk(ex::par, accessor.cells, update_h(accessor))
                                 | ex::bulk(ex::par, accessor.cells, update_e(time, dt, accessor)),
                               n_iterations))
       | ex::then(dump_vtk(write_results, accessor));
#else
  return ex::just()
       | ex::on(computer,
                repeat_n(n_iterations,
                         ex::bulk(ex::par, accessor.cells, update_h(accessor))
                           | ex::bulk(ex::par, accessor.cells, update_e(time, dt, accessor))))
       | ex::then(dump_vtk(write_results, accessor));
#endif
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
