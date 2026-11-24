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

#include "../../stdexec/execution.hpp"

#include "../../exec/completion_signatures.hpp"

#include "../detail/variant.cuh"

#include "common.cuh"

#include <cuda/std/tuple>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // continues_on as a facade algorithm. The stream_domain customizes continues_on(sndr, sch)
  // by re-tagging it with this tag (see transform_sender_for<continues_on_t> below).
  struct continues_on_t : stream_tag_base
  {};

  namespace _trnsfr
  {
    template <class Tag, class Storage, class... Args>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _continues_on_kernel(Storage* storage, Args... args)
    {
      ::new (storage) Storage();
      storage->template emplace<decayed_tuple_t<Tag, Args...>>(Tag(), static_cast<Args&&>(args)...);
    }

    // The key of the operation that moves to the scheduler
    struct sched_op
    {};

    struct algo;

    template <class Sender, class Receiver>
    struct state : _strm::opstate_base<Receiver>
    {
      using scheduler_t  = __decay_t<STDEXEC::__facade::__data_of_t<Sender>>;
      using child_t      = STDEXEC::__facade::__nth_child_of_t<Sender, 0>;
      using child_env_t  = terminal_stream_env_t<env_of_t<Receiver>>;
      using storage_t    = variant_storage_t<child_t, child_env_t>;
      using sched_sndr_t = schedule_result_t<scheduler_t const &>;
      using sched_rcvr_t = STDEXEC::__facade::__receiver_t<algo, sched_op, state, Sender, Receiver>;

      explicit state(Sender&& sndr, Receiver rcvr, std::unique_ptr<stream_provider> src_provider)
        : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr),
                                        STDEXEC::__facade::__get_data(sndr).ctx_)
        , src_provider_(std::move(src_provider))
        , sched_op_(STDEXEC::connect(STDEXEC::schedule(static_cast<scheduler_t const &>(
                                       STDEXEC::__facade::__get_data(sndr))),
                                     sched_rcvr_t{this}))
      {}

      ~state()
      {
        if (this->temp_storage_)
        {
          this->ctx_.managed_resource_->deallocate(this->temp_storage_, sizeof(storage_t));
        }
      }

      STDEXEC_IMMOVABLE(state);

      // Set when the work before the transition runs on another scheduler's stream
      std::unique_ptr<stream_provider> src_provider_;
      // The saved completion of the child: which one, and its arguments, if it has any
      unsigned int index_{0};
      storage_t*   storage_{nullptr};
      // A CUDA error, sent from the scheduler like the child's completions (see __fail)
      cudaError_t                                  error_{cudaSuccess};
      connect_result_t<sched_sndr_t, sched_rcvr_t> sched_op_;
    };

    template <class Tuple>
    struct _tuple_token;

    template <class Tag, class... Args>
    struct _tuple_token<::cuda::std::tuple<Tag, Args...>>
    {
      using __t = STDEXEC::__facade::__effects<Tag(Args...)>;
    };

    template <>
    struct _tuple_token<::cuda::std::tuple<set_noop>>
    {
      using __t = STDEXEC::__facade::__effects<>;
    };

    template <class Storage>
    struct _storage_traits;

    template <class... Tuples>
    struct _storage_traits<variant_t<Tuples...>>
    {
      using __token_t = STDEXEC::__facade::__union_t<typename _tuple_token<Tuples>::__t...>;

      template <class Tuple>
      static constexpr bool __holds = (__same_as<Tuple, Tuples> || ...);

      // Sends the saved completion
      template <class State>
      static auto __send(State& state) noexcept -> __token_t
      {
        _storage_traits::__send_at(state, std::index_sequence_for<Tuples...>{});
        return {};
      }

      template <class State, std::size_t... Is>
      static void __send_at(State& state, std::index_sequence<Is...>) noexcept
      {
        ((state.index_ == Is ? _storage_traits::__send_one<Tuples>(state) : void()), ...);
      }

      template <class Tuple, class State>
      static void __send_one(State& state) noexcept
      {
        if constexpr (!__same_as<Tuple, ::cuda::std::tuple<set_noop>>)
        {
          constexpr std::size_t nargs = ::cuda::std::tuple_size_v<Tuple> - 1;
          if constexpr (nargs == 0)
          {
            using tag_t = ::cuda::std::tuple_element_t<0, Tuple>;
            (void) _strm::_propagate(state, tag_t());
          }
          else if constexpr (__same_as<Tuple, decayed_tuple_t<set_error_t, cudaError_t>>)
          {
            // The error is the child's, if it sent one; otherwise it is the operation's own.
            if (state.storage_ == nullptr)
            {
              (void) _strm::_propagate(state, set_error_t(), state.error_);
            }
            else
            {
              _storage_traits::__send_args(state,
                                           state.storage_->template get<Tuple>(),
                                           std::make_index_sequence<nargs>{});
            }
          }
          else
          {
            _storage_traits::__send_args(state,
                                         state.storage_->template get<Tuple>(),
                                         std::make_index_sequence<nargs>{});
          }
        }
      }

      template <class State, class Tag, class... Bs, std::size_t... Is>
      static void __send_args(State&                          state,
                              ::cuda::std::tuple<Tag, Bs...>& tpl,
                              std::index_sequence<Is...>) noexcept
      {
        (void) _strm::_propagate(state, Tag(), std::move(::cuda::std::get<Is + 1>(tpl))...);
      }
    };

    // Copies the arguments of the child's completion to temporary storage on the stream.
    // If they are trivially copyable, a kernel constructs the storage on the device, to
    // avoid moving managed memory; otherwise it is constructed on the host and prefetched.
    template <class State, class Tag, class... Args>
    auto __save(State& state, Tag, Args&&... args) noexcept -> cudaError_t
    {
      using storage_t                    = State::storage_t;
      using tuple_t                      = decayed_tuple_t<Tag, Args...>;
      constexpr bool construct_on_device = trivially_copyable<__decay_t<Args>...>;

      void* memory = nullptr;
      STDEXEC_TRY
      {
        memory = state.ctx_.managed_resource_->allocate(sizeof(storage_t));
      }
      STDEXEC_CATCH_ALL
      {
        return cudaErrorMemoryAllocation;
      }
      state.temp_storage_ = memory;
      auto* storage       = static_cast<storage_t*>(memory);

      if constexpr (!construct_on_device)
      {
        ::new (storage) storage_t();
        storage->template emplace<tuple_t>(Tag(), static_cast<Args&&>(args)...);
      }

      auto fail = [&](cudaError_t status) noexcept
      {
        if constexpr (!construct_on_device)
        {
          storage->~storage_t();
        }
        return status;
      };

      int dev_id{};
      if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaGetDevice(&dev_id)); status != cudaSuccess)
      {
        return fail(status);
      }

      int concurrent_managed_access{};
      if (cudaError_t status = STDEXEC_LOG_CUDA_API(
            cudaDeviceGetAttribute(&concurrent_managed_access,
                                   cudaDevAttrConcurrentManagedAccess,
                                   dev_id));
          status != cudaSuccess)
      {
        return fail(status);
      }

      cudaStream_t stream = state.get_stream();

      if (concurrent_managed_access)
      {
        if (cudaError_t status = STDEXEC_LOG_CUDA_API(
              cudaMemPrefetchAsync(storage, sizeof(storage_t), dev_id, stream));
            status != cudaSuccess)
        {
          return fail(status);
        }
      }

      if constexpr (construct_on_device)
      {
        _continues_on_kernel<Tag, storage_t, __decay_t<Args>...>
          <<<1, 1, 0, stream>>>(storage, args...);
        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError()); status != cudaSuccess)
        {
          return fail(status);
        }
      }

      state.storage_ = storage;
      state.defer_temp_storage_destruction(storage);
      return cudaSuccess;
    }

    // continues_on(sndr, sch): the child runs on its own scheduler's stream, if that is a
    // different stream scheduler. When it completes, its results are copied to the stream
    // of sch, and an operation scheduled on sch sends them from there.
    struct algo : stream_algorithm
    {
      using stream_algorithm::__env_for;
      using stream_algorithm::__get_env;

      using __sched_value_t =
        STDEXEC::__facade::__via<sched_op,
                                 set_value_t,
                                 STDEXEC::__facade::__via<STDEXEC::__facade::__child<0>,
                                                          set_value_t,
                                                          STDEXEC::__facade::__from_start>>;

      // Customizations are found in the domain of the scheduler, which is where the values
      // complete even if the child sends none.
      using __dispatch_source = __sched_value_t;

      template <class Tag>
      using __empty_channel_source = __if_c<__same_as<Tag, set_value_t>, __sched_value_t, void>;

      template <class Sender, class Env>
      static auto __env_for(STDEXEC::__facade::__child<0>, Sender const &, Env&& env) noexcept
      {
        return _strm::make_terminal_stream_env(static_cast<Env&&>(env),
                                               static_cast<stream_provider*>(nullptr));
      }

      template <class State>
      static auto __get_env(STDEXEC::__facade::__child<0>, State const & state) noexcept
      {
        return _strm::make_terminal_stream_env(STDEXEC::get_env(state.rcvr_),
                                               state.src_provider_ ? state.src_provider_.get()
                                                                   : state.get_stream_provider());
      }

      template <class Sender, class Env>
      static auto __env_for(sched_op, Sender const &, Env&& env) noexcept
      {
        return _strm::make_stream_env(static_cast<Env&&>(env),
                                      static_cast<stream_provider*>(nullptr));
      }

      template <class State>
      static auto __get_env(sched_op, State const & state) noexcept
      {
        return state.make_env();
      }

      using stream_algorithm::__nested_sender;

      template <class Sender, class... Env>
      static auto __nested_sender(sched_op, Sender const & sndr, Env const &...)
      {
        return STDEXEC::schedule(STDEXEC::__facade::__get_data(sndr));
      }

      template <class Sender, class... Env>
      static auto __nested_attrs(sched_op, Sender const & sndr, Env const &...) noexcept
      {
        return STDEXEC::__facade::__schedule_attrs<decltype(STDEXEC::__facade::__get_data(sndr))>{
          STDEXEC::__facade::__get_data(sndr)};
      }

      // Work before the transition keeps running on its own scheduler's stream.
      template <class Sender, class Receiver>
      static auto
      __src_provider(Sender const & sndr, Receiver const & rcvr) -> std::unique_ptr<stream_provider>
      {
        using child_t = STDEXEC::__facade::__nth_child_of_t<Sender, 0>;
        using env_t   = terminal_stream_env_t<env_of_t<Receiver>>;
        auto attrs    = STDEXEC::__facade::__nested_attrs_of<algo>(STDEXEC::__facade::__child<0>(),
                                                                sndr,
                                                                STDEXEC::get_env(rcvr));
        auto env = algo::__env_for(STDEXEC::__facade::__child<0>(), sndr, STDEXEC::get_env(rcvr));
        if constexpr (stream_sender<child_t, env_t>
                      && __callable<get_completion_scheduler_t<set_value_t>,
                                    decltype(attrs) const &,
                                    decltype(env) const &>)
        {
          auto src_sched = get_completion_scheduler<set_value_t>(attrs, env);
          if constexpr (gpu_stream_scheduler<decltype(src_sched), env_t>)
          {
            if (src_sched != STDEXEC::__facade::__get_data(sndr))
            {
              return std::make_unique<stream_provider>(src_sched.ctx_);
            }
          }
        }
        return nullptr;
      }

      template <class Sender, class Receiver>
      static auto __get_state(Sender&& sndr, Receiver rcvr) -> state<Sender, Receiver>
      {
        auto src_provider = algo::__src_provider(sndr, rcvr);
        return state<Sender, Receiver>(static_cast<Sender&&>(sndr),
                                       static_cast<Receiver&&>(rcvr),
                                       std::move(src_provider));
      }

      template <class State, class Children>
      static auto __start(State& state, Children& children) noexcept
      {
        cudaError_t status = state.stream_provider_.status_;
        if (status == cudaSuccess && state.src_provider_)
        {
          status = state.src_provider_->status_;
        }
        return STDEXEC::__facade::__choose(
          status == cudaSuccess,
          [&]() noexcept { return children.__start_all(); },
          [&]() noexcept { return algo::__fail(state, status); });
      }

      // A CUDA error is sent from the scheduler too, so that all of the operation's
      // completions happen in the same place. (The schedule operation only completes its
      // receiver, so this works even when the stream could not be created.)
      template <class State>
      static auto __fail(State& state, cudaError_t status) noexcept
      {
        using storage_t = State::storage_t;
        state.error_    = status;
        state.storage_  = nullptr;
        state.index_ =
          __mapply<__mfind_i<decayed_tuple_t<set_error_t, cudaError_t>>, storage_t>::value;
        return STDEXEC::__facade::__start_nested<sched_op>(state.sched_op_);
      }

      // The child completed: save its results on the scheduler's stream, and move there.
      template <class State, class Tag, class... Args>
      static auto
      __complete(STDEXEC::__facade::__child<0>, State& state, Tag, Args&&... args) noexcept
      {
        using storage_t = State::storage_t;
        using tuple_t   = decayed_tuple_t<Tag, Args...>;
        using traits_t  = _storage_traits<storage_t>;

        if constexpr (!traits_t::template __holds<tuple_t>)
        {
          // (A completion that the child's signatures do not mention, as a cudaError_t
          // from an enqueue_opstate)
          return _strm::_propagate(state, Tag(), static_cast<Args&&>(args)...);
        }
        else
        {
          state.index_ = __mapply<__mfind_i<tuple_t>, storage_t>::value;
          if constexpr (sizeof...(Args) != 0)
          {
            if (cudaError_t status = _trnsfr::__save(state, Tag(), static_cast<Args&&>(args)...);
                status != cudaSuccess)
            {
              return algo::__fail(state, status);
            }
          }
          return STDEXEC::__facade::__start_nested<sched_op>(state.sched_op_);
        }
      }

      // On the scheduler: send the saved completion.
      template <class State>
      static auto __complete(sched_op, State& state, set_value_t) noexcept
      {
        return _storage_traits<typename State::storage_t>::__send(state);
      }

      // The schedule operation failed or was stopped.
      template <class State, class Tag, class... Args>
      static auto __complete(sched_op, State& state, Tag tag, Args&&... args) noexcept
      {
        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }
    };

  }  // namespace _trnsfr

  template <>
  struct transform_sender_for<STDEXEC::continues_on_t>
  {
    template <class Env, class Sched, class Sender>
    auto operator()(Env const &, __ignore, Sched sched, Sender&& sndr) const
    {
      static_assert(gpu_stream_scheduler<Sched, Env>);
      return STDEXEC::__facade::__sender{_strm::continues_on_t(),
                                         static_cast<Sched&&>(sched),
                                         static_cast<Sender&&>(sndr)};
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::continues_on_t>
  : nv::execution::_strm::_trnsfr::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
