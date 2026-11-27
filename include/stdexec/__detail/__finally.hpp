/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after execution_fwd.hpp
#  include "__continues_on.hpp"
#  include "__domain.hpp"
#  include "__env.hpp"
#  include "__optional.hpp"
#  include "__schedulers.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__senders.hpp"
#  include "__storage.hpp"
#  include "__tuple.hpp"
#  include "__utility.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#    include <utility>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  struct _THE_FINAL_SENDER_MUST_BE_A_SENDER_OF_VOID_;
  struct _INVALID_ARGUMENT_TO_THE_FINALLY_ALGORITHM_;

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct __finally_t
  {
    template <sender _Initial, sender _Final>
    constexpr auto operator()(_Initial&& __initial, _Final&& __final) const  //
      -> auto
    {
      return STDEXEC::__check_well_formed(__facade::__sender{__finally_t(),
                                                             __(),
                                                             static_cast<_Initial&&>(__initial),
                                                             static_cast<_Final&&>(__final)});
    }

    template <sender _Final>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Final&& __final) const
    {
      return __closure(*this, static_cast<_Final&&>(__final));
    }
  };

  STDEXEC_MODULE_EXPORT_AUTHORING
  inline constexpr __finally_t __finally_{};

  namespace __final
  {
    struct __algo;

    // Does the initial sender complete where it starts, however it completes? Then the final
    // sender starts where the operation started, which the receiver's environment says.
    template <class _Initial, class _Env>
    inline constexpr bool __initial_stays_put_v = __stays_put<__decay_t<_Initial>, _Env>;

    // If the final sender never completes with a value, the results of the initial sender are
    // never sent, so they are not saved.
    template <bool _SavesResults>
    struct __storage_fn
    {
      template <class _Completions>
      using __f = __mapply_q<__results_storage, _Completions>;
    };

    template <>
    struct __storage_fn<false>
    {
      template <class _Completions>
      using __f = __results_storage<>;
    };

    template <class _Sender, class _Receiver>
    struct __state
    {
      using __env_t     = env_of_t<_Receiver>;
      using __initial_t = __facade::__nth_child_of_t<_Sender, 0>;
      using __final_t   = __facade::__nth_child_of_t<_Sender, 1>;
      using __env2_t    = decltype(__trnsfr::__mk_env2(__declval<__decay_t<__initial_t> const &>(),
                                                    __declval<__decay_t<__env_t> const &>()));
      using __initial_rcvr_t =
        __facade::__receiver_t<__algo, __facade::__child<0>, __state, _Sender, _Receiver>;
      using __final_rcvr_t =
        __facade::__receiver_t<__algo, __facade::__child<1>, __state, _Sender, _Receiver>;
      using __initial_op_t = connect_result_t<__initial_t, __initial_rcvr_t>;
      using __final_op_t   = connect_result_t<__final_t, __final_rcvr_t>;

      static constexpr bool __saves_results =
        __facade::__completions_of_t<__algo, __facade::__child<1>, _Sender, _Receiver>::__count(
          set_value)
        != 0;
      using __storage_t = __storage_fn<__saves_results>::template __f<
        __facade::__completions_of_t<__algo, __facade::__child<0>, _Sender, _Receiver>>;
      static constexpr bool __initial_stays_put = __initial_stays_put_v<__initial_t, __env_t>;

      // The final sender's operation is connected first. The initial sender's is destroyed
      // before the final sender starts.
      constexpr explicit __state(_Sender&& __sndr, _Receiver&& __rcvr)
        noexcept(__nothrow_connectable<__final_t, __final_rcvr_t>
                 && __nothrow_connectable<__initial_t, __initial_rcvr_t>)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __env2_(__trnsfr::__mk_env2(__facade::__get_child<0>(__sndr), STDEXEC::get_env(__rcvr_)))
        , __final_op_(STDEXEC::connect(__facade::__get_child<1>(static_cast<_Sender&&>(__sndr)),
                                       __final_rcvr_t{this}))
      {
        __initial_op_.__emplace_from(STDEXEC::connect,
                                     __facade::__get_child<0>(static_cast<_Sender&&>(__sndr)),
                                     __initial_rcvr_t{this});
      }

      STDEXEC_IMMOVABLE(__state);

      _Receiver                  __rcvr_;
      __env2_t const             __env2_;
      __storage_t                __storage_{};
      __final_op_t               __final_op_;
      __optional<__initial_op_t> __initial_op_{};
    };

    // __finally_(initial, final): when the initial sender completes, its results are saved,
    // its operation is destroyed, and the final sender is started. When that completes with a
    // value, the saved results are sent. Its errors and stopped completions are sent instead.
    struct __algo : __facade::__defaults
    {
      static constexpr bool __connects_children = false;

      using __facade::__defaults::__env_for;
      using __facade::__defaults::__get_env;

      // The final sender sees where the initial sender completed (see __trnsfr::__mk_env2),
      // joined with the receiver's environment. Unless the initial sender completes where it
      // starts, that is not where the operation started, so the receiver's start scheduler and
      // domain are hidden from it. (The initial sender sees the receiver's environment.)
      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __env_for(__facade::__child<1>,
                                      _Sender const & __sndr,
                                      _Env&&          __env) noexcept
      {
        auto __env2 = __trnsfr::__mk_env2(__facade::__get_child<0>(__sndr), __env);
        if constexpr (__initial_stays_put_v<__facade::__nth_child_of_t<_Sender, 0>, _Env>)
          return __env::__join(std::move(__env2), static_cast<_Env&&>(__env));
        else
          return __env::__join(std::move(__env2), __hide_scheduler{static_cast<_Env&&>(__env)});
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__facade::__child<1>, _State const & __state) noexcept
      {
        using __env2_t = __decay_t<decltype(__state.__env2_)>;
        if constexpr (_State::__initial_stays_put)
          return __env::__join(__env2_t(__state.__env2_), STDEXEC::get_env(__state.__rcvr_));
        else
          return __env::__join(__env2_t(__state.__env2_),
                               __hide_scheduler{STDEXEC::get_env(__state.__rcvr_)});
      }

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__state<_Sender, _Receiver>, _Sender, _Receiver>)
          -> __state<_Sender, _Receiver>
      {
        return __state<_Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                           static_cast<_Receiver&&>(__rcvr));
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return __facade::__start_nested<__facade::__child<0>>(*__state.__initial_op_);
      }

      // The initial sender completed: save its results (if they can be sent), and start the
      // final sender. If decay-copying the results throws, the exception is saved as an error
      // instead, and sent after the final sender completes, like any other result. If they
      // cannot be decay-copied, that is a type error.
      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State&               __state,
                                       [[maybe_unused]] _Tag __tag,
                                       [[maybe_unused]]
                                       _Args&&... __args) noexcept
      {
        if constexpr (!_State::__saves_results)
        {
          return __algo::__start_final(__state);
        }
        else if constexpr (!__decay_copyable<_Args...>)
        {
          return __facade::__decay_copy_error<__finally_t, _Args...>();
        }
        else
        {
          if constexpr (__nothrow_decay_copyable<_Args...>)
          {
            __state.__storage_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
          }
          else
          {
            STDEXEC_TRY
            {
              __state.__storage_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
            }
            STDEXEC_CATCH_ALL
            {
              __state.__storage_.__emplace_from(__mktuple, set_error, std::current_exception());
            }
          }
          return __algo::__start_final(__state);
        }
      }

      // The final sender completed with a value: send the saved results.
      template <class _State, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<1>,
                                       _State& __state,
                                       set_value_t,
                                       _Args&&...) noexcept
      {
        if constexpr (sizeof...(_Args) != 0)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_TO_THE_FINALLY_ALGORITHM_),
                         _WHERE_(_IN_ALGORITHM_, __finally_t),
                         _WHY_(_THE_FINAL_SENDER_MUST_BE_A_SENDER_OF_VOID_),
                         _WITH_SENDER_<__decay_t<typename _State::__final_t>>>());
        }
        else
        {
          return __trnsfr::__send_saved(__state.__storage_, __state.__rcvr_);
        }
      }

      // The final sender failed or was stopped.
      template <class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<1>,
                                       _State& __state,
                                       _Tag    __tag,
                                       _Args&&... __args) noexcept
      {
        return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args&&>(__args)...);
      }

     private:
      // Destroys the initial sender's operation (its results, if they are needed, are saved),
      // and starts the final sender.
      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __start_final(_State& __state) noexcept
      {
        __state.__initial_op_.reset();
        return __facade::__start_nested<__facade::__child<1>>(__state.__final_op_);
      }
    };
  }  // namespace __final

  template <>
  struct __facade::__algorithm<__finally_t> : __final::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
