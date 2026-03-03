/*
 * Copyright (c) 2023 Runner-2019
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

#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__optional.hpp"
#include "../stdexec/execution.hpp"

#include "completion_signatures.hpp"
#include "sequence.hpp"
#include "trampoline_scheduler.hpp"

#include <exception>
#include <type_traits>

namespace experimental::execution
{
  struct _EXPECTING_A_SENDER_OF_ONE_VALUE_THAT_IS_CONVERTIBLE_TO_BOOL_;
  struct _EXPECTING_A_SENDER_OF_VOID_;

  namespace __repeat
  {
    using namespace STDEXEC;

    struct repeat_t;
    struct repeat_until_t;

    template <class _Receiver>
    struct __opstate_base
    {
      constexpr explicit __opstate_base(_Receiver &&__rcvr) noexcept
        : __rcvr_{static_cast<_Receiver &&>(__rcvr)}
      {
        static_assert(__nothrow_constructible_from<trampoline_scheduler>,
                      "trampoline_scheduler c'tor is always expected to be noexcept");
      }

      virtual constexpr void __cleanup() noexcept = 0;
      virtual constexpr void __repeat() noexcept  = 0;

      _Receiver            __rcvr_;
      trampoline_scheduler __sched_{};

     protected:
      ~__opstate_base() noexcept = default;
    };

    template <class _Boolean, bool _Expected>
    concept __bool_constant = __decay_t<_Boolean>::value == _Expected;

    template <class _Receiver>
    struct __receiver
    {
      using receiver_concept = STDEXEC::receiver_t;

      template <class... _Booleans>
      constexpr void set_value(_Booleans &&...__bools) noexcept
      {
        if constexpr ((__bool_constant<_Booleans, true> && ...))
        {
          // Always done:
          __state_->__cleanup();
          STDEXEC::set_value(std::move(__state_->__rcvr_));
        }
        else if constexpr ((__bool_constant<_Booleans, false> && ...))
        {
          // Never done:
          __state_->__repeat();
        }
        else
        {
          // Mixed results:
          constexpr bool __is_nothrow = (std::is_nothrow_convertible_v<_Booleans, bool> && ...);
          STDEXEC_TRY
          {
            // If the child sender completed with true, we're done
            bool const __done = (static_cast<bool>(static_cast<_Booleans &&>(__bools)) && ...);
            if (__done)
            {
              __state_->__cleanup();
              STDEXEC::set_value(std::move(__state_->__rcvr_));
            }
            else
            {
              __state_->__repeat();
            }
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (!__is_nothrow)
            {
              __state_->__cleanup();
              STDEXEC::set_error(std::move(__state_->__rcvr_), std::current_exception());
            }
          }
        }
      }

      template <class _Error>
      constexpr void set_error(_Error &&__err) noexcept
      {
        STDEXEC_TRY
        {
          auto __err_copy = static_cast<_Error &&>(__err);  // make a local copy of the error...
          __state_->__cleanup();  // ... because this could potentially invalidate it.
          STDEXEC::set_error(std::move(__state_->__rcvr_), static_cast<_Error &&>(__err_copy));
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!__nothrow_decay_copyable<_Error>)
          {
            __state_->__cleanup();
            STDEXEC::set_error(std::move(__state_->__rcvr_), std::current_exception());
          }
        }
      }

      constexpr void set_stopped() noexcept
      {
        __state_->__cleanup();
        STDEXEC::set_stopped(std::move(__state_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Receiver>>
      {
        return __fwd_env(STDEXEC::get_env(__state_->__rcvr_));
      }

      __opstate_base<_Receiver> *__state_;
    };

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wtsan")

    template <class _Child, class _Receiver>
    struct __opstate final : __opstate_base<_Receiver>
    {
      using __receiver_t = __receiver<_Receiver>;
      using __bouncy_sndr_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler>, _Child &>;
      using __child_op_t = STDEXEC::connect_result_t<__bouncy_sndr_t, __receiver_t>;

      constexpr explicit __opstate(_Child __child, _Receiver __rcvr)
        noexcept(__nothrow_move_constructible<_Child> && noexcept(__connect()))
        : __opstate_base<_Receiver>(std::move(__rcvr))
        , __child_(std::move(__child))
      {
        __connect();
      }

      constexpr void start() noexcept
      {
        STDEXEC::start(*__child_op_);
      }

      constexpr auto __connect() noexcept(
        __nothrow_invocable<STDEXEC::schedule_t, trampoline_scheduler &>
        && __nothrow_invocable<sequence_t, schedule_result_t<trampoline_scheduler>, _Child &>
        && __nothrow_connectable<__bouncy_sndr_t, __receiver_t>) -> __child_op_t &
      {
        return __child_op_.__emplace_from(STDEXEC::connect,
                                          exec::sequence(STDEXEC::schedule(this->__sched_),
                                                         __child_),
                                          __receiver_t{this});
      }

      constexpr void __cleanup() noexcept final
      {
        __child_op_.reset();
      }

      constexpr void __repeat() noexcept final
      {
        STDEXEC_TRY
        {
          STDEXEC::start(__connect());
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!noexcept(__connect()))
          {
            STDEXEC::set_error(static_cast<_Receiver &&>(this->__rcvr_), std::current_exception());
          }
        }
      }

      _Child                            __child_;
      STDEXEC::__optional<__child_op_t> __child_op_;
    };

    template <class _Child, class _Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __opstate(_Child, _Receiver) -> __opstate<_Child, _Receiver>;

    STDEXEC_PRAGMA_POP()

    struct __repeat_until_impl : __sexpr_defaults
    {
      template <class _Child>
      static constexpr auto __transform_values = []<class... _Args>()
      {
        if constexpr (sizeof...(_Args) != 1 || (!__std::convertible_to<_Args, bool> || ...))
        {
          return exec::throw_compile_time_error<
            _WHAT_(_INVALID_ARGUMENT_),
            _WHERE_(_IN_ALGORITHM_, repeat_until_t),
            _WHY_(_EXPECTING_A_SENDER_OF_ONE_VALUE_THAT_IS_CONVERTIBLE_TO_BOOL_),
            _WITH_PRETTY_SENDER_<_Child>>();
        }
        else if constexpr ((__bool_constant<_Args, false> && ...))
        {
          return STDEXEC::completion_signatures{};
        }
        else if constexpr ((std::is_nothrow_convertible_v<_Args, bool> && ...))
        {
          return STDEXEC::completion_signatures<set_value_t()>();
        }
        else
        {
          return STDEXEC::completion_signatures<set_value_t(), set_error_t(std::exception_ptr)>();
        }
      };

      static constexpr auto __transform_errors = []<class _Error>() noexcept
      {
        if constexpr (__nothrow_decay_copyable<_Error> || __decays_to<_Error, std::exception_ptr>)
        {
          return STDEXEC::completion_signatures<set_error_t(_Error)>();
        }
        else
        {
          return STDEXEC::completion_signatures<set_error_t(_Error),
                                                set_error_t(std::exception_ptr)>();
        }
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        using __child_t   = __child_of<_Sender>;
        using __bouncer_t = schedule_result_t<trampoline_scheduler>;

        STDEXEC_COMPLSIGS_LET(__completions, get_completion_signatures<__child_t, _Env...>())
        {
          using __eptr_completion_t        = set_error_t(std::exception_ptr);
          constexpr auto __eptr_completion = (__eptr_completion_t *) nullptr;
          constexpr auto __sigs =
            exec::transform_completion_signatures(__completions,
                                                  __transform_values<__child_t>,
                                                  __transform_errors);
          // The repeat_until sender is a dependent sender if one of the following is
          // true:
          //   - the child sender is a dependent sender, or
          //   - the trampoline scheduler's sender is a dependent sender, or
          //   - sizeof...(_Env) == 0 and the child sender does not have a
          //     set_error(exception_ptr) completion.
          constexpr bool __is_dependent = (sizeof...(_Env) == 0)
                                       && (dependent_sender<__bouncer_t>
                                           || !__sigs.__contains(__eptr_completion));
          if constexpr (__is_dependent)
          {
            return exec::throw_compile_time_error<dependent_sender_error,
                                                  _WITH_PRETTY_SENDER_<__child_t>>();
          }
          else
          {
            constexpr bool __has_nothrow_connect =
              (__nothrow_connectable<__child_t, __receiver_archetype<_Env>> || ...);
            constexpr auto __eptr_sigs    = __eptr_completion_unless<__has_nothrow_connect>();
            constexpr auto __bouncer_sigs = exec::transform_completion_signatures(
              get_completion_signatures<__bouncer_t, _Env...>(),
              exec::ignore_completion());  // drop the set_value_t() completion from the
                                           // trampoline scheduler.

            return exec::concat_completion_signatures(__sigs, __eptr_sigs, __bouncer_sigs);
          }
        }
      };

      static constexpr auto __connect =
        []<class _Sender, class _Receiver>(_Sender &&__sndr, _Receiver __rcvr) noexcept(
          __nothrow_constructible_from<__opstate<__child_of<_Sender>, _Receiver>,
                                       __child_of<_Sender>,
                                       _Receiver>)
      {
        auto &[__tag, __ign, __child] = __sndr;
        return __opstate(STDEXEC::__forward_like<_Sender>(__child), std::move(__rcvr));
      };
    };

    struct repeat_until_t
    {
      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr) const
      {
        return __make_sexpr<repeat_until_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const
      {
        return __closure(*this);
      }
    };

    struct repeat_t
    {
      struct _never
      {
        STDEXEC_ATTRIBUTE(host, device, always_inline)
        constexpr std::false_type operator()() const noexcept
        {
          return {};
        }
      };

      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr) const
      {
        return __make_sexpr<repeat_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const
      {
        return __closure(*this);
      }

      template <class _CvSender, class _Env>
      static constexpr auto
      transform_sender(STDEXEC::set_value_t, _CvSender &&__sndr, _Env const &) noexcept
      {
        using namespace STDEXEC;
        using __child_t               = __child_of<_CvSender>;
        using __values_t              = value_types_of_t<__child_t, _Env, __mlist, __mlist>;
        auto &[__tag, __ign, __child] = __sndr;

        if constexpr (__same_as<__values_t, __mlist<>> || __same_as<__values_t, __mlist<__mlist<>>>)
        {
          return repeat_until_t()(then(static_cast<__child_t &&>(__child), _never{}));
        }
        else
        {
          return __not_a_sender<_WHAT_(_INVALID_ARGUMENT_, _EXPECTING_A_SENDER_OF_VOID_),
                                _WHERE_(_IN_ALGORITHM_, repeat_until_t),
                                _WITH_PRETTY_SENDER_<__child_t>,
                                _WITH_ENVIRONMENT_(_Env)>();
        }
      }
    };
  }  // namespace __repeat

  using __repeat::repeat_t;
  inline constexpr repeat_t repeat{};

  using __repeat::repeat_until_t;
  inline constexpr repeat_until_t repeat_until{};

  /// deprecated interfaces
  using repeat_effect_t [[deprecated("use exec::repeat_t instead")]]             = repeat_t;
  using repeat_effect_until_t [[deprecated("use exec::repeat_until_t instead")]] = repeat_until_t;
  [[deprecated("use exec::repeat instead")]]
  inline constexpr repeat_t const &repeat_effect = repeat;
  [[deprecated("use exec::repeat_until instead")]]
  inline constexpr repeat_until_t const &repeat_effect_until = repeat_until;
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC
{
  template <>
  struct __sexpr_impl<exec::repeat_t> : exec::__repeat::__repeat_until_impl
  {};

  template <>
  struct __sexpr_impl<exec::repeat_until_t> : exec::__repeat::__repeat_until_impl
  {};
}  // namespace STDEXEC
