/*
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

#include "__basic_sender.hpp"
#include "__completion_behavior.hpp"
#include "__finally.hpp"
#include "__schedulers.hpp"
#include "__senders.hpp"
#include "__unstoppable.hpp"

namespace STDEXEC
{
  struct _CANNOT_MAKE_SENDER_AFFINE_TO_THE_CURRENT_SCHEDULER_
  {};
  struct _THE_SCHEDULER_IN_THE_CURRENT_EXECUTION_ENVIRONMENT_IS_NOT_INFALLIBLE_
  {};

  namespace __affine_on
  {
    // For a given completion tag, a sender is "already affine" if either it doesn't send
    // that tag, or if its completion behavior for that tag is already "inline" or
    // "__asynchronous_affine".
    template <class _Tag, class _Sender, class _Env>
    concept __already_affine = (!__sends<_Tag, _Sender, _Env>)
                            || (__get_completion_behavior<_Tag, _Sender, _Env>()
                                >= __completion_behavior::__asynchronous_affine);

    // For the purpose of the affine_on algorithm, a sender that is "already affine" for
    // all three of the standard completion tags does not need to be adapted to become
    // affine.
    template <class _Sender, class _Env>
    concept __is_affine = __already_affine<set_value_t, _Sender, _Env>
                       && __already_affine<set_error_t, _Sender, _Env>
                       && __already_affine<set_stopped_t, _Sender, _Env>;
  }  // namespace __affine_on

  struct affine_on_t
  {
    template <sender _Sender>
    constexpr auto operator()(_Sender &&__sndr) const -> __well_formed_sender auto
    {
      return __make_sexpr<affine_on_t>({}, static_cast<_Sender &&>(__sndr));
    }

    constexpr auto operator()() const noexcept
    {
      return __closure(*this);
    }

    template <class _Sender, class _Env>
    static constexpr auto transform_sender(set_value_t, _Sender &&__sndr, _Env const &__env)
    {
      static_assert(sender_expr_for<_Sender, affine_on_t>);
      auto &[__tag, __ign, __child] = __sndr;
      using __child_t               = decltype(__child);
      using __cv_child_t            = __copy_cvref_t<_Sender, __child_t>;
      using __sched_t = __call_result_or_t<get_scheduler_t, __not_a_scheduler<>, _Env const &>;

      if constexpr (!sender_in<__cv_child_t, _Env>)
      {  // NOLINT(bugprone-branch-clone)
        // The child sender is not compatible with the environment, so we can't adapt
        // it. Instead, just return the child as-is, which will result in an appropriate
        // compile-time error when the child sender is used.
        return STDEXEC::__forward_like<_Sender>(__child);
      }
      else if constexpr (__affine_on::__is_affine<__cv_child_t, _Env>)
      {
        // Check the child's completion behavior. If it is "inline" or "async_affine", then
        // we can just return the child sender. Otherwise, we need to wrap it.
        return STDEXEC::__forward_like<_Sender>(__child);
      }
      else if constexpr (__same_as<__sched_t, __not_a_scheduler<>>)
      {
        // The environment doesn't have a scheduler, so we can't adapt the sender to be
        // affine. Instead, return a type describing the problem.
        return __not_a_sender<_WHAT_(_CANNOT_MAKE_SENDER_AFFINE_TO_THE_CURRENT_SCHEDULER_),
                              _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_),
                              _WHERE_(_IN_ALGORITHM_, affine_on_t)>{};
      }
      else if constexpr (!__infallible_scheduler<__sched_t, __unstoppable_env_t<_Env>>)
      {
        // The scheduler in the environment isn't infallible, so we can't adapt the sender to be
        // affine. Instead, return a type describing the problem.
        return __not_a_sender<
          _WHAT_(_CANNOT_MAKE_SENDER_AFFINE_TO_THE_CURRENT_SCHEDULER_),
          _WHY_(_THE_SCHEDULER_IN_THE_CURRENT_EXECUTION_ENVIRONMENT_IS_NOT_INFALLIBLE_),
          _WHERE_(_IN_ALGORITHM_, affine_on_t),
          _WITH_SCHEDULER_(__sched_t)>{};
      }
      else
      {
        // The child sender is compatible with the environment, but isn't already affine, and
        // the environment has an infallible scheduler, so we can adapt the sender to run on
        // that scheduler, which will make it affine.
        return STDEXEC::__finally_(STDEXEC::__forward_like<_Sender>(__child),
                                   unstoppable(schedule(get_scheduler(__env))));
      }
    }
  };

  inline constexpr affine_on_t affine_on{};

  namespace __affine_on
  {
    template <class _Attrs>
    struct __attrs
    {
      template <class _Tag, class... _Env>
        requires __queryable_with<_Attrs, __get_completion_behavior_t<_Tag>, _Env const &...>
      constexpr auto query(__get_completion_behavior_t<_Tag>, _Env const &...) const noexcept
      {
        using __behavior_t =
          __query_result_t<_Attrs, __get_completion_behavior_t<_Tag>, _Env const &...>;

        // When the child sender completes inline, we can return "inline" here instead of
        // "__asynchronous_affine".
        if constexpr (__behavior_t::value == __completion_behavior::__inline_completion)
        {
          return __completion_behavior::__inline_completion;
        }
        else
        {
          return __completion_behavior::__asynchronous_affine;
        }
      }
    };
  }  // namespace __affine_on

  template <>
  struct __sexpr_impl<affine_on_t> : __sexpr_defaults
  {
    static constexpr auto __get_attrs =  //
      []<class _Child>(__ignore, __ignore, _Child const &) noexcept
    {
      return __affine_on::__attrs<env_of_t<_Child>>{};
    };
  };
}  // namespace STDEXEC
