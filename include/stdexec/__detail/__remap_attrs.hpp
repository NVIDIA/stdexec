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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__completion_behavior.hpp"
#  include "__concepts.hpp"
#  include "__domain.hpp"
#  include "__gather_completion_signatures.hpp"
#  include "__get_completion_signatures.hpp"
#  include "__meta.hpp"
#  include "__query.hpp"
#  include "__schedulers.hpp"
#  include "__sender_concepts.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //! @brief Returns @c true if @c _Child is known, in environment @c _Env..., to
  //! complete through the @c _From channel only with arguments for which the
  //! metafunction @c _Pred yields @c true. Returns @c false if that cannot be
  //! determined (e.g., the child is a dependent sender and no environment was
  //! provided).
  template <class _From, class _Child, class _Pred, class... _Env>
  [[nodiscard]]
  consteval auto __all_completions_satisfy() noexcept -> bool
  {
    if constexpr (sender_in<_Child, _Env...>)
    {
      return __gather_completion_signatures_t<_From,
                                              __completion_signatures_of_t<_Child, _Env...>,
                                              _Pred,
                                              __qq<__mand_t>>::value;
    }
    else
    {
      return false;
    }
  }

  template <class _Fun>
  struct __nothrow_invocable_fn
  {
    template <class... _Args>
    using __f = __mbool<__nothrow_invocable<_Fun, _Args...>>;
  };

  //! @brief The attributes of a sender adaptor that does not introduce
  //! asynchrony, but that may deliver a child's completion on a different
  //! channel than the one the child used (e.g., @c upon_error turns an error
  //! completion into a value completion).
  //!
  //! @tparam _Map A policy type with a static member function template
  //! @code
  //!   template <class _Tag, class _From, class _Child, class... _Env>
  //!   static consteval bool __feeds() noexcept;
  //! @endcode
  //! that returns @c true if a @c _From completion of @c _Child can result in a
  //! @c _Tag completion of the adaptor.
  //!
  //! The completion scheduler for a channel of the adaptor is the child's
  //! completion scheduler for the one child channel that feeds it. If more than
  //! one child channel can feed it, the completion scheduler is unknown. The
  //! completion domain and completion behavior are the combination of those
  //! of all the child channels that feed it.
  //!
  //! The adaptor connects its child with the receiver's environment unchanged,
  //! so the child's attributes are queried with the same environment.
  template <class _Map, class _Child>
  struct __remap_attrs
  {
   private:
    // Whether a _From completion of the child can actually result in a _Tag
    // completion of the adaptor. A child channel that is known never to be used
    // does not count.
    template <class _Tag, class _From, class... _Env>
    static constexpr bool __feeds_v = _Map::template __feeds<_Tag, _From, _Child, _Env...>()
                                   && !__never_sends<_From, _Child, _Env...>;

    // The child channels that can feed the adaptor's _Tag channel, applied to
    // _Fn.
    template <class _Fn, class _Tag, class... _Env>
    using __sources_t =
      __mcall<__mconcat<_Fn>,
              __if_c<__feeds_v<_Tag, set_value_t, _Env...>, __mlist<set_value_t>, __mlist<>>,
              __if_c<__feeds_v<_Tag, set_error_t, _Env...>, __mlist<set_error_t>, __mlist<>>,
              __if_c<__feeds_v<_Tag, set_stopped_t, _Env...>, __mlist<set_stopped_t>, __mlist<>>>;

    template <class... _Env>
    struct __domain_fn
    {
      template <class... _From>
      using __f =
        __common_domain_t<__completion_domain_t<_From, env_of_t<_Child>, _Env const &...>...>;
    };

    template <class _Tag, class... _Env>
    using __source_t = __sources_t<__q<__mfront>, _Tag, _Env...>;

    template <class _Tag, class... _Env>
    static constexpr std::size_t __source_count_v = __sources_t<__msize, _Tag, _Env...>::value;

    template <class... _Env>
    struct __behavior_fn
    {
      template <class... _From>
      static constexpr auto __value() noexcept
      {
        if constexpr (sizeof...(_From) == 0)
        {
          return __completion_behavior::__unknown;
        }
        else
        {
          return __completion_behavior::__common(
            STDEXEC::__get_completion_behavior<_From, _Child, _Env...>()...);
        }
      }

      template <class... _From>
      using __f = decltype(__value<_From...>());
    };

   public:
    template <class _Tag, class... _Env>
      requires(__source_count_v<_Tag, _Env...> == 1)
           && __callable<get_completion_scheduler_t<__source_t<_Tag, _Env...>>,
                         env_of_t<_Child>,
                         _Env const &...>
    [[nodiscard]]
    constexpr auto query(get_completion_scheduler_t<_Tag>, _Env const &...__env) const noexcept
    {
      return get_completion_scheduler<__source_t<_Tag, _Env...>>(get_env(__child_), __env...);
    }

    template <class _Tag, class... _Env>
      requires(__source_count_v<_Tag, _Env...> != 0)
    [[nodiscard]]
    constexpr auto query(get_completion_domain_t<_Tag>, _Env const &...) const noexcept
      -> __ensure_valid_domain_t<__sources_t<__domain_fn<_Env...>, _Tag, _Env...>>
    {
      return {};
    }

    template <class _Tag, class... _Env>
    [[nodiscard]]
    constexpr auto query(__get_completion_behavior_t<_Tag>, _Env const &...) const noexcept
    {
      return __sources_t<__behavior_fn<_Env...>, _Tag, _Env...>{};
    }

    template <class _Attrs = env_of_t<_Child>, __forwarding_query _Query, class... _Args>
      requires(!__completion_query<_Query>) && __queryable_with<_Attrs, _Query, _Args...>
    [[nodiscard]]
    constexpr auto query(_Query, _Args &&...__args) const
      noexcept(__nothrow_queryable_with<_Attrs, _Query, _Args...>)
        -> __query_result_t<_Attrs, _Query, _Args...>
    {
      return get_env(__child_).query(_Query(), static_cast<_Args &&>(__args)...);
    }

    _Child const &__child_;
  };

  //! @brief A policy for @c __remap_attrs for adaptors like @c then,
  //! @c upon_error, and @c upon_stopped, which pass the @c _SetTag completions
  //! of the child to @c _Fun and complete with the result on the @c set_value
  //! channel (or with @c set_error if @c _Fun throws). Other completions are
  //! passed through unchanged.
  template <class _SetTag, class _Fun>
  struct __upon_channel_map
  {
    template <class _Tag, class _From, class _Child, class... _Env>
    [[nodiscard]]
    static consteval auto __feeds() noexcept -> bool
    {
      if constexpr (__same_as<_From, _SetTag>)
      {
        // Only check whether _Fun can throw when it matters, so that we don't
        // needlessly instantiate _Fun's call operator.
        if constexpr (__same_as<_Tag, set_error_t>)
        {
          return !STDEXEC::__all_completions_satisfy<_From,
                                                     _Child,
                                                     __nothrow_invocable_fn<_Fun>,
                                                     _Env...>();
        }
        else
        {
          return __same_as<_Tag, set_value_t>;
        }
      }
      else
      {
        return __same_as<_Tag, _From>;
      }
    }
  };
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
