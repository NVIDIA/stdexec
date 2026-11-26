/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#  include "__completion_signatures_of.hpp"
#  include "__diagnostics.hpp"
#  include "__execution_legacy.hpp"  // IWYU pragma: export
#  include "__meta.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_facade.hpp"
#  include "__senders.hpp"  // IWYU pragma: keep for __check_well_formed
#  include "__transform_completion_signatures.hpp"

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.bulk]
  namespace __bulk
  {
    //! Wrapper for a policy object.
    //!
    //! If we wrap a standard execution policy, we don't store anything, as we
    //! know the type.
    //! Stores the execution policy object if it's a non-standard one.
    //! Provides a way to query the execution policy object.
    template <class _Pol>
    struct __policy_wrapper
    {
      _Pol __pol_;

      /*implicit*/ constexpr __policy_wrapper(_Pol __pol)
        : __pol_{__pol}
      {}

      [[nodiscard]]
      constexpr _Pol const & __get() const noexcept
      {
        return __pol_;
      }
    };

    template <>
    struct __policy_wrapper<sequenced_policy>
    {
      /*implicit*/ constexpr __policy_wrapper(sequenced_policy const &) {}

      [[nodiscard]]
      constexpr sequenced_policy const & __get() const noexcept
      {
        return seq;
      }
    };

    template <>
    struct __policy_wrapper<parallel_policy>
    {
      /*implicit*/ constexpr __policy_wrapper(parallel_policy const &) {}

      [[nodiscard]]
      constexpr parallel_policy const & __get() const noexcept
      {
        return par;
      }
    };

    template <>
    struct __policy_wrapper<parallel_unsequenced_policy>
    {
      /*implicit*/ constexpr __policy_wrapper(parallel_unsequenced_policy const &) {}

      [[nodiscard]]
      constexpr parallel_unsequenced_policy const & __get() const noexcept
      {
        return par_unseq;
      }
    };

    template <>
    struct __policy_wrapper<unsequenced_policy>
    {
      /*implicit*/ constexpr __policy_wrapper(unsequenced_policy const &) {}

      [[nodiscard]]
      constexpr unsequenced_policy const & __get() const noexcept
      {
        return unseq;
      }
    };

    template <class _Pol, class _Shape, class _Fun>
    struct __data
    {
      STDEXEC_ATTRIBUTE(no_unique_address)
      __policy_wrapper<_Pol> __pol_;
      _Shape                 __shape_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Fun __fun_;
    };

    template <class _Pol, class _Shape, class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __data(_Pol const &, _Shape, _Fun) -> __data<_Pol, _Shape, _Fun>;

    template <class _AlgoTag>
    struct __generic_bulk_t  // NOLINT(bugprone-crtp-constructor-accessibility)
    {
      template <sender _Sender,
                typename _Policy,
                __std::integral           _Shape,
                __std::copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto
      operator()(_Sender&& __sndr, _Policy&& __pol, _Shape __shape, _Fun __fun) const -> auto
      {
        return STDEXEC::__check_well_formed(__facade::__sender{
          _AlgoTag(),
          __data{__pol, __shape, static_cast<_Fun&&>(__fun)},
          static_cast<_Sender&&>(__sndr)
        });
      }

      template <typename _Policy, __std::integral _Shape, __std::copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Policy&& __pol, _Shape __shape, _Fun __fun) const
      {
        return __closure(*this,
                         static_cast<_Policy&&>(__pol),
                         static_cast<_Shape&&>(__shape),
                         static_cast<_Fun&&>(__fun));
      }

      template <sender _Sender, __std::integral _Shape, __std::copy_constructible _Fun>
      [[deprecated("The bulk algorithm now requires an execution policy such "
                   "as " STDEXEC_PP_STRINGIZE(STDEXEC) "::par as an argument.")]]
      STDEXEC_ATTRIBUTE(host, device) auto
      operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const
      {
        return (*this)(static_cast<_Sender&&>(__sndr),
                       par,
                       static_cast<_Shape&&>(__shape),
                       static_cast<_Fun&&>(__fun));
      }

      template <__std::integral _Shape, __std::copy_constructible _Fun>
      [[deprecated("The bulk algorithm now requires an execution policy such "
                   "as " STDEXEC_PP_STRINGIZE(STDEXEC) "::par as an argument.")]]
      STDEXEC_ATTRIBUTE(always_inline) auto operator()(_Shape __shape, _Fun __fun) const
      {
        return (*this)(par, static_cast<_Shape&&>(__shape), static_cast<_Fun&&>(__fun));
      }
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Fun>
    struct __as_bulk_chunked_fn
    {
      _Fun __fun_;

      template <class _Shape, class... _Args>
      constexpr void operator()(_Shape __begin, _Shape __end, _Args&... __args)
        noexcept(__nothrow_callable<_Fun&, _Shape, _Args&...>)
      {
        for (; __begin != __end; ++__begin)
        {
          __fun_(__begin, __args...);
        }
      }
    };

    template <class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __as_bulk_chunked_fn(_Fun) -> __as_bulk_chunked_fn<_Fun>;

    // Calls the function with the shape and the child's values: once with the whole range,
    // for bulk_chunked, and once per index otherwise
    template <class _AlgoTag, class _Data, class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void __run(_Data& __data, _Args&... __args)
    {
      using __shape_t = decltype(__data.__shape_);
      if constexpr (__same_as<_AlgoTag, bulk_chunked_t>)
      {
        if (__shape_t{} < __data.__shape_)
        {
          __data.__fun_(__shape_t(), __shape_t(__data.__shape_), __args...);
        }
      }
      else
      {
        auto const __shape = __data.__shape_;
        for (__shape_t __i{}; __i != __shape; ++__i)
        {
          __data.__fun_(__shape_t(__i), __args...);
        }
      }
    }

    // bulk(sndr, pol, shape, fn), bulk_chunked(...) and bulk_unchunked(...): when the child
    // completes with values, the function is called with the shape and the values, and then
    // the values are sent. Other completions of the child are passed through. This calls the
    // function sequentially. Schedulers customize the algorithms to parallelize it. (bulk is
    // lowered to bulk_chunked by its tag's transform_sender, so that customizations of
    // bulk_chunked apply to it. It calls the function once per index.)
    template <class _AlgoTag>
    struct __algo : __facade::__defaults
    {
      // The child sees the forwarded environment of the receiver
      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __env_for(__facade::__child<0>, _Sender const &, _Env&& __env) noexcept
        -> __fwd_env_t<_Env>
      {
        return __fwd_env(static_cast<_Env&&>(__env));
      }

      template <class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(__facade::__child<0>, _State const & __state) noexcept
        -> decltype(auto)
      {
        return __fwd_env(STDEXEC::get_env(__state.__rcvr_));
      }

      // The child completed with values: call the function, and then send them.
      template <class _State, class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __complete(__facade::__child<0>,
                                       _State& __state,
                                       set_value_t,
                                       _Args&&... __args) noexcept
      {
        using __data_t     = decltype(__state.__data_);
        using __shape_t    = decltype(__data_t::__shape_);
        using __fun_t      = decltype(__data_t::__fun_);
        using __arg_pack_t = __if_c<__same_as<_AlgoTag, bulk_chunked_t>,
                                    __tuple<__shape_t, __shape_t, _Args&...>,
                                    __tuple<__shape_t, _Args&...>>;

        if constexpr (!__applicable<__fun_t&, __arg_pack_t>)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
                         _WHERE_(_IN_ALGORITHM_, _AlgoTag),
                         _WITH_FUNCTION_(__fun_t&),
                         __mapply<__qf<_WITH_ARGUMENTS_>, __arg_pack_t>>());
        }
        else if constexpr (__nothrow_applicable<__fun_t&, __arg_pack_t>)
        {
          __bulk::__run<_AlgoTag>(__state.__data_, __args...);
          return __facade::__send(__state.__rcvr_, set_value_t(), static_cast<_Args&&>(__args)...);
        }
        else
        {
          using __result_t =
            __facade::__union_t<__facade::__effects<set_value_t(_Args...)>,
                                __facade::__effects<set_error_t(std::exception_ptr)>>;
          STDEXEC_TRY
          {
            __bulk::__run<_AlgoTag>(__state.__data_, __args...);
          }
          STDEXEC_CATCH_ALL
          {
            return __result_t(
              __facade::__send(__state.__rcvr_, set_error_t(), std::current_exception()));
          }
          return __result_t(
            __facade::__send(__state.__rcvr_, set_value_t(), static_cast<_Args&&>(__args)...));
        }
      }

      // The child's other completions are passed through.
      using __facade::__defaults::__complete;
    };
  }  // namespace __bulk

  //! @brief A pipeable sender adaptor that applies a function to each index
  //!        in @c [0, shape) under a given execution policy.
  //!
  //! @c bulk is the parallel-loop primitive of the sender model. You give
  //! it a sender, an execution policy (e.g. @c stdexec::par), an integral
  //! shape, and a callable; you get back a sender that, when started,
  //! invokes `f(i, vs...)` for every @c i in @c [0, shape) — where
  //! @c vs... are the predecessor's value-completion datums. The
  //! execution policy controls whether the invocations may run in
  //! parallel.
  //!
  //! See [exec.bulk] in the C++26 working draft.
  //!
  //! The signature of the operator overloads (inherited from a detail
  //! base) is:
  //!
  //! @code{.cpp}
  //! template <sender Sender, /* execution_policy */ Policy,
  //!           integral Shape, copy_constructible Fun>
  //!   auto operator()(Sender&& sndr, Policy&& pol,
  //!                   Shape shape, Fun fun) const -> sender auto;   // direct
  //!
  //! template <class Policy, integral Shape, copy_constructible Fun>
  //!   auto operator()(Policy&& pol, Shape shape, Fun fun) const;    // closure
  //! @endcode
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::bulk(sndr, stdexec::par, 1024, fn);
  //! auto s2 = sndr | stdexec::bulk(stdexec::par, 1024, fn);
  //! @endcode
  //!
  //! **Completion signatures.**
  //!
  //! @c bulk forwards the predecessor's completion signatures, optionally
  //! adding @c set_error_t(std::exception_ptr) if invoking @c fn may throw
  //! (or if internal allocation may throw):
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged from sndr
  //! set_error_t(Es)...               // forwarded unchanged from sndr
  //! set_error_t(std::exception_ptr)  // added if fn may throw
  //! set_stopped_t()                  // forwarded unchanged (if present)
  //! @endcode
  //!
  //! @c fn is invoked with the *value-completion datums* of @c sndr
  //! preserved across all invocations — every iteration sees the same
  //! @c vs...
  //!
  //! **Execution policy.**
  //!
  //! The policy argument follows the `<execution>` conventions:
  //! @c stdexec::seq for sequenced execution, @c stdexec::par for
  //! permitted-parallel, @c stdexec::par_unseq for permitted parallel and
  //! vectorized. A custom scheduler's domain may interpret these
  //! differently — e.g. a GPU domain may lower @c par to a CUDA kernel
  //! launch.
  //!
  //! **Implementation note: lowering to** @c bulk_chunked **.**
  //!
  //! Internally, @c bulk is implemented in terms of
  //! @ref bulk_chunked_t — its @c transform_sender member rewrites
  //! `bulk(sndr, pol, n, f)` into a @c bulk_chunked over the same
  //! shape with @c f wrapped in a per-chunk loop. If a domain customizes
  //! @c bulk_chunked, @c bulk picks up that customization automatically.
  //!
  //! @see stdexec::bulk_chunked    — explicit-chunk variant
  //! @see stdexec::bulk_unchunked  — strict per-index variant (no chunking
  //!                                 allowed)
  //! @see stdexec::when_all        — concurrent composition without an index
  //!                                 space
  struct bulk_t : __bulk::__generic_bulk_t<bulk_t>
  {
    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      auto& [__tag, __data, __child] = __sndr;
      // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` is customized, we
      // will see the customization.
      return bulk_chunked(STDEXEC::__forward_like<_Sender>(__child),
                          __data.__pol_.__get(),
                          __data.__shape_,
                          __bulk::__as_bulk_chunked_fn(
                            STDEXEC::__forward_like<_Sender>(__data).__fun_));
    }
  };

  //! @brief A pipeable sender adaptor that invokes a function with chunked
  //!        sub-ranges of an integer index space.
  //!
  //! Where @ref bulk_t passes a *single index* to its callable,
  //! @c bulk_chunked passes a *half-open range* @c [begin, end) covering
  //! some subset of @c [0, shape). The implementation may split @c [0,
  //! shape) into any number of chunks (including one chunk equal to the
  //! whole range, or @c shape chunks of one element each) — the only
  //! guarantee is that every index in @c [0, shape) is covered by exactly
  //! one chunk.
  //!
  //! See [exec.bulk] in the C++26 working draft.
  //!
  //! The signature of the operator overloads (inherited from a detail
  //! base) is:
  //!
  //! @code{.cpp}
  //! template <sender Sender, /* execution_policy */ Policy,
  //!           integral Shape, copy_constructible Fun>
  //!   auto operator()(Sender&& sndr, Policy&& pol,
  //!                   Shape shape, Fun fun) const -> sender auto;   // direct
  //!
  //! template <class Policy, integral Shape, copy_constructible Fun>
  //!   auto operator()(Policy&& pol, Shape shape, Fun fun) const;    // closure
  //! @endcode
  //!
  //! The callable is invoked as `fun(begin, end, vs...)`, where
  //! @c vs... are the predecessor's value-completion datums (shared across
  //! all chunks).
  //!
  //! **When to use** @c bulk_chunked **vs.** @c bulk **:**
  //!
  //! Use @c bulk when the per-iteration body is small and the loop is the
  //! payload — @c bulk's lowering to @c bulk_chunked will let the runtime
  //! pick chunk sizes for you. Use @c bulk_chunked directly when the body
  //! benefits from per-chunk amortization (allocations, accumulators,
  //! vectorization setup) that you want to do once per chunk rather than
  //! once per index.
  //!
  //! @see stdexec::bulk            — index-at-a-time variant (lowers to this)
  //! @see stdexec::bulk_unchunked  — strict per-index variant (no chunking)
  struct bulk_chunked_t : __bulk::__generic_bulk_t<bulk_chunked_t>
  {};

  //! @brief A pipeable sender adaptor that invokes a function once per index
  //!        in @c [0, shape), *without* permission to chunk.
  //!
  //! @c bulk_unchunked has the same per-index invocation pattern as
  //! @ref bulk_t — `fun(i, vs...)` for every @c i — but explicitly
  //! forbids the implementation from combining multiple indices into a
  //! single call. Spec-recommended (but not required) practice is for
  //! each iteration to run on a *distinct* execution agent.
  //!
  //! Use this only when the body of the loop has per-iteration state or
  //! synchronization that *cannot* be batched — e.g., per-thread-local
  //! accumulators, per-index hardware resources, observable side effects
  //! that must be one-per-index. For ordinary parallel loops, prefer
  //! @ref bulk_t (which lowers to @c bulk_chunked and lets the runtime
  //! make chunk-size decisions).
  //!
  //! See [exec.bulk] in the C++26 working draft.
  //!
  //! The signature of the operator overloads (inherited from a detail
  //! base) is:
  //!
  //! @code{.cpp}
  //! template <sender Sender, /* execution_policy */ Policy,
  //!           integral Shape, copy_constructible Fun>
  //!   auto operator()(Sender&& sndr, Policy&& pol,
  //!                   Shape shape, Fun fun) const -> sender auto;   // direct
  //!
  //! template <class Policy, integral Shape, copy_constructible Fun>
  //!   auto operator()(Policy&& pol, Shape shape, Fun fun) const;    // closure
  //! @endcode
  //!
  //! @see stdexec::bulk            — chunking-permitted index-at-a-time variant
  //! @see stdexec::bulk_chunked    — explicit-chunk variant
  struct bulk_unchunked_t : __bulk::__generic_bulk_t<bulk_unchunked_t>
  {};

  //! @brief The customization point object for the @c bulk sender adaptor.
  //!
  //! @c bulk is an instance of @c bulk_t. See @ref bulk_t for the full
  //! description, the lowering to @c bulk_chunked, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr bulk_t bulk{};

  //! @brief The customization point object for the @c bulk_chunked sender
  //! adaptor.
  //!
  //! @c bulk_chunked is an instance of @c bulk_chunked_t. See
  //! @ref bulk_chunked_t for the full description.
  //!
  //! @hideinitializer
  inline constexpr bulk_chunked_t bulk_chunked{};

  //! @brief The customization point object for the @c bulk_unchunked sender
  //! adaptor.
  //!
  //! @c bulk_unchunked is an instance of @c bulk_unchunked_t. See
  //! @ref bulk_unchunked_t for the full description and when to reach for
  //! it.
  //!
  //! @hideinitializer
  inline constexpr bulk_unchunked_t bulk_unchunked{};

  template <>
  struct __facade::__algorithm<bulk_t> : __bulk::__algo<bulk_t>
  {};

  template <>
  struct __facade::__algorithm<bulk_chunked_t> : __bulk::__algo<bulk_chunked_t>
  {};

  template <>
  struct __facade::__algorithm<bulk_unchunked_t> : __bulk::__algo<bulk_unchunked_t>
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
