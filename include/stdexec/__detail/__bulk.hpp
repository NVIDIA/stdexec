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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__completion_signatures_of.hpp"
#include "__diagnostics.hpp"
#include "__execution_legacy.hpp"  // IWYU pragma: export
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"  // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"

#include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.bulk]
  namespace __bulk
  {
    //! Wrapper for a policy object.
    //!
    //! If we wrap a standard execution policy, we don't store anything, as we know the type.
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
    struct __bulk_traits;

    template <>
    struct __bulk_traits<bulk_t>
    {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <>
    struct __bulk_traits<bulk_chunked_t>
    {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_chunked_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried = __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>,
                                          _Fun,
                                          _Shape,
                                          _Shape>;
    };

    template <>
    struct __bulk_traits<bulk_unchunked_t>
    {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_unchunked_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _AlgoTag, class _Fun, class _Shape, class _CvSender, class... _Env>
    using __with_error_invoke_t =
      __if<__value_types_t<
             __completion_signatures_of_t<_CvSender, _Env...>,
             __mtransform<__q<__decay_ref>,
                          typename __bulk_traits<_AlgoTag>::template __fun_curried<_Fun, _Shape>>,
             __q<__mand>>,
           completion_signatures<>,
           __eptr_completion_t>;

    template <class _AlgoTag, class _Fun, class _Shape, class _CvSender, class... _Env>
    using __completion_signatures = __transform_completion_signatures_t<
      __completion_signatures_of_t<_CvSender, _Env...>,
      __with_error_invoke_t<_AlgoTag, _Fun, _Shape, _CvSender, _Env...>>;

    template <class _AlgoTag>
    struct __generic_bulk_t  // NOLINT(bugprone-crtp-constructor-accessibility)
    {
      template <sender _Sender,
                typename _Policy,
                __std::integral           _Shape,
                __std::copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Sender&& __sndr, _Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __well_formed_sender auto
      {
        return __make_sexpr<_AlgoTag>(__data{__pol, __shape, static_cast<_Fun&&>(__fun)},
                                      static_cast<_Sender&&>(__sndr));
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

    template <class _Fun>
    struct __as_bulk_chunked_fn
    {
      _Fun __fun_;

      template <class _Shape, class... _Args>
      constexpr void operator()(_Shape __begin, _Shape __end, _Args&... __args)
        noexcept(__nothrow_callable<_Fun&, _Shape, decltype(__args)...>)
      {
        for (; __begin != __end; ++__begin)
        {
          __fun_(__begin, __args...);
        }
      }
    };

    template <class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __as_bulk_chunked_fn(_Fun) -> __as_bulk_chunked_fn<_Fun>;

    template <class _Child>
    struct __attrs : env<__fwd_env_t<env_of_t<_Child>>>
    {
      using __base_t = env<__fwd_env_t<env_of_t<_Child>>>;
      using __base_t::query;

      constexpr explicit __attrs(_Child const & __child) noexcept
        : __base_t{__fwd_env(STDEXEC::get_env(__child))}
      {}

      template <class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(__get_completion_behavior_t<set_value_t>, _Env&&...) const noexcept
      {
        return STDEXEC::__get_completion_behavior<set_value_t, _Child, _Env...>();
      }
    };

    template <class _AlgoTag>
    struct __impl_base : __sexpr_defaults
    {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      // Forward the child sender's environment (which contains completion scheduler)
      static constexpr auto __get_attrs =  //
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept
      {
        return __attrs{__child};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, _AlgoTag>);
        // TODO: port this to use constant evaluation
        return __completion_signatures<_AlgoTag,
                                       __fun_t<_Sender>,
                                       __shape_t<_Sender>,
                                       __child_of<_Sender>,
                                       _Env...>{};
      };
    };

    struct __chunked_impl : __impl_base<bulk_chunked_t>
    {
      //! This implements the core default behavior for `bulk_chunked`:
      //! When setting value, it calls the function with the entire range.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto __complete =
        []<class _Tag, class _State, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        if constexpr (__std::same_as<_Tag, set_value_t>)
        {
          // Intercept set_value and dispatch to the bulk operation.
          using __shape_t             = decltype(__state.__data_.__shape_);
          using __fun_t               = decltype(__state.__data_.__fun_);
          constexpr bool __is_nothrow = __nothrow_callable<__fun_t, __shape_t, __shape_t, _Args...>;
          STDEXEC_TRY
          {
            __state.__data_.__fun_(static_cast<__shape_t>(0), __state.__data_.__shape_, __args...);
            _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (!__is_nothrow)
            {
              STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            }
          }
        }
        else
        {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __unchunked_impl : __impl_base<bulk_unchunked_t>
    {
      //! This implements the core default behavior for `bulk_unchunked`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in concurrently. That is customized by the scheduler.
      static constexpr auto __complete =
        []<class _Tag, class _State, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        if constexpr (__std::same_as<_Tag, set_value_t>)
        {
          using __shape_t             = decltype(__state.__data_.__shape_);
          using __fun_t               = decltype(__state.__data_.__fun_);
          constexpr bool __is_nothrow = __nothrow_callable<__fun_t, __shape_t, _Args...>;
          auto const     __shape      = __state.__data_.__shape_;
          STDEXEC_TRY
          {
            for (__shape_t __i{}; __i != __shape; ++__i)
            {
              __state.__data_.__fun_(__i, __args...);
            }
            _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (!__is_nothrow)
            {
              STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            }
          }
        }
        else
        {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __impl : __impl_base<bulk_t>
    {
      // Implementation is handled by lowering to `bulk_chunked` in the tag's `transform_sender`.
    };
  }  // namespace __bulk

  //! @brief A pipeable sender adaptor that applies a function to each index
  //!        in @c [0, shape) under a given execution policy.
  //!
  //! @c bulk is the parallel-loop primitive of the sender model. You give
  //! it a sender, an execution policy (e.g. @c stdexec::par), an integral
  //! shape, and a callable; you get back a sender that, when started,
  //! invokes <tt>f(i, vs...)</tt> for every @c i in @c [0, shape) — where
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
  //! <tt>bulk(sndr, pol, n, f)</tt> into a @c bulk_chunked over the same
  //! shape with @c f wrapped in a per-chunk loop. If a domain customizes
  //! @c bulk_chunked, @c bulk picks up that customization automatically.
  //!
  //! @see stdexec::bulk_chunked    — explicit-chunk variant
  //! @see stdexec::bulk_unchunked  — strict per-index variant (no chunking allowed)
  //! @see stdexec::when_all        — concurrent composition without an index space
  struct bulk_t : __bulk::__generic_bulk_t<bulk_t>
  {
    template <class _Sender>
    static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore)
    {
      auto& [__tag, __data, __child] = __sndr;
      // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` is customized, we will see the customization.
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
  //! The callable is invoked as <tt>fun(begin, end, vs...)</tt>, where
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
  //! @ref bulk_t — <tt>fun(i, vs...)</tt> for every @c i — but explicitly
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
  //! @c bulk is an instance of @ref bulk_t. See @ref bulk_t for the full
  //! description, the lowering to @c bulk_chunked, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr bulk_t bulk{};

  //! @brief The customization point object for the @c bulk_chunked sender adaptor.
  //!
  //! @c bulk_chunked is an instance of @ref bulk_chunked_t. See
  //! @ref bulk_chunked_t for the full description.
  //!
  //! @hideinitializer
  inline constexpr bulk_chunked_t bulk_chunked{};

  //! @brief The customization point object for the @c bulk_unchunked sender adaptor.
  //!
  //! @c bulk_unchunked is an instance of @ref bulk_unchunked_t. See
  //! @ref bulk_unchunked_t for the full description and when to reach for
  //! it.
  //!
  //! @hideinitializer
  inline constexpr bulk_unchunked_t bulk_unchunked{};

  template <>
  struct __sexpr_impl<bulk_t> : __bulk::__impl
  {};

  template <>
  struct __sexpr_impl<bulk_chunked_t> : __bulk::__chunked_impl
  {};

  template <>
  struct __sexpr_impl<bulk_unchunked_t> : __bulk::__unchunked_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
