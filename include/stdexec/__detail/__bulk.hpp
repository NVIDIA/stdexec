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
#include "__execution_legacy.hpp"

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__completion_signatures_of.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp" // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.bulk]
  namespace __bulk {
    struct bulk_t;
    struct bulk_chunked_t;
    struct bulk_unchunked_t;

    //! Wrapper for a policy object.
    //!
    //! If we wrap a standard execution policy, we don't store anything, as we know the type.
    //! Stores the execution policy object if it's a non-standard one.
    //! Provides a way to query the execution policy object.
    template <class _Pol>
    struct __policy_wrapper {
      _Pol __pol_;

      /*implicit*/ constexpr __policy_wrapper(_Pol __pol)
        : __pol_{__pol} {
      }

      [[nodiscard]]
      constexpr const _Pol& __get() const noexcept {
        return __pol_;
      }
    };

    template <>
    struct __policy_wrapper<sequenced_policy> {
      /*implicit*/ constexpr __policy_wrapper(const sequenced_policy&) {
      }

      [[nodiscard]]
      constexpr const sequenced_policy& __get() const noexcept {
        return seq;
      }
    };

    template <>
    struct __policy_wrapper<parallel_policy> {
      /*implicit*/ constexpr __policy_wrapper(const parallel_policy&) {
      }

      [[nodiscard]]
      constexpr const parallel_policy& __get() const noexcept {
        return par;
      }
    };

    template <>
    struct __policy_wrapper<parallel_unsequenced_policy> {
      /*implicit*/ constexpr __policy_wrapper(const parallel_unsequenced_policy&) {
      }

      [[nodiscard]]
      constexpr const parallel_unsequenced_policy& __get() const noexcept {
        return par_unseq;
      }
    };

    template <>
    struct __policy_wrapper<unsequenced_policy> {
      /*implicit*/ constexpr __policy_wrapper(const unsequenced_policy&) {
      }

      [[nodiscard]]
      constexpr const unsequenced_policy& __get() const noexcept {
        return unseq;
      }
    };

    template <class _Pol, class _Shape, class _Fun>
    struct __data {
      STDEXEC_ATTRIBUTE(no_unique_address)
      __policy_wrapper<_Pol> __pol_;
      _Shape __shape_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Fun __fun_;
    };

    template <class _Pol, class _Shape, class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __data(const _Pol&, _Shape, _Fun) -> __data<_Pol, _Shape, _Fun>;

    template <class _AlgoTag>
    struct __bulk_traits;

    template <>
    struct __bulk_traits<bulk_t> {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <>
    struct __bulk_traits<bulk_chunked_t> {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_chunked_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried = __mbind_front<
        __mtry_catch_q<__nothrow_invocable_t, __on_not_callable>,
        _Fun,
        _Shape,
        _Shape
      >;
    };

    template <>
    struct __bulk_traits<bulk_unchunked_t> {
      using __on_not_callable = __mbind_front_q<__callable_error_t, bulk_unchunked_t>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _AlgoTag, class _Fun, class _Shape, class _CvSender, class... _Env>
    using __with_error_invoke_t = __if<
      __value_types_t<
        __completion_signatures_of_t<_CvSender, _Env...>,
        __mtransform<
          __q<__decay_ref>,
          typename __bulk_traits<_AlgoTag>::template __fun_curried<_Fun, _Shape>
        >,
        __q<__mand>
      >,
      completion_signatures<>,
      __eptr_completion
    >;


    template <class _AlgoTag, class _Fun, class _Shape, class _CvSender, class... _Env>
    using __completion_signatures = transform_completion_signatures<
      __completion_signatures_of_t<_CvSender, _Env...>,
      __with_error_invoke_t<_AlgoTag, _Fun, _Shape, _CvSender, _Env...>
    >;

    template <class _AlgoTag>
    struct __generic_bulk_t { // NOLINT(bugprone-crtp-constructor-accessibility)
      template <
        sender _Sender,
        typename _Policy,
        __std::integral _Shape,
        __std::copy_constructible _Fun
      >
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Sender&& __sndr, _Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __well_formed_sender auto {
        return __make_sexpr<_AlgoTag>(
          __data{__pol, __shape, static_cast<_Fun&&>(__fun)}, static_cast<_Sender&&>(__sndr));
      }

      template <typename _Policy, __std::integral _Shape, __std::copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Policy&& __pol, _Shape __shape, _Fun __fun) const {
        return __closure(
          *this,
          static_cast<_Policy&&>(__pol),
          static_cast<_Shape&&>(__shape),
          static_cast<_Fun&&>(__fun));
      }

      template <sender _Sender, __std::integral _Shape, __std::copy_constructible _Fun>
      [[deprecated(
        "The bulk algorithm now requires an execution policy such as STDEXEC::par as an "
        "argument.")]]
      STDEXEC_ATTRIBUTE(host, device) auto
        operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const {
        return (*this)(
          static_cast<_Sender&&>(__sndr),
          par,
          static_cast<_Shape&&>(__shape),
          static_cast<_Fun&&>(__fun));
      }

      template <__std::integral _Shape, __std::copy_constructible _Fun>
      [[deprecated(
        "The bulk algorithm now requires an execution policy such as STDEXEC::par as an "
        "argument.")]]
      STDEXEC_ATTRIBUTE(always_inline) auto operator()(_Shape __shape, _Fun __fun) const {
        return (*this)(par, static_cast<_Shape&&>(__shape), static_cast<_Fun&&>(__fun));
      }
    };

    template <class _Fun>
    struct __as_bulk_chunked_fn {
      _Fun __fun_;

      template <class _Shape, class... _Args>
      constexpr void operator()(_Shape __begin, _Shape __end, _Args&... __args)
        noexcept(__nothrow_callable<_Fun&, _Shape, decltype(__args)...>) {
        for (; __begin != __end; ++__begin) {
          __fun_(__begin, __args...);
        }
      }
    };

    template <class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __as_bulk_chunked_fn(_Fun) -> __as_bulk_chunked_fn<_Fun>;

    struct bulk_t : __generic_bulk_t<bulk_t> {
      struct __transform_sender_fn {
        template <class _Data, class _Child>
        constexpr auto operator()(__ignore, _Data&& __data, _Child&& __child) const {
          // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` is customized, we will see the customization.
          return bulk_chunked(
            static_cast<_Child&&>(__child),
            __data.__pol_.__get(),
            __data.__shape_,
            __as_bulk_chunked_fn(std::move(__data.__fun_)));
        }
      };

      template <class _Sender>
      static constexpr auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        return __apply(__transform_sender_fn(), static_cast<_Sender&&>(__sndr));
      }
    };

    struct bulk_chunked_t : __generic_bulk_t<bulk_chunked_t> { };

    struct bulk_unchunked_t : __generic_bulk_t<bulk_unchunked_t> { };

    template <class _AlgoTag>
    struct __impl_base : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      // Forward the child sender's environment (which contains completion scheduler)
      static constexpr auto get_attrs = [](__ignore, __ignore, const auto& __child) noexcept {
        return __fwd_env(STDEXEC::get_env(__child));
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, _AlgoTag>);
        // TODO: port this to use constant evaluation
        return __completion_signatures<
          _AlgoTag,
          __fun_t<_Sender>,
          __shape_t<_Sender>,
          __child_of<_Sender>,
          _Env...
        >{};
      };
    };

    struct __chunked_impl : __impl_base<bulk_chunked_t> {
      //! This implements the core default behavior for `bulk_chunked`:
      //! When setting value, it calls the function with the entire range.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto complete = []<class _Tag, class _State, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        if constexpr (__std::same_as<_Tag, set_value_t>) {
          // Intercept set_value and dispatch to the bulk operation.
          using __shape_t = decltype(__state.__data_.__shape_);
          using __fun_t = decltype(__state.__data_.__fun_);
          constexpr bool __is_nothrow = __nothrow_callable<__fun_t, __shape_t, __shape_t, _Args...>;
          STDEXEC_TRY {
            __state.__data_.__fun_(static_cast<__shape_t>(0), __state.__data_.__shape_, __args...);
            _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__is_nothrow) {
              STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __unchunked_impl : __impl_base<bulk_unchunked_t> {
      //! This implements the core default behavior for `bulk_unchunked`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in concurrently. That is customized by the scheduler.
      static constexpr auto complete = []<class _Tag, class _State, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        if constexpr (__std::same_as<_Tag, set_value_t>) {
          using __shape_t = decltype(__state.__data_.__shape_);
          using __fun_t = decltype(__state.__data_.__fun_);
          constexpr bool __is_nothrow = __nothrow_callable<__fun_t, __shape_t, _Args...>;
          const auto __shape = __state.__data_.__shape_;
          STDEXEC_TRY {
            for (__shape_t __i{}; __i != __shape; ++__i) {
              __state.__data_.__fun_(__i, __args...);
            }
            _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__is_nothrow) {
              STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __impl : __impl_base<bulk_t> {
      // Implementation is handled by lowering to `bulk_chunked` in the tag's `transform_sender`.
    };
  } // namespace __bulk

  using __bulk::bulk_t;
  using __bulk::bulk_chunked_t;
  using __bulk::bulk_unchunked_t;
  inline constexpr bulk_t bulk{};
  inline constexpr bulk_chunked_t bulk_chunked{};
  inline constexpr bulk_unchunked_t bulk_unchunked{};

  template <>
  struct __sexpr_impl<bulk_t> : __bulk::__impl { };

  template <>
  struct __sexpr_impl<bulk_chunked_t> : __bulk::__chunked_impl { };

  template <>
  struct __sexpr_impl<bulk_unchunked_t> : __bulk::__unchunked_impl { };
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
