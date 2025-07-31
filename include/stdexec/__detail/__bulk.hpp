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

#include "__execution_legacy.hpp"
#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__meta.hpp"
#include "__senders_core.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__transform_completion_signatures.hpp"
#include "__transform_sender.hpp"
#include "__senders.hpp" // IWYU pragma: keep for __well_formed_sender

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
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

      /*implicit*/ __policy_wrapper(_Pol __pol)
        : __pol_{__pol} {
      }

      const _Pol& __get() const noexcept {
        return __pol_;
      }
    };

    template <>
    struct __policy_wrapper<sequenced_policy> {
      /*implicit*/ __policy_wrapper(const sequenced_policy&) {
      }

      const sequenced_policy& __get() const noexcept {
        return seq;
      }
    };

    template <>
    struct __policy_wrapper<parallel_policy> {
      /*implicit*/ __policy_wrapper(const parallel_policy&) {
      }

      const parallel_policy& __get() const noexcept {
        return par;
      }
    };

    template <>
    struct __policy_wrapper<parallel_unsequenced_policy> {
      /*implicit*/ __policy_wrapper(const parallel_unsequenced_policy&) {
      }

      const parallel_unsequenced_policy& __get() const noexcept {
        return par_unseq;
      }
    };

    template <>
    struct __policy_wrapper<unsequenced_policy> {
      /*implicit*/ __policy_wrapper(const unsequenced_policy&) {
      }

      const unsequenced_policy& __get() const noexcept {
        return unseq;
      }
    };

    template <class _Pol, class _Shape, class _Fun>
    struct __data {
      STDEXEC_ATTRIBUTE(no_unique_address) __policy_wrapper<_Pol> __pol_;
      _Shape __shape_;
      STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_;
      static constexpr auto __mbrs_ =
        __mliterals<&__data::__pol_, &__data::__shape_, &__data::__fun_>();
    };
    template <class _Pol, class _Shape, class _Fun>
    __data(const _Pol&, _Shape, _Fun) -> __data<_Pol, _Shape, _Fun>;

    template <class _AlgoTag>
    struct __bulk_traits;

    template <>
    struct __bulk_traits<bulk_t> {
      using __on_not_callable =
        __callable_error<"In stdexec::bulk(Sender, Policy, Shape, Function)..."_mstr>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <>
    struct __bulk_traits<bulk_chunked_t> {
      using __on_not_callable =
        __callable_error<"In stdexec::bulk_chunked(Sender, Policy, Shape, Function)..."_mstr>;

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
      using __on_not_callable =
        __callable_error<"In stdexec::bulk_unchunked(Sender, Policy, Shape, Function)..."_mstr>;

      // Curried function, after passing the required indices.
      template <class _Fun, class _Shape>
      using __fun_curried =
        __mbind_front<__mtry_catch_q<__nothrow_invocable_t, __on_not_callable>, _Fun, _Shape>;
    };

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _AlgoTag, class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __with_error_invoke_t = __if<
      __value_types_t<
        __completion_signatures_of_t<_CvrefSender, _Env...>,
        __mtransform<
          __q<__decay_ref>,
          typename __bulk_traits<_AlgoTag>::template __fun_curried<_Fun, _Shape>
        >,
        __q<__mand>
      >,
      completion_signatures<>,
      __eptr_completion
    >;


    template <class _AlgoTag, class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __completion_signatures = transform_completion_signatures<
      __completion_signatures_of_t<_CvrefSender, _Env...>,
      __with_error_invoke_t<_AlgoTag, _Fun, _Shape, _CvrefSender, _Env...>
    >;

    template <class _AlgoTag>
    struct __generic_bulk_t { // NOLINT(bugprone-crtp-constructor-accessibility)
      template <sender _Sender, typename _Policy, integral _Shape, copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(host, device)
      auto operator()(_Sender&& __sndr, _Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<_AlgoTag>(
            __data{__pol, __shape, static_cast<_Fun&&>(__fun)}, static_cast<_Sender&&>(__sndr)));
      }

      template <typename _Policy, integral _Shape, copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __binder_back<_AlgoTag, _Policy, _Shape, _Fun> {
        return {
          {static_cast<_Policy&&>(__pol),
           static_cast<_Shape&&>(__shape),
           static_cast<_Fun&&>(__fun)},
          {},
          {}
        };
      }

      template <sender _Sender, integral _Shape, copy_constructible _Fun>
      [[deprecated(
        "The bulk algorithm now requires an execution policy such as stdexec::par as an "
        "argument.")]]
      STDEXEC_ATTRIBUTE(host, device) auto
        operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const {
        return (*this)(
          static_cast<_Sender&&>(__sndr),
          par,
          static_cast<_Shape&&>(__shape),
          static_cast<_Fun&&>(__fun));
      }

      template <integral _Shape, copy_constructible _Fun>
      [[deprecated(
        "The bulk algorithm now requires an execution policy such as stdexec::par as an "
        "argument.")]]
      STDEXEC_ATTRIBUTE(always_inline) auto operator()(_Shape __shape, _Fun __fun) const {
        return (*this)(par, static_cast<_Shape&&>(__shape), static_cast<_Fun&&>(__fun));
      }
    };

    struct bulk_t : __generic_bulk_t<bulk_t> {
      template <class _Env>
      static auto __transform_sender_fn(const _Env&) {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          using __shape_t = std::remove_cvref_t<decltype(__data.__shape_)>;
          auto __new_f =
            [__func = std::move(
               __data.__fun_)](__shape_t __begin, __shape_t __end, auto&&... __vs) mutable
#if !STDEXEC_MSVC()
            // MSVCBUG https://developercommunity.visualstudio.com/t/noexcept-expression-in-lambda-template-n/10718680
            noexcept(noexcept(__data.__fun_(__begin++, __vs...)))
#endif
          {
            while (__begin != __end)
              __func(__begin++, __vs...);
          };

          // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` is customized, we will see the customization.
          return bulk_chunked(
            static_cast<_Child&&>(__child),
            __data.__pol_.__get(),
            __data.__shape_,
            std::move(__new_f));
        };
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn(__env));
      }
    };

    struct bulk_chunked_t : __generic_bulk_t<bulk_chunked_t> { };

    struct bulk_unchunked_t : __generic_bulk_t<bulk_unchunked_t> { };

    template <class _AlgoTag>
    struct __bulk_impl_base : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      static constexpr auto get_completion_signatures =
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept -> __completion_signatures<
                                                                          _AlgoTag,
                                                                          __fun_t<_Sender>,
                                                                          __shape_t<_Sender>,
                                                                          __child_of<_Sender>,
                                                                          _Env...
                                                                        > {
        static_assert(sender_expr_for<_Sender, bulk_t>);
        return {};
      };
    };

    struct __bulk_chunked_impl : __bulk_impl_base<bulk_chunked_t> {
      //! This implements the core default behavior for `bulk_chunked`:
      //! When setting value, it calls the function with the entire range.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto complete =
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          // Intercept set_value and dispatch to the bulk operation.
          using __shape_t = decltype(__state.__shape_);
          if constexpr (noexcept(__state.__fun_(__shape_t{}, __shape_t{}, __args...))) {
            // The noexcept version that doesn't need try/catch:
            __state.__fun_(static_cast<__shape_t>(0), __state.__shape_, __args...);
            _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
          } else {
            STDEXEC_TRY {
              __state.__fun_(static_cast<__shape_t>(0), __state.__shape_, __args...);
              _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
            }
            STDEXEC_CATCH_ALL {
              stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __bulk_unchunked_impl : __bulk_impl_base<bulk_unchunked_t> {
      //! This implements the core default behavior for `bulk_unchunked`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in concurrently. That is customized by the scheduler.
      static constexpr auto complete =
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          using __shape_t = decltype(__state.__shape_);
          if constexpr (noexcept(__state.__fun_(__shape_t{}, __args...))) {
            // The noexcept version that doesn't need try/catch:
            for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
              __state.__fun_(__i, __args...);
            }
            _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
          } else {
            STDEXEC_TRY {
              for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
                __state.__fun_(__i, __args...);
              }
              _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
            }
            STDEXEC_CATCH_ALL {
              stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __bulk_impl : __bulk_impl_base<bulk_t> {
      // Implementation is handled by lowering to `bulk_chunked` in `transform_sender`.
    };
  } // namespace __bulk

  using __bulk::bulk_t;
  using __bulk::bulk_chunked_t;
  using __bulk::bulk_unchunked_t;
  inline constexpr bulk_t bulk{};
  inline constexpr bulk_chunked_t bulk_chunked{};
  inline constexpr bulk_unchunked_t bulk_unchunked{};

  template <>
  struct __sexpr_impl<bulk_t> : __bulk::__bulk_impl { };

  template <>
  struct __sexpr_impl<bulk_chunked_t> : __bulk::__bulk_chunked_impl { };

  template <>
  struct __sexpr_impl<bulk_unchunked_t> : __bulk::__bulk_unchunked_impl { };
} // namespace stdexec

STDEXEC_PRAGMA_POP()
