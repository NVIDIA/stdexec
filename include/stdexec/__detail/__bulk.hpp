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
    inline constexpr __mstring __bulk_context =
      "In stdexec::bulk(Sender, Policy, Shape, Function)..."_mstr;
    inline constexpr __mstring __bulk_chunked_context =
      "In stdexec::bulk_chunked(Sender, Policy, Shape, Function)..."_mstr;
    inline constexpr __mstring __bulk_unchunked_context =
      "In stdexec::bulk_unchunked(Sender, Shape, Function)..."_mstr;
    using __on_not_callable = __callable_error<__bulk_context>;
    using __on_not_callable2 = __callable_error<__bulk_chunked_context>;

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
      /*implicit*/ __policy_wrapper(sequenced_policy) {
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
      STDEXEC_ATTRIBUTE((no_unique_address)) __policy_wrapper<_Pol> __pol_;
      _Shape __shape_;
      STDEXEC_ATTRIBUTE((no_unique_address)) _Fun __fun_;
      static constexpr auto __mbrs_ =
        __mliterals<&__data::__pol_, &__data::__shape_, &__data::__fun_>();
    };
    template <class _Pol, class _Shape, class _Fun>
    __data(_Pol, _Shape, _Fun) -> __data<_Pol, _Shape, _Fun>;

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _Catch, class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __with_error_invoke_t = //
      __if<
        __value_types_t<
          __completion_signatures_of_t<_CvrefSender, _Env...>,
          __mtransform<
            __q<__decay_ref>,
            __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun, _Shape>>,
          __q<__mand>>,
        completion_signatures<>,
        __eptr_completion>;
    template <class _Catch, class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __with_error_invoke2_t = //
          __if<
            __value_types_t<
              __completion_signatures_of_t<_CvrefSender, _Env...>,
              __mtransform<
                __q<__decay_ref>,
                __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun, _Shape, _Shape>>,
              __q<__mand>>,
            completion_signatures<>,
            __eptr_completion>;
    
    template <class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __completion_signatures = //
      transform_completion_signatures<
        __completion_signatures_of_t<_CvrefSender, _Env...>,
        __with_error_invoke_t<__on_not_callable, _Fun, _Shape, _CvrefSender, _Env...>>;

        template <class _Fun, class _Shape, class _CvrefSender, class... _Env>
    using __completion_signatures2 = //
        transform_completion_signatures<
          __completion_signatures_of_t<_CvrefSender, _Env...>,
          __with_error_invoke2_t<__on_not_callable2, _Fun, _Shape, _Shape, _CvrefSender, _Env...>>;
      // TODO (now): use tag to provide appropriate error message

    struct bulk_t;
    struct bulk_chunked_t;
    struct bulk_unchunked_t;

    template <class _Tag>
    struct __generic_bulk_t {
      template <sender _Sender, typename _Policy, integral _Shape, copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE((host, device)) auto operator()(_Sender&& __sndr, _Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<_Tag>(
            __data{__pol, __shape, static_cast<_Fun&&>(__fun)}, static_cast<_Sender&&>(__sndr)));
      }

      template <typename _Policy, integral _Shape, copy_constructible _Fun>
        requires is_execution_policy_v<std::remove_cvref_t<_Policy>>
      STDEXEC_ATTRIBUTE((always_inline)) auto operator()(_Policy&& __pol, _Shape __shape, _Fun __fun) const
        -> __binder_back<_Tag, _Policy, _Shape, _Fun> {
        return {
          {static_cast<_Policy&&>(__pol),
           static_cast<_Shape&&>(__shape),
           static_cast<_Fun&&>(__fun)},
          {},
          {}
        };
      }

      // This describes how to use the pieces of a bulk sender to find
      // legacy customizations of the bulk algorithm.
      using _Sender = __1;
      using _Pol = __nth_member<0>(__0);
      using _Shape = __nth_member<1>(__0);
      using _Fun = __nth_member<2>(__0);
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          _Tag,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Pol,
          _Shape,
          _Fun),
        tag_invoke_t(_Tag, _Sender, _Pol, _Shape, _Fun)>;
    };

    struct bulk_t : __generic_bulk_t<bulk_t> { };

    struct bulk_chunked_t : __generic_bulk_t<bulk_chunked_t> { };

    struct bulk_unchunked_t {
      template <sender _Sender, integral _Shape, copy_constructible _Fun>
      STDEXEC_ATTRIBUTE((host, device)) auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const
        -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<bulk_unchunked_t>(
            __data{par, __shape, static_cast<_Fun&&>(__fun)}, static_cast<_Sender&&>(__sndr)));
      }

      template <integral _Shape, copy_constructible _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) auto operator()(_Shape __shape, _Fun __fun) const
        -> __binder_back<bulk_unchunked_t, _Shape, _Fun> {
        return {
          {static_cast<_Shape&&>(__shape), static_cast<_Fun&&>(__fun)},
          {},
          {}
        };
      }

      // This describes how to use the pieces of a bulk sender to find
      // legacy customizations of the bulk algorithm.
      using _Sender = __1;
      using _Shape = __nth_member<1>(__0);
      using _Fun = __nth_member<2>(__0);
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          bulk_unchunked_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Shape,
          _Fun),
        tag_invoke_t(bulk_unchunked_t, _Sender, _Shape, _Fun)>;
    };

    struct __bulk_base_impl : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> __completion_signatures<__fun_t<_Sender>, __shape_t<_Sender>, __child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, bulk_t>);
        return {};
      };
    };

    struct __bulk_base2_impl : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> __completion_signatures2<__fun_t<_Sender>, __shape_t<_Sender>, __child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, bulk_t>);
        return {};
      };
    };

    struct __bulk_chunked_impl : __bulk_base2_impl {
      //! This implements the core default behavior for `bulk`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto complete = //
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
            try {
              __state.__fun_(static_cast<__shape_t>(0), __state.__shape_, __args...);
              _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
            } catch (...) {
              stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __bulk_unchunked_impl : __bulk_base_impl {
      //! This implements the core default behavior for `bulk`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto complete = //
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          // Intercept set_value and dispatch to the bulk operation.
          using __shape_t = decltype(__state.__shape_);
          constexpr bool __scheduler_available =
            requires { get_completion_scheduler<set_value_t>(get_env(__rcvr)); };
          if constexpr (__scheduler_available) {
            // This default implementation doesn't run a scheduler with concurrent progres guarantees.
            constexpr auto __guarantee = get_forward_progress_guarantee(
              get_completion_scheduler<set_value_t>(get_env(__rcvr)));
            static_assert(__guarantee != forward_progress_guarantee::concurrent);
          }
          if constexpr (noexcept(__state.__fun_(__shape_t{}, __args...))) {
            // The noexcept version that doesn't need try/catch:
            for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
              __state.__fun_(__i, __args...);
            }
            _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
          } else {
            try {
              for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
                __state.__fun_(__i, __args...);
              }
              _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
            } catch (...) {
              stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };

    struct __bulk_impl : __bulk_base_impl {
      //! This implements the core default behavior for `bulk`:
      //! When setting value, it loops over the shape and invokes the function.
      //! Note: This is not done in parallel. That is customized by the scheduler.
      //! See, e.g., static_thread_pool::bulk_receiver::__t.
      static constexpr auto complete = //
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          using __shape_t = decltype(__state.__shape_);

          constexpr bool __nothrow = noexcept(__state.__fun_(__state.__shape_, __args...));
          auto __new_f =
            [__func = std::move(__state.__fun_)](
              __shape_t __begin, __shape_t __end, auto&&... __vs) mutable noexcept(__nothrow) {
              while (__begin != __end)
                __func(__begin++, __vs...);
            };

          auto __chunked_data = __data{__state.__pol_, __state.__shape_, std::move(__new_f)};
          __bulk_chunked_impl::complete(
            _Tag(), __chunked_data, __rcvr, _Tag(), std::forward<_Args>(__args)...);
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
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
