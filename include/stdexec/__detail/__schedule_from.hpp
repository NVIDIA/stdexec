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
#include "__concepts.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__operation_states.hpp"
#include "__senders_core.hpp"
#include "__schedulers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__variant.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  namespace __schfr {
    template <class... _Ts>
    using __tuple_t = __tuple_for<__decay_t<_Ts>...>;

    template <class... _Ts>
    using __variant_t = __variant_for<__monostate, _Ts...>;

    // Compute a variant type that is capable of storing the results of the
    // input sender when it completes. The variant has type:
    //   variant<
    //     monostate,
    //     tuple<set_stopped_t>,
    //     tuple<set_value_t, __decay_t<_Values1>...>,
    //     tuple<set_value_t, __decay_t<_Values2>...>,
    //        ...
    //     tuple<set_error_t, __decay_t<_Error1>>,
    //     tuple<set_error_t, __decay_t<_Error2>>,
    //        ...
    //   >
    template <class _CvrefSender, class _Env>
    using __variant_for = //
      __for_each_completion_signature<
        __completion_signatures_of_t<_CvrefSender, _Env>,
        __tuple_t,
        __munique<__qq<__variant_for>>::__f>;

    template <class... _Values>
    using __decay_value_sig = set_value_t (*)(__decay_t<_Values>...);

    template <class _Error>
    using __decay_error_sig = set_error_t (*)(__decay_t<_Error>);

    template <class... _Ts>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Ts> && ...)>;

    template <class _CvrefSender, class... _Env>
    using __all_nothrow_decay_copyable_results = //
      __for_each_completion_signature<
        __completion_signatures_of_t<_CvrefSender, _Env...>,
        __all_nothrow_decay_copyable,
        __mand_t>;

    template <class _Scheduler, class _CvrefSender, class... _Env>
    using __completions_t = //
      __mtry_q<__concat_completion_signatures>::__f<
        __transform_completion_signatures<
          __completion_signatures_of_t<_CvrefSender, _Env...>,
          __decay_value_sig,
          __decay_error_sig,
          set_stopped_t (*)(),
          __completion_signature_ptrs>,
        transform_completion_signatures<
          __completion_signatures_of_t<schedule_result_t<_Scheduler>, _Env...>,
          __eptr_completion_if_t<__all_nothrow_decay_copyable_results<_CvrefSender, _Env...>>,
          __mconst<completion_signatures<>>::__f>>;

    template <class _SchedulerId>
    struct __environ {
      using _Scheduler = stdexec::__t<_SchedulerId>;

      struct __t {
        using __id = __environ;

        _Scheduler __sched_;

        template <__one_of<set_value_t, set_stopped_t> _Tag>
        auto query(get_completion_scheduler_t<_Tag>) const noexcept {
          return __sched_;
        }

        auto query(get_domain_t) const noexcept {
          return query_or(get_domain, __sched_, default_domain());
        }
      };
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state;

    template <class _State>
    STDEXEC_ATTRIBUTE((always_inline))
    auto
      __make_visitor_fn(_State* __state) noexcept {
      return [__state]<class _Tup>(_Tup& __tupl) noexcept -> void {
        if constexpr (__same_as<_Tup, __monostate>) {
          std::terminate(); // reaching this indicates a bug in schedule_from
        } else {
          __tupl.apply(
            [&]<class... _Args>(auto __tag, _Args&... __args) noexcept -> void {
              __tag(std::move(__state->__receiver()), static_cast<_Args&&>(__args)...);
            },
            __tupl);
        }
      };
    }

    // This receiver is to be completed on the execution context associated with the scheduler. When
    // the source sender completes, the completion information is saved off in the operation state
    // so that when this receiver completes, it can read the completion out of the operation state
    // and forward it to the output receiver after transitioning to the scheduler's context.
    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __receiver2 {
      using receiver_concept = receiver_t;

      void set_value() noexcept {
        __state_->__data_.visit(__schfr::__make_visitor_fn(__state_), __state_->__data_);
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept {
        stdexec::set_error(
          static_cast<_Receiver&&>(__state_->__receiver()), static_cast<_Error&&>(__err));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(static_cast<_Receiver&&>(__state_->__receiver()));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return stdexec::get_env(__state_->__receiver());
      }

      __state<_Scheduler, _Sexpr, _Receiver>* __state_;
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state
      : __enable_receiver_from_this<_Sexpr, _Receiver>
      , __immovable {
      using __variant_t = __variant_for<__child_of<_Sexpr>, env_of_t<_Receiver>>;
      using __receiver2_t = __receiver2<_Scheduler, _Sexpr, _Receiver>;

      __variant_t __data_;
      connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t> __state2_;
      STDEXEC_APPLE_CLANG(__state* __self_;)

      explicit __state(_Scheduler __sched)
        : __data_()
        , __state2_(connect(schedule(__sched), __receiver2_t{this}))
            STDEXEC_APPLE_CLANG(, __self_(this)) {
      }
    };

    struct schedule_from_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        using _Env = __t<__environ<__id<__decay_t<_Scheduler>>>>;
        auto __env = _Env{{static_cast<_Scheduler&&>(__sched)}};
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<schedule_from_t>(std::move(__env), static_cast<_Sender&&>(__sndr)));
      }

      using _Sender = __1;
      using _Env = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(schedule_from_t, get_completion_scheduler_t<set_value_t>(_Env&), _Sender)>;
    };

    struct __schedule_from_impl : __sexpr_defaults {
      template <class _Sender>
      using __scheduler_t =
        __decay_t<__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<_Sender>>>;

      static constexpr auto get_attrs = //
        []<class _Data, class _Child>(const _Data& __data, const _Child& __child) noexcept {
          return __env::__join(__data, stdexec::get_env(__child));
        };

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> __completions_t<__scheduler_t<_Sender>, __child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, schedule_from_t>);
        return {};
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&) {
          static_assert(sender_expr_for<_Sender, schedule_from_t>);
          auto __sched = get_completion_scheduler<set_value_t>(stdexec::get_env(__sndr));
          using _Scheduler = decltype(__sched);
          return __state<_Scheduler, _Sender, _Receiver>{__sched};
        };

      static constexpr auto complete = //
        []<class _State, class _Receiver, class _Tag, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag __tag,
          _Args&&... __args) noexcept -> void {
        STDEXEC_APPLE_CLANG(__state.__self_ == &__state ? void() : std::terminate());
        // Write the tag and the args into the operation state so that we can forward the completion
        // from within the scheduler's execution context.
        if constexpr (__nothrow_callable<__tup::__mktuple_t, _Tag, _Args...>) {
          __state.__data_.emplace_from(__tup::__mktuple, __tag, static_cast<_Args&&>(__args)...);
        } else {
          try {
            __state.__data_.emplace_from(__tup::__mktuple, __tag, static_cast<_Args&&>(__args)...);
          } catch (...) {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            return;
          }
        }

        // Enqueue the schedule operation so the completion happens on the scheduler's execution
        // context.
        stdexec::start(__state.__state2_);
      };
    };
  } // namespace __schfr

  using __schfr::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  template <>
  struct __sexpr_impl<schedule_from_t> : __schfr::__schedule_from_impl { };
} // namespace stdexec
