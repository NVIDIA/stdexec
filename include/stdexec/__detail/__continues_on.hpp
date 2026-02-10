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
#include "__concepts.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__operation_states.hpp"
#include "__schedule_from.hpp"
#include "__schedulers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__utility.hpp"
#include "__variant.hpp" // IWYU pragma: keep for __variant

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.continues_on]
  namespace __trnsfr {
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
    template <class _CvSender, class _Env>
    using __results_of_t = __for_each_completion_signature_t<
      __completion_signatures_of_t<_CvSender, _Env>,
      __decayed_tuple,
      __munique<__qq<STDEXEC::__variant>>::__f
    >;

    template <class... _Values>
    using __decay_value_sig = set_value_t (*)(__decay_t<_Values>...);

    template <class _Error>
    using __decay_error_sig = set_error_t (*)(__decay_t<_Error>);

    template <class _Scheduler, class _Completions, class... _Env>
    using __completions_impl_t = __mtry_q<__concat_completion_signatures_t>::__f<
      __transform_completion_signatures_t<
        _Completions,
        __decay_value_sig,
        __decay_error_sig,
        set_stopped_t (*)(),
        __completion_signature_ptrs_t
      >,
      transform_completion_signatures<
        __completion_signatures_of_t<schedule_result_t<_Scheduler>, _Env...>,
        __eptr_completion_unless_t<__nothrow_decay_copyable_results_t<_Completions>>,
        __mconst<completion_signatures<>>::__f
      >
    >;

    template <class _Scheduler, class _CvSender, class... _Env>
    using __completions_t =
      __completions_impl_t<_Scheduler, __completion_signatures_of_t<_CvSender, _Env...>, _Env...>;

    template <class _State>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto __make_visitor_fn(_State* __state) noexcept {
      return [__state]<class _Tup>(_Tup& __tupl) noexcept -> void {
        STDEXEC::__apply(
          [&]<class _Tag, class... _Args>(_Tag, _Args&... __args) noexcept -> void {
            _Tag()(static_cast<_State&&>(*__state).__rcvr_, static_cast<_Args&&>(__args)...);
          },
          __tupl);
      };
    }

    template <class _Sexpr, class _Receiver>
    struct __state_base : __immovable {
      using __variant_t = __results_of_t<__child_of<_Sexpr>, env_of_t<_Receiver>>;

      _Receiver __rcvr_;
      __variant_t __data_{__no_init};
    };

    // This receiver is to be completed on the execution context associated with the scheduler. When
    // the source sender completes, the completion information is saved off in the operation state
    // so that when this receiver completes, it can read the completion out of the operation state
    // and forward it to the output receiver after transitioning to the scheduler's context.
    template <class _Sexpr, class _Receiver>
    struct __receiver2 {
      using receiver_concept = receiver_t;

      constexpr void set_value() noexcept {
        STDEXEC::__visit(__trnsfr::__make_visitor_fn(__state_), __state_->__data_);
      }

      template <class _Error>
      constexpr void set_error(_Error&& __err) noexcept {
        STDEXEC::set_error(
          static_cast<_Receiver&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__state_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __state_base<_Sexpr, _Receiver>* __state_;
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state : __state_base<_Sexpr, _Receiver> {
      using __variant_t = __results_of_t<__child_of<_Sexpr>, env_of_t<_Receiver>>;
      using __receiver2_t = __receiver2<_Sexpr, _Receiver>;

      constexpr explicit __state(_Scheduler __sched, _Receiver&& __rcvr)
        : __state::__state_base{{}, static_cast<_Receiver&&>(__rcvr)}
        , __state2_(connect(schedule(__sched), __receiver2_t{this})) {
      }

      connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t> __state2_;
    };

    struct continues_on_t {
      template <scheduler _Scheduler, sender _Sender>
      constexpr auto
        operator()(_Sender&& __sndr, _Scheduler __sched) const -> __well_formed_sender auto {
        return __make_sexpr<continues_on_t>(
          static_cast<_Scheduler&&>(__sched), schedule_from(static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Scheduler __sched) const noexcept {
        return __closure(*this, static_cast<_Scheduler&&>(__sched));
      }
    };

    //! @brief The @c continues_on sender's attributes.
    template <class _Scheduler, class _Sender>
    struct __attrs {
     private:
      //! @brief Returns `true` when:
      //! - _SetTag is set_error_t, and
      //! - _Sender has value completions, and
      //! - at least one of the value completions is not nothrow decay-copyable.
      //! In that case, error completions can come from the sender's value completions.
      template <class _SetTag, class... _Env>
      static consteval bool __has_decay_copy_errors() noexcept {
        if constexpr (__same_as<_SetTag, set_error_t>) {
          if constexpr (__sends<set_value_t, _Sender, __fwd_env_t<_Env>...>) {
            return !__cmplsigs::__partitions_of_t<
              completion_signatures_of_t<_Sender, __fwd_env_t<_Env>...>
            >::__nothrow_decay_copyable::__values::value;
          }
        }
        return false;
      }

      const _Scheduler& __sch_;
      const _Sender& __sndr_;

     public:
      constexpr explicit __attrs(const _Scheduler& __sch, const _Sender& __sndr) noexcept
        : __sch_(__sch)
        , __sndr_(__sndr) {
      }

      //! @brief Queries the completion scheduler for a given @c _SetTag.
      //! @tparam _SetTag The completion tag to query for.
      //! @tparam _Env The environment to consider when querying for the completion
      //! scheduler.
      //!
      //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
      //! succeeded and execution continues on @c _Scheduler.
      //!
      //! Otherwise, if @c _Sender never completes with @c _SetTag, and either @c _SetTag is
      //! @c set_stopped_t or decay-copying @c _Sender's value results cannot throw, then a
      //! @c _SetTag completion can only come from the scheduler's sender. In this case, return
      //! the scheduler's completion scheduler if it has one.
      //!
      //! Otherwise, if the scheduler's sender never completes with @c _SetTag, then a
      //! @c _SetTag completion can only come from the original sender, so return the
      //! original sender's completion scheduler.
      template <class _SetTag, class... _Env>
        requires(__same_as<_SetTag, set_value_t>
                 || __never_sends<_SetTag, _Sender, __fwd_env_t<_Env>...>)
             && (!__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<_SetTag>, const _Env&... __env) const noexcept
        -> __call_result_t<get_completion_scheduler_t<_SetTag>, _Scheduler, __fwd_env_t<_Env>...> {
        return get_completion_scheduler<_SetTag>(__sch_, __fwd_env(__env)...);
      }

      //! @overload
      template <class _SetTag, class... _Env>
        requires __never_sends<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<_SetTag>, const _Env&... __env) const noexcept
        -> __call_result_t<
          get_completion_scheduler_t<_SetTag>,
          env_of_t<_Sender>,
          __fwd_env_t<_Env>...
        > {
        return get_completion_scheduler<_SetTag>(get_env(__sndr_), __fwd_env(__env)...);
      }

      //! @brief Queries the completion domain for a given @c _SetTag.
      //! @tparam _SetTag The completion tag to query for.
      //! @tparam _Env The environment to consider when querying for the completion domain.
      //!
      //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
      //! succeeded and execution continues on @c _Scheduler.
      //!
      //! Otherwise, if @c _SetTag is @c set_stopped_t or if decay-copying @c _Sender's value
      //! results cannot throw, then a @c _SetTag completion can happen on the sender's
      //! completion domain (if it has one) or the scheduler's completion domain (if it has
      //! one).
      //!
      //! @note Otherwise, @c _SetTag is @c set_error_t and decay-copying @c _Sender's value
      //! results can throw, so error completions can also come from the sender's value
      //! completions.
      template <__same_as<set_value_t> _SetTag, class... _Env>
      [[nodiscard]]
      constexpr auto
        query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept -> __unless_one_of_t<
          __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>,
          indeterminate_domain<>
        > {
        return {};
      }

      //! @overload
      template <__one_of<set_error_t, set_stopped_t> _SetTag, class... _Env>
        requires(!__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto
        query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept -> __unless_one_of_t<
          __common_domain_t<
            __completion_domain_of_t<_SetTag, _Sender, __fwd_env_t<_Env>...>,
            __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>
          >,
          indeterminate_domain<>
        > {
        return {};
      }

      //! @overload
      template <class _SetTag, class... _Env>
        requires(__has_decay_copy_errors<_SetTag, _Env...>())
      [[nodiscard]]
      constexpr auto
        query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept -> __unless_one_of_t<
          __common_domain_t<
            __completion_domain_of_t<_SetTag, _Sender, __fwd_env_t<_Env>...>,
            __completion_domain_of_t<_SetTag, schedule_result_t<_Scheduler>, __fwd_env_t<_Env>...>,
            __completion_domain_of_t<set_value_t, _Sender, __fwd_env_t<_Env>...>
          >,
          indeterminate_domain<>
        > {
        return {};
      }

      //! @brief Queries the completion behavior of the combined sender.
      //! @tparam _Env The environment to consider when querying for the completion behavior.
      //! @note The completion behavior is the minimum between the scheduler's sender and
      //! the original sender.
      template <class _Tag, class... _Env>
      [[nodiscard]]
      constexpr auto query(get_completion_behavior_t<_Tag>, const _Env&...) const noexcept {
        using _SchSender = schedule_result_t<_Scheduler>;
        constexpr auto cb_sched =
          STDEXEC::get_completion_behavior<_Tag, _SchSender, __fwd_env_t<_Env>...>();
        constexpr auto cb_sndr =
          STDEXEC::get_completion_behavior<_Tag, _Sender, __fwd_env_t<_Env>...>();
        return completion_behavior::weakest(cb_sched, cb_sndr);
      }

      //! @brief Forwards other queries to the underlying sender's environment.
      //! @pre @c _Tag is a forwarding query but not a completion query.
      template <__forwarding_query _Tag, class... _Args>
        requires(!__is_completion_query<_Tag>)
             && __queryable_with<env_of_t<_Sender>, _Tag, _Args...>
      [[nodiscard]]
      constexpr auto query(_Tag, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Sender>, _Tag, _Args...>)
          -> __query_result_t<env_of_t<_Sender>, _Tag, _Args...> {
        return get_env(__sndr_).query(_Tag{}, static_cast<_Args&&>(__args)...);
      }
    };

    struct __continues_on_impl : __sexpr_defaults {
     private:
      template <class _Sender, class... _Env>
      using __scheduler_t = __decay_t<__data_of<_Sender>>;

      template <class _Child, class... _Env>
      static consteval auto __get_child_completions() {
        STDEXEC_COMPLSIGS_LET(
          __child_completions, STDEXEC::get_completion_signatures<_Child, __fwd_env_t<_Env>...>()) {
          // continues_on has the completions of the child sender, but with value and
          // error result types decayed.
          return __transform_completion_signatures(
            __child_completions,
            __decay_arguments<set_value_t, continues_on_t>(),
            __decay_arguments<set_error_t, continues_on_t>());
        }
      }

      template <class _Scheduler, class... _Env>
      static consteval auto __get_scheduler_completions() {
        using __sndr_t = schedule_result_t<_Scheduler>;
        STDEXEC_COMPLSIGS_LET(
          __sched_completions,
          STDEXEC::get_completion_signatures<__sndr_t, __fwd_env_t<_Env>...>()) {
          // The scheduler contributes only error and stopped completions; we ignore value
          // completions here
          return __transform_completion_signatures(__sched_completions, __ignore_completion());
        }
      }

      template <class _Sender, class _Receiver>
      using __state_for_t = __state<__decay_t<__tuple_element_t<1, _Sender>>, _Sender, _Receiver>;

     public:
      static constexpr auto get_attrs =
        [](__ignore, const auto& __data, const auto& __child) noexcept {
          return __attrs{__data, __child};
        };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, continues_on_t>);
        using __scheduler_t = __decay_t<__data_of<_Sender>>;
        using __child_t = __child_of<_Sender>;
        auto __child_completions = __get_child_completions<__child_t, _Env...>();
        return __concat_completion_signatures(
          __child_completions,
          __get_scheduler_completions<__scheduler_t, _Env...>(),
          __eptr_completion_unless_t<
            __nothrow_decay_copyable_results_t<decltype(__child_completions)>
          >());
      }

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr)
        -> __state_for_t<_Sender, _Receiver>
        requires sender_in<__child_of<_Sender>, __fwd_env_t<env_of_t<_Receiver>>>
      {
        static_assert(sender_expr_for<_Sender, continues_on_t>);
        auto& [__tag, __sched, __child] = __sndr;
        return __state_for_t<_Sender, _Receiver>{__sched, static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto complete = []<class _State, class _Tag, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag __tag,
                                         _Args&&... __args) noexcept -> void {
        // Write the tag and the args into the operation state so that we can forward the completion
        // from within the scheduler's execution context.
        if constexpr (__nothrow_callable<__mktuple_t, _Tag, _Args...>) {
          __state.__data_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
        } else {
          STDEXEC_TRY {
            __state.__data_.__emplace_from(__mktuple, __tag, static_cast<_Args&&>(__args)...);
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
            return;
          }
        }

        // Enqueue the schedule operation so the completion happens on the scheduler's execution
        // context.
        STDEXEC::start(__state.__state2_);
      };
    };
  } // namespace __trnsfr

  using __trnsfr::continues_on_t;
  inline constexpr continues_on_t continues_on{};

  template <>
  struct __sexpr_impl<continues_on_t> : __trnsfr::__continues_on_impl { };
} // namespace STDEXEC
