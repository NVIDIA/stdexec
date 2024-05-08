/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__detail/__as_awaitable.hpp"
#include "__detail/__basic_sender.hpp"
#include "__detail/__bulk.hpp"
#include "__detail/__completion_signatures.hpp"
#include "__detail/__connect_awaitable.hpp"
#include "__detail/__cpo.hpp"
#include "__detail/__debug.hpp"
#include "__detail/__domain.hpp"
#include "__detail/__ensure_started.hpp"
#include "__detail/__env.hpp"
#include "__detail/__execute.hpp"
#include "__detail/__inline_scheduler.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__intrusive_slist.hpp"
#include "__detail/__just.hpp"
#include "__detail/__let.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__operation_states.hpp"
#include "__detail/__receivers.hpp"
#include "__detail/__receiver_adaptor.hpp"
#include "__detail/__schedulers.hpp"
#include "__detail/__senders.hpp"
#include "__detail/__sender_adaptor_closure.hpp"
#include "__detail/__split.hpp"
#include "__detail/__start_detached.hpp"
#include "__detail/__submit.hpp"
#include "__detail/__then.hpp"
#include "__detail/__transform_sender.hpp"
#include "__detail/__transform_completion_signatures.hpp"
#include "__detail/__type_traits.hpp"
#include "__detail/__upon_error.hpp"
#include "__detail/__upon_stopped.hpp"
#include "__detail/__utility.hpp"
#include "__detail/__with_awaitable_senders.hpp"

#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>
#include <cstddef>
#include <exception>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wundefined-inline")
STDEXEC_PRAGMA_IGNORE_GNU("-Wsubobject-linkage")
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

STDEXEC_PRAGMA_IGNORE_EDG(1302)
STDEXEC_PRAGMA_IGNORE_EDG(497)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Tag = set_value_t>
  concept __completes_on =
    __decays_to<__call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;

  // NOT TO SPEC
  template <class _Tag, const auto& _Predicate>
  concept tag_category = //
    requires {
      typename __mbool<bool{_Predicate(_Tag{})}>;
      requires bool { _Predicate(_Tag{}) };
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  // [execution.senders.adaptors.stopped_as_error]
  namespace __stopped_as_xxx {
    struct stopped_as_optional_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<stopped_as_optional_t>(__(), static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()() const noexcept -> __binder_back<stopped_as_optional_t> {
        return {};
      }
    };

    struct __stopped_as_optional_impl : __sexpr_defaults {
      template <class... _Tys>
        requires(sizeof...(_Tys) == 1)
      using __set_value_t = completion_signatures<set_value_t(std::optional<__decay_t<_Tys>>...)>;

      template <class _Ty>
      using __set_error_t = completion_signatures<set_error_t(_Ty)>;

      static constexpr auto get_completion_signatures =       //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept //
        -> make_completion_signatures<
          __child_of<_Self>,
          _Env,
          completion_signatures<set_error_t(std::exception_ptr)>,
          __set_value_t,
          __set_error_t,
          completion_signatures<>> {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&&, _Receiver&) noexcept
        requires __single_typed_sender<__child_of<_Self>, env_of_t<_Receiver>>
      {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        using _Value = __decay_t<__single_sender_value_t<__child_of<_Self>, env_of_t<_Receiver>>>;
        return __mtype<_Value>();
      };

      static constexpr auto complete = //
        []<class _State, class _Receiver, __completion_tag _Tag, class... _Args>(
          __ignore,
          _State&,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          try {
            static_assert(constructible_from<__t<_State>, _Args...>);
            stdexec::set_value(
              static_cast<_Receiver&&>(__rcvr),
              std::optional<__t<_State>>{static_cast<_Args&&>(__args)...});
          } catch (...) {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        } else if constexpr (same_as<_Tag, set_error_t>) {
          stdexec::set_error(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        } else {
          stdexec::set_value(
            static_cast<_Receiver&&>(__rcvr), std::optional<__t<_State>>{std::nullopt});
        }
      };
    };

    struct stopped_as_error_t {
      template <sender _Sender, __movable_value _Error>
      auto operator()(_Sender&& __sndr, _Error __err) const {
        return static_cast<_Sender&&>(__sndr)
             | let_stopped([__err2 = static_cast<_Error&&>(__err)]() mutable //
                           noexcept(std::is_nothrow_move_constructible_v<_Error>) {
                             return just_error(static_cast<_Error&&>(__err2));
                           });
      }

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Error __err) const -> __binder_back<stopped_as_error_t, _Error> {
        return {{static_cast<_Error&&>(__err)}};
      }
    };
  } // namespace __stopped_as_xxx

  using __stopped_as_xxx::stopped_as_optional_t;
  inline constexpr stopped_as_optional_t stopped_as_optional{};
  using __stopped_as_xxx::stopped_as_error_t;
  inline constexpr stopped_as_error_t stopped_as_error{};

  template <>
  struct __sexpr_impl<stopped_as_optional_t> : __stopped_as_xxx::__stopped_as_optional_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  namespace __loop {
    class run_loop;

    struct __task : __immovable {
      __task* __next_ = this;

      union {
        void (*__execute_)(__task*) noexcept;
        __task* __tail_;
      };

      void __execute() noexcept {
        (*__execute_)(this);
      }
    };

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __task {
        using __id = __operation;

        run_loop* __loop_;
        STDEXEC_ATTRIBUTE((no_unique_address))
        _Receiver __rcvr_;

        static void __execute_impl(__task* __p) noexcept {
          auto& __rcvr = static_cast<__t*>(__p)->__rcvr_;
          try {
            if (get_stop_token(get_env(__rcvr)).stop_requested()) {
              set_stopped(static_cast<_Receiver&&>(__rcvr));
            } else {
              set_value(static_cast<_Receiver&&>(__rcvr));
            }
          } catch (...) {
            set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        }

        explicit __t(__task* __tail) noexcept
          : __task{.__tail_ = __tail} {
        }

        __t(__task* __next, run_loop* __loop, _Receiver __rcvr)
          : __task{{}, __next, {&__execute_impl}}
          , __loop_{__loop}
          , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
        }

        STDEXEC_ATTRIBUTE((always_inline))
        STDEXEC_MEMFN_DECL(
          void start)(this __t& __self) noexcept {
          __self.__start_();
        }

        void __start_() noexcept;
      };
    };

    class run_loop {
      template <class... Ts>
      using __completion_signatures_ = completion_signatures<Ts...>;

      template <class>
      friend struct __operation;
     public:
      struct __scheduler {
        using __t = __scheduler;
        using __id = __scheduler;
        auto operator==(const __scheduler&) const noexcept -> bool = default;

       private:
        struct __schedule_task {
          using __t = __schedule_task;
          using __id = __schedule_task;
          using sender_concept = sender_t;
          using completion_signatures = //
            __completion_signatures_<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;

         private:
          friend __scheduler;

          template <class _Receiver>
          using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

          template <class _Receiver>
          STDEXEC_MEMFN_DECL(auto connect)(this const __schedule_task& __self, _Receiver __rcvr)
            -> __operation<_Receiver> {
            return __self.__connect_(static_cast<_Receiver&&>(__rcvr));
          }

          template <class _Receiver>
          auto __connect_(_Receiver&& __rcvr) const -> __operation<_Receiver> {
            return {&__loop_->__head_, __loop_, static_cast<_Receiver&&>(__rcvr)};
          }

          struct __env {
            run_loop* __loop_;

            template <class _CPO>
            STDEXEC_MEMFN_DECL(auto query)(this const __env& __self, get_completion_scheduler_t<_CPO>) noexcept
              -> __scheduler {
              return __self.__loop_->get_scheduler();
            }
          };

          STDEXEC_MEMFN_DECL(auto get_env)(this const __schedule_task& __self) noexcept -> __env {
            return __env{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop) {
          }

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        STDEXEC_MEMFN_DECL(auto schedule)(this const __scheduler& __self) noexcept -> __schedule_task {
          return __self.__schedule();
        }

        STDEXEC_MEMFN_DECL(auto query)(this const __scheduler&, get_forward_progress_guarantee_t) noexcept
          -> stdexec::forward_progress_guarantee {
          return stdexec::forward_progress_guarantee::parallel;
        }

        // BUGBUG NOT TO SPEC
        STDEXEC_MEMFN_DECL(
          auto execute_may_block_caller)(this const __scheduler&) noexcept -> bool {
          return false;
        }

        [[nodiscard]]
        auto __schedule() const noexcept -> __schedule_task {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
      };

      auto get_scheduler() noexcept -> __scheduler {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void __push_back_(__task* __task);
      auto __pop_front_() -> __task*;

      std::mutex __mutex_;
      std::condition_variable __cv_;
      __task __head_{.__tail_ = &__head_};
      bool __stop_ = false;
    };

    template <class _ReceiverId>
    inline void __operation<_ReceiverId>::__t::__start_() noexcept {
      try {
        __loop_->__push_back_(this);
      } catch (...) {
        set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
      }
    }

    inline void run_loop::run() {
      for (__task* __task; (__task = __pop_front_()) != &__head_;) {
        __task->__execute();
      }
    }

    inline void run_loop::finish() {
      std::unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__task* __task) {
      std::unique_lock __lock{__mutex_};
      __task->__next_ = &__head_;
      __head_.__tail_ = __head_.__tail_->__next_ = __task;
      __cv_.notify_one();
    }

    inline auto run_loop::__pop_front_() -> __task* {
      std::unique_lock __lock{__mutex_};
      __cv_.wait(__lock, [this] { return __head_.__next_ != &__head_ || __stop_; });
      if (__head_.__tail_ == __head_.__next_)
        __head_.__tail_ = &__head_;
      return std::exchange(__head_.__next_, __head_.__next_->__next_);
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  namespace __schedule_from {
    template <class... _Ts>
    using __value_tuple = __tup::__tuple_for<__decay_t<_Ts>...>;

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
    using __variant_for_t = __compl_sigs::__maybe_for_all_sigs<
      __completion_signatures_of_t<_CvrefSender, _Env>,
      __q<__value_tuple>,
      __nullable_variant_t>;

    template <class _Tag>
    using __decay_signature =
      __transform<__q<__decay_t>, __mcompose<__q<completion_signatures>, __qf<_Tag>>>;

    template <class... _Ts>
    using __all_nothrow_decay_copyable_ = __mbool<(__nothrow_decay_copyable<_Ts> && ...)>;

    template <class _CvrefSender, class _Env>
    using __all_values_and_errors_nothrow_decay_copyable = //
      __compl_sigs::__maybe_for_all_sigs<
        __completion_signatures_of_t<_CvrefSender, _Env>,
        __q<__all_nothrow_decay_copyable_>,
        __q<__mand>>;

    template <class _CvrefSender, class _Env>
    using __with_error_t = //
      __if<
        __all_values_and_errors_nothrow_decay_copyable<_CvrefSender, _Env>,
        completion_signatures<>,
        __with_exception_ptr>;

    template <class _Scheduler, class _CvrefSender, class _Env>
    using __completions_t = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __try_make_completion_signatures<
          schedule_result_t<_Scheduler>,
          _Env,
          __with_error_t<_CvrefSender, _Env>,
          __mconst<completion_signatures<>>>,
        __decay_signature<set_value_t>,
        __decay_signature<set_error_t>>;

    template <class _SchedulerId>
    struct __environ {
      using _Scheduler = stdexec::__t<_SchedulerId>;

      struct __t
        : __env::__with<
            stdexec::__t<_SchedulerId>,
            get_completion_scheduler_t<set_value_t>,
            get_completion_scheduler_t<set_stopped_t>> {
        using __id = __environ;

        explicit __t(_Scheduler __sched) noexcept
          : __t::__with{std::move(__sched)} {
        }

        template <same_as<get_domain_t> _Key>
        friend auto tag_invoke(_Key, const __t& __self) noexcept {
          return query_or(get_domain, __self.__value_, default_domain());
        }
      };
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state;

    // This receiver is to be completed on the execution context
    // associated with the scheduler. When the source sender
    // completes, the completion information is saved off in the
    // operation state so that when this receiver completes, it can
    // read the completion out of the operation state and forward it
    // to the output receiver after transitioning to the scheduler's
    // context.
    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __receiver2 : receiver_adaptor<__receiver2<_Scheduler, _Sexpr, _Receiver>> {
      explicit __receiver2(__state<_Scheduler, _Sexpr, _Receiver>* __state) noexcept
        : __state_{__state} {
      }

      auto base() && noexcept -> _Receiver&& {
        return std::move(__state_->__receiver());
      }

      auto base() const & noexcept -> const _Receiver& {
        return __state_->__receiver();
      }

      void set_value() && noexcept {
        STDEXEC_ASSERT(!__state_->__data_.valueless_by_exception());
        std::visit(
          [__state = __state_]<class _Tup>(_Tup& __tupl) noexcept -> void {
            if constexpr (same_as<_Tup, std::monostate>) {
              std::terminate(); // reaching this indicates a bug in schedule_from
            } else {
              __tup::__apply(
                [&]<class... _Args>(auto __tag, _Args&... __args) noexcept -> void {
                  __tag(std::move(__state->__receiver()), static_cast<_Args&&>(__args)...);
                },
                __tupl);
            }
          },
          __state_->__data_);
      }

      __state<_Scheduler, _Sexpr, _Receiver>* __state_;
    };

    template <class _Scheduler, class _Sexpr, class _Receiver>
    struct __state
      : __enable_receiver_from_this<_Sexpr, _Receiver>
      , __immovable {
      using __variant_t = __variant_for_t<__child_of<_Sexpr>, env_of_t<_Receiver>>;
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
        []<class _Sender, class _Env>(_Sender&&, const _Env&) noexcept
        -> __completions_t<__scheduler_t<_Sender>, __child_of<_Sender>, _Env> {
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

      static constexpr auto complete = []<class _Tag, class... _Args>(
                                         __ignore,
                                         auto& __state,
                                         auto& __rcvr,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        STDEXEC_APPLE_CLANG(__state.__self_ == &__state ? void() : std::terminate());
        // Write the tag and the args into the operation state so that
        // we can forward the completion from within the scheduler's
        // execution context.
        using __async_result = __value_tuple<_Tag, _Args...>;
        constexpr bool __nothrow_ =
          noexcept(__async_result{_Tag(), static_cast<_Args&&>(__args)...});
        auto __emplace_result = [&]() noexcept(__nothrow_) {
          return __async_result{_Tag(), static_cast<_Args&&>(__args)...};
        };
        if constexpr (__nothrow_) {
          __state.__data_.template emplace<__async_result>(__conv{__emplace_result});
        } else {
          try {
            __state.__data_.template emplace<__async_result>(__conv{__emplace_result});
          } catch (...) {
            set_error(std::move(__rcvr), std::current_exception());
            return;
          }
        }
        // Enqueue the schedule operation so the completion happens
        // on the scheduler's execution context.
        stdexec::start(__state.__state2_);
      };
    };
  } // namespace __schedule_from

  using __schedule_from::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  template <>
  struct __sexpr_impl<schedule_from_t> : __schedule_from::__schedule_from_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.continue_on]
  namespace __continue_on {
    using __schedule_from::__environ;

    template <class _Env>
    using __scheduler_t = __result_of<get_completion_scheduler<set_value_t>, _Env>;

    template <class _Sender>
    using __lowered_t = //
      __result_of<schedule_from, __scheduler_t<__data_of<_Sender>>, __child_of<_Sender>>;

    struct continue_on_t {
      template <sender _Sender, scheduler _Scheduler>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        using _Env = __t<__environ<__id<__decay_t<_Scheduler>>>>;
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<continue_on_t>(
            _Env{{static_cast<_Scheduler&&>(__sched)}}, static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Scheduler&& __sched) const
        -> __binder_back<continue_on_t, __decay_t<_Scheduler>> {
        return {{static_cast<_Scheduler&&>(__sched)}};
      }

      //////////////////////////////////////////////////////////////////////////////////////////////
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<
          tag_invoke_t(
            continue_on_t,
            get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
            _Sender,
            get_completion_scheduler_t<set_value_t>(_Env)),
          tag_invoke_t(continue_on_t, _Sender, get_completion_scheduler_t<set_value_t>(_Env))>;

      template <class _Env>
      static auto __transform_sender_fn(const _Env&) {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          auto __sched = get_completion_scheduler<set_value_t>(__data);
          return schedule_from(std::move(__sched), static_cast<_Child&&>(__child));
        };
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn(__env));
      }
    };

    struct __continue_on_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data, class _Child>(const _Data& __data, const _Child& __child) noexcept
        -> decltype(auto) {
        return __env::__join(__data, stdexec::get_env(__child));
      };
    };
  } // namespace __continue_on

  using __continue_on::continue_on_t;
  inline constexpr continue_on_t continue_on{};

  using transfer_t = continue_on_t;
  inline constexpr transfer_t transfer{};

  template <>
  struct __sexpr_impl<continue_on_t> : __continue_on::__continue_on_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    // This is a helper for finding legacy cusutomizations of transfer_just.
    inline auto __transfer_just_tag_invoke() {
      return []<class... _Ts>(_Ts&&... __ts) -> tag_invoke_result_t<transfer_just_t, _Ts...> {
        return tag_invoke(transfer_just, static_cast<_Ts&&>(__ts)...);
      };
    }

    template <class _Env>
    auto __make_transform_fn(const _Env&) {
      return [&]<class _Scheduler, class... _Values>(_Scheduler&& __sched, _Values&&... __vals) {
        return transfer(just(static_cast<_Values&&>(__vals)...), static_cast<_Scheduler&&>(__sched));
      };
    }

    template <class _Env>
    auto __transform_sender_fn(const _Env& __env) {
      return [&]<class _Data>(__ignore, _Data&& __data) {
        return __tup::__apply(__make_transform_fn(__env), static_cast<_Data&&>(__data));
      };
    }

    struct transfer_just_t {
      using _Data = __0;
      using __legacy_customizations_t = //
        __types<__tup::__apply_t(decltype(__transfer_just_tag_invoke()), _Data)>;

      template <scheduler _Scheduler, __movable_value... _Values>
      auto
        operator()(_Scheduler&& __sched, _Values&&... __vals) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_just_t>(
            __tuple{static_cast<_Scheduler&&>(__sched), static_cast<_Values&&>(__vals)...}));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(static_cast<_Sender&&>(__sndr), __transform_sender_fn(__env));
      }
    };

    inline auto __make_env_fn() noexcept {
      return []<class _Scheduler>(const _Scheduler& __sched, const auto&...) noexcept {
        using _Env = __t<__schedule_from::__environ<__id<_Scheduler>>>;
        return _Env{__sched};
      };
    }

    struct __transfer_just_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data) noexcept {
          return __tup::__apply(__make_env_fn(), __data);
        };
    };
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  template <>
  struct __sexpr_impl<transfer_just_t> : __transfer_just::__transfer_just_impl { };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write_ {
    struct __write_t {
      template <sender _Sender, class... _Envs>
      auto operator()(_Sender&& __sndr, _Envs... __envs) const {
        return __make_sexpr<__write_t>(
          __env::__join(static_cast<_Envs&&>(__envs)...), static_cast<_Sender&&>(__sndr));
      }

      template <class... _Envs>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Envs... __envs) const -> __binder_back<__write_t, _Envs...> {
        return {{static_cast<_Envs&&>(__envs)...}};
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, const auto& __state, __ignore) noexcept {
          return __env::__join(__state, static_cast<_Env&&>(__env));
        };
      }

      template <sender_expr_for<__write_t> _Self, class _Env>
      static auto transform_env(const _Self& __self, _Env&& __env) noexcept {
        return __sexpr_apply(__self, __transform_env_fn(static_cast<_Env&&>(__env)));
      }
    };

    struct __write_impl : __sexpr_defaults {
      static constexpr auto get_env = //
        [](__ignore, const auto& __state, const auto& __rcvr) noexcept {
          return __env::__join(__state, stdexec::get_env(__rcvr));
        };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept
        -> stdexec::__completion_signatures_of_t<
          __child_of<_Self>,
          __env::__join_t<const __decay_t<__data_of<_Self>>&, _Env>> {
        static_assert(sender_expr_for<_Self, __write_t>);
        return {};
      };
    };
  } // namespace __write_

  using __write_::__write_t;
  inline constexpr __write_t __write{};

  template <>
  struct __sexpr_impl<__write_t> : __write_::__write_impl { };

  namespace __detail {
    template <class _Env, class _Scheduler>
    STDEXEC_ATTRIBUTE((always_inline))
    auto
      __mkenv_sched(_Env&& __env, _Scheduler __sched) {
      auto __env2 = __env::__join(
        __env::__with(__sched, get_scheduler),
        __env::__without(static_cast<_Env&&>(__env), get_domain));
      using _Env2 = decltype(__env2);

      struct __env_t : _Env2 { };

      return __env_t{static_cast<_Env2&&>(__env2)};
    }

    template <class _Ty, class = __name_of<__decay_t<_Ty>>>
    struct __always {
      _Ty __val_;

      auto operator()() noexcept -> _Ty {
        return static_cast<_Ty&&>(__val_);
      }
    };

    template <class _Ty>
    __always(_Ty) -> __always<_Ty>;
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.start_on]
  namespace __start_on {
    struct start_on_t {
      using _Sender = __1;
      using _Scheduler = __0;
      using __legacy_customizations_t = __types<tag_invoke_t(start_on_t, _Scheduler, _Sender)>;

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<start_on_t>(
            static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr)));
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, auto __sched, __ignore) noexcept {
          return __detail::__mkenv_sched(static_cast<_Env&&>(__env), __sched);
        };
      }

      template <class _Sender, class _Env>
      static auto transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn(static_cast<_Env&&>(__env)));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          []<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
            return let_value(schedule(__data), __detail::__always{static_cast<_Child&&>(__child)});
          });
      }
    };
  } // namespace __start_on

  using __start_on::start_on_t;
  inline constexpr start_on_t start_on{};

  using on_t = start_on_t;
  inline constexpr on_t on{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  namespace __into_variant {
    template <class _Sender, class _Env>
      requires sender_in<_Sender, _Env>
    using __into_variant_result_t = value_types_of_t<_Sender, _Env>;

    template <class _Sender, class _Env>
    using __variant_t = __try_value_types_of_t<_Sender, _Env>;

    template <class _Variant>
    using __variant_completions =
      completion_signatures<set_value_t(_Variant), set_error_t(std::exception_ptr)>;

    template <class _Sender, class _Env>
    using __compl_sigs = //
      __try_make_completion_signatures<
        _Sender,
        _Env,
        __meval<__variant_completions, __variant_t<_Sender, _Env>>,
        __mconst<completion_signatures<>>>;

    struct into_variant_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<into_variant_t>(__(), std::forward<_Sender>(__sndr)));
      }

      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()() const noexcept -> __binder_back<into_variant_t> {
        return {};
      }
    };

    struct __into_variant_impl : __sexpr_defaults {
      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&&, _Receiver&) noexcept {
          using __variant_t = value_types_of_t<__child_of<_Self>, env_of_t<_Receiver>>;
          return __mtype<__variant_t>();
        };

      static constexpr auto complete = //
        []<class _State, class _Receiver, class _Tag, class... _Args>(
          __ignore,
          _State,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          using __variant_t = __t<_State>;
          try {
            set_value(
              static_cast<_Receiver&&>(__rcvr),
              __variant_t{std::tuple<_Args&&...>{static_cast<_Args&&>(__args)...}});
          } catch (...) {
            set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };

      static constexpr auto get_completion_signatures =       //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept //
        -> __compl_sigs<__child_of<_Self>, _Env> {
        static_assert(sender_expr_for<_Self, into_variant_t>);
        return {};
      };
    };
  } // namespace __into_variant

  using __into_variant::into_variant_t;
  inline constexpr into_variant_t into_variant{};

  template <>
  struct __sexpr_impl<into_variant_t> : __into_variant::__into_variant_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    enum __state_t {
      __started,
      __error,
      __stopped
    };

    struct __on_stop_request {
      inplace_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Env>
    auto __mkenv(_Env&& __env, const inplace_stop_source& __stop_source) noexcept {
      return __env::__join(
        __env::__with(__stop_source.get_token(), get_stop_token), static_cast<_Env&&>(__env));
    }

    template <class _Env>
    using __env_t = //
      decltype(__mkenv(__declval<_Env>(), __declval<inplace_stop_source&>()));

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __mvalid<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    template <
      __mstring _Context = "In stdexec::when_all()..."_mstr,
      __mstring _Diagnostic =
        "The given sender can complete successfully in more that one way. "
        "Use stdexec::when_all_with_variant() instead."_mstr>
    struct _INVALID_WHEN_ALL_ARGUMENT_;

    template <class _Sender, class _Env>
    using __too_many_value_completions_error =
      __mexception<_INVALID_WHEN_ALL_ARGUMENT_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>>;

    template <class _Sender, class _Env, class _ValueTuple, class... _Rest>
    using __value_tuple_t = __minvoke<
      __if_c<(0 == sizeof...(_Rest)), __mconst<_ValueTuple>, __q<__too_many_value_completions_error>>,
      _Sender,
      _Env>;

    template <class _Env, class _Sender>
    using __single_values_of_t = //
      __try_value_types_of_t<
        _Sender,
        _Env,
        __transform<__q<__decay_t>, __q<__types>>,
        __mbind_front_q<__value_tuple_t, _Sender, _Env>>;

    template <class _Env, class... _Senders>
    using __set_values_sig_t = //
      __meval<
        completion_signatures,
        __minvoke<__mconcat<__qf<set_value_t>>, __single_values_of_t<_Env, _Senders>...>>;

    template <class... _Args>
    using __all_nothrow_decay_copyable_ = __mbool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Env, class... _Senders>
    using __all_nothrow_decay_copyable = //
      __mand<__compl_sigs::__maybe_for_all_sigs<
        __completion_signatures_of_t<_Senders, _Env>,
        __q<__all_nothrow_decay_copyable_>,
        __q<__mand>>...>;

    template <class _Env, class... _Senders>
    using __completions_t = //
      __concat_completion_signatures_t<
        __if<
          __all_nothrow_decay_copyable<_Env, _Senders...>,
          completion_signatures<set_stopped_t()>,
          completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>,
        __minvoke<
          __with_default<__mbind_front_q<__set_values_sig_t, _Env>, completion_signatures<>>,
          _Senders...>,
        __try_make_completion_signatures<
          _Senders,
          _Env,
          completion_signatures<>,
          __mconst<completion_signatures<>>,
          __mcompose<__q<completion_signatures>, __qf<set_error_t>, __q<__decay_t>>>...>;

    struct __not_an_error { };

    struct __tie_fn {
      template <class... _Ty>
      auto operator()(_Ty&... __vals) noexcept -> std::tuple<_Ty&...> {
        return std::tuple<_Ty&...>{__vals...};
      }
    };

    template <class _Tag, class _Receiver>
    auto __complete_fn(_Tag, _Receiver& __rcvr) noexcept {
      return [&]<class... _Ts>(_Ts&... __ts) noexcept {
        if constexpr (!same_as<__types<_Ts...>, __types<__not_an_error>>) {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Ts&&>(__ts)...);
        }
      };
    }

    template <class _Receiver, class _ValuesTuple>
    void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept {
      __tup::__apply(
        [&](auto&... __opt_vals) noexcept -> void {
          __apply(
            __complete_fn(set_value, __rcvr), //
            std::tuple_cat(__tup::__apply(__tie_fn{}, *__opt_vals)...));
        },
        __values);
    }

    template <class... Ts>
    using __decayed_custom_tuple = __tup::__tuple_for<__decay_t<Ts>...>;

    template <class _Env, class _Sender>
    using __values_opt_tuple_t = //
      value_types_of_t<_Sender, __env_t<_Env>, __decayed_custom_tuple, std::optional>;

    template <class _Env, __max1_sender<__env_t<_Env>>... _Senders>
    struct __traits {
      // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
      using __values_tuple = //
        __minvoke<
          __with_default<
            __transform<__mbind_front_q<__values_opt_tuple_t, _Env>, __q<__tup::__tuple_for>>,
            __ignore>,
          _Senders...>;

      using __nullable_variant_t_ = __munique<__mbind_front_q<std::variant, __not_an_error>>;

      using __error_types = //
        __minvoke<
          __mconcat<__transform<__q<__decay_t>, __nullable_variant_t_>>,
          error_types_of_t<_Senders, __env_t<_Env>, __types>...>;

      using __errors_variant = //
        __if<
          __all_nothrow_decay_copyable<_Env, _Senders...>,
          __error_types,
          __minvoke<__push_back_unique<__q<std::variant>>, __error_types, std::exception_ptr>>;
    };

    struct _INVALID_ARGUMENTS_TO_WHEN_ALL_ { };

    template <class _ErrorsVariant, class _ValuesTuple, class _StopToken>
    struct __when_all_state {
      using __stop_callback_t = typename _StopToken::template callback_type<__on_stop_request>;

      template <class _Receiver>
      void __arrive(_Receiver& __rcvr) noexcept {
        if (0 == --__count_) {
          __complete(__rcvr);
        }
      }

      template <class _Receiver>
      void __complete(_Receiver& __rcvr) noexcept {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(std::memory_order_relaxed)) {
        case __started:
          if constexpr (!same_as<_ValuesTuple, __ignore>) {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr, __values_);
          }
          break;
        case __error:
          if constexpr (!same_as<_ErrorsVariant, std::variant<std::monostate>>) {
            // One or more child operations completed with an error:
            std::visit(__complete_fn(set_error, __rcvr), __errors_);
          }
          break;
        case __stopped:
          stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr));
          break;
        default:;
        }
      }

      std::atomic<std::size_t> __count_;
      inplace_stop_source __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      std::atomic<__state_t> __state_{__started};
      _ErrorsVariant __errors_{};
      STDEXEC_ATTRIBUTE((no_unique_address))
      _ValuesTuple __values_{};
      std::optional<__stop_callback_t> __on_stop_{};
    };

    template <class _Env>
    static auto __mk_state_fn(const _Env&) noexcept {
      return [&]<__max1_sender<__env_t<_Env>>... _Child>(__ignore, __ignore, _Child&&...) {
        using _Traits = __traits<_Env, _Child...>;
        using _ErrorsVariant = typename _Traits::__errors_variant;
        using _ValuesTuple = typename _Traits::__values_tuple;
        using _State = __when_all_state<_ErrorsVariant, _ValuesTuple, stop_token_of_t<_Env>>;
        return _State{
          sizeof...(_Child),
          inplace_stop_source{},
          __started,
          _ErrorsVariant{},
          _ValuesTuple{},
          std::nullopt};
      };
    }

    template <class _Env>
    using __mk_state_fn_t = decltype(__when_all::__mk_state_fn(__declval<_Env>()));

    struct when_all_t {
      // Used by the default_domain to find legacy customizations:
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(when_all_t, _Sender...)>;

      template <sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = __domain::__common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain, __make_sexpr<when_all_t>(__(), static_cast<_Senders&&>(__sndrs)...));
      }
    };

    struct __when_all_impl : __sexpr_defaults {
      template <class _Self, class _Env>
      using __error_t = __mexception<
        _INVALID_ARGUMENTS_TO_WHEN_ALL_,
        __children_of<_Self, __q<_WITH_SENDERS_>>,
        _WITH_ENVIRONMENT_<_Env>>;

      template <class _Self, class _Env>
      using __completions = //
        __children_of<_Self, __mbind_front_q<__completions_t, __env_t<_Env>>>;

      static constexpr auto get_attrs = //
        []<class... _Child>(__ignore, const _Child&...) noexcept {
          using _Domain = __domain::__common_domain_t<_Child...>;
          if constexpr (same_as<_Domain, default_domain>) {
            return empty_env();
          } else {
            return __env::__with(_Domain(), get_domain);
          }
        };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept {
          static_assert(sender_expr_for<_Self, when_all_t>);
          return __minvoke<__mtry_catch<__q<__completions>, __q<__error_t>>, _Self, _Env>();
        };

      static constexpr auto get_env = //
        []<class _State, class _Receiver>(
          __ignore,
          _State& __state,
          const _Receiver& __rcvr) noexcept //
        -> __env_t<env_of_t<const _Receiver&>> {
        return __mkenv(stdexec::get_env(__rcvr), __state.__stop_source_);
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver& __rcvr)
        -> __sexpr_apply_result_t<_Self, __mk_state_fn_t<env_of_t<_Receiver>>> {
        return __sexpr_apply(
          static_cast<_Self&&>(__self), __when_all::__mk_state_fn(stdexec::get_env(__rcvr)));
      };

      static constexpr auto start = //
        []<class _State, class _Receiver, class... _Operations>(
          _State& __state,
          _Receiver& __rcvr,
          _Operations&... __child_ops) noexcept -> void {
        // register stop callback:
        __state.__on_stop_.emplace(
          get_stop_token(stdexec::get_env(__rcvr)), __on_stop_request{__state.__stop_source_});
        if (__state.__stop_source_.stop_requested()) {
          // Stop has already been requested. Don't bother starting
          // the child operations.
          stdexec::set_stopped(std::move(__rcvr));
        } else {
          (stdexec::start(__child_ops), ...);
          if constexpr (sizeof...(__child_ops) == 0) {
            __state.__complete(__rcvr);
          }
        }
      };

      template <class _State, class _Receiver, class _Error>
      static void __set_error(_State& __state, _Receiver&, _Error&& __err) noexcept {
        // TODO: What memory orderings are actually needed here?
        if (__error != __state.__state_.exchange(__error)) {
          __state.__stop_source_.request_stop();
          // We won the race, free to write the error into the operation
          // state without worry.
          if constexpr (__nothrow_decay_copyable<_Error>) {
            __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
          } else {
            try {
              __state.__errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
            } catch (...) {
              __state.__errors_.template emplace<std::exception_ptr>(std::current_exception());
            }
          }
        }
      }

      static constexpr auto complete = //
        []<class _Index, class _State, class _Receiver, class _Set, class... _Args>(
          _Index,
          _State& __state,
          _Receiver& __rcvr,
          _Set,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Set, set_error_t>) {
          __set_error(__state, __rcvr, static_cast<_Args&&>(__args)...);
        } else if constexpr (same_as<_Set, set_stopped_t>) {
          __state_t __expected = __started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (__state.__state_.compare_exchange_strong(__expected, __stopped)) {
            __state.__stop_source_.request_stop();
          }
        } else if constexpr (!same_as<decltype(_State::__values_), __ignore>) {
          // We only need to bother recording the completion values
          // if we're not already in the "error" or "stopped" state.
          if (__state.__state_ == __started) {
            auto& __opt_values = __tup::__get<__v<_Index>>(__state.__values_);
            using _Tuple = __decayed_custom_tuple<_Args...>;
            static_assert(
              same_as<decltype(*__opt_values), _Tuple&>,
              "One of the senders in this when_all() is fibbing about what types it sends");
            if constexpr ((__nothrow_decay_copyable<_Args> && ...)) {
              __opt_values.emplace(_Tuple{{static_cast<_Args&&>(__args)}...});
            } else {
              try {
                __opt_values.emplace(_Tuple{{static_cast<_Args&&>(__args)}...});
              } catch (...) {
                __set_error(__state, __rcvr, std::current_exception());
              }
            }
          }
        }

        __state.__arrive(__rcvr);
      };
    };

    struct when_all_with_variant_t {
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(when_all_with_variant_t, _Sender...)>;

      template <sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const -> __well_formed_sender auto {
        auto __domain = __domain::__common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<when_all_with_variant_t>(__(), static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the when_all_with_variant into a regular when_all (looking for
        // early when_all customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class... _Child>(__ignore, __ignore, _Child&&... __child) {
            return when_all_t()(into_variant(static_cast<_Child&&>(__child))...);
          });
      }
    };

    struct __when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class... _Child>(__ignore, const _Child&...) noexcept {
          using _Domain = __domain::__common_domain_t<_Child...>;
          if constexpr (same_as<_Domain, default_domain>) {
            return empty_env();
          } else {
            return __env::__with(_Domain(), get_domain);
          }
        };
    };

    struct transfer_when_all_t {
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(
          transfer_when_all_t,
          get_completion_scheduler_t<set_value_t>(const _Env&),
          _Sender...)>;

      template <scheduler _Scheduler, sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto
        operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        using _Env = __t<__schedule_from::__environ<__id<__decay_t<_Scheduler>>>>;
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_t>(
            _Env{static_cast<_Scheduler&&>(__sched)}, static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the transfer_when_all into a regular transform | when_all
        // (looking for early customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer(
              when_all_t()(static_cast<_Child&&>(__child)...),
              get_completion_scheduler<set_value_t>(__data));
          });
      }
    };

    struct __transfer_when_all_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data, const auto&...) noexcept -> const _Data& {
        return __data;
      };
    };

    struct transfer_when_all_with_variant_t {
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(
          transfer_when_all_with_variant_t,
          get_completion_scheduler_t<set_value_t>(const _Env&),
          _Sender...)>;

      template <scheduler _Scheduler, sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto
        operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const -> __well_formed_sender auto {
        using _Env = __t<__schedule_from::__environ<__id<__decay_t<_Scheduler>>>>;
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_with_variant_t>(
            _Env{{static_cast<_Scheduler&&>(__sched)}}, static_cast<_Senders&&>(__sndrs)...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        // transform the transfer_when_all_with_variant into regular transform_when_all
        // and into_variant calls/ (looking for early customizations), then transform it
        // again to look for late customizations.
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer_when_all_t()(
              get_completion_scheduler<set_value_t>(static_cast<_Data&&>(__data)),
              into_variant(static_cast<_Child&&>(__child))...);
          });
      }
    };

    struct __transfer_when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data, const auto&...) noexcept -> const _Data& {
        return __data;
      };
    };
  } // namespace __when_all

  using __when_all::when_all_t;
  inline constexpr when_all_t when_all{};

  using __when_all::when_all_with_variant_t;
  inline constexpr when_all_with_variant_t when_all_with_variant{};

  using __when_all::transfer_when_all_t;
  inline constexpr transfer_when_all_t transfer_when_all{};

  using __when_all::transfer_when_all_with_variant_t;
  inline constexpr transfer_when_all_with_variant_t transfer_when_all_with_variant{};

  template <>
  struct __sexpr_impl<when_all_t> : __when_all::__when_all_impl { };

  template <>
  struct __sexpr_impl<when_all_with_variant_t> : __when_all::__when_all_with_variant_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_t> : __when_all::__transfer_when_all_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_with_variant_t>
    : __when_all::__transfer_when_all_with_variant_impl { };

  namespace __read {
    template <class _Tag, class _ReceiverId>
    using __result_t = __call_result_t<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    template <class _Tag, class _ReceiverId>
    concept __nothrow_t = __nothrow_callable<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    inline constexpr __mstring __query_failed_diag =
      "The current execution environment doesn't have a value for the given query."_mstr;

    template <class _Tag>
    struct _WITH_QUERY_;

    template <class _Tag, class _Env>
    using __query_failed_error = //
      __mexception<              //
        _NOT_CALLABLE_<"In stdexec::read()..."_mstr, __query_failed_diag>,
        _WITH_QUERY_<_Tag>,
        _WITH_ENVIRONMENT_<_Env>>;

    template <class _Tag, class _Env>
      requires __callable<_Tag, _Env>
    using __completions_t = //
      __if_c<
        __nothrow_callable<_Tag, _Env>,
        completion_signatures<set_value_t(__call_result_t<_Tag, _Env>)>,
        completion_signatures<
          set_value_t(__call_result_t<_Tag, _Env>),
          set_error_t(std::exception_ptr)>>;

    template <class _Tag, class _Ty>
    struct __state {
      using __query = _Tag;
      using __result = _Ty;
      std::optional<_Ty> __result_;
    };

    template <class _Tag, class _Ty>
      requires same_as<_Ty, _Ty&&>
    struct __state<_Tag, _Ty> {
      using __query = _Tag;
      using __result = _Ty;
    };

    struct __read_t {
      template <class _Tag>
      constexpr auto operator()(_Tag) const noexcept {
        return __make_sexpr<__read_t>(_Tag());
      }
    };

    struct __read_impl : __sexpr_defaults {
      using is_dependent = void;

      template <class _Tag, class _Env>
      using __completions_t =
        __minvoke<__mtry_catch_q<__read::__completions_t, __q<__query_failed_error>>, _Tag, _Env>;

      static constexpr auto get_completion_signatures =            //
        []<class _Self, class _Env>(const _Self&, _Env&&) noexcept //
        -> __completions_t<__data_of<_Self>, _Env> {
        static_assert(sender_expr_for<_Self, __read_t>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(const _Self&, _Receiver&) noexcept {
          using __query = __data_of<_Self>;
          using __result = __call_result_t<__query, env_of_t<_Receiver>>;
          return __state<__query, __result>();
        };

      static constexpr auto start = //
        []<class _State, class _Receiver>(_State& __state, _Receiver& __rcvr) noexcept -> void {
        using __query = typename _State::__query;
        using __result = typename _State::__result;
        if constexpr (same_as<__result, __result&&>) {
          // The query returns a reference type; pass it straight through to the receiver.
          stdexec::__set_value_invoke(std::move(__rcvr), __query(), stdexec::get_env(__rcvr));
        } else {
          constexpr bool _Nothrow = __nothrow_callable<__query, env_of_t<_Receiver>>;
          auto __query_fn = [&]() noexcept(_Nothrow) -> __result&& {
            __state.__result_.emplace(__conv{[&]() noexcept(_Nothrow) {
              return __query()(stdexec::get_env(__rcvr));
            }});
            return std::move(*__state.__result_);
          };
          stdexec::__set_value_invoke(std::move(__rcvr), __query_fn);
        }
      };
    };
  } // namespace __read

  inline constexpr __read::__read_t read{};

  template <>
  struct __sexpr_impl<__read::__read_t> : __read::__read_impl { };

  namespace __queries {
    template <class _Tag>
    inline auto get_scheduler_t::operator()() const noexcept {
      return read(get_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_scheduler_t, const _Env&>
    inline auto get_scheduler_t::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_scheduler_t, const _Env&>>);
      return tag_invoke(get_scheduler_t{}, __env);
    }

    template <class _Tag>
    inline auto get_delegatee_scheduler_t::operator()() const noexcept {
      return read(get_delegatee_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
    inline auto get_delegatee_scheduler_t::operator()(const _Env& __t) const noexcept
      -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_delegatee_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>>);
      return tag_invoke(get_delegatee_scheduler_t{}, std::as_const(__t));
    }

    template <class _Tag>
    inline auto get_allocator_t::operator()() const noexcept {
      return read(get_allocator);
    }

    template <class _Tag>
    inline auto get_stop_token_t::operator()() const noexcept {
      return read(get_stop_token);
    }

    template <__completion_tag _CPO>
    template <__has_completion_scheduler_for<_CPO> _Queryable>
    auto get_completion_scheduler_t<_CPO>::operator()(const _Queryable& __queryable) const noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&> {
      static_assert(
        nothrow_tag_invocable<get_completion_scheduler_t<_CPO>, const _Queryable&>,
        "get_completion_scheduler<_CPO> should be noexcept");
      static_assert(
        scheduler<tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&>>);
      return tag_invoke(*this, __queryable);
    }
  } // namespace __queries

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on_v2 {
    inline constexpr __mstring __on_context = "In stdexec::on(Scheduler, Sender)..."_mstr;
    inline constexpr __mstring __no_scheduler_diag =
      "stdexec::on() requires a scheduler to transition back to."_mstr;
    inline constexpr __mstring __no_scheduler_details =
      "The provided environment lacks a value for the get_scheduler() query."_mstr;

    template <
      __mstring _Context = __on_context,
      __mstring _Diagnostic = __no_scheduler_diag,
      __mstring _Details = __no_scheduler_details>
    struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_ { };

    struct on_t;

    template <class _Sender, class _Env>
    struct __no_scheduler_in_environment {
      using sender_concept = sender_t;

      STDEXEC_MEMFN_DECL(auto get_completion_signatures)(this const __no_scheduler_in_environment&, const auto&) noexcept {
        return __mexception<
          _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_<>,
          _WITH_SENDER_<_Sender>,
          _WITH_ENVIRONMENT_<_Env>>{};
      }
    };

    template <class _Scheduler, class _Closure>
    struct __continue_on_data {
      _Scheduler __sched_;
      _Closure __clsur_;
    };
    template <class _Scheduler, class _Closure>
    __continue_on_data(_Scheduler, _Closure) -> __continue_on_data<_Scheduler, _Closure>;

    template <class _Scheduler>
    struct __with_sched {
      _Scheduler __sched_;

      STDEXEC_MEMFN_DECL(auto query)(this const __with_sched& __self, get_scheduler_t) noexcept -> _Scheduler {
        return __self.__sched_;
      }

      STDEXEC_MEMFN_DECL(auto query)(this const __with_sched& __self, get_domain_t) noexcept {
        return query_or(get_domain, __self.__sched_, default_domain());
      }
    };

    template <class _Scheduler>
    __with_sched(_Scheduler) -> __with_sched<_Scheduler>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<on_t>(static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr)));
      }

      template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
        -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<on_t>(
            __continue_on_data{static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)},
            static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Scheduler&& __sched, _Closure&& __clsur) const {
        return __binder_back<on_t, __decay_t<_Scheduler>, __decay_t<_Closure>>{
          {static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)}
        };
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        __transform_env_fn(_Env&& __env) noexcept {
        return [&]<class _Data>(__ignore, _Data&& __data, __ignore) noexcept -> decltype(auto) {
          if constexpr (scheduler<_Data>) {
            return __detail::__mkenv_sched(static_cast<_Env&&>(__env), static_cast<_Data&&>(__data));
          } else {
            return static_cast<_Env>(static_cast<_Env&&>(__env));
          }
        };
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        __transform_sender_fn(const _Env& __env) noexcept {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          if constexpr (scheduler<_Data>) {
            // This branch handles the case where `on` was called like `on(sch, snd)`
            auto __old = query_or(get_scheduler, __env, __none_such{});
            if constexpr (same_as<decltype(__old), __none_such>) {
              if constexpr (__is_root_env<_Env>) {
                return continue_on(
                  start_on(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)),
                  std::move(__inln::__scheduler{}));
              } else {
                return __none_such{};
              }
            } else {
              return continue_on(
                start_on(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)),
                std::move(__old));
            }
          } else {
            // This branch handles the case where `on` was called like `on(snd, sch, clsur)`
            auto __old = query_or(
              get_completion_scheduler<set_value_t>,
              get_env(__child),
              query_or(get_scheduler, __env, __none_such{}));
            if constexpr (same_as<decltype(__old), __none_such>) {
              return __none_such{};
            } else {
              auto&& [__sched, __clsur] = static_cast<_Data&&>(__data);
              return __write(                                                       //
                continue_on(                                                        //
                  __forward_like<_Data>(__clsur)(                                   //
                    continue_on(                                                    //
                      __write(static_cast<_Child&&>(__child), __with_sched{__old}), //
                      __sched)),                                                    //
                  __old),
                __with_sched{__sched});
            }
          }
        };
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn(static_cast<_Env&&>(__env)));
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto
        transform_sender(_Sender&& __sndr, const _Env& __env) {
        auto __tfx_sndr_fn = __transform_sender_fn(__env);
        using _TfxSndrFn = decltype(__tfx_sndr_fn);
        using _NewSndr = __sexpr_apply_result_t<_Sender, _TfxSndrFn>;
        if constexpr (same_as<_NewSndr, __none_such>) {
          return __no_scheduler_in_environment<_Sender, _Env>{};
        } else {
          return __sexpr_apply(
            static_cast<_Sender&&>(__sndr), static_cast<_TfxSndrFn&&>(__tfx_sndr_fn));
        }
      }
    };
  } // namespace __on_v2

  namespace v2 {
    using __on_v2::on_t;
    inline constexpr on_t on{};

    using continue_on_t = v2::on_t;
    inline constexpr continue_on_t continue_on{}; // for back-compat
  }                                               // namespace v2

  template <>
  struct __sexpr_impl<v2::on_t> : __sexpr_defaults {
    using is_dependent = void;
  };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    inline auto __make_env(run_loop& __loop) noexcept {
      return __env::__with(__loop.get_scheduler(), get_scheduler, get_delegatee_scheduler);
    }

    struct __env : __result_of<__make_env, run_loop&> {
      __env();

      explicit __env(run_loop& __loop) noexcept
        : __result_of<__make_env, run_loop&>{__sync_wait::__make_env(__loop)} {
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _Sender, class _Continuation>
    using __sync_wait_result_impl = //
      __try_value_types_of_t<
        _Sender,
        __env,
        __transform<__q<__decay_t>, _Continuation>,
        __q<__msingle>>;

    template <class _Sender>
    using __sync_wait_result_t = __mtry_eval<__sync_wait_result_impl, _Sender, __q<std::tuple>>;

    template <class _Sender>
    using __sync_wait_with_variant_result_t =
      __mtry_eval<__sync_wait_result_impl, __result_of<into_variant, _Sender>, __q<__midentity>>;

    template <class... _Values>
    struct __state {
      using _Tuple = std::tuple<_Values...>;
      std::variant<std::monostate, _Tuple, std::exception_ptr, set_stopped_t> __data_{};
    };

    template <class... _Values>
    struct __receiver {
      struct __t {
        using receiver_concept = receiver_t;
        using __id = __receiver;
        __state<_Values...>* __state_;
        run_loop* __loop_;

        template <class _Error>
        void __set_error(_Error __err) noexcept {
          if constexpr (__decays_to<_Error, std::exception_ptr>)
            __state_->__data_.template emplace<2>(static_cast<_Error&&>(__err));
          else if constexpr (__decays_to<_Error, std::error_code>)
            __state_->__data_.template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
          else
            __state_->__data_.template emplace<2>(
              std::make_exception_ptr(static_cast<_Error&&>(__err)));
          __loop_->finish();
        }

        template <class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        STDEXEC_MEMFN_DECL(
          void set_value)(this __t&& __rcvr, _As&&... __as) noexcept {
          try {
            __rcvr.__state_->__data_.template emplace<1>(static_cast<_As&&>(__as)...);
            __rcvr.__loop_->finish();
          } catch (...) {
            __rcvr.__set_error(std::current_exception());
          }
        }

        template <class _Error>
        STDEXEC_MEMFN_DECL(void set_error)(this __t&& __rcvr, _Error __err) noexcept {
          __rcvr.__set_error(static_cast<_Error&&>(__err));
        }

        STDEXEC_MEMFN_DECL(void set_stopped)(this __t&& __rcvr) noexcept {
          __rcvr.__state_->__data_.template emplace<3>(set_stopped_t{});
          __rcvr.__loop_->finish();
        }

        STDEXEC_MEMFN_DECL(auto get_env)(this const __t& __rcvr) noexcept -> __env {
          return __env(*__rcvr.__loop_);
        }
      };
    };

    template <class _Sender>
    using __receiver_t = __t<__sync_wait_result_impl<_Sender, __q<__receiver>>>;

    // These are for hiding the metaprogramming in diagnostics
    template <class _Sender>
    struct __sync_receiver_for {
      using __t = __receiver_t<_Sender>;
    };
    template <class _Sender>
    using __sync_receiver_for_t = __t<__sync_receiver_for<_Sender>>;

    template <class _Sender>
    struct __value_tuple_for {
      using __t = __sync_wait_result_t<_Sender>;
    };
    template <class _Sender>
    using __value_tuple_for_t = __t<__value_tuple_for<_Sender>>;

    template <class _Sender>
    struct __variant_for {
      using __t = __sync_wait_with_variant_result_t<_Sender>;
    };
    template <class _Sender>
    using __variant_for_t = __t<__variant_for<_Sender>>;

    inline constexpr __mstring __sync_wait_context_diag = //
      "In stdexec::sync_wait()..."_mstr;
    inline constexpr __mstring __too_many_successful_completions_diag =
      "The argument to stdexec::sync_wait() is a sender that can complete successfully in more "
      "than one way. Use stdexec::sync_wait_with_variant() instead."_mstr;

    template <__mstring _Context, __mstring _Diagnostic>
    struct _INVALID_ARGUMENT_TO_SYNC_WAIT_;

    template <__mstring _Diagnostic>
    using __invalid_argument_to_sync_wait =
      _INVALID_ARGUMENT_TO_SYNC_WAIT_<__sync_wait_context_diag, _Diagnostic>;

    template <__mstring _Diagnostic, class _Sender, class _Env = __env>
    using __sync_wait_error = __mexception<
      __invalid_argument_to_sync_wait<_Diagnostic>,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_Env>>;

    template <class _Sender, class>
    using __too_many_successful_completions_error =
      __sync_wait_error<__too_many_successful_completions_diag, _Sender>;

    template <class _Sender>
    concept __valid_sync_wait_argument = __ok<__minvoke<
      __mtry_catch_q<__single_value_variant_sender_t, __q<__too_many_successful_completions_error>>,
      _Sender,
      __env>>;

#if STDEXEC_NVHPC()
    // It requires some hoop-jumping to get the NVHPC compiler to report a meaningful
    // diagnostic for SFINAE failures.
    template <class _Sender>
    auto __diagnose_error() {
      if constexpr (!sender_in<_Sender, __env>) {
        using _Completions = __completion_signatures_of_t<_Sender, __env>;
        if constexpr (__merror<_Completions>) {
          return _Completions();
        } else {
          constexpr __mstring __diag =
            "The stdexec::sender_in<Sender, Environment> concept check has failed."_mstr;
          return __sync_wait_error<__diag, _Sender>();
        }
      } else if constexpr (!__valid_sync_wait_argument<_Sender>) {
        return __sync_wait_error<__too_many_successful_completions_diag, _Sender>();
      } else if constexpr (!sender_to<_Sender, __sync_receiver_for_t<_Sender>>) {
        constexpr __mstring __diag =
          "Failed to connect the given sender to sync_wait's internal receiver. "
          "The stdexec::connect(Sender, Receiver) expression is ill-formed."_mstr;
        return __sync_wait_error<__diag, _Sender>();
      } else {
        constexpr __mstring __diag = "Unknown concept check failure."_mstr;
        return __sync_wait_error<__diag, _Sender>();
      }
    }

    template <class _Sender>
    using __error_description_t = decltype(__sync_wait::__diagnose_error<_Sender>());
#endif

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <sender_in<__env> _Sender>
        requires __valid_sync_wait_argument<_Sender>
              && __has_implementation_for<sync_wait_t, __early_domain_of_t<_Sender>, _Sender>
      auto operator()(_Sender&& __sndr) const -> std::optional<__value_tuple_for_t<_Sender>> {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

#if STDEXEC_NVHPC()
      // This is needed to get sensible diagnostics from nvc++
      template <class _Sender, class _Error = __error_description_t<_Sender>>
      auto operator()(_Sender&&, [[maybe_unused]] _Error __diagnostic = {}) const
        -> std::optional<std::tuple<int>> = delete;
#endif

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        // For legacy reasons:
        tag_invoke_t(
          sync_wait_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(sync_wait_t, _Sender)>;

      // clang-format off
      /// @brief Synchronously wait for the result of a sender, blocking the
      ///         current thread.
      ///
      /// `sync_wait` connects and starts the given sender, and then drives a
      ///         `run_loop` instance until the sender completes. Additional work
      ///         can be delegated to the `run_loop` by scheduling work on the
      ///         scheduler returned by calling `get_delegatee_scheduler` on the
      ///         receiver's environment.
      ///
      /// @pre The sender must have a exactly one value completion signature. That
      ///         is, it can only complete successfully in one way, with a single
      ///         set of values.
      ///
      /// @retval success Returns an engaged `std::optional` containing the result
      ///         values in a `std::tuple`.
      /// @retval canceled Returns an empty `std::optional`.
      /// @retval error Throws the error.
      ///
      /// @throws std::rethrow_exception(error) if the error has type
      ///         `std::exception_ptr`.
      /// @throws std::system_error(error) if the error has type
      ///         `std::error_code`.
      /// @throws error otherwise
      // clang-format on
      template <class _Sender>
        requires sender_to<_Sender, __sync_receiver_for_t<_Sender>>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        using state_t = __sync_wait_result_impl<_Sender, __q<__state>>;
        state_t __state{};
        run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state =
          connect(static_cast<_Sender&&>(__sndr), __receiver_t<_Sender>{&__state, &__loop});
        start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          std::rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait_with_variant]
    struct sync_wait_with_variant_t {
      struct __impl;

      template <sender_in<__env> _Sender>
        requires __callable<
          apply_sender_t,
          __early_domain_of_t<_Sender>,
          sync_wait_with_variant_t,
          _Sender>
      auto operator()(_Sender&& __sndr) const -> std::optional<__variant_for_t<_Sender>> {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

#if STDEXEC_NVHPC()
      template <
        class _Sender,
        class _Error = __error_description_t<__result_of<into_variant, _Sender>>>
      auto operator()(_Sender&&, [[maybe_unused]] _Error __diagnostic = {}) const
        -> std::optional<std::tuple<std::variant<std::tuple<>>>> = delete;
#endif

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        // For legacy reasons:
        tag_invoke_t(
          sync_wait_with_variant_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(sync_wait_with_variant_t, _Sender)>;

      template <class _Sender>
        requires __callable<sync_wait_t, __result_of<into_variant, _Sender>>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__variant_for_t<_Sender>> {
        if (auto __opt_values = sync_wait_t()(into_variant(static_cast<_Sender&&>(__sndr)))) {
          return std::move(std::get<0>(*__opt_values));
        }
        return std::nullopt;
      }
    };
  } // namespace __sync_wait

  using __sync_wait::sync_wait_t;
  inline constexpr sync_wait_t sync_wait{};

  using __sync_wait::sync_wait_with_variant_t;
  inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct __ignore_sender {
    using sender_concept = sender_t;

    template <sender _Sender>
    constexpr __ignore_sender(_Sender&&) noexcept {
    }
  };

  template <auto _Reason = "You cannot pipe one sender into another."_mstr>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;
} // namespace stdexec

#if STDEXEC_MSVC()
namespace stdexec {
  // MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-codegen-in-await_suspend-aroun/10454102

  // MSVC incorrectly allocates the return buffer for await_suspend calls within the suspended coroutine
  // frame. When the suspended coroutine is destroyed within await_suspend, the continuation coroutine handle
  // is not only used after free, but also overwritten by the debug malloc implementation when NRVO is in play.

  // This workaround delays the destruction of the suspended coroutine by wrapping the continuation in another
  // coroutine which destroys the former and transfers execution to the original continuation.

  // The wrapping coroutine is thread-local and is reused within the thread for each destroy-and-continue sequence.
  // The wrapping coroutine itself is destroyed at thread exit.

  namespace __destroy_and_continue_msvc {
    struct __task {
      struct promise_type {
        __task get_return_object() noexcept {
          return {__coro::coroutine_handle<promise_type>::from_promise(*this)};
        }

        static std::suspend_never initial_suspend() noexcept {
          return {};
        }

        static std::suspend_never final_suspend() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
          return {};
        }

        static void return_void() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }

        static void unhandled_exception() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }
      };

      __coro::coroutine_handle<> __coro_;
    };

    struct __continue_t {
      static constexpr bool await_ready() noexcept {
        return false;
      }

      __coro::coroutine_handle<> await_suspend(__coro::coroutine_handle<>) noexcept {
        return __continue_;
      }

      static void await_resume() noexcept {
      }

      __coro::coroutine_handle<> __continue_;
    };

    struct __context {
      __coro::coroutine_handle<> __destroy_;
      __coro::coroutine_handle<> __continue_;
    };

    inline __task __co_impl(__context& __c) {
      while (true) {
        co_await __continue_t{__c.__continue_};
        __c.__destroy_.destroy();
      }
    }

    struct __context_and_coro {
      __context_and_coro() {
        __context_.__continue_ = __coro::noop_coroutine();
        __coro_ = __co_impl(__context_).__coro_;
      }

      ~__context_and_coro() {
        __coro_.destroy();
      }

      __context __context_;
      __coro::coroutine_handle<> __coro_;
    };

    inline __coro::coroutine_handle<>
      __impl(__coro::coroutine_handle<> __destroy, __coro::coroutine_handle<> __continue) {
      static thread_local __context_and_coro __c;
      __c.__context_.__destroy_ = __destroy;
      __c.__context_.__continue_ = __continue;
      return __c.__coro_;
    }
  } // namespace __destroy_and_continue_msvc
} // namespace stdexec

#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue)                                      \
    (::stdexec::__destroy_and_continue_msvc::__impl(__destroy, __continue))
#else
#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue) (__destroy.destroy(), __continue)
#endif

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(stdexec::__ignore_sender, _Sender&&) noexcept -> stdexec::__ignore_sender;

#include "__detail/__p2300.hpp"

STDEXEC_PRAGMA_POP()
