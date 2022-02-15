/*
 * Copyright (c) NVIDIA
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

#include <atomic>
#include <barrier>
#include <cassert>
#include <condition_variable>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>
#include <type_traits>
#include <variant>

#include <__utility.hpp>
#include <functional.hpp>
#include <concepts.hpp>
#include <coroutine.hpp>
#include <stop_token.hpp>

#if defined(__clang__)
#define _STRINGIZE(__arg) #__arg
#define _PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define _PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define _PRAGMA_IGNORE(__arg) _Pragma(_STRINGIZE(GCC diagnostic ignored __arg))
#else
#define _PRAGMA_PUSH()
#define _PRAGMA_POP()
#define _PRAGMA_IGNORE(__arg)
#endif

_PRAGMA_PUSH()
_PRAGMA_IGNORE("-Wundefined-inline")
_PRAGMA_IGNORE("-Wundefined-internal")

namespace std::execution {
  enum class forward_progress_guarantee {
    concurrent,
    parallel,
    weakly_parallel
  };

  /////////////////////////////////////////////////////////////////////////////
  // env_of
  namespace __env {
    namespace __impl {
      struct __empty_env {};
      struct no_env {
        friend void tag_invoke(auto, same_as<no_env> auto, auto&&...) = delete;
      };

      template <__class _Tag, class _Value, class _BaseEnvId = __x<__empty_env>>
          requires copy_constructible<unwrap_reference_t<_Value>>
        struct __env {
          using _BaseEnv = __t<_BaseEnvId>;
          [[no_unique_address]] unwrap_reference_t<_Value> __value_;
          [[no_unique_address]] _BaseEnv __base_env_{};

          // Forward only the receiver queries:
          template <__none_of<_Tag> _Tag2, class... _As>
              requires __callable<_Tag2, const _BaseEnv&, _As...>
            friend auto tag_invoke(_Tag2 __tag, const __env& __self, _As&&... __as) noexcept
              -> __call_result_t<_Tag2, const _BaseEnv&, _As...> {
              return ((_Tag2&&) __tag)(__self.__base_env_, (_As&&) __as...);
            }

          template <__one_of<_Tag> _Tag2>
            friend auto tag_invoke(_Tag2, const __env& __self, auto&&...)
              noexcept(is_nothrow_copy_constructible_v<unwrap_reference_t<_Value>>)
              -> unwrap_reference_t<_Value> {
              return __self.__value_;
            }
        };

      template <__class _Tag>
        struct __make_env_t {
          template <class _Value>
            __env<_Tag, decay_t<_Value>> operator()(_Value&& __value) const
              noexcept(is_nothrow_copy_constructible_v<unwrap_reference_t<decay_t<_Value>>>) {
              return {(_Value&&) __value};
            }

          template <class _Value, class _BaseEnv>
            auto operator()(_Value&& __value, _BaseEnv&& __base_env) const
              -> __env<_Tag, decay_t<_Value>, __x<decay_t<_BaseEnv>>> {
              return {(_Value&&) __value, (_BaseEnv&&) __base_env};
            }

            auto operator()(auto&&, no_env) const noexcept
              -> no_env {
              return {};
            }
        };
    } // namespace __impl
    using __impl::__empty_env;
    using __impl::no_env;

    // For getting an evaluation environment from a receiver
    struct get_env_t {
      template <class _EnvProvider>
          requires tag_invocable<get_env_t, const _EnvProvider&>
        constexpr auto operator()(const _EnvProvider& __with_env) const
          noexcept(nothrow_tag_invocable<get_env_t, const _EnvProvider&>)
          -> tag_invoke_result_t<get_env_t, const _EnvProvider&> {
          using _Env = tag_invoke_result_t<get_env_t, const _EnvProvider&>;
          static_assert(!same_as<_Env, no_env>);
          return tag_invoke(*this, __with_env);
        }
    };

    // For making an evaluation environment from a key/value pair, and optionally
    // another environment.
    template <__class _Tag>
      inline constexpr __impl::__make_env_t<_Tag> make_env {};
  } // namespace __env
  using __env::get_env_t;
  inline constexpr __env::get_env_t get_env{};
  using __env::no_env;
  using __env::make_env;
  using __env::__empty_env;

  template <class _EnvProvider>
    using env_of_t = decay_t<__call_result_t<get_env_t, _EnvProvider>>;

  template <__class _Tag, class _Value, class _BaseEnv = __empty_env>
    using make_env_t =
      decltype(make_env<_Tag>(__declval<_Value>(), __declval<_BaseEnv>()));

  template <class _EnvProvider>
    concept environment_provider =
      requires (_EnvProvider& __ep) {
        { get_env(std::as_const(__ep)) } -> __none_of<no_env, void>;
      };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __receivers {
    struct set_value_t {
      template <class _Receiver, class... _As>
        requires tag_invocable<set_value_t, _Receiver, _As...>
      void operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(set_value_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }
    };

    struct set_error_t {
      template <class _Receiver, class _Error>
        requires tag_invocable<set_error_t, _Receiver, _Error>
      void operator()(_Receiver&& __rcvr, _Error&& __err) const noexcept {
        static_assert(nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void) tag_invoke(set_error_t{}, (_Receiver&&) __rcvr, (_Error&&) __err);
      }
    };

    struct set_stopped_t {
      template <class _Receiver>
        requires tag_invocable<set_stopped_t, _Receiver>
      void operator()(_Receiver&& __rcvr) const noexcept {
        static_assert(nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) tag_invoke(set_stopped_t{}, (_Receiver&&) __rcvr);
      }
    };
  } // namespace __receivers
  using __receivers::set_value_t;
  using __receivers::set_error_t;
  using __receivers::set_stopped_t;
  inline constexpr set_value_t set_value{};
  inline constexpr set_error_t set_error{};
  inline constexpr set_stopped_t set_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __completion_signatures {
    template <same_as<set_value_t> _Tag, class _Ty = __q<__types>, class... _Args>
      __types<__minvoke<_Ty, _Args...>> __test(_Tag(*)(_Args...));
    template <same_as<set_error_t> _Tag, class _Ty = __q<__types>, class _Error>
      __types<__minvoke1<_Ty, _Error>> __test(_Tag(*)(_Error));
    template <same_as<set_stopped_t> _Tag, class _Ty = __q<__types>>
      __types<__minvoke<_Ty>> __test(_Tag(*)());
    template <class, class = void>
      __types<> __test(...);

    struct __none_such {};
    struct __dependent {};
  } // namespace __completion_signatures

  template <class _Env>
    using dependent_completion_signatures =
      __if_c<
        same_as<_Env, no_env>,
        __completion_signatures::__dependent,
        __completion_signatures::__none_such>;

  template <class _Sig>
    concept __completion_signature =
      __typename<decltype(__completion_signatures::__test((_Sig*) nullptr))>;

  template <__completion_signature... _Sigs>
    struct completion_signatures {
      template <class _Sig, class _Tag, class _Ty = __q<__types>>
        using __signal_args_t =
          decltype(__completion_signatures::__test<_Tag, _Ty>((_Sig*) nullptr));

      template <class _Tag>
        using __count_of =
          integral_constant<
            size_t,
            (__mapply<__mcount, __signal_args_t<_Sigs, _Tag>>::value + ...)>;

      template <template <class...> class _Tuple, template <class...> class _Variant>
        using __value_types =
          __minvoke<
            __concat<__q<_Variant>>,
            __signal_args_t<_Sigs, set_value_t, __q<_Tuple>>...>;

      template <template <class...> class _Variant>
        using __error_types =
          __minvoke<
            __concat<__q<_Variant>>,
            __signal_args_t<_Sigs, set_error_t, __q1<__id>>...>;
    };

  template <class...>
    struct __concat_completion_signatures {
      using type = dependent_completion_signatures<no_env>;
    };

  template <__is_instance_of<completion_signatures>... _Completions>
    struct __concat_completion_signatures<_Completions...> {
      using type =
        __minvoke<
          __concat<__munique<__q<completion_signatures>>>,
          _Completions...>;
    };

  template <class... _Completions>
    using __concat_completion_signatures_t =
      __t<__concat_completion_signatures<_Completions...>>;

  template <class _Traits, class _Env>
    concept __valid_completion_signatures =
      __is_instance_of<_Traits, completion_signatures> ||
      (
        same_as<_Traits, dependent_completion_signatures<no_env>> &&
        same_as<_Env, no_env>
      );

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Sig>
    struct _MISSING_COMPLETION_SIGNAL_;
  template <class _Tag, class... _Args>
    struct _MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)> {
      template <class _Receiver>
        struct _WITH_RECEIVER_ : false_type {};
    };

  namespace __receiver_concepts {
    struct __found_completion_signature {
      template <class>
        using _WITH_RECEIVER_ = true_type;
    };

    template <class _Receiver, class _Tag, class... _Args>
      using __missing_completion_signal_t =
        __if<
          __bool<nothrow_tag_invocable<_Tag, _Receiver, _Args...>>,
          __found_completion_signature,
          _MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)>>;

    template <class _Receiver, class _Tag, class... _Args>
      auto __has_completion(_Tag(*)(_Args...)) ->
        __missing_completion_signal_t<_Receiver, _Tag, _Args...>;

    template <class _Receiver, class... _Sigs>
      auto __has_completions(completion_signatures<_Sigs...>*) ->
        decltype((__has_completion<_Receiver>((_Sigs*)0), ...));

    template <class _Completion, class _Receiver>
      concept __is_valid_completion =
        _Completion::template _WITH_RECEIVER_<_Receiver>::value;
  } // namespace __receiver_concepts

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Receiver>
    concept receiver =
      environment_provider<__cref_t<_Receiver>> &&
      move_constructible<remove_cvref_t<_Receiver>> &&
      constructible_from<remove_cvref_t<_Receiver>, _Receiver>;

  template <class _Receiver, class _Completions>
    concept receiver_of =
      receiver<_Receiver> &&
      requires (_Completions* __required_completions) {
        { __receiver_concepts::__has_completions<remove_cvref_t<_Receiver>>(
            __required_completions) } ->
          __receiver_concepts::__is_valid_completion<remove_cvref_t<_Receiver>>;
      };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.sndtraits]
  namespace __get_completion_signatures {
    struct get_completion_signatures_t {
      template <class _Sender, class _Env = no_env>
      constexpr auto operator()(_Sender&& __sndr, const _Env& = {}) const noexcept {
        static_assert(sizeof(_Sender), "Incomplete type used with get_completion_signatures");
        static_assert(sizeof(_Env), "Incomplete type used with get_completion_signatures");
        if constexpr (tag_invocable<get_completion_signatures_t, _Sender, _Env>) {
          using _Completions =
            tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
          static_assert(__valid_completion_signatures<_Completions, _Env>);
          return _Completions{};
        } else if constexpr (requires { typename remove_cvref_t<_Sender>::completion_signatures; }) {
          using _Completions =
            typename remove_cvref_t<_Sender>::completion_signatures;
          static_assert(__valid_completion_signatures<_Completions, _Env>);
          return _Completions{};
        } else if constexpr (__awaitable<_Sender>) {
          using _Result = __await_result_t<_Sender>;
          if constexpr (is_void_v<_Result>) {
            return completion_signatures<set_value_t(), set_error_t(exception_ptr)>{};
          } else {
            return completion_signatures<set_value_t(_Result), set_error_t(exception_ptr)>{};
          }
        } else {
          return __completion_signatures::__none_such{};
        }
      }
    };
  } // namespace __get_completion_signatures

  using __get_completion_signatures::get_completion_signatures_t;
  inline constexpr get_completion_signatures_t get_completion_signatures {};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  // NOT TO SPEC (YET)
  template <class _Sender, class _Env>
    concept __sender =
      requires (_Sender&& __sndr, _Env&& __env) {
        { get_completion_signatures((_Sender&&) __sndr, (_Env&&) __env) } ->
          __valid_completion_signatures<_Env>;
      };

  template <class _Sender, class _Env = no_env>
    concept sender =
      __sender<_Sender, _Env> &&
      __sender<_Sender, no_env> &&
      move_constructible<remove_cvref_t<_Sender>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.traits]
  template <class _Sender, class _Env>
    using __completion_signatures_of_t =
      __call_result_t<
        get_completion_signatures_t,
        _Sender,
        _Env>;

  // __checked_completion_signatures is for catching logic bugs in a typed
  // sender's metadata. If sender<S> and sender<S, Ctx> are both true, then they
  // had better report the same metadata. This completion signatures wrapper
  // enforces that at compile time.
  template <class _Sender, class _Env>
    struct __checked_completion_signatures {
     private:
      using _WithEnv = __completion_signatures_of_t<_Sender, _Env>;
      using _WithoutEnv = __completion_signatures_of_t<_Sender, no_env>;
      static_assert(
        __one_of<
          _WithoutEnv,
          _WithEnv,
          dependent_completion_signatures<no_env>>);
     public:
      using type = _WithEnv;
    };

  template <class _Sender, class _Env = no_env>
      requires sender<_Sender, _Env>
    using completion_signatures_of_t =
      __t<__checked_completion_signatures<_Sender, _Env>>;

  template <class _Receiver, class _Sender>
    concept __receiver_from =
      receiver_of<
        _Receiver,
        completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  struct __not_a_variant {
    __not_a_variant() = delete;
  };
  template <class... _Ts>
    using __variant =
      __minvoke<
        __if_c<
          sizeof...(_Ts) != 0,
          __transform<__q1<decay_t>, __munique<__q<variant>>>,
          __constant<__not_a_variant>>,
        _Ts...>;

  template <class... _Ts>
    using __decayed_tuple = tuple<decay_t<_Ts>...>;

  template <class _Sender,
            class _Env = no_env,
            template <class...> class _Tuple = __decayed_tuple,
            template <class...> class _Variant = __variant>
      requires sender<_Sender, _Env>
    using value_types_of_t =
      typename completion_signatures_of_t<_Sender, _Env>::template
        __value_types<_Tuple, _Variant>;

  template <class _Sender,
            class _Env = no_env,
            template <class...> class _Variant = __variant>
      requires sender<_Sender, _Env>
    using error_types_of_t =
      typename completion_signatures_of_t<_Sender, _Env>::template
        __error_types<_Variant>;

  template <class _Sender, class _Env = no_env>
      requires sender<_Sender, _Env>
    inline constexpr bool sends_stopped =
      (completion_signatures_of_t<_Sender, _Env>
        ::template __count_of<set_stopped_t>::value != 0);

  template <class _Sender,
            class _Env = no_env,
            class _Tuple = __q<__decayed_tuple>,
            class _Variant = __q<__variant>>
      requires sender<_Sender, _Env>
    using __value_types_of_t =
      value_types_of_t<
        _Sender, _Env, _Tuple::template __f, _Variant::template __f>;

  template <class _Sender,
            class _Env = no_env,
            class _Variant = __q<__variant>>
      requires sender<_Sender, _Env>
    using __error_types_of_t =
      error_types_of_t<_Sender, _Env, _Variant::template __f>;

  template <class _Sender, class _Env = no_env>
    using __single_sender_value_t =
      __value_types_of_t<_Sender, _Env, __single_or<void>, __q<__single_t>>;

  template <class _Sender, class _Env = no_env>
    concept __single_typed_sender =
      sender<_Sender, _Env> &&
      __valid<__single_sender_value_t, _Sender, _Env>;

  template <class _Sender, class _Env = no_env>
    using __single_value_variant_sender_t =
      value_types_of_t<_Sender, _Env, __types, __single_t>;

  template <class _Sender, class _Env = no_env>
    concept __single_value_variant_sender =
      sender<_Sender, _Env> &&
      __valid<__single_value_variant_sender_t, _Sender, _Env>;

  /////////////////////////////////////////////////////////////////////////////
  namespace __completion_signatures {
    template <class... _Args>
      using __default_set_value = completion_signatures<set_value_t(_Args...)>;

    template <class _Err>
      using __default_set_error = completion_signatures<set_error_t(_Err)>;

    template <__is_instance_of<completion_signatures>... _Sigs>
      using __ensure_concat = __minvoke<__concat<__q<completion_signatures>>, _Sigs...>;

    template<class _Sender, class _Env, class _Sigs, class _SetValue, class _SetError, class _SetStopped>
      using __compl_sigs_t =
        __concat_completion_signatures_t<
          _Sigs,
          __value_types_of_t<_Sender, _Env, _SetValue, __q<__ensure_concat>>,
          __error_types_of_t<_Sender, _Env, __transform<_SetError, __q<__ensure_concat>>>,
          __if_c<sends_stopped<_Sender, _Env>, _SetStopped, completion_signatures<>>>;

    template<class _Sender, class _Env, class _Sigs, class _SetValue, class _SetError, class _SetStopped>
      auto __make(int) ->
        __compl_sigs_t<_Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;

    template<class, class _Env, class, class, class, class>
      auto __make(long) -> dependent_completion_signatures<_Env>;
  } // namespace __completion_signatures

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC
  //
  // make_completion_signatures
  // ==========================
  //
  // `make_completion_signatures` takes a sender, and environment, and a bunch
  // of other template arguments for munging the completion signatures of a
  // sender in interesting ways.
  //
  //  ```c++
  //  template <class... Args>
  //    using __default_set_value = completion_signatures<set_value_t(Args...)>;
  //
  //  template <class Err>
  //    using __default_set_error = completion_signatures<set_error_t(Err)>;
  //
  //  template <
  //    sender Sndr,
  //    class Env = no_env,
  //    class AddlSigs = completion_signatures<>,
  //    template <class...> class SetValue = __default_set_value,
  //    template <class> class SetError = __default_set_error,
  //    class SetStopped = completion_signatures<set_stopped_t()>>
  //      requires sender<Sndr, Env>
  //  using make_completion_signatures =
  //    completion_signatures< ... >;
  //  ```
  //
  //  * `SetValue` : an alias template that accepts a set of value types and
  //    returns an instance of `completion_signatures`.
  //  * `SetError` : an alias template that accepts an error types and returns a
  //    an instance of `completion_signatures`.
  //  * `SetStopped` : an instantiation of `completion_signatures` with a list
  //    of completion signatures `Sigs...` to the added to the list if the
  //    sender can complete with a stopped signal.
  //  * `AddlSigs` : an instantiation of `completion_signatures` with a list of
  //    completion signatures `Sigs...` to the added to the list
  //    unconditionally.
  //
  //  `make_completion_signatures` does the following:
  //  * Let `VCs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `value_types_of_t<Sndr, Env, SetValue,
  //    __typelist>`, and let `Vs...` be the concatenation of the packs that are
  //    template arguments to each `completion_signature` in `VCs...`.
  //  * Let `ECs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `error_types_of_t<Sndr, Env, __errorlist>`, where
  //    `__errorlist` is an alias template such that `__errorlist<Ts...>` names
  //    `__typelist<SetError<Ts>...>`, and let `Es...` by the concatenation of
  //    the packs that are the template arguments to each `completion_signature`
  //    in `ECs...`.
  //  * Let `Ss...` be an empty pack if `sends_stopped<Sndr, Env>` is
  //    `false`; otherwise, a pack containing the template arguments of the
  //    `completion_signatures` instantiation named by `SetStopped`.
  //  * Let `MoreSigs...` be a pack of the template arguments of the
  //    `completion_signatures` instantiation named by `AddlSigs`.
  //
  //  Then `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` names the type `completion_signatures< Sigs... >` where
  //  `Sigs...` is the unique set of types in `[Vs..., Es..., Ss...,
  //  MoreSigs...]`.
  //
  //  If any of the above type computations are ill-formed,
  //  `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` is an alias for an empty struct
  template<
    class _Sender,
    class _Env = no_env,
    __valid_completion_signatures<_Env> _Sigs =
      completion_signatures<>,
    template <class...> class _SetValue =
      __completion_signatures::__default_set_value,
    template <class> class _SetError =
      __completion_signatures::__default_set_error,
    __valid_completion_signatures<_Env> _SetStopped =
      completion_signatures<set_stopped_t()>>
      requires sender<_Sender, _Env>
  using make_completion_signatures =
    decltype(__completion_signatures::
      __make<_Sender, _Env, _Sigs, __q<_SetValue>, __q1<_SetError>, _SetStopped>(0));

  // Needed fairly often
  using __with_exception_ptr =
    completion_signatures<set_error_t(exception_ptr)>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __schedule {
    struct schedule_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler> &&
          sender<tag_invoke_result_t<schedule_t, _Scheduler>>
      auto operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        return tag_invoke(schedule_t{}, (_Scheduler&&) __sched);
      }
    };
  }
  using __schedule::schedule_t;
  inline constexpr schedule_t schedule{};

  // [execution.schedulers.queries], scheduler queries
  namespace __scheduler_queries {
    namespace __impl {
      template <class _T>
        using __cref_t = const remove_reference_t<_T>&;

      struct forwarding_scheduler_query_t {
        template <class _Tag>
        constexpr bool operator()(_Tag __tag) const noexcept {
          if constexpr (tag_invocable<forwarding_scheduler_query_t, _Tag>) {
            static_assert(noexcept(tag_invoke(*this, (_Tag&&) __tag) ? true : false));
            return tag_invoke(*this, (_Tag&&) __tag) ? true : false;
          } else {
            return false;
          }
        }
      };

      struct get_forward_progress_guarantee_t {
        template <class _T>
          requires tag_invocable<get_forward_progress_guarantee_t, __cref_t<_T>>
        tag_invoke_result_t<get_forward_progress_guarantee_t, __cref_t<_T>> operator()(
            _T&& __t) const
          noexcept(nothrow_tag_invocable<get_forward_progress_guarantee_t, __cref_t<_T>>) {
          return tag_invoke(get_forward_progress_guarantee_t{}, std::as_const(__t)) ? true : false;
        }
        execution::forward_progress_guarantee operator()(auto&&) const noexcept {
          return execution::forward_progress_guarantee::weakly_parallel;
        }
      };
    } // namespace __impl

    using __impl::forwarding_scheduler_query_t;
    using __impl::get_forward_progress_guarantee_t;

    template <class _Tag>
      concept __scheduler_query =
        forwarding_scheduler_query(_Tag{});
  } // namespace __scheduler_queries
  using __scheduler_queries::forwarding_scheduler_query_t;
  inline constexpr forwarding_scheduler_query_t forwarding_scheduler_query{};

  using __scheduler_queries::get_forward_progress_guarantee_t;
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};

  namespace __sender_queries {
    namespace __impl {
      template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO>
        struct get_completion_scheduler_t;
    }
    using __impl::get_completion_scheduler_t;
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template <class _Scheduler>
    concept scheduler =
      copy_constructible<remove_cvref_t<_Scheduler>> &&
      equality_comparable<remove_cvref_t<_Scheduler>> &&
      requires(_Scheduler&& __sched, const __sender_queries::get_completion_scheduler_t<set_value_t> __tag) {
        { schedule((_Scheduler&&) __sched) } -> sender;
        { tag_invoke(__tag, schedule((_Scheduler&&) __sched)) } -> same_as<remove_cvref_t<_Scheduler>>;
      };

  // NOT TO SPEC
  template <scheduler _Scheduler>
    using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.general.queries], general queries
  namespace __general_queries {
    namespace __impl {
      // TODO: implement allocator concept
      template <class _T0>
        concept __allocator = true;

      struct get_scheduler_t {
        template <__none_of<no_env> _Env>
          requires nothrow_tag_invocable<get_scheduler_t, const _Env&> &&
            scheduler<tag_invoke_result_t<get_scheduler_t, const _Env&>>
        auto operator()(const _Env& __env) const
          noexcept(nothrow_tag_invocable<get_scheduler_t, const _Env&>)
          -> tag_invoke_result_t<get_scheduler_t, const _Env&> {
          return tag_invoke(get_scheduler_t{}, __env);
        }
        auto operator()() const noexcept;
      };

      struct get_delegatee_scheduler_t {
        template <class _T>
          requires nothrow_tag_invocable<get_delegatee_scheduler_t, __cref_t<_T>> &&
            scheduler<tag_invoke_result_t<get_delegatee_scheduler_t, __cref_t<_T>>>
        auto operator()(_T&& __t) const
          noexcept(nothrow_tag_invocable<get_delegatee_scheduler_t, __cref_t<_T>>)
          -> tag_invoke_result_t<get_delegatee_scheduler_t, __cref_t<_T>> {
          return tag_invoke(get_delegatee_scheduler_t{}, std::as_const(__t));
        }
        auto operator()() const noexcept;
      };

      struct get_allocator_t {
        template <__none_of<no_env> _Env>
          requires nothrow_tag_invocable<get_allocator_t, const _Env&> &&
            __allocator<tag_invoke_result_t<get_allocator_t, const _Env&>>
        auto operator()(const _Env& __env) const
          noexcept(nothrow_tag_invocable<get_allocator_t, const _Env&>)
          -> tag_invoke_result_t<get_allocator_t, const _Env&> {
          return tag_invoke(get_allocator_t{}, __env);
        }
        auto operator()() const noexcept;
      };

      struct get_stop_token_t {
        template <__none_of<no_env> _Env>
        never_stop_token operator()(const _Env&) const noexcept {
          return {};
        }
        template <__none_of<no_env> _Env>
          requires tag_invocable<get_stop_token_t, const _Env&> &&
            stoppable_token<tag_invoke_result_t<get_stop_token_t, const _Env&>>
        auto operator()(const _Env& __env) const
          noexcept(nothrow_tag_invocable<get_stop_token_t, const _Env&>)
          -> tag_invoke_result_t<get_stop_token_t, const _Env&> {
          return tag_invoke(get_stop_token_t{}, __env);
        }
        auto operator()() const noexcept;
      };
    } // namespace __impl

    using __impl::get_allocator_t;
    using __impl::get_scheduler_t;
    using __impl::get_delegatee_scheduler_t;
    using __impl::get_stop_token_t;
  } // namespace __general_queries
  using __general_queries::get_allocator_t;
  using __general_queries::get_scheduler_t;
  using __general_queries::get_delegatee_scheduler_t;
  using __general_queries::get_stop_token_t;
  inline constexpr get_scheduler_t get_scheduler{};
  inline constexpr get_delegatee_scheduler_t get_delegatee_scheduler{};
  inline constexpr get_allocator_t get_allocator{};
  inline constexpr get_stop_token_t get_stop_token{};

  template <class _T>
    using stop_token_of_t =
      remove_cvref_t<decltype(get_stop_token(__declval<_T>()))>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  namespace __start {
    struct start_t {
      template <class _O>
        requires tag_invocable<start_t, _O&>
      void operator()(_O& o) const noexcept(nothrow_tag_invocable<start_t, _O&>) {
        (void) tag_invoke(start_t{}, o);
      }
    };
  }
  using __start::start_t;
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _O>
    concept operation_state =
      destructible<_O> &&
      is_object_v<_O> &&
      requires (_O& o) {
        {start(o)} noexcept;
      };

  namespace __as_awaitable {
    struct as_awaitable_t;
  }
  using __as_awaitable::as_awaitable_t;
  extern const as_awaitable_t as_awaitable;

  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  namespace __connect_awaitable_ {
    struct __promise_base {
      __coro::suspend_always initial_suspend() noexcept {
        return {};
      }
      [[noreturn]] __coro::suspend_always final_suspend() noexcept {
        terminate();
      }
      [[noreturn]] void unhandled_exception() noexcept {
        terminate();
      }
      [[noreturn]] void return_void() noexcept {
        terminate();
      }
      template <class _Fun>
      auto yield_value(_Fun&& __fun) noexcept {
        struct awaiter {
          _Fun&& __fun_;
          bool await_ready() noexcept {
            return false;
          }
          void await_suspend(__coro::coroutine_handle<>)
            noexcept(__nothrow_callable<_Fun>) {
            // If this throws, the runtime catches the exception,
            // resumes the __connect_awaitable coroutine, and immediately
            // rethrows the exception. The end result is that an
            // exception_ptr to the exception gets passed to set_error.
            ((_Fun &&) __fun_)();
          }
          [[noreturn]] void await_resume() noexcept {
            terminate();
          }
        };
        return awaiter{(_Fun &&) __fun};
      }
    };

    struct __operation_base {
      __coro::coroutine_handle<> __coro_;

      explicit __operation_base(__coro::coroutine_handle<> __hcoro) noexcept
        : __coro_(__hcoro) {}

      __operation_base(__operation_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {})) {}

      ~__operation_base() {
        if (__coro_)
          __coro_.destroy();
      }

      friend void tag_invoke(start_t, __operation_base& __self) noexcept {
        __self.__coro_.resume();
      }
    };

    template <class _ReceiverId>
      struct __promise;

    template <class _ReceiverId>
      struct __operation : __operation_base {
        using promise_type = __promise<_ReceiverId>;
        using __operation_base::__operation_base;
      };

    template <class _ReceiverId>
      struct __promise : __promise_base {
        using _Receiver = __t<_ReceiverId>;

        explicit __promise(auto&, _Receiver& __rcvr) noexcept
          : __rcvr_(__rcvr)
        {}

        __coro::coroutine_handle<> unhandled_stopped() noexcept {
          set_stopped(std::move(__rcvr_));
          // Returning noop_coroutine here causes the __connect_awaitable
          // coroutine to never resume past the point where it co_await's
          // the awaitable.
          return __coro::noop_coroutine();
        }

        __operation<_ReceiverId> get_return_object() noexcept {
          return __operation<_ReceiverId>{
            __coro::coroutine_handle<__promise>::from_promise(*this)};
        }

        template <class _Awaitable>
        _Awaitable&& await_transform(_Awaitable&& __await) noexcept {
          return (_Awaitable&&) __await;
        }

        template <class _Awaitable>
          requires tag_invocable<as_awaitable_t, _Awaitable, __promise&>
        auto await_transform(_Awaitable&& __await)
            noexcept(nothrow_tag_invocable<as_awaitable_t, _Awaitable, __promise&>)
            -> tag_invoke_result_t<as_awaitable_t, _Awaitable, __promise&> {
          return tag_invoke(as_awaitable, (_Awaitable&&) __await, *this);
        }

        // Pass through the get_env receiver query
        friend auto tag_invoke(get_env_t, const __promise& __self)
          -> env_of_t<_Receiver> {
          return get_env(__self.__rcvr_);
        }

        _Receiver& __rcvr_;
      };

    template <receiver _Receiver>
      using __promise_t = __promise<__x<remove_cvref_t<_Receiver>>>;

    template <receiver _Receiver>
      using __operation_t = __operation<__x<remove_cvref_t<_Receiver>>>;

    struct __connect_awaitable_t {
     private:
      template <class _Awaitable, class _Receiver>
      static __operation_t<_Receiver> __co_impl(_Awaitable __await, _Receiver __rcvr) {
        using __result_t = __await_result_t<_Awaitable, __promise_t<_Receiver>>;
        exception_ptr __eptr;
        try {
          // This is a bit mind bending control-flow wise.
          // We are first evaluating the co_await expression.
          // Then the result of that is passed into a lambda
          // that curries a reference to the result into another
          // lambda which is then returned to 'co_yield'.
          // The 'co_yield' expression then invokes this lambda
          // after the coroutine is suspended so that it is safe
          // for the receiver to destroy the coroutine.
          auto __fun = [&](auto&&... __as) noexcept {
            return [&]() noexcept -> void {
              set_value((_Receiver&&) __rcvr, (add_rvalue_reference_t<__result_t>) __as...);
            };
          };
          if constexpr (is_void_v<__result_t>)
            co_yield (co_await (_Awaitable &&) __await, __fun());
          else
            co_yield __fun(co_await (_Awaitable &&) __await);
        } catch (...) {
          __eptr = current_exception();
        }
        co_yield [&]() noexcept -> void {
          set_error((_Receiver&&) __rcvr, (exception_ptr&&) __eptr);
        };
      }

      template <receiver _Receiver, class _Awaitable>
        using __completions_t =
          completion_signatures<
            __minvoke1< // set_value_t() or set_value_t(T)
              __remove<void, __qf<set_value_t>>,
              __await_result_t<_Awaitable, __promise_t<_Receiver>>>,
            set_error_t(exception_ptr),
            set_stopped_t()>;

     public:
      template <class _Receiver, __awaitable<__promise_t<_Receiver>> _Awaitable>
          requires receiver_of<_Receiver, __completions_t<_Receiver, _Awaitable>>
        __operation_t<_Receiver> operator()(_Awaitable&& __await, _Receiver&& __rcvr) const {
          return __co_impl((_Awaitable&&) __await, (_Receiver&&) __rcvr);
        }
    };
  } // namespace __connect_awaitable_
  using __connect_awaitable_::__connect_awaitable_t;
  inline constexpr __connect_awaitable_t __connect_awaitable{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    struct __is_debug_env_t {
      template <class _Env>
          requires tag_invocable<__is_debug_env_t, _Env>
        void operator()(_Env&&) const noexcept;
    };
    template <class _Env>
      using __debug_env_t =
        make_env_t<__is_debug_env_t, int, _Env>;

    template <class _Sig>
      struct __completion;

    template <class _Tag, class... _Args>
      struct __completion<_Tag(_Args...)> {
        friend void tag_invoke(_Tag, __completion&&, _Args&&...) noexcept;
      };

    template <class _Env, class _Sigs>
      struct __debug_receiver;

    template <class _Env, class... _Sigs>
      struct __debug_receiver<_Env, completion_signatures<_Sigs...>>
        : __completion<_Sigs>... {
        friend __debug_env_t<_Env> tag_invoke(get_env_t, __debug_receiver) noexcept;
      };

    template <class _Env>
      struct __any_debug_receiver {
        friend void tag_invoke(set_value_t, __any_debug_receiver&&, auto&&...) noexcept;
        friend void tag_invoke(set_error_t, __any_debug_receiver&&, auto&&) noexcept;
        friend void tag_invoke(set_stopped_t, __any_debug_receiver&&) noexcept;
        friend __debug_env_t<_Env> tag_invoke(get_env_t, __any_debug_receiver) noexcept;
      };

    struct connect_t;

    template <class _Sender, class _Receiver>
      concept __connectable_sender_with =
        sender<_Sender, env_of_t<_Receiver>> &&
        __receiver_from<_Receiver, _Sender> &&
        tag_invocable<connect_t, _Sender, _Receiver>;

    struct connect_t {
      struct __debug_op_state {
        __debug_op_state(auto&&);
        friend void tag_invoke(start_t, __debug_op_state&) noexcept;
      };

      template <class _Sender, class _Receiver>
        requires __connectable_sender_with<_Sender, _Receiver>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(nothrow_tag_invocable<connect_t, _Sender, _Receiver>)
        -> tag_invoke_result_t<connect_t, _Sender, _Receiver> {
        static_assert(
          operation_state<tag_invoke_result_t<connect_t, _Sender, _Receiver>>,
          "execution::connect(sender, receiver) must return a type that "
          "satisfies the operation_state concept");
        return tag_invoke(connect_t{}, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
      }
      template <class _Awaitable, class _Receiver>
        requires (!__connectable_sender_with<_Awaitable, _Receiver>) &&
          __callable<__connect_awaitable_t, _Awaitable, _Receiver>
      auto operator()(_Awaitable&& __await, _Receiver&& __rcvr) const
        -> __connect_awaitable_::__operation_t<_Receiver> {
        return __connect_awaitable((_Awaitable&&) __await, (_Receiver&&) __rcvr);
      }
      // This overload is purely for the purposes of debugging why a
      // sender will not connect. Use the __debug_sender function below.
      template <class _Sender, class _Receiver>
        requires (!__connectable_sender_with<_Sender, _Receiver>) &&
           (!__callable<__connect_awaitable_t, _Sender, _Receiver>) &&
           tag_invocable<__is_debug_env_t, env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        -> __debug_op_state {
        // This should generate an instantiate backtrace that contains useful
        // debugging information.
        using std::__tag_invoke::tag_invoke;
        return tag_invoke(*this, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // `__debug_sender`
    // ===============
    //
    // Understanding why a particular sender doesn't connect to a particular
    // receiver is nigh impossible in the current design due to limitations in
    // how the compiler reports overload resolution failure in the presence of
    // constraints. `__debug_sender` is a utility to assist with the process. It
    // gives you the deep template instantiation backtrace that you need to
    // understand where in a chain of senders the problem is occurring.
    //
    // ```c++
    // template <class _Sigs, class _Env = __empty_env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env = {});
    //
    // template <class _Sender>
    //   void __debug_sender(_Sender&& __sndr);
    //
    // template <class _Env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env);
    // ```
    //
    // **Usage:**
    //
    // To find out where in a chain of senders, a sender is failing to connect
    // to a receiver, pass it to `__debug_sender`, optionally with an
    // environment argument; e.g. `__debug_sender(sndr [, env])`
    //
    // To find out why a sender will not connect to a receiver of a particular
    // signature, specify the error and value types as an explicit template
    // argument that names an instantiation of `completion_signatures`; e.g.:
    // `__debug_sender<completion_signatures<set_value_t(int)>>(sndr [, env])`.
    //
    // **How it works:**
    //
    // The `__debug_sender` function `connect`'s the sender to a
    // `__debug_receiver`, whose environment is augmented with a special
    // `__is_debug_env_t` query. An additional fall-back overload is added to
    // the `connect` CPO that recognizes receivers whose environments respond to
    // that query and lets them through. Then in a non-immediate context, it
    // looks for a `tag_invoke(connect_t...)` overload for the input sender and
    // receiver. This will recurse until it hits the `tag_invoke` call that is
    // causing the failure.
    //
    // At least with clang, this gives me a nice backtrace, at the bottom of
    // which is the faulty `tag_invoke` overload with a mention of the
    // constraint that failed.
    template <class _Sigs, class _Env = __empty_env, class _Sender>
      void __debug_sender(_Sender&& __sndr, _Env = {}) {
        using _Receiver = __debug_receiver<_Env, _Sigs>;
        (void) connect_t{}((_Sender&&) __sndr, _Receiver{});
      }

    template <class _Sender>
      void __debug_sender(_Sender&& __sndr) {
        using _Receiver = __any_debug_receiver<__empty_env>;
        (void) connect_t{}((_Sender&&) __sndr, _Receiver{});
      }

    template <class _Env, class _Sender>
      void __debug_sender(_Sender&& __sndr, _Env) {
        using _Receiver = __any_debug_receiver<_Env>;
        (void) connect_t{}((_Sender&&) __sndr, _Receiver{});
      }
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect {};

  template <class _Sender, class _Receiver>
    using connect_result_t = __call_result_t<connect_t, _Sender, _Receiver>;

  template <class _Sender, class _Receiver>
    concept __has_nothrow_connect =
      noexcept(connect(__declval<_Sender>(), __declval<_Receiver>()));

  using __connect::__debug_sender;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  template <class _Sender, class _Receiver>
    concept sender_to =
      sender<_Sender, env_of_t<_Receiver>> &&
      __receiver_from<_Receiver, _Sender> &&
      requires (_Sender&& __sndr, _Receiver&& __rcvr) {
        connect((_Sender&&) __sndr, (_Receiver&&) __rcvr);
      };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.queries], sender queries
  namespace __sender_queries {
    namespace __impl {
      template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO>
        struct get_completion_scheduler_t {
          template <sender _Sender>
            requires tag_invocable<get_completion_scheduler_t, const _Sender&> &&
              scheduler<tag_invoke_result_t<get_completion_scheduler_t, const _Sender&>>
          auto operator()(const _Sender& __sndr) const noexcept
              -> tag_invoke_result_t<get_completion_scheduler_t, const _Sender&> {
            // NOT TO SPEC:
            static_assert(
              nothrow_tag_invocable<get_completion_scheduler_t, const _Sender&>,
              "get_completion_scheduler<_CPO> should be noexcept");
            return tag_invoke(*this, __sndr);
          }
        };

      struct forwarding_sender_query_t {
        template <class _Tag>
        constexpr bool operator()(_Tag __tag) const noexcept {
          if constexpr (nothrow_tag_invocable<forwarding_sender_query_t, _Tag> &&
                        is_invocable_r_v<bool, tag_t<tag_invoke>,
                                         forwarding_sender_query_t, _Tag>) {
            return tag_invoke(*this, (_Tag&&) __tag);
          } else {
            return false;
          }
        }
      };
    } // namespace __impl

    using __impl::get_completion_scheduler_t;
    using __impl::forwarding_sender_query_t;

    template <class _Tag>
      concept __sender_query =
        forwarding_sender_query(_Tag{});
  } // namespace __sender_queries
  using __sender_queries::get_completion_scheduler_t;
  using __sender_queries::forwarding_sender_query_t;

  template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO>
    inline constexpr get_completion_scheduler_t<_CPO> get_completion_scheduler{};

  inline constexpr forwarding_sender_query_t forwarding_sender_query{};

  template <class _Sender, class _CPO>
    concept __has_completion_scheduler =
      __callable<get_completion_scheduler_t<_CPO>, _Sender>;

  template <class _Sender, class _CPO>
    using __completion_scheduler_for =
      __call_result_t<get_completion_scheduler_t<_CPO>, _Sender>;

  template <class _Fun, class _CPO, class _Sender, class... _As>
    concept __tag_invocable_with_completion_scheduler =
      __has_completion_scheduler<_Sender, _CPO> &&
      tag_invocable<_Fun, __completion_scheduler_for<_Sender, _CPO>, _Sender, _As...>;

  /////////////////////////////////////////////////////////////////////////////
  // execution::as_awaitable [execution.coro_utils.as_awaitable]
  namespace __as_awaitable {
    namespace __impl {
      struct __void {};
      template <class _Value>
        using __value_or_void_t =
          __if<is_same<_Value, void>, __void, _Value>;
      template <class _Value>
        using __expected_t =
          variant<monostate, __value_or_void_t<_Value>, exception_ptr>;

      template <class _Value>
        struct __receiver_base {
          template <class... _Us>
            requires constructible_from<__value_or_void_t<_Value>, _Us...>
          friend void tag_invoke(set_value_t, __receiver_base&& __self, _Us&&... __us)
              noexcept try {
            __self.__result_->template emplace<1>((_Us&&) __us...);
            __self.__continuation_.resume();
          } catch(...) {
            set_error((__receiver_base&&) __self, current_exception());
          }

          template <class _Error>
          friend void tag_invoke(set_error_t, __receiver_base&& __self, _Error&& __err) noexcept {
            if constexpr (__decays_to<_Error, exception_ptr>)
              __self.__result_->template emplace<2>((_Error&&) __err);
            else if constexpr (__decays_to<_Error, error_code>)
              __self.__result_->template emplace<2>(make_exception_ptr(system_error(__err)));
            else
              __self.__result_->template emplace<2>(make_exception_ptr((_Error&&) __err));
            __self.__continuation_.resume();
          }

          __expected_t<_Value>* __result_;
          __coro::coroutine_handle<> __continuation_;
        };

      template <typename _PromiseId, typename _Value>
        struct __sender_awaitable_base {
          using _Promise = __t<_PromiseId>;
          struct __receiver : __receiver_base<_Value> {
            friend void tag_invoke(set_stopped_t, __receiver&& __self) noexcept {
              auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
                __self.__continuation_.address());
              __coro::coroutine_handle<> __stopped_continuation =
                __continuation.promise().unhandled_stopped();
              __stopped_continuation.resume();
            }

            // Forward get_env query to the coroutine promise
            friend auto tag_invoke(get_env_t, const __receiver& __self)
              -> env_of_t<_Promise> {
              auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
                __self.__continuation_.address());
              return get_env(__continuation.promise());
            }
          };

        bool await_ready() const noexcept {
          return false;
        }

        _Value await_resume() {
          switch (__result_.index()) {
          case 0: // receiver contract not satisfied
            assert(!"_Should never get here");
            break;
          case 1: // set_value
            if constexpr (!is_void_v<_Value>)
              return (_Value&&) std::get<1>(__result_);
            else
              return;
          case 2: // set_error
            std::rethrow_exception(std::get<2>(__result_));
          }
          terminate();
        }

       protected:
        __expected_t<_Value> __result_;
      };

      template <typename _PromiseId, typename _SenderId>
      struct __sender_awaitable
        : __sender_awaitable_base<
            _PromiseId,
            __single_sender_value_t<__t<_SenderId>, env_of_t<__t<_PromiseId>>>> {
       private:
        using _Promise = __t<_PromiseId>;
        using _Sender = __t<_SenderId>;
        using _Env = env_of_t<_Promise>;
        using _Value = __single_sender_value_t<_Sender, _Env>;
        using _Base = __sender_awaitable_base<_PromiseId, _Value>;
        using __receiver = typename _Base::__receiver;
        connect_result_t<_Sender, __receiver> __op_state_;
       public:
        __sender_awaitable(_Sender&& sender, __coro::coroutine_handle<_Promise> __hcoro)
            noexcept(__has_nothrow_connect<_Sender, __receiver>)
          : __op_state_(connect((_Sender&&)sender, __receiver{{&this->__result_, __hcoro}}))
        {}

        void await_suspend(__coro::coroutine_handle<_Promise>) noexcept {
          start(__op_state_);
        }
      };
      template <class _Promise, class _Sender>
        using __sender_awaitable_t =
          __sender_awaitable<__x<_Promise>, __x<_Sender>>;

      template <class _T, class _Promise>
        concept __custom_tag_invoke_awaiter =
          tag_invocable<as_awaitable_t, _T, _Promise&> &&
          __awaitable<tag_invoke_result_t<as_awaitable_t, _T, _Promise&>, _Promise>;

      template <class _Sender, class _Promise>
        using __receiver =
          typename __sender_awaitable_base<
            __x<_Promise>,
            __single_sender_value_t<_Sender, env_of_t<_Promise>>
          >::__receiver;

      template <class _Sender, class _Promise>
        concept __awaitable_sender =
          __single_typed_sender<_Sender, env_of_t<_Promise>> &&
          sender_to<_Sender, __receiver<_Sender, _Promise>> &&
          requires (_Promise& __promise) {
            { __promise.unhandled_stopped() } -> convertible_to<__coro::coroutine_handle<>>;
          };
    } // namespace __impl

    struct as_awaitable_t {
      template <class _T, class _Promise>
      static constexpr bool __is_noexcept() noexcept {
        if constexpr (__impl::__custom_tag_invoke_awaiter<_T, _Promise>) {
          return nothrow_tag_invocable<as_awaitable_t, _T, _Promise&>;
        } else if constexpr (__awaitable<_T>) {
          return true;
        } else if constexpr (__impl::__awaitable_sender<_T, _Promise>) {
          using _Sender = __impl::__sender_awaitable_t<_Promise, _T>;
          return is_nothrow_constructible_v<_Sender, _T, __coro::coroutine_handle<_Promise>>;
        } else {
          return true;
        }
      }
      template <class _T, class _Promise>
      decltype(auto) operator()(_T&& __t, _Promise& __promise) const
          noexcept(__is_noexcept<_T, _Promise>()) {
        if constexpr (__impl::__custom_tag_invoke_awaiter<_T, _Promise>) {
          return tag_invoke(*this, (_T&&) __t, __promise);
        } else if constexpr (__awaitable<_T>) {
          return (_T&&) __t;
        } else if constexpr (__impl::__awaitable_sender<_T, _Promise>) {
          auto __hcoro = __coro::coroutine_handle<_Promise>::from_promise(__promise);
          return __impl::__sender_awaitable_t<_Promise, _T>{(_T&&) __t, __hcoro};
        } else {
          return (_T&&) __t;
        }
      }
    };
  } // namespace __as_awaitable
  using __as_awaitable::as_awaitable_t;
  inline constexpr as_awaitable_t as_awaitable;

  namespace __with_awaitable_senders {
    namespace __impl {
      struct __with_awaitable_senders_base {
        template <class _OtherPromise>
        void set_continuation(__coro::coroutine_handle<_OtherPromise> __hcoro) noexcept {
          static_assert(!is_void_v<_OtherPromise>);
          __continuation_ = __hcoro;
          if constexpr (requires(_OtherPromise& __other) { __other.unhandled_stopped(); }) {
            __stopped_callback_ = [](void* __address) noexcept -> __coro::coroutine_handle<> {
              // This causes the rest of the coroutine (the part after the co_await
              // of the sender) to be skipped and invokes the calling coroutine's
              // stopped handler.
              return __coro::coroutine_handle<_OtherPromise>::from_address(__address)
                  .promise().unhandled_stopped();
            };
          }
          // If _OtherPromise doesn't implement unhandled_stopped(), then if a "stopped" unwind
          // reaches this point, it's considered an unhandled exception and terminate()
          // is called.
        }

        __coro::coroutine_handle<> continuation() const noexcept {
          return __continuation_;
        }

        __coro::coroutine_handle<> unhandled_stopped() noexcept {
          return (*__stopped_callback_)(__continuation_.address());
        }

       private:
        __coro::coroutine_handle<> __continuation_{};
        __coro::coroutine_handle<> (*__stopped_callback_)(void*) noexcept =
          [](void*) noexcept -> __coro::coroutine_handle<> {
            std::terminate();
          };
      };
    } // namespace __impl

    template <class _Promise>
    struct with_awaitable_senders : __impl::__with_awaitable_senders_base {
      template <class _Value>
      auto await_transform(_Value&& __val)
        -> __call_result_t<as_awaitable_t, _Value, _Promise&> {
        static_assert(derived_from<_Promise, with_awaitable_senders>);
        return as_awaitable((_Value&&) __val, static_cast<_Promise&>(*this));
      }
    };
  }
  using __with_awaitable_senders::with_awaitable_senders;

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __submit
  namespace __submit_ {
    namespace __impl {
      template <class _SenderId, class _ReceiverId>
        struct __operation {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          struct __receiver {
            __operation* __op_state_;
            // Forward all the receiver ops, and delete the operation state.
            template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _As>
              requires __callable<_Tag, _Receiver, _As...>
            friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as)
                noexcept(__nothrow_callable<_Tag, _Receiver, _As...>) {
              // Delete the state as cleanup:
              unique_ptr<__operation> __g{__self.__op_state_};
              return __tag((_Receiver&&) __self.__op_state_->__rcvr_, (_As&&) __as...);
            }
            // Forward all receiever queries.
            friend auto tag_invoke(get_env_t, const __receiver& __self)
              -> env_of_t<_Receiver> {
              return get_env((const _Receiver&) __self.__op_state_->__rcvr_);
            }
          };
          _Receiver __rcvr_;
          connect_result_t<_Sender, __receiver> __op_state_;
          __operation(_Sender&& __sndr, __decays_to<_Receiver> auto&& __rcvr)
            : __rcvr_((decltype(__rcvr)&&) __rcvr)
            , __op_state_(connect((_Sender&&) __sndr, __receiver{this}))
          {}
        };
    } // namespace __impl

    struct __submit_t {
      template <receiver _Receiver, sender_to<_Receiver> _Sender>
      void operator()(_Sender&& __sndr, _Receiver&& __rcvr) const noexcept(false) {
        start((new __impl::__operation<__x<_Sender>, __x<decay_t<_Receiver>>>{
            (_Sender&&) __sndr, (_Receiver&&) __rcvr})->__op_state_);
      }
    };
  } // namespace __submit_
  using __submit_::__submit_t;
  inline constexpr __submit_t __submit{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  namespace __start_detached {
    namespace __impl {
      struct __detached_receiver {
        friend void tag_invoke(set_value_t, __detached_receiver&&, auto&&...) noexcept {}
        [[noreturn]]
        friend void tag_invoke(set_error_t, __detached_receiver&&, auto&&) noexcept {
          terminate();
        }
        friend void tag_invoke(set_stopped_t, __detached_receiver&&) noexcept {}
        friend __empty_env tag_invoke(get_env_t, const __detached_receiver&) noexcept {
          return {};
        }
      };
    } // namespace __impl

    struct start_detached_t {
      template <sender _Sender>
        requires tag_invocable<start_detached_t, _Sender>
      void operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<start_detached_t, _Sender>) {
        (void) tag_invoke(start_detached_t{}, (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!tag_invocable<start_detached_t, _Sender>) &&
          sender_to<_Sender, __impl::__detached_receiver>
      void operator()(_Sender&& __sndr) const noexcept(false) {
        __submit((_Sender&&) __sndr, __impl::__detached_receiver{});
      }
    };
  } // namespace __start_detached
  using __start_detached::start_detached_t;
  inline constexpr start_detached_t start_detached{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __just {
    template <class _CPO, class... _Ts>
      struct __sender {
        tuple<_Ts...> __vals_;

        using completion_signatures = completion_signatures<_CPO(_Ts...)>;

        template <class _ReceiverId>
          struct __operation {
            using _Receiver = __t<_ReceiverId>;
            tuple<_Ts...> __vals_;
            _Receiver __rcvr_;

            friend void tag_invoke(start_t, __operation& __op_state) noexcept {
              static_assert(__nothrow_callable<_CPO, _Receiver, _Ts...>);
              std::apply([&__op_state](_Ts&... __ts) {
                _CPO{}((_Receiver&&) __op_state.__rcvr_, (_Ts&&) __ts...);
              }, __op_state.__vals_);
            }
          };

        template <class _Receiver>
          requires (copy_constructible<_Ts> &&...)
        friend auto tag_invoke(connect_t, const __sender& __sndr, _Receiver&& __rcvr)
          noexcept((is_nothrow_copy_constructible_v<_Ts> &&...))
          -> __operation<__x<remove_cvref_t<_Receiver>>> {
          return {__sndr.__vals_, (_Receiver&&) __rcvr};
        }

        template <class _Receiver>
        friend auto tag_invoke(connect_t, __sender&& __sndr, _Receiver&& __rcvr)
          noexcept((is_nothrow_move_constructible_v<_Ts> &&...))
          -> __operation<__x<remove_cvref_t<_Receiver>>> {
          return {((__sender&&) __sndr).__vals_, (_Receiver&&) __rcvr};
        }
      };

    inline constexpr struct __just_t {
      template <__movable_value... _Ts>
      __sender<set_value_t, decay_t<_Ts>...> operator()(_Ts&&... __ts) const
        noexcept((is_nothrow_constructible_v<decay_t<_Ts>, _Ts> &&...)) {
        return {{(_Ts&&) __ts...}};
      }
    } just {};

    inline constexpr struct __just_error_t {
      template <__movable_value _Error>
      __sender<set_error_t, decay_t<_Error>> operator()(_Error&& __err) const
        noexcept(is_nothrow_constructible_v<decay_t<_Error>, _Error>) {
        return {{(_Error&&) __err}};
      }
    } just_error {};

    inline constexpr struct __just_stopped_t {
      __sender<set_stopped_t> operator()() const noexcept {
        return {{}};
      }
    } just_stopped {};
  }
  using __just::just;
  using __just::just_error;
  using __just::just_stopped;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  namespace __execute_ {
    namespace __impl {
      template <class _Fun>
        struct __as_receiver {
          _Fun __fun_;
          friend void tag_invoke(set_value_t, __as_receiver&& __rcvr) noexcept try {
            __rcvr.__fun_();
          } catch(...) {
            set_error((__as_receiver&&) __rcvr, exception_ptr());
          }
          [[noreturn]]
          friend void tag_invoke(set_error_t, __as_receiver&&, exception_ptr) noexcept {
            terminate();
          }
          friend void tag_invoke(set_stopped_t, __as_receiver&&) noexcept {}
        };
    }

    struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const
        noexcept(noexcept(
          submit(schedule((_Scheduler&&) __sched), __impl::__as_receiver<_Fun>{(_Fun&&) __fun}))) {
        (void) submit(schedule((_Scheduler&&) __sched), __impl::__as_receiver<_Fun>{(_Fun&&) __fun});
      }
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> &&
          move_constructible<_Fun> &&
          tag_invocable<execute_t, _Scheduler, _Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const
        noexcept(nothrow_tag_invocable<execute_t, _Scheduler, _Fun>) {
        (void) tag_invoke(execute_t{}, (_Scheduler&&) __sched, (_Fun&&) __fun);
      }
    };
  }
  using __execute_::execute_t;
  inline constexpr execute_t execute{};

  // NOT TO SPEC:
  namespace __closure {
    template <__class _D>
      struct sender_adaptor_closure;
  }
  using __closure::sender_adaptor_closure;

  template <class _T>
    concept __sender_adaptor_closure =
      derived_from<remove_cvref_t<_T>, sender_adaptor_closure<remove_cvref_t<_T>>> &&
      move_constructible<remove_cvref_t<_T>> &&
      constructible_from<remove_cvref_t<_T>, _T>;

  namespace __closure {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      [[no_unique_address]] _T0 __t0_;
      [[no_unique_address]] _T1 __t1_;

      template <sender _Sender>
        requires __callable<_T0, _Sender> && __callable<_T1, __call_result_t<_T0, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) && {
        return ((_T1&&) __t1_)(((_T0&&) __t0_)((_Sender&&) __sndr));
      }

      template <sender _Sender>
        requires __callable<const _T0&, _Sender> && __callable<const _T1&, __call_result_t<const _T0&, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) const & {
        return __t1_(__t0_((_Sender&&) __sndr));
      }
    };

    template <__class _D>
      struct sender_adaptor_closure
      {};

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    __compose<remove_cvref_t<_T0>, remove_cvref_t<_T1>> operator|(_T0&& __t0, _T1&& __t1) {
      return {(_T0&&) __t0, (_T1&&) __t1};
    }

    template <sender _Sender, __sender_adaptor_closure _Closure>
      requires __callable<_Closure, _Sender>
    __call_result_t<_Closure, _Sender> operator|(_Sender&& __sndr, _Closure&& __clsur) {
      return ((_Closure&&) __clsur)((_Sender&&) __sndr);
    }

    template <class _Fun, class... _As>
    struct __binder_back : sender_adaptor_closure<__binder_back<_Fun, _As...>> {
      [[no_unique_address]] _Fun __fun_;
      tuple<_As...> __as_;

      template <sender _Sender>
        requires __callable<_Fun, _Sender, _As...>
      __call_result_t<_Fun, _Sender, _As...> operator()(_Sender&& __sndr) &&
        noexcept(__nothrow_callable<_Fun, _Sender, _As...>) {
        return std::apply([&__sndr, this](_As&... __as) {
            return ((_Fun&&) __fun_)((_Sender&&) __sndr, (_As&&) __as...);
          }, __as_);
      }

      template <sender _Sender>
        requires __callable<const _Fun&, _Sender, const _As&...>
      __call_result_t<const _Fun&, _Sender, const _As&...> operator()(_Sender&& __sndr) const &
        noexcept(__nothrow_callable<const _Fun&, _Sender, const _As&...>) {
        return std::apply([&__sndr, this](const _As&... __as) {
            return __fun_((_Sender&&) __sndr, __as...);
          }, __as_);
      }
    };
  } // namespace __closure
  using __closure::__binder_back;

  namespace __tag_invoke_adaptors {
    // A derived-to-base cast that works even when the base is not
    // accessible from derived.
    template <class _T, class _U>
      __member_t<_U, _T> __c_cast(_U&& u) noexcept requires __decays_to<_T, _T> {
        static_assert(is_reference_v<__member_t<_U, _T>>);
        static_assert(is_base_of_v<_T, remove_reference_t<_U>>);
        return (__member_t<_U, _T>) (_U&&) u;
      }
    namespace __no {
      struct __nope {};
      struct __receiver : __nope {};
      void tag_invoke(set_error_t, __receiver, exception_ptr) noexcept;
      void tag_invoke(set_stopped_t, __receiver) noexcept;
      __empty_env tag_invoke(get_env_t, __receiver) noexcept;
    }
    using __not_a_receiver = __no::__receiver;

    template <class _Base>
      struct __adaptor {
        struct __t {
          template <class _T1>
            requires constructible_from<_Base, _T1>
          explicit __t(_T1&& __base) : __base_((_T1&&) __base) {}

         private:
          [[no_unique_address]] _Base __base_;

         protected:
          _Base& base() & noexcept { return __base_; }
          const _Base& base() const & noexcept { return __base_; }
          _Base&& base() && noexcept { return (_Base&&) __base_; }
        };
      };
    template <derived_from<__no::__nope> _Base>
      struct __adaptor<_Base> {
        struct __t : __no::__nope { };
      };
    template <class _Base>
      using __adaptor_base = typename __adaptor<_Base>::__t;

    template <class _Sender, class _Receiver>
      concept __has_connect =
        requires(_Sender&& __sndr, _Receiver&& __rcvr) {
          ((_Sender&&) __sndr).connect((_Receiver&&) __rcvr);
        };

    template <__class _Derived, sender _Base>
      struct __sender_adaptor {
        class __t : __adaptor_base<_Base> {
          using connect = void;

          template <__decays_to<_Derived> _Self, receiver _Receiver>
            requires __has_connect<_Self, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            noexcept(noexcept(((_Self&&) __self).connect((_Receiver&&) __rcvr)))
            -> decltype(((_Self&&) __self).connect((_Receiver&&) __rcvr)) {
            return ((_Self&&) __self).connect((_Receiver&&) __rcvr);
          }

          template <__decays_to<_Derived> _Self, receiver _Receiver>
            requires requires {typename decay_t<_Self>::connect;} &&
              sender_to<__member_t<_Self, _Base>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            noexcept(__has_nothrow_connect<__member_t<_Self, _Base>, _Receiver>)
            -> connect_result_t<__member_t<_Self, _Base>, _Receiver> {
            return execution::connect(((_Self&&) __self).base(), (_Receiver&&) __rcvr);
          }

          template <__sender_queries::__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend auto tag_invoke(_Tag __tag, const _Derived& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>)
            -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Base&, _As...> {
            return ((_Tag&&) __tag)(__self.base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _Receiver, class... _As>
      concept __has_set_value =
        requires(_Receiver&& __rcvr, _As&&... __as) {
          ((_Receiver&&) __rcvr).set_value((_As&&) __as...);
        };

    template <class _Receiver, class _Error>
      concept __has_set_error =
        requires(_Receiver&& __rcvr, _Error&& __err) {
          ((_Receiver&&) __rcvr).set_error((_Error&&) __err);
        };

    template <class _Receiver>
      concept __has_set_stopped =
        requires(_Receiver&& __rcvr) {
          ((_Receiver&&) __rcvr).set_stopped();
        };

    template <class _Receiver>
      concept __has_get_env =
        requires(const _Receiver& __rcvr) {
          __rcvr.get_env();
        };

    template <__class _Derived, class _Base>
      struct __receiver_adaptor {
        class __t : __adaptor_base<_Base> {
          friend _Derived;
          using set_value = void;
          using set_error = void;
          using set_stopped = void;
          using get_env = void;

          static constexpr bool __has_base = !derived_from<_Base, __no::__nope>;

          template <class _D>
            using __base_from_derived_t = decltype(__declval<_D>().base());

          using __get_base_t =
            __if<
              __bool<__has_base>,
              __mbind_back<__defer<__member_t>, _Base>,
              __q1<__base_from_derived_t>>;

          template <class _D>
            using __base_t = __minvoke1<__get_base_t, _D&&>;

          template <class _D>
            static __base_t<_D> __get_base(_D&& __self) noexcept {
              if constexpr (__has_base) {
                return __c_cast<__t>((_D&&) __self).base();
              } else {
                return ((_D&&) __self).base();
              }
            }

          template <class... _As>
            requires __has_set_value<_Derived, _As...>
          friend void tag_invoke(set_value_t, _Derived&& __self, _As&&... __as) noexcept {
            static_assert(noexcept(((_Derived&&) __self).set_value((_As&&) __as...)));
            ((_Derived&&) __self).set_value((_As&&) __as...);
          }

          template <class _D = _Derived, class... _As>
            requires requires {typename _D::set_value;} &&
              tag_invocable<set_value_t, __base_t<_D>, _As...>
          friend void tag_invoke(set_value_t, _Derived&& __self, _As&&... __as) noexcept {
            execution::set_value(__get_base((_Derived&&) __self), (_As&&) __as...);
          }

          template <class _Error>
            requires __has_set_error<_Derived, _Error>
          friend void tag_invoke(set_error_t, _Derived&& __self, _Error&& __err) noexcept {
            static_assert(noexcept(((_Derived&&) __self).set_error((_Error&&) __err)));
            ((_Derived&&) __self).set_error((_Error&&) __err);
          }

          template <class _Error, class _D = _Derived>
            requires requires {typename _D::set_error;} &&
              tag_invocable<set_error_t, __base_t<_D>, _Error>
          friend void tag_invoke(set_error_t, _Derived&& __self, _Error&& __err) noexcept {
            execution::set_error(__get_base((_Derived&&) __self), (_Error&&) __err);
          }

          template <class _D = _Derived>
            requires __has_set_stopped<_D>
          friend void tag_invoke(set_stopped_t, _Derived&& __self) noexcept {
            static_assert(noexcept(((_Derived&&) __self).set_stopped()));
            ((_Derived&&) __self).set_stopped();
          }

          template <class _D = _Derived>
            requires requires {typename _D::set_stopped;} &&
              tag_invocable<set_stopped_t, __base_t<_D>>
          friend void tag_invoke(set_stopped_t, _Derived&& __self) noexcept {
            execution::set_stopped(__get_base((_Derived&&) __self));
          }

          // Pass through the get_env receiver query
          template <class _D = _Derived>
            requires __has_get_env<_D>
          friend auto tag_invoke(get_env_t, const _Derived& __self) {
            return __self.get_env();
          }

          template <class _D = _Derived>
            requires requires {typename _D::get_env;}
          friend auto tag_invoke(get_env_t, const _Derived& __self) {
            return execution::get_env(__get_base(__self));
          }

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _OpState>
      concept __has_start =
        requires(_OpState& __op_state) {
          __op_state.start();
        };

    template <__class _Derived, operation_state _Base>
      struct __operation_state_adaptor {
        class __t : __adaptor_base<_Base> {
          using start = void;

          template <class _D = _Derived>
            requires __has_start<_D>
          friend void tag_invoke(start_t, _Derived& __self) noexcept {
            static_assert(noexcept(__self.start()));
            __self.start();
          }

          template <class _D = _Derived>
            requires requires {typename _D::start;}
          friend void tag_invoke(start_t, _Derived& __self) noexcept {
            execution::start(__c_cast<__t>(__self).base());
          }

          template <__none_of<start_t> _Tag, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend auto tag_invoke(_Tag __tag, const _Derived& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>)
            -> __call_result_if_t<__none_of<_Tag, start_t>, _Tag, const _Base&, _As...> {
            return ((_Tag&&) __tag)(__c_cast<__t>(__self).base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _Scheduler>
      concept __has_schedule =
        requires(_Scheduler&& __sched) {
          ((_Scheduler&&) __sched).schedule();
        };

    template <__class _Derived, scheduler _Base>
      struct __scheduler_adaptor {
        class __t : __adaptor_base<_Base> {
          using schedule = void;

          template <__decays_to<_Derived> _Self>
            requires __has_schedule<_Self>
          friend auto tag_invoke(schedule_t, _Self&& __self)
            noexcept(noexcept(((_Self&&) __self).schedule()))
            -> decltype(((_Self&&) __self).schedule()) {
            return ((_Self&&) __self).schedule();
          }

          template <__decays_to<_Derived> _Self>
            requires requires {typename decay_t<_Self>::schedule;} &&
              scheduler<__member_t<_Self, _Base>>
          friend auto tag_invoke(schedule_t, _Self&& __self)
            noexcept(noexcept(execution::schedule(__declval<__member_t<_Self, _Base>>())))
            -> schedule_result_t<_Self> {
            return execution::schedule(__c_cast<__t>((_Self&&) __self).base());
          }

          template <__none_of<schedule_t> _Tag, same_as<_Derived> _Self, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend auto tag_invoke(_Tag __tag, const _Self& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>)
            -> __call_result_if_t<__none_of<_Tag, schedule_t>, _Tag, const _Base&, _As...> {
            return ((_Tag&&) __tag)(__c_cast<__t>(__self).base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };
  } // namespace __tag_invoke_adaptors

  // NOT TO SPEC
  template <__class _Derived, sender _Base>
    using sender_adaptor =
      typename __tag_invoke_adaptors::__sender_adaptor<_Derived, _Base>::__t;

  template <__class _Derived, receiver _Base = __tag_invoke_adaptors::__not_a_receiver>
    using receiver_adaptor =
      typename __tag_invoke_adaptors::__receiver_adaptor<_Derived, _Base>::__t;

  // NOT TO SPEC
  template <__class _Derived, operation_state _Base>
    using operation_state_adaptor =
      typename __tag_invoke_adaptors::__operation_state_adaptor<_Derived, _Base>::__t;

  // NOT TO SPEC
  template <__class _Derived, scheduler _Base>
    using scheduler_adaptor =
      typename __tag_invoke_adaptors::__scheduler_adaptor<_Derived, _Base>::__t;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  namespace __then {
    template <class _ReceiverId, class _Fun>
      class __receiver
        : receiver_adaptor<__receiver<_ReceiverId, _Fun>, __t<_ReceiverId>> {
        using _Receiver = __t<_ReceiverId>;
        friend receiver_adaptor<__receiver, _Receiver>;
        [[no_unique_address]] _Fun __f_;

        // Customize set_value by invoking the invocable and passing the result
        // to the base class
        template <class... _As>
          requires invocable<_Fun, _As...> &&
            tag_invocable<set_value_t, _Receiver, invoke_result_t<_Fun, _As...>>
        void set_value(_As&&... __as) && noexcept try {
          execution::set_value(
              ((__receiver&&) *this).base(),
              std::invoke((_Fun&&) __f_, (_As&&) __as...));
        } catch(...) {
          execution::set_error(
              ((__receiver&&) *this).base(),
              current_exception());
        }
        // Handle the case when the invocable returns void
        template <class _R2 = _Receiver, class... _As>
          requires invocable<_Fun, _As...> &&
            same_as<void, invoke_result_t<_Fun, _As...>> &&
            tag_invocable<set_value_t, _R2>
        void set_value(_As&&... __as) && noexcept try {
          invoke((_Fun&&) __f_, (_As&&) __as...);
          execution::set_value(((__receiver&&) *this).base());
        } catch(...) {
          execution::set_error(
              ((__receiver&&) *this).base(),
              current_exception());
        }

       public:
        explicit __receiver(_Receiver __rcvr, _Fun __fun)
          : receiver_adaptor<__receiver, _Receiver>((_Receiver&&) __rcvr)
          , __f_((_Fun&&) __fun)
        {}
      };

    template <class _SenderId, class _Fun>
      class __sender
        : sender_adaptor<__sender<_SenderId, _Fun>, __t<_SenderId>> {
        using _Sender = __t<_SenderId>;
        friend sender_adaptor<__sender, _Sender>;
        template <class _Receiver>
          using __receiver = __receiver<__x<remove_cvref_t<_Receiver>>, _Fun>;

        [[no_unique_address]] _Fun __fun_;

        template <class... _Args>
            requires invocable<_Fun, _Args...>
          using __set_value =
            completion_signatures<
              __minvoke1<
                __remove<void, __qf<set_value_t>>,
                invoke_result_t<_Fun, _Args...>>>;

        template <class _Env>
          using __completion_signatures =
            make_completion_signatures<
              _Sender, _Env, __with_exception_ptr, __set_value>;

        template <class _Receiver>
          requires sender_to<_Sender, __receiver<_Receiver>>
        auto connect(_Receiver&& __rcvr) &&
          noexcept(__has_nothrow_connect<_Sender, __receiver<_Receiver>>)
          -> connect_result_t<_Sender, __receiver<_Receiver>> {
          return execution::connect(
              ((__sender&&) *this).base(),
              __receiver<_Receiver>{(_Receiver&&) __rcvr, (_Fun&&) __fun_});
        }

        template <class _Env>
        friend auto tag_invoke(get_completion_signatures_t, const __sender&, _Env) ->
          __completion_signatures<_Env>;

       public:
        explicit __sender(_Sender __sndr, _Fun __fun)
          : sender_adaptor<__sender, _Sender>{(_Sender&&) __sndr}
          , __fun_((_Fun&&) __fun)
        {}
      };

    struct then_t {
      template <class _Sender, class _Fun>
        using __sender = __sender<__x<remove_cvref_t<_Sender>>, _Fun>;

      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(then_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>) &&
          tag_invocable<then_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_t, _Sender, _Fun>) {
        return tag_invoke(then_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>) &&
          (!tag_invocable<then_t, _Sender, _Fun>) &&
          sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }
      template <class _Fun>
      __binder_back<then_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }
  using __then::then_t;
  inline constexpr then_t then{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_error]
  namespace __upon_error {
    struct upon_error_t {
      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<upon_error_t, set_error_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_error_t, __completion_scheduler_for<_Sender, set_error_t>, _Sender, _Fun>) {
        auto __sched = get_completion_scheduler<set_error_t>(__sndr);
        return tag_invoke(upon_error_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<upon_error_t, set_error_t, _Sender, _Fun>) &&
          tag_invocable<upon_error_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_error_t, _Sender, _Fun>) {
        return tag_invoke(upon_error_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <class _Fun>
      __binder_back<upon_error_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }
  using __upon_error::upon_error_t;
  inline constexpr upon_error_t upon_error{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_stopped]
  namespace __upon_stopped {
    struct upon_stopped_t {
      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<upon_stopped_t, set_stopped_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_stopped_t, __completion_scheduler_for<_Sender, set_stopped_t>, _Sender, _Fun>) {
        auto __sched = get_completion_scheduler<set_stopped_t>(__sndr);
        return tag_invoke(upon_stopped_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<upon_stopped_t, set_stopped_t, _Sender, _Fun>) &&
          tag_invocable<upon_stopped_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_stopped_t, _Sender, _Fun>) {
        return tag_invoke(upon_stopped_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <class _Fun>
      __binder_back<upon_stopped_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }
  using __upon_stopped::upon_stopped_t;
  inline constexpr upon_stopped_t upon_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.bulk]
  namespace __bulk {
    struct bulk_t {
      template <sender _Sender, integral _Shape, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<bulk_t, set_value_t, _Sender, _Shape, _Fun>
      sender auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const
        noexcept(nothrow_tag_invocable<bulk_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Shape, _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(bulk_t{}, std::move(__sched), (_Sender&&) __sndr, (_Shape&&) __shape, (_Fun&&) __fun);
      }
      template <sender _Sender, integral _Shape, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<bulk_t, set_value_t, _Sender, _Shape, _Fun>) &&
          tag_invocable<bulk_t, _Sender, _Shape, _Fun>
      sender auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const
        noexcept(nothrow_tag_invocable<bulk_t, _Sender, _Shape, _Fun>) {
        return tag_invoke(bulk_t{}, (_Sender&&) __sndr, (_Shape&&) __shape, (_Fun&&) __fun);
      }
      template <integral _Shape, class _Fun>
      __binder_back<bulk_t, _Shape, _Fun> operator()(_Shape __shape, _Fun __fun) const {
        return {{}, {}, {(_Shape&&) __shape, (_Fun&&) __fun}};
      }
    };
  }
  using __bulk::bulk_t;
  inline constexpr bulk_t bulk{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.split]
  namespace __split {

    struct __operation_base {
      virtual void __notify() noexcept = 0;
      virtual ~__operation_base() = default;
    };

    template <class _Value, class _Error>
      class __receiver {
        __operation_base &__op_state_;
        in_place_stop_source &__stop_source_;
        variant<_Value, _Error, set_stopped_t> &__data_;

      public:
        template <class... _As>
        friend void tag_invoke(set_value_t, __receiver&& __self, _As&&... __as) noexcept {
          try {
            __self.__data_.template emplace<0>(tuple<decay_t<_As>...>{(_As &&) __as...});
          } catch (...) {
            __self.__data_.template emplace<1>(current_exception());
          }
          __self.__op_state_.__notify();
        }

        template <class _T>
        friend void tag_invoke(set_error_t, __receiver&& __self, _T&& __error) noexcept {
          __self.__data_.template emplace<1>(__error);
          __self.__op_state_.__notify();
        }

        friend void tag_invoke(set_stopped_t tag, __receiver&& __self) noexcept {
          ((__receiver&&)__self).__data_.template emplace<2>(tag);
          __self.__op_state_.__notify();
        }

        friend auto tag_invoke(get_env_t, const __receiver& __self) {
          return make_env<get_stop_token_t>(__self.__stop_source_.get_token());
        }

        __receiver(__operation_base &__op_state,
                   in_place_stop_source &__stop_source,
                   variant<_Value, _Error, set_stopped_t> &__data) noexcept
          : __op_state_(__op_state)
          , __stop_source_(__stop_source)
          , __data_(__data) {
        }
      };

    enum class __state_t { __created, __started, __completed };

    template <class _Sender>
      struct __sh_state : __operation_base {
        using __nullable_variant_t = __bind_front<__munique<__q<variant>>, tuple<>>;
        using __error = __error_types_of_t<_Sender,
                                           __empty_env,
                                           __bind_front<__q<__variant>,
                                                        exception_ptr>>;
        using __value = __value_types_of_t<_Sender,
                                           __empty_env,
                                           __q<__decayed_tuple>,
                                           __nullable_variant_t>;
        using __receiver = __receiver<__value, __error>;

        in_place_stop_source __stop_source_{};
        vector<__operation_base*> __operation_states_;
        connect_result_t<_Sender, __receiver> __op_state2_;
        variant<__value, __error, set_stopped_t> __data_;

        mutex __mutex_;
        __state_t __state_{__state_t::__created};

        explicit __sh_state(_Sender& __sndr)
          : __op_state2_(
                  connect((_Sender&&) __sndr,
                          __receiver{*this, __stop_source_, __data_}))
        {}

        void __notify() noexcept override {
          {
            lock_guard __lock{__mutex_};
            __state_ = __state_t::__completed;
          }

          for(auto __op_state: __operation_states_) {
            __op_state->__notify();
          }
        }
      };

    template <class _SenderId, class _ReceiverId>
      class __operation : public __operation_base {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;

        struct __on_stop_requested {
          in_place_stop_source& __stop_source_;
          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };
        using __on_stop = optional<typename stop_token_of_t<
            env_of_t<_Receiver> &>::template callback_type<__on_stop_requested>>;

        _Receiver __recvr_;
        __on_stop __on_stop_{};
        shared_ptr<__sh_state<_Sender>> __shared_state_;

        template <class... _As>
          requires tag_invocable<set_value_t, _Receiver, decay_t<_As>&...>
        void __apply_values(tuple<_As...>& __tupl) noexcept {
          try {
            std::apply([&](auto&... __args) -> void {
              execution::set_value((_Receiver&&) __recvr_, __args...);
            }, __tupl);
          } catch(...) {
            execution::set_error((_Receiver&&) __recvr_, current_exception());
          }
        }

        template <class... _As>
          requires (!tag_invocable<set_value_t, _Receiver, decay_t<_As>&...>)
        void __apply_values(tuple<_As...>&) noexcept {
          assert(!"_Should never get here");
        }

        void __propagate_signal() noexcept {
          auto &__data = __shared_state_->__data_;

          if (__data.index() == 0) {
            visit([&](auto& __tupl) -> void {
              __apply_values(__tupl);
            }, std::get<0>(__data));
          }
          if (__data.index() == 1) {
            visit([&](auto& __err) -> void {
              execution::set_error((_Receiver&&) __recvr_, __err);
            }, std::get<1>(__data));
          }
          else if (__data.index() == 2) {
            execution::set_stopped((_Receiver&&) __recvr_);
          }
        }

      public:
        __operation(_Receiver&& __rcvr,
                    shared_ptr<__sh_state<_Sender>> __shared_state)
          : __recvr_((_Receiver&&)__rcvr)
          , __shared_state_(move(__shared_state)) {
        }

        void __notify() noexcept override {
          __on_stop_.reset();
          __propagate_signal();
        }

        friend void tag_invoke(start_t, __operation& __self) noexcept try {
          __sh_state<_Sender> *__shared_state = __self.__shared_state_.get();
          unique_lock __lock{__shared_state->__mutex_};
          __state_t __state = __shared_state->__state_;

          if (__state == __state_t::__completed) {
            __lock.unlock();
            __self.__propagate_signal();
          }
          else {
            __self.__on_stop_.emplace(
                get_stop_token(get_env(__self.__recvr_)),
                __on_stop_requested{__shared_state->__stop_source_});

            if (__shared_state->__stop_source_.stop_requested()) {
              __lock.unlock();
              execution::set_stopped((_Receiver&&) __self.__recvr_);
            }
            else {
              __shared_state->__operation_states_.push_back(&__self);

              if (__state != __state_t::__started) {
                __shared_state->__state_ = __state_t::__started;
                __lock.unlock();

                execution::start(__shared_state->__op_state2_);
              }
            }
          }
        } catch (...) {
          execution::set_error((_Receiver&&) __self.__recvr_, current_exception());
        }
      };

    template <class _SenderId>
      class __sender {
        using _Sender = __t<_SenderId>;
        using __sh_state = __sh_state<_Sender>;
        template <class _Receiver>
          using __operation = __operation<_SenderId, __x<remove_cvref_t<_Receiver>>>;

        _Sender __sndr_;
        shared_ptr<__sh_state> __shared_state_;

      public:
        template <__decays_to<__sender> _Self, receiver _Receiver>
            requires receiver_of<_Receiver, completion_signatures_of_t<_Self>>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __recvr)
            noexcept(__has_nothrow_connect<__member_t<_Self, _Sender>, _Receiver>)
            -> __operation<_Receiver> {
            return __operation<_Receiver>{(_Receiver &&) __recvr,
                                          __self.__shared_state_};
          }

        template <__sender_queries::__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...> &&
              __none_of<_Tag,
                        get_completion_scheduler_t<set_value_t>,
                        get_completion_scheduler_t<set_error_t>,
                        get_completion_scheduler_t<set_stopped_t>>
          friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

        template <class... _Tys>
        using __set_value_t = completion_signatures<set_value_t(decay_t<_Tys>&...)>;

        template <class _Ty>
        using __set_error_t = completion_signatures<set_error_t(decay_t<_Ty>&)>;

        template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
            make_completion_signatures<
              __member_t<_Self, _Sender>,
              _Env,
              completion_signatures<set_error_t(exception_ptr&)>,
              __set_value_t,
              __set_error_t>;

        explicit __sender(_Sender __sndr)
            : __sndr_((_Sender&&) __sndr)
            , __shared_state_{make_shared<__sh_state>(__sndr_)}
        {}
      };

    struct split_t {
      template <class _Sender>
        using __sender = __sender<__x<remove_cvref_t<_Sender>>>;

      template <sender _Sender>
        requires __tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<split_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(split_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!__tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>) &&
          tag_invocable<split_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<split_t, _Sender>) {
        return tag_invoke(split_t{}, (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!__tag_invocable_with_completion_scheduler<split_t, set_value_t, _Sender>) &&
          (!tag_invocable<split_t, _Sender>)
      __sender<_Sender> operator()(_Sender&& __sndr) const {
        return __sender<_Sender>{(_Sender&&) __sndr};
      }
      __binder_back<split_t> operator()() const {
        return {{}, {}, {}};
      }
    };
  }
  using __split::split_t;
  inline constexpr split_t split{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ensure_started]
  namespace __ensure_started {
    struct ensure_started_t {
      template <sender _Sender>
        requires __tag_invocable_with_completion_scheduler<ensure_started_t, set_value_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<ensure_started_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(ensure_started_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!__tag_invocable_with_completion_scheduler<ensure_started_t, set_value_t, _Sender>) &&
          tag_invocable<ensure_started_t, _Sender>
      sender auto operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<ensure_started_t, _Sender>) {
        return tag_invoke(ensure_started_t{}, (_Sender&&) __sndr);
      }
    };
  }
  using __ensure_started::ensure_started_t;
  inline constexpr ensure_started_t ensure_started{};

  //////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.let_value]
  // [execution.senders.adaptors.let_error]
  // [execution.senders.adaptors.let_stopped]
  namespace __let {
    namespace __impl {
      using __nullable_variant_t = __munique<__mbind_front<__q<variant>, monostate>>;

      template <class... _Ts>
        struct __as_tuple {
          __decayed_tuple<_Ts...> operator()(_Ts...) const;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __receiver;

      template <class... _Ts>
        struct __which_tuple_ : _Ts... {
          using _Ts::operator()...;
        };

      struct __which_tuple_base {
        template <class... _Ts>
          __decayed_tuple<_Ts...> operator()(_Ts&&...) const;
      };

      template <sender, class, class>
        struct __which_tuple : __which_tuple_base {};

      template <class _Sender, class _Env>
          requires sender<_Sender, _Env>
        struct __which_tuple<_Sender, _Env, set_value_t>
          : value_types_of_t<_Sender, _Env, __as_tuple, __which_tuple_> {};

      template <class _Sender, class _Env>
          requires sender<_Sender, _Env>
        struct __which_tuple<_Sender, _Env, set_error_t>
          : __error_types_of_t<
              _Sender,
              _Env,
              __transform<__q<__as_tuple>, __q<__which_tuple_>>> {};

      template <class _Fun>
        struct __applyable_fn {
          __ operator()(auto&&...) const;
          template <class... _As>
              requires invocable<_Fun, _As...>
            invoke_result_t<_Fun, _As...> operator()(_As&&...) const {
                terminate(); // this is never called; but we need a body
            }
        };

      template <class _Fun, class _Tuple>
        concept __applyable =
          requires (__applyable_fn<_Fun> __fun, _Tuple&& __tupl) {
            {std::apply(__fun, (_Tuple&&) __tupl)} -> __none_of<__>;
          };
      template <class _Fun, class _Tuple>
          requires __applyable<_Fun, _Tuple>
        using __apply_result_t =
          decltype(std::apply(__applyable_fn<_Fun>{}, __declval<_Tuple>()));

      template <class _T>
        using __decay_ref = decay_t<_T>&;

      template <class _Fun, class... _As>
        using __result_sender_t = __call_result_t<_Fun, __decay_ref<_As>...>;

      template <class _Sender, class _Receiver, class _Fun, class _Let>
          requires sender<_Sender, env_of_t<_Receiver>>
        struct __storage;

      // Storage for let_value:
      template <class _Sender, class _Receiver, class _Fun>
        struct __storage<_Sender, _Receiver, _Fun, set_value_t> {
          template <class... _As>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _As...>, _Receiver>;

          // Compute a variant of tuples to hold all the values of the input
          // sender:
          using __args_t =
            __value_types_of_t<_Sender, env_of_t<_Receiver>, __q<__decayed_tuple>, __nullable_variant_t>;
          __args_t __args_;

          // Compute a variant of operation states:
          using __op_state3_t =
            __value_types_of_t<_Sender, env_of_t<_Receiver>, __q<__op_state_for_t>, __nullable_variant_t>;
          __op_state3_t __op_state3_;
        };

      // Storage for let_error:
      template <class _Sender, class _Receiver, class _Fun>
        struct __storage<_Sender, _Receiver, _Fun, set_error_t> {
          template <class _Error>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _Error>, _Receiver>;

          // Compute a variant of tuples to hold all the errors of the input
          // sender:
          using __args_t =
            __error_types_of_t<
              _Sender,
              env_of_t<_Receiver>,
              __transform<__q<__decayed_tuple>, __nullable_variant_t>>;
          __args_t __args_;

          // Compute a variant of operation states:
          using __op_state3_t =
            __error_types_of_t<
              _Sender,
              env_of_t<_Receiver>,
              __transform<__q1<__op_state_for_t>, __nullable_variant_t>>;
          __op_state3_t __op_state3_;
        };

      // Storage for let_stopped
      template <class _Sender, class _Receiver, class _Fun>
        struct __storage<_Sender, _Receiver, _Fun, set_stopped_t> {
          variant<tuple<>> __args_;
          variant<monostate, connect_result_t<__call_result_t<_Fun>, _Receiver>> __op_state3_;
        };

      template <class _Env, class _Fun, class _Set, class _Sig>
        struct __tfx_signal;

      template <class _Env, class _Fun, class _Set, class _Ret, class... _Args>
          requires (!same_as<_Set, _Ret>)
        struct __tfx_signal<_Env, _Fun, _Set, _Ret(_Args...)> {
          using type = completion_signatures<_Ret(_Args...)>;
        };

      template <class _Env, class _Fun, class _Set, class... _Args>
          requires invocable<_Fun, _Args...> &&
            sender<invoke_result_t<_Fun, _Args...>, _Env>
        struct __tfx_signal<_Env, _Fun, _Set, _Set(_Args...)> {
          using type =
            make_completion_signatures<
              invoke_result_t<_Fun, _Args...>,
              _Env,
              completion_signatures<set_error_t(exception_ptr)>>;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __operation;

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __receiver {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          _Receiver&& base() && noexcept { return (_Receiver&&) __op_state_->__rcvr_;}
          const _Receiver& base() const & noexcept { return __op_state_->__rcvr_;}

          template <class... _As>
            using __which_tuple_t =
              __call_result_t<__which_tuple<_Sender, env_of_t<_Receiver>, _Let>, _As...>;

          template <class... _As>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _As...>, _Receiver>;

          template <__one_of<_Let> _Tag, class... _As>
              requires __applyable<_Fun, __which_tuple_t<_As...>&> &&
                sender_to<__apply_result_t<_Fun, __which_tuple_t<_As...>&>, _Receiver>
            friend void tag_invoke(_Tag, __receiver&& __self, _As&&... __as) noexcept try {
              using __tuple_t = __which_tuple_t<_As...>;
              using __op_state_t = __mapply<__q<__op_state_for_t>, __tuple_t>;
              auto& __args =
                __self.__op_state_->__storage_.__args_.template emplace<__tuple_t>((_As&&) __as...);
              start(__self.__op_state_->__storage_.__op_state3_.template emplace<__op_state_t>(
                __conv{[&] {
                  return connect(std::apply(std::move(__self.__op_state_->__fun_), __args), std::move(__self).base());
                }}
              ));
            } catch(...) {
              set_error(std::move(__self).base(), current_exception());
            }

          template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _As>
              requires __none_of<_Tag, _Let> && tag_invocable<_Tag, _Receiver, _As...>
            friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept try {
              __tag(std::move(__self).base(), (_As&&) __as...);
            } catch(...) {
              set_error(std::move(__self).base(), current_exception());
            }

          friend auto tag_invoke(get_env_t, const __receiver& __self)
            -> env_of_t<_Receiver> {
            return get_env(__self.base());
          }

          __operation<_SenderId, _ReceiverId, _Fun, _Let>* __op_state_;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __operation {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_t = __receiver<_SenderId, _ReceiverId, _Fun, _Let>;

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            start(__self.__op_state2_);
          }

          template <class _Receiver2>
            __operation(_Sender&& __sndr, _Receiver2&& __rcvr, _Fun __fun)
              : __op_state2_(connect((_Sender&&) __sndr, __receiver_t{this}))
              , __rcvr_((_Receiver2&&) __rcvr)
              , __fun_((_Fun&&) __fun)
            {}

          connect_result_t<_Sender, __receiver_t> __op_state2_;
          _Receiver __rcvr_;
          _Fun __fun_;
          [[no_unique_address]] __storage<_Sender, _Receiver, _Fun, _Let> __storage_;
        };

      template <class _SenderId, class _Fun, class _SetId>
        struct __sender {
          using _Sender = __t<_SenderId>;
          using _Set = __t<_SetId>;
          template <class _Self, class _Receiver>
            using __operation_t =
              __operation<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _Fun,
                _Set>;
          template <class _Self, class _Receiver>
            using __receiver_t =
              __receiver<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _Fun,
                _Set>;

          template <class _Env, class _Sig>
            using __tfx_signal_t = __t<__tfx_signal<_Env, _Fun, _Set, _Sig>>;

          template <class _Env>
            using __tfx_signal = __mbind_front_q1<__tfx_signal_t, _Env>;

          template <class _Self, class _Env>
            using __completions =
              __mapply<
                __transform<
                  __tfx_signal<_Env>,
                  __q<__concat_completion_signatures_t>>,
                completion_signatures_of_t<__member_t<_Self, _Sender>, _Env>>;

          template <__decays_to<__sender> _Self, class _Receiver>
              requires
                sender_to<__member_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation_t<_Self, _Receiver> {
              return __operation_t<_Self, _Receiver>{
                  ((_Self&&) __self).__sndr_,
                  (_Receiver&&) __rcvr,
                  ((_Self&&) __self).__fun_
              };
            }

          template <__sender_queries::__sender_query _Tag, class... _As>
              requires __callable<_Tag, const _Sender&, _As...>
            friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
              noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
              -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            }

          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> dependent_completion_signatures<_Env>;
          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> __completions<__member_t<_Self, _Sender>, _Env> requires true;

          _Sender __sndr_;
          _Fun __fun_;
        };

      template <class _LetTag, class _SetTag>
        struct __let_xxx_t {
          using type = _SetTag;
          template <class _Sender, class _Fun>
            using __sender = __impl::__sender<__x<remove_cvref_t<_Sender>>, _Fun, _LetTag>;

          template <sender _Sender, __movable_value _Fun>
            requires __tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>
          sender auto operator()(_Sender&& __sndr, _Fun __fun) const
            noexcept(nothrow_tag_invocable<_LetTag, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Fun>) {
            auto __sched = get_completion_scheduler<set_value_t>(__sndr);
            return tag_invoke(_LetTag{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
          }
          template <sender _Sender, __movable_value _Fun>
            requires (!__tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>) &&
              tag_invocable<_LetTag, _Sender, _Fun>
          sender auto operator()(_Sender&& __sndr, _Fun __fun) const
            noexcept(nothrow_tag_invocable<_LetTag, _Sender, _Fun>) {
            return tag_invoke(_LetTag{}, (_Sender&&) __sndr, (_Fun&&) __fun);
          }
          template <sender _Sender, __movable_value _Fun>
            requires (!__tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>) &&
              (!tag_invocable<_LetTag, _Sender, _Fun>) &&
              sender<__sender<_Sender, _Fun>>
          __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
            return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
          }
          template <class _Fun>
          __binder_back<_LetTag, _Fun> operator()(_Fun __fun) const {
            return {{}, {}, {(_Fun&&) __fun}};
          }
        };
    } // namespace __impl

    struct let_value_t
      : __let::__impl::__let_xxx_t<let_value_t, set_value_t>
    {};

    struct let_error_t
      : __let::__impl::__let_xxx_t<let_error_t, set_error_t>
    {};

    struct let_stopped_t
      : __let::__impl::__let_xxx_t<let_stopped_t, set_stopped_t>
    {};
  } // namespace __let
  using __let::let_value_t;
  inline constexpr let_value_t let_value{};
  using __let::let_error_t;
  inline constexpr let_error_t let_error{};
  using __let::let_stopped_t;
  inline constexpr let_stopped_t let_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  // [execution.senders.adaptors.stopped_as_error]
  namespace __stopped_as_xxx {
    template <class _SenderId, class _ReceiverId>
      struct __operation;

    template <class _SenderId, class _ReceiverId>
      struct __receiver : receiver_adaptor<__receiver<_SenderId, _ReceiverId>> {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;
        _Receiver&& base() && noexcept { return (_Receiver&&) __op_->__rcvr_; }
        const _Receiver& base() const & noexcept { return __op_->__rcvr_; }

        template <class _Ty>
          void set_value(_Ty&& __a) && noexcept try {
            using _Value = __single_sender_value_t<_Sender, env_of_t<_Receiver>>;
            static_assert(constructible_from<_Value, _Ty>);
            execution::set_value(
                ((__receiver&&) *this).base(),
                optional<_Value>{(_Ty&&) __a});
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }
        void set_stopped() && noexcept {
          using _Value = __single_sender_value_t<_Sender, env_of_t<_Receiver>>;
          execution::set_value(((__receiver&&) *this).base(), optional<_Value>{nullopt});
        }

        __operation<_SenderId, _ReceiverId>* __op_;
      };

    template <class _SenderId, class _ReceiverId>
      struct __operation {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;
        using __receiver_t = __receiver<_SenderId, _ReceiverId>;

        __operation(_Sender&& __sndr, _Receiver&& __rcvr)
          : __op_state_(connect((_Sender&&) __sndr, __receiver_t{{}, this}))
          , __rcvr_((_Receiver&&) __rcvr)
        {}

        friend void tag_invoke(start_t, __operation& __self) noexcept {
          start(__self.__op_state_);
        }

        connect_result_t<_Sender, __receiver_t> __op_state_;
        _Receiver __rcvr_;
      };

    template <class _SenderId>
      struct __sender {
        using _Sender = __t<_SenderId>;
        template <class _Self, class _Receiver>
          using __operation_t =
            __operation<__x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>>;
        template <class _Self, class _Receiver>
          using __receiver_t =
            __receiver<__x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>>;

        template <__decays_to<__sender> _Self, receiver _Receiver>
            requires __single_typed_sender<__member_t<_Self, _Sender>, env_of_t<_Receiver>> &&
              sender_to<__member_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            -> __operation_t<_Self, _Receiver> {
            return {((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
          }

        template <__sender_queries::__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

        template <class... _Tys>
            requires (sizeof...(_Tys) == 1)
          using __set_value_t =
            completion_signatures<set_value_t(optional<_Tys>...)>;

        template <class _Ty>
          using __set_error_t =
            completion_signatures<set_error_t(_Ty)>;

        template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
            make_completion_signatures<
              __member_t<_Self, _Sender>,
              _Env,
              completion_signatures<set_error_t(exception_ptr)>,
              __set_value_t,
              __set_error_t,
              completion_signatures<>>;

        _Sender __sndr_;
      };

    struct __stopped_as_optional_t {
      template <sender _Sender>
        auto operator()(_Sender&& __sndr) const
          -> __sender<__x<decay_t<_Sender>>> {
          return {(_Sender&&) __sndr};
        }
      __binder_back<__stopped_as_optional_t> operator()() const noexcept {
        return {};
      }
    };

    struct __stopped_as_error_t {
      template <sender _Sender, __movable_value _Error>
        auto operator()(_Sender&& __sndr, _Error __err) const {
          return (_Sender&&) __sndr
            | let_stopped([__err2 = (_Error&&) __err] () mutable {
                return just_error((_Error&&) __err2);
              });
        }
      template <__movable_value _Error>
        auto operator()(_Error __err) const
          -> __binder_back<__stopped_as_error_t, _Error> {
          return {{}, {}, {(_Error&&) __err}};
        }
    };
  } // namespace __stopped_as_xxx
  using __stopped_as_xxx::__stopped_as_optional_t;
  inline constexpr __stopped_as_optional_t stopped_as_optional{};
  using __stopped_as_xxx::__stopped_as_error_t;
  inline constexpr __stopped_as_error_t stopped_as_error{};

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  namespace __loop {
    class run_loop;

    namespace __impl {
      struct __task {
        virtual void __execute_() noexcept = 0;
        __task* __next_ = nullptr;
      };

      template <typename _ReceiverId>
        class __operation final : __task {
          using _Receiver = __t<_ReceiverId>;

          friend void tag_invoke(start_t, __operation& __op_state) noexcept {
            __op_state.__start_();
          }

          void __execute_() noexcept override try {
            if (get_stop_token(get_env(__rcvr_)).stop_requested()) {
              set_stopped((_Receiver&&) __rcvr_);
            } else {
              set_value((_Receiver&&) __rcvr_);
            }
          } catch(...) {
            set_error((_Receiver&&) __rcvr_, current_exception());
          }

          void __start_() noexcept;

          [[no_unique_address]] _Receiver __rcvr_;
          run_loop* const __loop_;

         public:
          template <typename _Receiver2>
          explicit __operation(_Receiver2&& __rcvr, run_loop* __loop)
            : __rcvr_((_Receiver2 &&) __rcvr)
            , __loop_(__loop) {}
        };
    } // namespace __impl

    class run_loop {
      template <class>
        friend class __impl::__operation;
     public:
      class __scheduler {
        struct __schedule_task {
          using completion_signatures =
            completion_signatures<
              set_value_t(),
              set_error_t(exception_ptr),
              set_stopped_t()>;
         private:
          friend __scheduler;

          template <typename _Receiver>
          friend __impl::__operation<__x<decay_t<_Receiver>>>
          tag_invoke(connect_t, const __schedule_task& __self, _Receiver&& __rcvr) {
            return __impl::__operation<__x<decay_t<_Receiver>>>{(_Receiver &&) __rcvr, __self.__loop_};
          }

          template <class _CPO>
          friend __scheduler
          tag_invoke(get_completion_scheduler_t<_CPO>, const __schedule_task& __self) noexcept {
            return __scheduler{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop)
          {}

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept : __loop_(__loop) {}

       public:
        friend __schedule_task tag_invoke(schedule_t, const __scheduler& __self) noexcept {
          return __self.__schedule();
        }

        friend execution::forward_progress_guarantee tag_invoke(
            get_forward_progress_guarantee_t, const __scheduler&) noexcept {
          return execution::forward_progress_guarantee::parallel;
        }

        bool operator==(const __scheduler&) const noexcept = default;

       private:
        __schedule_task __schedule() const noexcept {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
      };

      __scheduler get_scheduler() {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void __push_back_(__impl::__task* __task);
      __impl::__task* __pop_front_();

      mutex __mutex_;
      condition_variable __cv_;
      __impl::__task* __head_ = nullptr;
      __impl::__task* __tail_ = nullptr;
      bool __stop_ = false;
    };

    namespace __impl {
      template <typename _ReceiverId>
      inline void __operation<_ReceiverId>::__start_() noexcept try {
        __loop_->__push_back_(this);
      } catch(...) {
        set_error((_Receiver&&) __rcvr_, current_exception());
      }
    }

    inline void run_loop::run() {
      while (auto* __task = __pop_front_()) {
        __task->__execute_();
      }
    }

    inline void run_loop::finish() {
      unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__impl::__task* __task) {
      unique_lock __lock{__mutex_};
      if (__head_ == nullptr) {
        __head_ = __task;
      } else {
        __tail_->__next_ = __task;
      }
      __tail_ = __task;
      __task->__next_ = nullptr;
      __cv_.notify_one();
    }

    inline __impl::__task* run_loop::__pop_front_() {
      unique_lock __lock{__mutex_};
      while (__head_ == nullptr) {
        if (__stop_)
          return nullptr;
        __cv_.wait(__lock);
      }
      auto* __task = __head_;
      __head_ = __task->__next_;
      if (__head_ == nullptr)
        __tail_ = nullptr;
      return __task;
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  namespace __schedule_from {
    // The completion information can be stored in situ within a variant in
    // the operation state
    template <class _Sender, class _Env>
        requires sender<_Sender, _Env>
      struct __completion_storage {
        // Compute a variant type that is capable of storing the results of the
        // input sender when it completes. The variant has type:
        //   variant<
        //     tuple<set_stopped_t>,
        //     tuple<set_value_t, decay_t<_Values1>...>,
        //     tuple<set_value_t, decay_t<_Values2>...>,
        //        ...
        //     tuple<set_error_t, decay_t<_Error1>>,
        //     tuple<set_error_t, decay_t<_Error2>>,
        //        ...
        //   >
        template <class... _Ts>
          using __bind_tuples =
            __mbind_front_q<__variant, tuple<set_stopped_t>, _Ts...>;

        using __bound_values_t =
          __value_types_of_t<
            _Sender,
            _Env,
            __mbind_front_q<__decayed_tuple, set_value_t>,
            __q<__bind_tuples>>;

        using __variant_t =
          __error_types_of_t<
            _Sender,
            _Env,
            __transform<
              __mbind_front_q<__decayed_tuple, set_error_t>,
              __bound_values_t>>;

        template <class _Receiver>
          struct __f : private __variant_t {
            __f() = default;
            using __variant_t::emplace;

            void __complete(_Receiver& __rcvr) noexcept try {
              std::visit([&](auto&& __tupl) -> void {
                std::apply([&](auto __tag, auto&&... __args) -> void {
                  __tag((_Receiver&&) __rcvr, (decltype(__args)&&) __args...);
                }, (decltype(__tupl)&&) __tupl);
              }, (__variant_t&&) *this);
            } catch(...) {
              set_error((_Receiver&&) __rcvr, current_exception());
            }
          };
      };

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
      struct __operation1;

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
      struct __receiver1;

    // This receiver is to be completed on the execution context
    // associated with the scheduler. When the source sender
    // completes, the completion information is saved off in the
    // operation state so that when this receiver completes, it can
    // read the completion out of the operation state and forward it
    // to the output receiver after transitioning to the scheduler's
    // context.
    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
      struct __receiver2 {
        using _Receiver = __t<_ReceiverId>;
        __operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>* __op_state_;

        // If the work is successfully scheduled on the new execution
        // context and is ready to run, forward the completion signal in
        // the operation state
        friend void tag_invoke(set_value_t, __receiver2&& __self) noexcept {
          __self.__op_state_->__data_.__complete(__self.__op_state_->__rcvr_);
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        friend void tag_invoke(_Tag, __receiver2&& __self, _Args&&... __args) noexcept {
          _Tag{}((_Receiver&&) __self.__op_state_->__rcvr_, (_Args&&) __args...);
        }

        friend auto tag_invoke(get_env_t, const __receiver2& __self)
          -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };

    // This receiver is connected to the input sender. When that
    // sender completes (on whatever context it completes on), save
    // the completion information into the operation state. Then,
    // schedule a second operation to __complete on the execution
    // context of the scheduler. That second receiver will read the
    // completion information out of the operation state and propagate
    // it to the output receiver from within the desired context.
    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
      struct __receiver1 {
        using _CvrefSender = __t<_CvrefSenderId>;
        using _Receiver = __t<_ReceiverId>;
        using __receiver2_t =
          __receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>;
        __operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>* __op_state_;

        template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        friend void tag_invoke(_Tag, __receiver1&& __self, _Args&&... __args) noexcept try {
          // Write the tag and the args into the operation state so that
          // we can forward the completion from within the scheduler's
          // execution context.
          __self.__op_state_->__data_.template emplace<__decayed_tuple<_Tag, _Args...>>(_Tag{}, (_Args&&) __args...);
          // Schedule the completion to happen on the scheduler's
          // execution context.
          __self.__op_state_->__state2_.emplace(
              __conv{[__op_state = __self.__op_state_] {
                return connect(schedule(__op_state->__sched_), __receiver2_t{__op_state});
              }});
          // Enqueue the scheduled operation:
          start(*__self.__op_state_->__state2_);
        } catch(...) {
          set_error((_Receiver&&) __self.__op_state_->__rcvr_, current_exception());
        }

        friend auto tag_invoke(get_env_t, const __receiver1& __self)
          -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
      struct __operation1 {
        using _Scheduler = __t<_SchedulerId>;
        using _CvrefSender = __t<_CvrefSenderId>;
        using _Receiver = __t<_ReceiverId>;
        using __receiver1_t =
          __receiver1<_SchedulerId, _CvrefSenderId, _ReceiverId>;
        using __receiver2_t =
          __receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>;

        _Scheduler __sched_;
        _Receiver __rcvr_;
        __minvoke<__completion_storage<_CvrefSender, env_of_t<_Receiver>>, _Receiver> __data_;
        connect_result_t<_CvrefSender, __receiver1_t> __state1_;
        optional<connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t>> __state2_;

        __operation1(_Scheduler __sched, _CvrefSender&& __sndr, __decays_to<_Receiver> auto&& __rcvr)
          : __sched_(__sched)
          , __rcvr_((decltype(__rcvr)&&) __rcvr)
          , __state1_(connect((_CvrefSender&&) __sndr, __receiver1_t{this})) {}

        friend void tag_invoke(start_t, __operation1& __op_state) noexcept {
          start(__op_state.__state1_);
        }
      };

    template <class _SchedulerId, class _SenderId>
      struct __sender {
        using _Scheduler = __t<_SchedulerId>;
        using _Sender = __t<_SenderId>;
        _Scheduler __sched_;
        _Sender __sndr_;

        template <__decays_to<__sender> _Self, class _Receiver>
          requires sender_to<__member_t<_Self, _Sender>, _Receiver>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            -> __operation1<_SchedulerId, __x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>> {
          return {__self.__sched_, ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
        }

        template <__one_of<set_value_t, set_stopped_t> _Tag>
        friend _Scheduler tag_invoke(get_completion_scheduler_t<_Tag>, const __sender& __self) noexcept {
          return __self.__sched_;
        }

        template <__sender_queries::__sender_query _Tag, class... _As>
          requires __callable<_Tag, const _Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
          noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
          -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
          return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
        }

        template <class...>
          using __value_t = completion_signatures<>;

        template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
            make_completion_signatures<
              __member_t<_Self, _Sender>,
              _Env,
              make_completion_signatures<
                schedule_result_t<_Scheduler>,
                _Env,
                completion_signatures<set_error_t(exception_ptr)>,
                __value_t>>;
      };

    struct schedule_from_t {
      // NOT TO SPEC: permit non-typed senders:
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<schedule_from_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<schedule_from_t, _Scheduler, _Sender>)
        -> tag_invoke_result_t<schedule_from_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      // NOT TO SPEC: permit non-typed senders:
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        -> __sender<__x<decay_t<_Scheduler>>, __x<decay_t<_Sender>>> {
        return {(_Scheduler&&) __sched, (_Sender&&) __sndr};
      }
    };
  } // namespace __schedule_from
  using __schedule_from::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.transfer]
  namespace __transfer {
    struct transfer_t {
      template <sender _Sender, scheduler _Scheduler>
        requires __tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>
      tag_invoke_result_t<transfer_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Scheduler>
      operator()(_Sender&& __sndr, _Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<transfer_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Scheduler>) {
        auto csch = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(transfer_t{}, std::move(csch), (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }
      template <sender _Sender, scheduler _Scheduler>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>) &&
          tag_invocable<transfer_t, _Sender, _Scheduler>
      tag_invoke_result_t<transfer_t, _Sender, _Scheduler>
      operator()(_Sender&& __sndr, _Scheduler&& __sched) const noexcept(nothrow_tag_invocable<transfer_t, _Sender, _Scheduler>) {
        return tag_invoke(transfer_t{}, (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }
      // NOT TO SPEC: permit non-typed senders:
      template <sender _Sender, scheduler _Scheduler>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>) &&
          (!tag_invocable<transfer_t, _Sender, _Scheduler>)
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const {
        return schedule_from((_Scheduler&&) __sched, (_Sender&&) __sndr);
      }
      template <scheduler _Scheduler>
      __binder_back<transfer_t, decay_t<_Scheduler>> operator()(_Scheduler&& __sched) const {
        return {{}, {}, {(_Scheduler&&) __sched}};
      }
    } ;
  } // namespace __transfer
  using __transfer::transfer_t;
  inline constexpr transfer_t transfer{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on {
    namespace __impl {
      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __operation;

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __receiver_ref
          : receiver_adaptor<__receiver_ref<_SchedulerId, _SenderId, _ReceiverId>> {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          __operation<_SchedulerId, _SenderId, _ReceiverId>* __op_state_;
          _Receiver&& base() && noexcept {
            return (_Receiver&&) __op_state_->__rcvr_;
          }
          const _Receiver& base() const & noexcept {
            return __op_state_->__rcvr_;
          }
          friend auto tag_invoke(get_env_t, const __receiver_ref& __self)
            -> make_env_t<get_scheduler_t, _Scheduler, env_of_t<_Receiver>> {
            return make_env<get_scheduler_t>(
              __self.__op_state_->__scheduler_,
              get_env(__self.base()));
          }
        };

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __receiver
          : receiver_adaptor<__receiver<_SchedulerId, _SenderId, _ReceiverId>> {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_ref_t =
            __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;
          __operation<_SchedulerId, _SenderId, _ReceiverId>* __op_state_;
          _Receiver&& base() && noexcept {
            return (_Receiver&&) __op_state_->__rcvr_;
          }
          const _Receiver& base() const & noexcept {
            return __op_state_->__rcvr_;
          }

          void set_value() && noexcept {
            // cache this locally since *this is going bye-bye.
            auto* __op_state = __op_state_;
            try {
              // This line will invalidate *this:
              start(__op_state->__data_.template emplace<1>(__conv{
                [__op_state] {
                  return connect((_Sender&&) __op_state->__sndr_,
                                  __receiver_ref_t{{}, __op_state});
                }
              }));
            } catch(...) {
              set_error((_Receiver&&) __op_state->__rcvr_,
                        current_exception());
            }
          }
        };

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __operation {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_t = __receiver<_SchedulerId, _SenderId, _ReceiverId>;
          using __receiver_ref_t = __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            start(std::get<0>(__self.__data_));
          }

          template <class _Sender2, class _Receiver2>
          __operation(_Scheduler __sched, _Sender2&& __sndr, _Receiver2&& __rcvr)
            : __data_{in_place_index<0>, __conv{[&, this]{
                return connect(schedule(__sched),
                                __receiver_t{{}, this});
              }}}
            , __scheduler_((_Scheduler&&) __sched)
            , __sndr_((_Sender2&&) __sndr)
            , __rcvr_((_Receiver2&&) __rcvr) {}

          variant<
              connect_result_t<schedule_result_t<_Scheduler>, __receiver_t>,
              connect_result_t<_Sender, __receiver_ref_t>> __data_;
          _Scheduler __scheduler_;
          _Sender __sndr_;
          _Receiver __rcvr_;
        };

      template <class _SchedulerId, class _SenderId>
        struct __sender {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          template <class _ReceiverId>
            using __receiver_ref_t =
              __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;
          template <class _ReceiverId>
            using __receiver_t =
              __receiver<_SchedulerId, _SenderId, _ReceiverId>;
          template <class _ReceiverId>
            using __operation_t =
              __operation<_SchedulerId, _SenderId, _ReceiverId>;

          _Scheduler __scheduler_;
          _Sender __sndr_;

          template <__decays_to<__sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<schedule_result_t<_Scheduler>,
                        __receiver_t<__x<decay_t<_Receiver>>>> &&
              sender_to<_Sender, __receiver_ref_t<__x<decay_t<_Receiver>>>>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            -> __operation_t<__x<decay_t<_Receiver>>> {
            return {((_Self&&) __self).__scheduler_,
                    ((_Self&&) __self).__sndr_,
                    (_Receiver&&) __rcvr};
          }

          template <__sender_queries::__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

          template <class...>
            using __value_t = completion_signatures<>;

          template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
            make_completion_signatures<
              schedule_result_t<_Scheduler>,
              _Env,
              make_completion_signatures<
                __member_t<_Self, _Sender>,
                make_env_t<get_scheduler_t, _Scheduler, _Env>,
                completion_signatures<set_error_t(exception_ptr)>>,
              __value_t>;
        };
    } // namespace __impl

    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<on_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<on_t, _Scheduler, _Sender>)
        -> tag_invoke_result_t<on_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        -> __impl::__sender<__x<decay_t<_Scheduler>>,
                            __x<decay_t<_Sender>>> {
        return {(_Scheduler&&) __sched, (_Sender&&) __sndr};
      }
    };
  } // namespace __on
  using __on::on_t;
  inline constexpr on_t on{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    struct transfer_just_t {
      template <scheduler _Scheduler, __movable_value... _Values>
        requires tag_invocable<transfer_just_t, _Scheduler, _Values...> &&
          sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        noexcept(nothrow_tag_invocable<transfer_just_t, _Scheduler, _Values...>)
        -> tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Values&&) __vals...);
      }
      template <scheduler _Scheduler, __movable_value... _Values>
        requires (!tag_invocable<transfer_just_t, _Scheduler, _Values...> ||
          !sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>)
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        -> decltype(transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched)) {
        return transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched);
      }
    };
  } // namespace __transfer_just
  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  namespace __into_variant {
    template <class _Sender, class _Env>
        requires sender<_Sender, _Env>
      using __into_variant_result_t =
        value_types_of_t<_Sender, _Env>;

    template <class _SenderId, class _ReceiverId>
      class __receiver
        : receiver_adaptor<__receiver<_SenderId, _ReceiverId>, __t<_ReceiverId>> {
        using _Sender = __t<_SenderId>;
        using _Receiver = __t<_ReceiverId>;
        friend receiver_adaptor<__receiver, _Receiver>;

        // Customize set_value by building a variant and passing the result
        // to the base class
        template <class... _As>
          void set_value(_As&&... __as) && noexcept try {
            using __variant_t =
              __into_variant_result_t<_Sender, env_of_t<_Receiver>>;
            static_assert(constructible_from<__variant_t, tuple<_As&&...>>);
            execution::set_value(
                ((__receiver&&) *this).base(),
                __variant_t{tuple<_As&&...>{(_As&&) __as...}});
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }

       public:
        using receiver_adaptor<__receiver, _Receiver>::receiver_adaptor;
      };

    template <class _SenderId>
      class __sender {
        using _Sender = __t<_SenderId>;
        friend sender_adaptor<__sender, _Sender>;
        template <class _Receiver>
          using __receiver_t = __receiver<_SenderId, __x<remove_cvref_t<_Receiver>>>;

        template <class...>
          using __value_t = completion_signatures<>;

        template <class _Env>
          using __completion_signatures =
            make_completion_signatures<
              _Sender,
              _Env,
              completion_signatures<
                set_value_t(__into_variant_result_t<_Sender, _Env>),
                set_error_t(exception_ptr)>,
              __value_t>;

        _Sender __sndr_;

        template <class _Receiver>
          requires sender_to<_Sender, __receiver_t<_Receiver>>
        friend auto tag_invoke(connect_t, __sender&& __self, _Receiver&& __rcvr)
          noexcept(__has_nothrow_connect<_Sender, __receiver_t<_Receiver>>)
          -> connect_result_t<_Sender, __receiver_t<_Receiver>> {
          return execution::connect(
              (_Sender&&) __self.__sndr_,
              __receiver_t<_Receiver>{(_Receiver&&) __rcvr});
        }

        template <__sender_queries::__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
            -> __call_result_if_t<__sender_queries::__sender_query<_Tag>, _Tag, const _Sender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

        template <class _Env>
          friend auto tag_invoke(get_completion_signatures_t, __sender&&, _Env) ->
            __completion_signatures<_Env>;

       public:
        explicit __sender(__decays_to<_Sender> auto&& __sndr)
          : __sndr_((decltype(__sndr)) __sndr) {}
      };

    struct __into_variant_t {
      template <sender _Sender>
        auto operator()(_Sender&& __sndr) const
          -> __sender<__x<remove_cvref_t<_Sender>>> {
          return __sender<__x<remove_cvref_t<_Sender>>>{(_Sender&&) __sndr};
        }
      auto operator()() const noexcept {
        return __binder_back<__into_variant_t>{};
      }
    };
  } // namespace __into_variant
  using __into_variant::__into_variant_t;
  inline constexpr __into_variant_t into_variant{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    namespace __impl {
      enum __state_t { __started, __error, __stopped };

      struct __on_stop_requested {
        in_place_stop_source& __stop_source_;
        void operator()() noexcept {
          __stop_source_.request_stop();
        }
      };

      template <class _Env>
        using __env_t =
          make_env_t<get_stop_token_t, in_place_stop_token, _Env>;

      template <class...>
        using __swallow_values = completion_signatures<>;

      template <class _Tag, class _Sender, class _Env>
        using __count_of =
          typename completion_signatures_of_t<_Sender, _Env>
            ::template __count_of<_Tag>;

      template <class _Env, class... _Senders>
        struct __traits {
          using type = dependent_completion_signatures<_Env>;
        };

      template <class _Env, class... _Senders>
          requires ((__count_of<set_value_t, _Senders, _Env>::value <= 1) &&...)
        struct __traits<_Env, _Senders...> {
          using __non_values =
            __concat_completion_signatures_t<
              completion_signatures<
                set_error_t(exception_ptr),
                set_stopped_t()>,
              make_completion_signatures<
                _Senders,
                _Env,
                completion_signatures<>,
                __swallow_values>...>;
          using __values =
            __minvoke<
              __concat<__qf<set_value_t>>,
              __value_types_of_t<
                _Senders,
                _Env,
                __q<__types>,
                __single_or<__types<>>>...>;
          using type =
            __if_c<
              (!__count_of<set_value_t, _Senders, _Env>::value ||...),
              __non_values,
              __minvoke2<
                __push_back<__q<completion_signatures>>, __non_values, __values>>;
        };

      template <class... _SenderIds>
        struct __sender {
          template <class... _Sndrs>
            explicit __sender(_Sndrs&&... __sndrs)
              : __sndrs_((_Sndrs&&) __sndrs...)
            {}

         private:
          template <class _CvrefEnv>
            using __completion_sigs =
              __t<__traits<
                __env_t<remove_cvref_t<_CvrefEnv>>,
                __member_t<_CvrefEnv, __t<_SenderIds>>...>>;

          template <class _Traits>
            using __sends_values =
              __bool<_Traits::template __count_of<set_value_t>::value != 0>;

          template <class _CvrefReceiverId>
            struct __operation;

          template <class _CvrefReceiverId, size_t _Index>
            struct __receiver : receiver_adaptor<__receiver<_CvrefReceiverId, _Index>> {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;
              using _Traits =
                __completion_sigs<
                  __member_t<_CvrefReceiverId, env_of_t<_Receiver>>>;

              _Receiver&& base() && noexcept {
                return (_Receiver&&) __op_state_->__recvr_;
              }
              const _Receiver& base() const & noexcept {
                return __op_state_->__recvr_;
              }
              template <class _Error>
                void __set_error(_Error&& __err, __state_t __expected) noexcept {
                  // TODO: _What memory orderings are actually needed here?
                  if (__op_state_->__state_.compare_exchange_strong(__expected, __error)) {
                    __op_state_->__stop_source_.request_stop();
                    // We won the race, free to write the error into the operation
                    // state without worry.
                    try {
                      __op_state_->__errors_.template emplace<decay_t<_Error>>((_Error&&) __err);
                    } catch(...) {
                      __op_state_->__errors_.template emplace<exception_ptr>(current_exception());
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class... _Values>
                void set_value(_Values&&... __vals) && noexcept {
                  if constexpr (__sends_values<_Traits>::value) {
                    // We only need to bother recording the completion values
                    // if we're not already in the "error" or "stopped" state.
                    if (__op_state_->__state_ == __started) {
                      try {
                        std::get<_Index>(__op_state_->__values_).emplace(
                            (_Values&&) __vals...);
                      } catch(...) {
                        __set_error(current_exception(), __started);
                      }
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class _Error>
                  requires tag_invocable<set_error_t, _Receiver, _Error>
                void set_error(_Error&& __err) && noexcept {
                  __set_error((_Error&&) __err, __started);
                }
              void set_stopped() && noexcept {
                __state_t __expected = __started;
                // Transition to the "stopped" state if and only if we're in the
                // "started" state. (_If this fails, it's because we're in an
                // error state, which trumps cancellation.)
                if (__op_state_->__state_.compare_exchange_strong(__expected, __stopped)) {
                  __op_state_->__stop_source_.request_stop();
                }
                __op_state_->__arrive();
              }
              friend auto tag_invoke(get_env_t, const __receiver& __self)
                -> __env_t<env_of_t<_Receiver>> {
                return make_env<get_stop_token_t>(
                  __self.__op_state_->__stop_source_.get_token(),
                  get_env(__self.base()));
              }
              __operation<_CvrefReceiverId>* __op_state_;
            };

          template <class _CvrefReceiverId>
            struct __operation {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;
              using _Env = env_of_t<_Receiver>;
              using _CvrefEnv = __member_t<_CvrefReceiverId, _Env>;
              using _Traits = __completion_sigs<_CvrefEnv>;

              template <class _Sender, size_t _Index>
                using __child_op_state =
                  connect_result_t<
                    __member_t<_WhenAll, _Sender>,
                    __receiver<_CvrefReceiverId, _Index>>;

              using _Indices = index_sequence_for<_SenderIds...>;

              template <size_t... _Is>
                static auto __connect_children(
                    __operation* __self, _WhenAll&& __when_all, index_sequence<_Is...>)
                    -> tuple<__child_op_state<__t<_SenderIds>, _Is>...> {
                  return tuple<__child_op_state<__t<_SenderIds>, _Is>...>{
                    __conv{[&__when_all, __self]() {
                      return execution::connect(
                          std::get<_Is>(((_WhenAll&&) __when_all).__sndrs_),
                          __receiver<_CvrefReceiverId, _Is>{{}, __self});
                    }}...
                  };
                }

              using __child_op_states_tuple_t =
                  decltype(__connect_children(nullptr, __declval<_WhenAll>(), _Indices{}));

              void __arrive() noexcept {
                if (0 == --__count_) {
                  __complete();
                }
              }

              void __complete() noexcept {
                // Stop callback is no longer needed. Destroy it.
                __on_stop_.reset();
                // All child operations have completed and arrived at the barrier.
                switch(__state_.load(memory_order_relaxed)) {
                case __started:
                  if constexpr (__sends_values<_Traits>::value) {
                    // All child operations completed successfully:
                    std::apply(
                      [this](auto&... __opt_vals) -> void {
                        std::apply(
                          [this](auto&... __all_vals) -> void {
                            try {
                              execution::set_value(
                                  (_Receiver&&) __recvr_, std::move(__all_vals)...);
                            } catch(...) {
                              execution::set_error(
                                  (_Receiver&&) __recvr_, current_exception());
                            }
                          },
                          std::tuple_cat(
                            std::apply(
                              [](auto&... __vals) { return std::tie(__vals...); },
                              *__opt_vals
                            )...
                          )
                        );
                      },
                      __values_
                    );
                  }
                  break;
                case __error:
                  std::visit([this](auto& __err) noexcept {
                    execution::set_error((_Receiver&&) __recvr_, std::move(__err));
                  }, __errors_);
                  break;
                case __stopped:
                  execution::set_stopped((_Receiver&&) __recvr_);
                  break;
                default:
                  ;
                }
              }

              __operation(_WhenAll&& when_all, _Receiver __rcvr)
                : __child_states_{__connect_children(this, (_WhenAll&&) when_all, _Indices{})}
                , __recvr_((_Receiver&&) __rcvr)
              {}

              friend void tag_invoke(start_t, __operation& __self) noexcept {
                // register stop callback:
                __self.__on_stop_.emplace(
                    get_stop_token(get_env(__self.__recvr_)),
                    __on_stop_requested{__self.__stop_source_});
                if (__self.__stop_source_.stop_requested()) {
                  // Stop has already been requested. Don't bother starting
                  // the child operations.
                  execution::set_stopped((_Receiver&&) __self.__recvr_);
                } else {
                  apply([](auto&&... __child_ops) noexcept -> void {
                    (execution::start(__child_ops), ...);
                  }, __self.__child_states_);
                }
              }

              // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
              using __child_values_tuple_t =
                __if<
                  __sends_values<_Traits>,
                  __minvoke<
                    __q<tuple>,
                    __value_types_of_t<
                      __t<_SenderIds>,
                      __env_t<_Env>,
                      __mcompose<__q1<optional>, __q<__decayed_tuple>>,
                      __single_or<void>>...>,
                  __>;

              __child_op_states_tuple_t __child_states_;
              _Receiver __recvr_;
              atomic<size_t> __count_{sizeof...(_SenderIds)};
              // Could be non-atomic here and atomic_ref everywhere except __completion_fn
              atomic<__state_t> __state_{__started};
              error_types_of_t<__sender, __env_t<_Env>, __variant> __errors_{};
              [[no_unique_address]] __child_values_tuple_t __values_{};
              in_place_stop_source __stop_source_{};
              optional<typename stop_token_of_t<env_of_t<_Receiver>&>::template
                  callback_type<__on_stop_requested>> __on_stop_{};
            };

          template <__decays_to<__sender> _Self, receiver _Receiver>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation<__member_t<_Self, __x<decay_t<_Receiver>>>> {
              return {(_Self&&) __self, (_Receiver&&) __rcvr};
            }

          template <__decays_to<__sender> _Self, class _Env>
            friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
              -> __completion_sigs<__member_t<_Self, _Env>>;

          tuple<__t<_SenderIds>...> __sndrs_;
        };

      template <class _Sender>
        using __into_variant_result_t =
          decltype(into_variant(__declval<_Sender>()));
    } // namespce __impl

    struct when_all_t {
      template <sender... _Senders>
        requires tag_invocable<when_all_t, _Senders...> &&
          sender<tag_invoke_result_t<when_all_t, _Senders...>> &&
          (sizeof...(_Senders) > 0)
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_t, _Senders...>)
        -> tag_invoke_result_t<when_all_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
          requires (!tag_invocable<when_all_t, _Senders...>) &&
            (sizeof...(_Senders) > 0)
      auto operator()(_Senders&&... __sndrs) const
        -> __impl::__sender<__x<decay_t<_Senders>>...> {
        return __impl::__sender<__x<decay_t<_Senders>>...>{
            (_Senders&&) __sndrs...};
      }
    };

    struct when_all_with_variant_t {
      template <sender... _Senders>
        requires tag_invocable<when_all_with_variant_t, _Senders...> &&
          sender<tag_invoke_result_t<when_all_with_variant_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_with_variant_t, _Senders...>)
        -> tag_invoke_result_t<when_all_with_variant_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
        requires (!tag_invocable<when_all_with_variant_t, _Senders...>) &&
          (__callable<__into_variant_t, _Senders> &&...)
      auto operator()(_Senders&&... __sndrs) const {
          return when_all_t{}(into_variant((_Senders&&) __sndrs)...);
      }
    };

    struct transfer_when_all_t {
      template <scheduler _Sched, sender... _Senders>
        requires tag_invocable<transfer_when_all_t, _Sched, _Senders...> &&
          sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_t, _Sched, _Senders...>)
        -> tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, sender... _Senders>
        requires ((!tag_invocable<transfer_when_all_t, _Sched, _Senders...>) ||
          (!sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>))
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer(when_all_t{}((_Senders&&) __sndrs...), (_Sched&&) __sched);
      }
    };

    struct transfer_when_all_with_variant_t {
      template <scheduler _Sched, sender... _Senders>
        requires tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...> &&
          sender<tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>)
        -> tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, sender... _Senders>
        requires (!tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>) &&
          (__callable<__into_variant_t, _Senders> &&...)
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer_when_all_t{}((_Sched&&) __sched, into_variant((_Senders&&) __sndrs)...);
      }
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

  namespace __read {
    template <class _Tag, class _ReceiverId>
      struct __operation {
        __t<_ReceiverId> __rcvr_;
        friend void tag_invoke(start_t, __operation& __self) noexcept try {
          auto __env = get_env(__self.__rcvr_);
          set_value(std::move(__self.__rcvr_), _Tag{}(__env));
        } catch(...) {
          set_error(std::move(__self.__rcvr_), current_exception());
        }
      };

    template <class _Tag>
      struct __sender {
        template <class _Env>
            requires __callable<_Tag, _Env>
          using __completions_t =
            completion_signatures<
              set_value_t(__call_result_t<_Tag, _Env>),
              set_error_t(exception_ptr)>;

        template <class _Receiver>
          requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
        friend auto tag_invoke(connect_t, __sender, _Receiver&& __rcvr)
          noexcept(is_nothrow_constructible_v<decay_t<_Receiver>, _Receiver>)
          -> __operation<_Tag, __x<decay_t<_Receiver>>> {
          return {(_Receiver&&) __rcvr};
        }

        template <class _Env>
          friend auto tag_invoke(get_completion_signatures_t, __sender, _Env) ->
            dependent_completion_signatures<_Env>;
        template <__typename _Env>
          friend auto tag_invoke(get_completion_signatures_t, __sender, _Env) ->
            __completions_t<_Env>;
      };

    struct __read_t {
      template <class _Tag>
      constexpr __sender<_Tag> operator()(_Tag) const noexcept {
        return {};
      }
    };
  } // namespace __read

  inline constexpr __read::__read_t read {};

  namespace __general_queries::__impl {
    inline auto get_scheduler_t::operator()() const noexcept {
      return read(get_scheduler);
    }
    inline auto get_delegatee_scheduler_t::operator()() const noexcept {
      return read(get_delegatee_scheduler);
    }
    inline auto get_allocator_t::operator()() const noexcept {
      return read(get_allocator);
    }
    inline auto get_stop_token_t::operator()() const noexcept {
      return read(get_stop_token);
    }
  }
} // namespace std::execution

namespace std::this_thread {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    namespace __impl {
      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));

      struct __env {
        execution::run_loop::__scheduler __sched_;

        friend auto tag_invoke(execution::get_scheduler_t, const __env& __self) noexcept
          -> execution::run_loop::__scheduler {
          return __self.__sched_;
        }

        friend auto tag_invoke(execution::get_delegatee_scheduler_t, const __env& __self) noexcept
          -> execution::run_loop::__scheduler {
          return __self.__sched_;
        }
      };

      // What should sync_wait(just_stopped()) return?
      template <class _Sender>
          requires execution::sender<_Sender, __env>
        using __sync_wait_result_t =
          execution::value_types_of_t<
            _Sender,
            __env,
            execution::__decayed_tuple,
            __single_t>;

      template <class _Sender>
        using __sync_wait_with_variant_result_t =
          __sync_wait_result_t<__into_variant_result_t<_Sender>>;

      template <class _SenderId>
        struct __state;

      template <class _SenderId>
        struct __receiver {
          using _Sender = __t<_SenderId>;
          __state<_SenderId>* __state_;
          execution::run_loop* __loop_;
          template <class _Sender2 = _Sender, class... _As>
            requires constructible_from<__sync_wait_result_t<_Sender2>, _As...>
          friend void tag_invoke(execution::set_value_t, __receiver&& __rcvr, _As&&... __as) noexcept try {
            __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
            __rcvr.__loop_->finish();
          } catch(...) {
            execution::set_error((__receiver&&) __rcvr, current_exception());
          }
          friend void tag_invoke(execution::set_error_t, __receiver&& __rcvr, exception_ptr __err) noexcept {
            __rcvr.__state_->__data_.template emplace<2>((exception_ptr&&) __err);
            __rcvr.__loop_->finish();
          }
          friend void tag_invoke(execution::set_stopped_t __d, __receiver&& __rcvr) noexcept {
            __rcvr.__state_->__data_.template emplace<3>(__d);
            __rcvr.__loop_->finish();
          }
          friend __env
          tag_invoke(execution::get_env_t, const __receiver& __rcvr) noexcept {
            return {__rcvr.__loop_->get_scheduler()};
          }
        };

      template <class _SenderId>
        struct __state {
          using _Tuple = __sync_wait_result_t<__t<_SenderId>>;
          variant<monostate, _Tuple, exception_ptr, execution::set_stopped_t> __data_;
        };

      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));
    } // namespace __impl

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      // TODO: constrain on return type
      template <execution::__single_value_variant_sender<__impl::__env> _Sender> // NOT TO SPEC
        requires
          execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>
      tag_invoke_result_t<
        sync_wait_t,
        execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
        _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<
          sync_wait_t,
          execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
          _Sender>) {
        auto __sched =
          execution::get_completion_scheduler<execution::set_value_t>(__sndr);
        return tag_invoke(sync_wait_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      // TODO: constrain on return type
      template <execution::__single_value_variant_sender<__impl::__env> _Sender> // NOT TO SPEC
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>) &&
          tag_invocable<sync_wait_t, _Sender>
      tag_invoke_result_t<sync_wait_t, _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<sync_wait_t, _Sender>) {
        return tag_invoke(sync_wait_t{}, (_Sender&&) __sndr);
      }
      template <execution::__single_value_variant_sender<__impl::__env> _Sender>
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>) &&
          (!tag_invocable<sync_wait_t, _Sender>) &&
          execution::sender<_Sender, __impl::__env> &&
          execution::sender_to<_Sender, __impl::__receiver<__x<_Sender>>>
      auto operator()(_Sender&& __sndr) const
        -> optional<__impl::__sync_wait_result_t<_Sender>> {
        using state_t = __impl::__state<__x<_Sender>>;
        state_t __state {};
        execution::run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state =
          execution::connect(
            (_Sender&&) __sndr,
            __impl::__receiver<__x<_Sender>>{&__state, &__loop});
        execution::start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait_with_variant]
    struct sync_wait_with_variant_t {
      template <execution::__single_value_variant_sender<__impl::__env> _Sender> // NOT TO SPEC
        requires
          execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>
      tag_invoke_result_t<
        sync_wait_with_variant_t,
        execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
        _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<
          sync_wait_with_variant_t,
          execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
          _Sender>) {
        auto __sched =
          execution::get_completion_scheduler<execution::set_value_t>(__sndr);
        return tag_invoke(
          sync_wait_with_variant_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      template <execution::__single_value_variant_sender<__impl::__env> _Sender> // NOT TO SPEC
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>) &&
          tag_invocable<sync_wait_with_variant_t, _Sender>
      tag_invoke_result_t<sync_wait_with_variant_t, _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<sync_wait_with_variant_t, _Sender>) {
        return tag_invoke(sync_wait_with_variant_t{}, (_Sender&&) __sndr);
      }
      template <execution::__single_value_variant_sender<__impl::__env> _Sender>
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>) &&
          (!tag_invocable<sync_wait_with_variant_t, _Sender>) &&
          invocable<sync_wait_t, __impl::__into_variant_result_t<_Sender>>
      optional<__impl::__sync_wait_with_variant_result_t<_Sender>>
      operator()(_Sender&& __sndr) const {
        return sync_wait(execution::into_variant((_Sender&&) __sndr));
      }
    };
  } // namespace __sync_wait
  using __sync_wait::sync_wait_t;
  inline constexpr sync_wait_t sync_wait{};
  using __sync_wait::sync_wait_with_variant_t;
  inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};
} // namespace std::this_thread

_PRAGMA_POP()
