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
#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__connect.hpp"
#  include "__diagnostics.hpp"
#  include "__domain.hpp"
#  include "__env.hpp"
#  include "__get_completion_signatures.hpp"
#  include "__memory.hpp"
#  include "__meta.hpp"
#  include "__operation_states.hpp"
#  include "__query.hpp"
#  include "__receivers.hpp"
#  include "__schedulers.hpp"
#  include "__sender_concepts.hpp"
#  include "__senders.hpp"
#  include "__subscribe.hpp"
#  include "__tuple.hpp"
#  include "__type_traits.hpp"
#  include "__utility.hpp"
#  include "__variant.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <cstddef>
#    include <cstdint>
#    include <exception>
#    include <type_traits>
#  endif

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

// The sender facade: a way to write a sender algorithm so that its completion
// signatures and its completion attributes (domain, behavior, scheduler) are
// derived from the code that performs its completions, rather than computed
// separately by hand. See MAINTAINERS.md for a guide to writing an algorithm.
//
// The design:
//
//   1. Effects. Algorithms complete with `__send`, which returns an *effect
//      token* -- a type that records what was done. A handler
//      returns the tokens of everything it did synchronously: `_Tag(_Args...)`
//      for a completion, `__starts<_Key>` for starting a child or nested
//      operation. Branches that complete differently are joined with
//      combinators (`__choose`, `__when`, `__visit`, `__set_value_from`) whose
//      result is the union of their branches' tokens.
//
//   2. Provenance. The facade owns every *trigger* -- `start()` and the
//      receivers of all child and nested operations -- so it knows where each
//      effect happened: `__from_start`, or `__via<_Key, _Tag, _Src>`, meaning
//      "inside nested op _Key's _Tag completion, where _Key was started from
//      _Src". The algorithm never mentions domains, schedulers, or behaviors.
//
//   3. Derivation. A consteval walk from `start()` through the tokens yields
//      the set of (signature, provenance) events. The completion signatures
//      are the event signatures. The attributes of a channel are a join over
//      the provenance of that channel's events, computed lazily when the
//      attributes are queried, so completion *schedulers* (runtime values)
//      work too:
//        - behavior:  join over the provenances; `__via` composes sequentially
//        - domain:    the one domain, or indeterminate_domain<all of them>
//        - scheduler: reported only if every event on the channel gets its
//                     scheduler from the same place (as events from a nested
//                     facade sender's completions of different kinds do, if
//                     it gets their schedulers from the same place)
//
//   4. Type errors. The walk *evaluates* each nested sender's
//      get_completion_signatures, so an ill-typed child fails the walk the
//      way it fails any other algorithm: by throwing when constexpr exceptions
//      are available, or by returning an __mexception otherwise. Children are
//      checked before any state type is computed, because state types may
//      depend on the children's completions. A handler (or `__get_state`)
//      that finds its own inputs ill-typed returns a token (or state type)
//      holding an __mexception, which the walk raises the same way. The
//      attributes are gated on the walk being a constant expression, so an
//      ill-typed sender has no completion attributes rather than causing hard
//      errors.
//
//   5. Environment dependence. A sender whose completions depend on its
//      environment is *dependent*: asked how it completes without one, it
//      reports dependent_sender_error. The walk asks the children without an
//      environment when it has none, so a dependent child makes its parent
//      dependent. An algorithm whose own completions depend on the
//      environment opts in by defining
//        template <class _Sender> static constexpr bool __depends_on_env = true;
//
// A facade algorithm is a class that derives from `__facade::__defaults` and
// overrides the hooks it needs. Its senders are `__facade::__sender`s, products
// of a tag, the algorithm's data, and its children, and `__facade::__algorithm`
// maps the tag to the algorithm:
//
//   template <>
//   struct __facade::__algorithm<then_t> : __upon::__algo<then_t, set_value_t>
//   {};
//
//   auto sndr = __facade::__sender{then_t(), fn, child};

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __facade
  {
    ////////////////////////////////////////////////////////////////////////////////////////
    // The type that a hook names: the type itself, or the type in an __mtype, which is how
    // a hook names a type that it cannot give as a value
    template <class _Ty>
    struct __unwrap_type
    {
      using __t = _Ty;
    };

    template <class _Ty>
    struct __unwrap_type<__mtype<_Ty>>
    {
      using __t = _Ty;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // 1. Effects
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key>
    struct __starts
    {};

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class... _Items>
    struct __effects
    {
      __effects() = default;

      // A token for fewer effects converts to a token for more effects, so a control-flow
      // path that does less can return into a superset. Paths that do nothing return
      // `__effects<>{}`.
#  if STDEXEC_NVHPC()
      // (nvc++ skips a call that returns a token if the token is converted without being
      // read, and so loses the call's effects. This conversion reads it.)
      template <class... _Others>
        requires(__one_of<_Others, _Items...> && ...)
      constexpr __effects(__effects<_Others...> __other) noexcept
        : __nvhpc_read_(__other.__nvhpc_read_)
      {}

      bool __nvhpc_read_ = false;
#  else
      template <class... _Others>
        requires(__one_of<_Others, _Items...> && ...)
      constexpr __effects(__effects<_Others...>) noexcept
      {}
#  endif
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class... _Tokens>
    using __union_t = __mcall<__mconcat<__munique<__qq<__effects>>>, _Tokens...>;

    // A handler that finds its inputs ill-typed returns a token holding an __mexception
    // that describes the problem. The walk raises it (see part 4 above).
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Error>
      requires __merror<_Error>
    [[nodiscard]]
    constexpr auto __type_error(_Error) noexcept -> __effects<_Error>
    {
      return {};
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // 2. Completing. __send is the only way an algorithm completes: it sends a completion to
    // the algorithm's receiver, and returns the token that records the completion.
    // __start_nested is the only way it starts a child or nested operation. (Call these
    // with qualified names, so that argument-dependent lookup does not look in the
    // receiver's namespaces.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Receiver, class _Tag, class... _As>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto __send(_Receiver &__rcvr, _Tag, _As &&...__as) noexcept
      -> __effects<_Tag(_As...)>
    {
      _Tag()(static_cast<_Receiver &&>(__rcvr), static_cast<_As &&>(__as)...);
      return {};
    }

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key, class _OpState>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto __start_nested(_OpState &__op) noexcept -> __effects<__starts<_Key>>
    {
      STDEXEC::start(__op);
      return {};
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // 3. Combinators for control flow whose paths complete differently
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Then, class _Else>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __choose(bool __cond, _Then __then_fn, _Else __else_fn) noexcept
      -> __union_t<__call_result_t<_Then &>, __call_result_t<_Else &>>
    {
      if (__cond)
        return __then_fn();
      else
        return __else_fn();
    }

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Then>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __when(bool __cond, _Then __then_fn) noexcept -> __call_result_t<_Then &>
    {
      if (__cond)
        return __then_fn();
      else
        return {};
    }

    // Visits a variant that holds a value.
    //
    // (The result type is a template parameter so that it is not an expression in the
    // signature, which nvc++ cannot mangle into the name of the lambda below.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Fn,
              class... _Ts,
              class _Result = __union_t<__effects<>, __call_result_t<_Fn &, _Ts &>...>>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __visit(__variant<_Ts...> &__var, _Fn __fn) noexcept -> _Result
    {
      if constexpr (sizeof...(_Ts) == 0)
      {
        return _Result();
      }
      else
      {
        return STDEXEC::__visit([&]<class _Alt>(_Alt &__alt) noexcept -> _Result
                                { return __fn(__alt); },
                                __var);
      }
    }

    // The token of sending the result of __fn(__args...) as a value, and the
    // exception as an error if the invocation can throw
    template <class _Fn, class... _Args>
    consteval auto __set_value_from_token() noexcept
    {
      using __value_t = __effects<__single_value_sig_t<__invoke_result_t<_Fn, _Args...>>>;
      if constexpr (__nothrow_invocable<_Fn, _Args...>)
        return __value_t();
      else
        return __union_t<__value_t, __effects<set_error_t(std::exception_ptr)>>();
    }

    // The type error that algorithms report when a completion's arguments, which they must
    // store, cannot be decay-copied
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _AlgoTag, class... _Args>
    [[nodiscard]]
    consteval auto __decay_copy_error() noexcept
    {
      // The first argument that cannot be decay-copied
      using __arg_t =
        __mcall<__mfind_if<__mcompose_q<__mnot, __decay_copyable_t>, __q<__mfront>>, _Args...>;
      return __facade::__type_error(__mexception<_WHAT_(_TYPE_IS_NOT_DECAY_COPYABLE_),
                                                 _WHERE_(_IN_ALGORITHM_, _AlgoTag),
                                                 _WITH_TYPE_<__arg_t>>());
    }

    // Invokes __fn(__args...) and sends the result as a value. If the invocation can
    // throw, sends the exception as an error. If __fn cannot be invoked with __args,
    // reports a type error on behalf of the algorithm _AlgoTag.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _AlgoTag, class _Receiver, class _Fn, class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __set_value_from(_Receiver &__rcvr, _Fn &&__fn, _Args &&...__args) noexcept
    {
      if constexpr (!__invocable<_Fn, _Args...>)
      {
        return __facade::__type_error(__callable_error_t<_AlgoTag, __decay_t<_Fn>, _Args...>());
      }
      else
      {
        STDEXEC::__set_value_from(static_cast<_Receiver &&>(__rcvr),
                                  static_cast<_Fn &&>(__fn),
                                  static_cast<_Args &&>(__args)...);
        return __facade::__set_value_from_token<_Fn, _Args...>();
      }
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    // 4. Provenance
    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __from_start
    {};  // synchronously within start()

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Key, class _Tag, class _Src>
    struct __via
    {};  // within nested op _Key's _Tag completion; _Key was started from _Src

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <std::size_t _Idx>
    struct __child
    {};  // the key of the _Idx-th child sender

    // (Not __event, which is a keyword of MSVC.)
    template <class _Sig, class _Src>
    struct __walk_event
    {};

    ////////////////////////////////////////////////////////////////////////////////////////
    // The senders of facade algorithms (defined at the end of this file)

    // The facade algorithm that gives the senders with tag _Tag their behavior.
    // An algorithm specializes it for its tag:
    //
    //   template <>
    //   struct __facade::__algorithm<then_t> : __upon::__algo<then_t, set_value_t>
    //   {};
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Tag>
    struct __algorithm;

    // A sender of a facade algorithm: a tag, the algorithm's data, and its children
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Tag, class _Data, class... _Child>
    struct __sender;

    ////////////////////////////////////////////////////////////////////////////////////////
    // Access to the parts of a sender. The hooks that run at compile time and during
    // attribute queries are given the sender, as a __sender const &.

    // The sender's data, with the sender's value category
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Sender>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __get_data(_Sender &&__sndr) noexcept -> decltype(auto)
    {
      return STDEXEC::__get<1>(static_cast<_Sender &&>(__sndr));
    }

    // The sender's _Idx-th child, with the sender's value category
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <std::size_t _Idx, class _Sender>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __get_child(_Sender &&__sndr) noexcept -> decltype(auto)
    {
      return STDEXEC::__get<_Idx + 2>(static_cast<_Sender &&>(__sndr));
    }

    // The facade sender with the parts of __sndr (a facade sender) and the tag _Tag. A sender
    // that is lowered to another can be built as that sender with a tag of its own, which
    // makes it a sender that a domain can customize, and lowered by re-tagging it. (See
    // __transfer_when_all and __transfer_just.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Tag, class _Sender>
    constexpr auto __retagged(_Sender &&__sndr)
    {
      return STDEXEC::__apply(
        []<class _Data, class... _Child>(__ignore, _Data &&__data, _Child &&...__child)
        {
          return __sender{_Tag(),
                          static_cast<_Data &&>(__data),
                          static_cast<_Child &&>(__child)...};
        },
        static_cast<_Sender &&>(__sndr));
    }

    // The I-th child of a sender, with the sender's value category. (Read from the sender's
    // member __child_t, not by applying __m_at to its template arguments, a metaprogram that
    // adds several types to each use.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Sender, std::size_t _Idx>
    using __nth_child_of_t =
      __copy_cvref_t<_Sender, typename __decay_t<_Sender>::template __child_t<_Idx>>;

    // The data of a sender, with the sender's value category
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Sender>
    using __data_of_t = __copy_cvref_t<_Sender, typename __decay_t<_Sender>::__data_t>;

    // The number of children of a sender
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Sender>
    inline constexpr std::size_t __nbr_children_v = __mapply<__msize, __decay_t<_Sender>>::value
                                                  - 2;

    // The environment that the operation keyed by _Key sees, given the
    // environment of the receiver connected to the algorithm's sender. See
    // __defaults::__env_for.
    template <class _Algo, class _Key, class _Sender, class _Env>
    using __env_t = decltype(_Algo::__env_for(_Key(),
                                              __declval<__decay_t<_Sender> const &>(),
                                              __declval<_Env>()));

    ////////////////////////////////////////////////////////////////////////////////////////
    // Sequence senders. An algorithm can name one of its children as the source of its
    // items (see __defaults::__item_source). When that child is a sequence sender, so is the
    // algorithm's sender: subscribing it subscribes the child, whose receiver forwards the
    // child's items to the algorithm's receiver. Connecting it connects the child, as any
    // other.

    // The key of the child whose items are the sender's, or void (see __defaults::__item_source)
    template <class _Algo, class _Sender>
    using __item_source_t = typename _Algo::template __item_source<__decay_t<_Sender>>;

    // Is the sender with these children, whose algorithm names the item source pointed to, a
    // sequence sender? (A function, not a variable template, whose specializations would be
    // symbols with the children in their names.)
    template <class... _Child>
    consteval auto __is_sequence_sender(void const *) noexcept -> bool
    {
      return false;
    }

    template <class... _Child, std::size_t _Idx>
    consteval auto __is_sequence_sender(__child<_Idx> const *) noexcept -> bool
    {
      return __sequence_sender_tagged<__m_at_c<_Idx, _Child...>>;
    }

    // The items of a sequence sender are the items of its item source, in the environment
    // that the item source sees: __mlist<_ItemSender, _ItemEnv...>, as exec::get_item_types
    // expects. (Without an environment, the item source sees the one the algorithm gives it
    // in an empty environment.)
    template <class _Algo, class _Sender, class _ItemSource, class... _Env>
    struct __item_types_from
    {};

    template <class _Algo, class _Sender, std::size_t _Idx, class... _Env>
    struct __item_types_from<_Algo, _Sender, __child<_Idx>, _Env...>
    {
      using __t =
        __mlist<__nth_child_of_t<_Sender, _Idx>, __env_t<_Algo, __child<_Idx>, _Sender, _Env>...>;
    };

    template <class _Algo, class _Sender, std::size_t _Idx>
    struct __item_types_from<_Algo, _Sender, __child<_Idx>>
    {
      using __t =
        __mlist<__nth_child_of_t<_Sender, _Idx>, __env_t<_Algo, __child<_Idx>, _Sender, env<>>>;
    };

    // The type of a state's receiver, computed only when _Item is known. (A receiver's state
    // can be incomplete when its type is: the state can hold an operation connected to it.)
    template <class _State, class _Item>
    struct __state_receiver
    {
      using __t = decltype(_State::__rcvr_);
    };

    // Does a receiver of the facade forward _Item to _Receiver, the receiver of the algorithm's
    // operation? It does if _Receiver accepts it, which it does when the algorithm's sender is
    // subscribed. (Only the item source is subscribed then, so only it sends items.)
    template <class _Receiver, class _Item>
    concept __forwards_items = requires(_Receiver &__rcvr, _Item &&__item) {
      __rcvr.set_next(static_cast<_Item &&>(__item));
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // The facade's runtime machinery: receivers and operation state. The tokens
    // that the handlers return are for the walk; at run time, the receivers and
    // the operation state discard them.

    // The receiver of a child operation that the facade connects. (Child
    // operations are stored outside the algorithm's state.) _Env is the type of
    // its environment, __env_t<_Algo, _Key, _Sender, env_of_t<_Receiver>> for
    // the algorithm's sender and receiver.
    //
    // (The receiver's type does not name the algorithm's sender. Each child
    // operation names its receiver, and the algorithm's operation names every
    // child operation, so its type, and the debug info that names it, would
    // grow with the square of the sender's size.)
    template <class _Algo, class _Key, class _State, class _Env>
    struct __receiver : _Algo::__receiver_base
    {
      using receiver_concept   = receiver_tag;
      using __child_receiver_t = void;  // see STDEXEC::__child_receiver

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr explicit __receiver(_State *__state) noexcept
        : __state_(__state)
      {}

      template <class... _As>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_value(_As &&...__as) noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_value_t(), static_cast<_As &&>(__as)...);
      }

      template <class _Error>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_error(_Error &&__err) noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_error_t(), static_cast<_Error &&>(__err));
      }

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_stopped() noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_stopped_t());
      }

      // Forwards the items of the item source of a subscribed sender (see __forwards_items)
      template <class _Item, class _Rcvr = __state_receiver<_State, _Item>::__t>
        requires __forwards_items<_Rcvr, _Item>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto set_next(_Item &&__item) & noexcept(
        noexcept(__declval<_Rcvr &>().set_next(__declval<_Item>())))
        -> decltype(__declval<_Rcvr &>().set_next(__declval<_Item>()))
      {
        return __state_->__rcvr_.set_next(static_cast<_Item &&>(__item));
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto get_env() const noexcept -> _Env
      {
        static_assert(__same_as<decltype(_Algo::__get_env(_Key(), *__state_)), _Env>,
                      "__get_env and __env_for must produce environments of the same type");
        return _Algo::__get_env(_Key(), *__state_);
      }

      _State *__state_;
    };

    // The receiver of an operation that an algorithm stores in its state: a
    // nested operation, or a child that the algorithm connects itself. Its
    // member functions are not constexpr, so that their bodies are instantiated
    // only once the state is complete. (The type of an operation can depend on
    // its receiver's member functions, for example when the operation
    // type-erases its receiver, and the state is incomplete while that type is
    // computed. Clang instantiates constexpr functions eagerly.)
    template <class _Algo, class _Key, class _State, class _Env>
    struct __nested_receiver : _Algo::__receiver_base
    {
      using receiver_concept   = receiver_tag;
      using __child_receiver_t = void;  // see STDEXEC::__child_receiver

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr explicit __nested_receiver(_State *__state) noexcept
        : __state_(__state)
      {}

      template <class... _As>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_value(_As &&...__as) noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_value_t(), static_cast<_As &&>(__as)...);
      }

      template <class _Error>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_error(_Error &&__err) noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_error_t(), static_cast<_Error &&>(__err));
      }

      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_stopped() noexcept
      {
        (void) _Algo::__complete(_Key(), *__state_, set_stopped_t());
      }

      // Forwards the items of the item source of a subscribed sender (see __forwards_items)
      template <class _Item, class _Rcvr = __state_receiver<_State, _Item>::__t>
        requires __forwards_items<_Rcvr, _Item>
      STDEXEC_ATTRIBUTE(always_inline)
      auto set_next(_Item &&__item) & noexcept(
        noexcept(__declval<_Rcvr &>().set_next(__declval<_Item>())))
        -> decltype(__declval<_Rcvr &>().set_next(__declval<_Item>()))
      {
        return __state_->__rcvr_.set_next(static_cast<_Item &&>(__item));
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      auto get_env() const noexcept -> _Env
      {
        static_assert(__same_as<decltype(_Algo::__get_env(_Key(), *__state_)), _Env>,
                      "__get_env and __env_for must produce environments of the same type");
        return _Algo::__get_env(_Key(), *__state_);
      }

      _State *__state_;
    };

    // The receiver of an operation that an algorithm stores in its state, when the state's
    // receiver is an archetype (as in the walk). Its completions do nothing, so analyzing an
    // algorithm does not instantiate the code that completes it. (That would instantiate
    // the code that completes the nested operation's receiver, which can be device code:
    // a CUDA stream operation completes a receiver that is not a stream receiver from a
    // kernel. And the algorithm's handlers need not be callable there.)
    template <class _Env>
    struct __nested_receiver_archetype : __receiver_archetype<_Env>
    {
      __nested_receiver_archetype() = default;

      template <class _State>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __nested_receiver_archetype(_State *) noexcept
      {}
    };

    // (A class keyed only by whether the state's receiver is an archetype, so that the debug
    // information, which names the class when it names the receiver's type by this alias, does
    // not repeat the types of the state, the sender and the receiver. The alias cannot name
    // the receiver's type directly: GCC would then look into _Algo where the alias is used,
    // and the algorithm can be incomplete there.)
    template <bool _IsArchetype>
    struct __receiver_of
    {
      template <class _Algo, class _Key, class _State, class _Sender, class _Receiver>
      using __f = __nested_receiver<_Algo,
                                    _Key,
                                    _State,
                                    __facade::__env_t<_Algo, _Key, _Sender, env_of_t<_Receiver>>>;
    };

    template <>
    struct __receiver_of<true>
    {
      template <class _Algo, class _Key, class _State, class _Sender, class _Receiver>
      using __f =
        __nested_receiver_archetype<__facade::__env_t<_Algo, _Key, _Sender, env_of_t<_Receiver>>>;
    };

    // The receiver to which an algorithm connects an operation it stores in its state
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Key, class _State, class _Sender, class _Receiver>
    using __receiver_t = __receiver_of<__std::derived_from<_Receiver, __receiver_archetype_base>>::
      template __f<_Algo, _Key, _State, _Sender, _Receiver>;

    template <std::size_t... _Is>
    auto __starts_all_impl(__indices<_Is...>) -> __effects<__starts<__child<_Is>>...>;

    template <std::size_t _Np>
    using __starts_all_t = decltype(__facade::__starts_all_impl(__make_indices<_Np>()));

    // The handle through which __start starts the children
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <std::size_t _Np, class _Ops>
    struct __children_ref
    {
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto __start_all() noexcept -> __starts_all_t<_Np>
      {
        if constexpr (!__same_as<_Ops, void>)  // _Ops is void during type-level analysis
        {
          STDEXEC::__apply([](auto &...__ops) noexcept { (STDEXEC::start(__ops), ...); }, *__ops_);
        }
        return {};
      }

      _Ops *__ops_;
    };

    // When an algorithm's sender is subscribed, the facade passes __subscribing_t to the
    // algorithm's __get_state, after the receiver. (The default __get_state ignores it.) An
    // algorithm that attaches its item source itself (see __connects_children) takes it, and
    // subscribes the item source (see __attach_child_t).
    //
    // An operation takes the subscription as a pack, _Subscribing..., which is empty unless
    // the sender is subscribed, so that a connected operation's types are the same as if
    // there were no subscribing. (Flags with the operation's types as arguments would be
    // variables, with symbols whose names repeat those types.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __subscribing_t
    {};

    // Does the algorithm name the type of its state with an alias, __state_type?
    template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
    concept __declares_state_type = requires {
      typename _Algo::template __state_type<_Sender, _Receiver, _Subscribing...>;
    };

    // (A class keyed only by whether the algorithm declares __state_type, so that the debug
    // information, which names the class when it names the state's type by this alias, does
    // not repeat the types of the sender and the receiver.)
    template <bool _Declared>
    struct __state_type_of
    {
      template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
      using __f = decltype(_Algo::__get_state(__declval<_Sender>(),
                                              __declval<_Receiver>(),
                                              _Subscribing()...));
    };

    template <>
    struct __state_type_of<true>
    {
      template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
      using __f = _Algo::template __state_type<_Sender, _Receiver, _Subscribing...>;
    };

    // The type of the algorithm's state: the one it declares with __state_type, or else the
    // type that __get_state returns. Naming the type of a call needs the function's exception
    // specification ([except.spec]), and its body if its return type is deduced. An algorithm
    // whose state holds operations declares __state_type, so that the walk, which needs only
    // the type, does not ask whether making the state can throw: that would name the
    // operations below the state.
    template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
    using __state_type_t =
      __state_type_of<__declares_state_type<_Algo, _Sender, _Receiver, _Subscribing...>>::
        template __f<_Algo, _Sender, _Receiver, _Subscribing...>;

    // Is the child keyed by __child<_Idx> subscribed? It is if it is the item source of the
    // sender, of type _Sender, and the sender is subscribed.
    template <class _Algo, class _Sender, std::size_t _Idx, class... _Subscribing>
    concept __subscribes_child = (sizeof...(_Subscribing) != 0)
                              && __same_as<__child<_Idx>, __item_source_t<_Algo, _Sender>>;

    // Is the child keyed by __child<_Idx> the item source, and a sequence sender? An algorithm
    // that attaches its children itself (see __connects_children) attaches such a child with
    // __attach_child_t, so that it is subscribed when the algorithm's sender is, and connects
    // the others. (Attaching passes the algorithm's state, whose type would then be repeated
    // in the debug information of each child's connection.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Sender, std::size_t _Idx>
    concept __attaches_child = __same_as<__child<_Idx>, __item_source_t<_Algo, _Sender>>
                            && __sequence_sender_tagged<__nth_child_of_t<_Sender, _Idx>>;

    // Is the child keyed by __child<_Idx> a sequence sender that is the item source, attached
    // in the walk (where its receiver, _Rcvr, is an archetype)? Then it is not attached: the
    // archetype cannot subscribe it, and connecting a sequence sender can be ill-formed. The
    // walk asks the child for its completions, not for its operation, so a placeholder stands
    // in for the operation, and attaching it is assumed to be potentially throwing.
    template <class _Algo, class _Sender, std::size_t _Idx, class _Rcvr>
    concept __walks_item_source = __std::derived_from<_Rcvr, __receiver_archetype_base>
                               && __attaches_child<_Algo, _Sender, _Idx>;

    struct __walk_op
    {
      using operation_state_concept = operation_state_tag;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void start() & noexcept {}
    };

    template <bool _Walking>
    struct __op_or_walk_op
    {
      template <template <class...> class _Fn, class... _Args>
      using __f = _Fn<_Args...>;
    };

    template <>
    struct __op_or_walk_op<true>
    {
      template <template <class...> class _Fn, class... _Args>
      using __f = __walk_op;
    };

    // Is the state's receiver an archetype, as it is in a walk? Then the state's operations
    // are never made or started (the state is for a walk, or for asking whether connecting
    // can throw), so it holds placeholders for them. (Whether making the state can throw is
    // still asked of the real operations.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Rcvr>
    concept __walking_receiver = __std::derived_from<_Rcvr, __receiver_archetype_base>;

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Rcvr, template <class...> class _Fn, class... _Args>
    using __state_op_t = __op_or_walk_op<__walking_receiver<_Rcvr>>::template __f<_Fn, _Args...>;

    // The type of the operations that a state holds, as _Lazy::__t (a class that names it, so
    // that it is not instantiated unless used), or _Walk if the state is for a walk.
    template <bool _Walking>
    struct __lazy_state_ops
    {
      template <class _Lazy, class _Walk>
      using __f = _Lazy::__t;
    };

    template <>
    struct __lazy_state_ops<true>
    {
      template <class _Lazy, class _Walk>
      using __f = _Walk;
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Rcvr, class _Lazy, class _Walk>
    using __lazy_state_ops_t =
      __lazy_state_ops<__walking_receiver<_Rcvr>>::template __f<_Lazy, _Walk>;

    struct __walk_attach_t
    {
      template <class _State, class _Child, class _Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_State &, _Child &&, _Rcvr &&) const noexcept(false) -> __walk_op
      {
        return {};
      }
    };

    // Connects the child keyed by __child<_Idx>, or, if _Subscribes, subscribes it. (A
    // specialization chooses, not constrained overloads: Clang 16 substitutes into a function
    // template's return type before it checks the constraints (see CWG 2369), so it would
    // compute the connecting overload's for an item source that can only be subscribed.)
    template <class _Algo, std::size_t _Idx, bool _Subscribes>
    struct __attach_child_fn
    {
      template <class _State, class _Child, class _Rcvr>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto operator()(_State &__state, _Child &&__child_sndr, _Rcvr &&__rcvr) const
        noexcept(noexcept(_Algo::__connect(__child<_Idx>(),
                                           __state,
                                           static_cast<_Child &&>(__child_sndr),
                                           static_cast<_Rcvr &&>(__rcvr))))
          -> decltype(_Algo::__connect(__child<_Idx>(),
                                       __state,
                                       static_cast<_Child &&>(__child_sndr),
                                       static_cast<_Rcvr &&>(__rcvr)))
      {
        return _Algo::__connect(__child<_Idx>(),
                                __state,
                                static_cast<_Child &&>(__child_sndr),
                                static_cast<_Rcvr &&>(__rcvr));
      }
    };

    template <class _Algo, std::size_t _Idx>
    struct __attach_child_fn<_Algo, _Idx, true>
    {
      template <class _State, class _Child, class _Rcvr>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_State &__state, _Child &&__child_sndr, _Rcvr &&__rcvr) const
        noexcept(noexcept(_Algo::__subscribe(__child<_Idx>(),
                                             __state,
                                             static_cast<_Child &&>(__child_sndr),
                                             static_cast<_Rcvr &&>(__rcvr))))
          -> decltype(_Algo::__subscribe(__child<_Idx>(),
                                         __state,
                                         static_cast<_Child &&>(__child_sndr),
                                         static_cast<_Rcvr &&>(__rcvr)))
      {
        return _Algo::__subscribe(__child<_Idx>(),
                                  __state,
                                  static_cast<_Child &&>(__child_sndr),
                                  static_cast<_Rcvr &&>(__rcvr));
      }
    };

    // The function object that attaches the child keyed by __child<_Idx> of a sender of type
    // _Sender to a receiver of type _Rcvr: it connects the child, or subscribes it (see
    // __subscribes_child), or, in the walk, neither (see __walks_item_source). An algorithm
    // that attaches its children itself uses it for its item source, so that the item source
    // is subscribed when the algorithm's sender is (see __subscribing_t and
    // __attaches_child).
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Sender, std::size_t _Idx, class _Rcvr, class... _Subscribing>
    using __attach_child_t = __if_c<
      __walks_item_source<_Algo, _Sender, _Idx, _Rcvr>,
      __walk_attach_t,
      __attach_child_fn<_Algo, _Idx, __subscribes_child<_Algo, _Sender, _Idx, _Subscribing...>>>;

    template <class _Attach, class _State, class _Child, class _Rcvr>
    using __attach_result_t =
      decltype(_Attach()(__declval<_State &>(), __declval<_Child>(), __declval<_Rcvr>()));

    template <class _Attach, class _State, class _Child, class _Rcvr>
    using __connect_result_t = connect_result_t<_Child, _Rcvr>;

    template <class _Attach, class _State, class _Child, class _Rcvr>
    concept __nothrow_attachable = requires(_State &__state, _Child &&__child, _Rcvr &&__rcvr) {
      {
        _Attach()(__state, static_cast<_Child &&>(__child), static_cast<_Rcvr &&>(__rcvr))
      } noexcept;
    };

    // The operation of the child keyed by __child<_Idx>, a sender of type _Child, that an
    // algorithm with a state of type _State attaches itself to a receiver of type _Rcvr:
    // attached with __attach_child_t or connected (see __attaches_child)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo,
              class _Sender,
              std::size_t _Idx,
              class _State,
              class _Child,
              class _Rcvr,
              class... _Subscribing>
    using __child_op_t =
      __minvoke_if_c<__attaches_child<_Algo, _Sender, _Idx>,
                     __q<__attach_result_t>,
                     __q<__connect_result_t>,
                     __attach_child_t<_Algo, _Sender, _Idx, _Rcvr, _Subscribing...>,
                     _State,
                     _Child,
                     _Rcvr>;

    // Can making that operation throw? (A disjunction, so that a sequence sender is not asked
    // whether it can be connected.)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo,
              class _Sender,
              std::size_t _Idx,
              class _State,
              class _Child,
              class _Rcvr,
              class... _Subscribing>
    concept __nothrow_attachable_child =
      (__attaches_child<_Algo, _Sender, _Idx>
       && __nothrow_attachable<__attach_child_t<_Algo, _Sender, _Idx, _Rcvr, _Subscribing...>,
                               _State,
                               _Child,
                               _Rcvr>)
      || (!__attaches_child<_Algo, _Sender, _Idx> && __nothrow_connectable<_Child, _Rcvr>);

    template <class _Algo, class _Sender, class _Receiver, class _Indices, class... _Subscribing>
    struct __children_ops;

    // The children's operations. When the algorithm's sender is subscribed, its item source
    // is subscribed, and the other children are connected.
    template <class _Algo, class _Sender, class _Receiver, std::size_t... _Is, class... _Subscribing>
    struct __children_ops<_Algo, _Sender, _Receiver, __indices<_Is...>, _Subscribing...>
    {
      using __state_t = __state_type_t<_Algo, _Sender, _Receiver, _Subscribing...>;

      template <std::size_t _Idx>
      using __receiver_t =
        __facade::__receiver<_Algo,
                             __child<_Idx>,
                             __state_t,
                             __facade::__env_t<_Algo, __child<_Idx>, _Sender, env_of_t<_Receiver>>>;

      template <std::size_t _Idx>
      using __attach_t =
        __attach_child_fn<_Algo, _Idx, __subscribes_child<_Algo, _Sender, _Idx, _Subscribing...>>;

      template <std::size_t _Idx>
      using __op_t = decltype(__attach_t<_Idx>()(__declval<__state_t &>(),
                                                 __declval<__nth_child_of_t<_Sender, _Idx>>(),
                                                 __declval<__receiver_t<_Idx>>()));

      using __t = __tuple<__op_t<_Is>...>;

      // (Not __nothrow, which the Windows SDK defines as a macro.)
      static constexpr bool __nothrow_v =
        (noexcept(__attach_t<_Is>()(__declval<__state_t &>(),
                                    __declval<__nth_child_of_t<_Sender, _Is>>(),
                                    __declval<__receiver_t<_Is>>()))
         && ...);

      // (Unless the sender is subscribed, the children are connected with __connect directly:
      // a function object's call operator would repeat the types of the state and the
      // receiver in the debug information.)
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto
      __connect(_Sender &&__sndr, __state_t &__state) noexcept(__nothrow_v) -> __t
      {
        if constexpr (sizeof...(_Subscribing) != 0)
        {
          return __t{__attach_t<_Is>()(__state,
                                       __facade::__get_child<_Is>(static_cast<_Sender &&>(__sndr)),
                                       __receiver_t<_Is>{&__state})...};
        }
        else
        {
          return __t{_Algo::__connect(__child<_Is>(),
                                      __state,
                                      __facade::__get_child<_Is>(static_cast<_Sender &&>(__sndr)),
                                      __receiver_t<_Is>{&__state})...};
        }
      }
    };

    template <class _Algo, class _Sender>
    inline constexpr std::size_t __connected_children_v = _Algo::__connects_children
                                                          ? __nbr_children_v<_Sender>
                                                          : 0;

    template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
    using __children_ops_t = __children_ops<_Algo,
                                            _Sender,
                                            _Receiver,
                                            __make_indices<__connected_children_v<_Algo, _Sender>>,
                                            _Subscribing...>;

    // The operation state of a facade sender, which is connected or, if it is a sequence
    // sender, subscribed (see __subscribing_t)
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Sender, class _Receiver, class... _Subscribing>
    struct __opstate
    {
      using operation_state_concept = operation_state_tag;
      using __state_t               = __state_type_t<_Algo, _Sender, _Receiver, _Subscribing...>;
      using __children_t            = __children_ops_t<_Algo, _Sender, _Receiver, _Subscribing...>;
      using __ops_t                 = __children_t::__t;

      static constexpr bool __nothrow_v = noexcept(_Algo::__get_state(__declval<_Sender>(),
                                                                      __declval<_Receiver>(),
                                                                      _Subscribing()...))
                                       && __children_t::__nothrow_v;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __opstate(_Sender &&__sndr, _Receiver __rcvr) noexcept(__nothrow_v)
        : __state_(_Algo::__get_state(static_cast<_Sender &&>(__sndr),
                                      static_cast<_Receiver &&>(__rcvr),
                                      _Subscribing()...))
        , __ops_(__children_t::__connect(static_cast<_Sender &&>(__sndr), __state_))
      {}

      STDEXEC_IMMOVABLE(__opstate);

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void start() & noexcept
      {
        __children_ref<__connected_children_v<_Algo, _Sender>, __ops_t> __children{&__ops_};
        (void) _Algo::__start(__state_, __children);
      }

      __state_t __state_;
      __ops_t   __ops_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // The walk: a consteval traversal from start() through the effect tokens that finds
    // every (signature, provenance) event. It evaluates each nested sender's
    // get_completion_signatures, so an ill-typed nested sender fails the walk the same way
    // it would fail any other algorithm: by throwing, or by returning the error.

    // The sender of the child or nested operation keyed by _Key
    template <class _Algo, class _Sender, class _Key, class... _Env>
    struct __nested_sender_of
    {
      using __t = __unwrap_type<
        decltype(_Algo::__nested_sender(_Key(),
                                        __declval<__decay_t<_Sender> const &>(),
                                        __declval<__decay_t<_Env> const &>()...))>::__t;
    };

    template <class _Algo, class _Sender, std::size_t _Idx, class... _Env>
    struct __nested_sender_of<_Algo, _Sender, __child<_Idx>, _Env...>
    {
      using __t = __nth_child_of_t<_Sender, _Idx>;
    };

    // Is the nested sender keyed by _Key known only by its type (because it is created at
    // run time), or can attribute queries inspect it?
    template <class _Algo, class _Sender, class _Key, class... _Env>
    inline constexpr bool __type_only_v =
      __is_instance_of<decltype(_Algo::__nested_sender(_Key(),
                                                       __declval<__decay_t<_Sender> const &>(),
                                                       __declval<__decay_t<_Env> const &>()...)),
                       __mtype>;

    template <class _Algo, class _Sender, std::size_t _Idx, class... _Env>
    inline constexpr bool __type_only_v<_Algo, _Sender, __child<_Idx>, _Env...> = false;

    // The receiver archetype of a walk that has no environment. Its environment is env<>,
    // but __completions_of_t recognizes it, and asks for completions without one.
    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __no_env_receiver : __receiver_archetype_base
    {
      [[nodiscard]]
      STDEXEC_ATTRIBUTE(host, device) constexpr auto get_env() const noexcept -> env<>
      {
        return {};
      }
    };

    template <class... _Env>
    struct __walk_receiver
    {
      using __t = __no_env_receiver;
    };

    template <class _Env>
    struct __walk_receiver<_Env>
    {
      using __t = __receiver_archetype<_Env>;
    };

    // The completion signatures of the child or nested operation keyed by _Key of a sender
    // connected to a receiver of type _Receiver, as the walk sees them. An algorithm whose
    // operation state depends on them must use this, so that its state agrees with the
    // walk even when the walk has no environment.
    template <class _Algo, class _Key, class _Sender, class _Receiver>
    struct __completions_of
    {
      using __env_t = env_of_t<_Receiver>;
      using __t     = __completion_signatures_of_t<
            typename __nested_sender_of<_Algo, _Sender, _Key, __env_t>::__t,
            __facade::__env_t<_Algo, _Key, _Sender, __env_t>>;
    };

    template <class _Algo, class _Key, class _Sender>
    struct __completions_of<_Algo, _Key, _Sender, __no_env_receiver>
    {
      using __t =
        __completion_signatures_of_t<typename __nested_sender_of<_Algo, _Sender, _Key>::__t>;
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Key, class _Sender, class _Receiver>
    using __completions_of_t = __completions_of<_Algo, _Key, _Sender, _Receiver>::__t;

    template <class _Algo, class _Sender, class... _Env>
    struct __ctx
    {
      using __algo_t = _Algo;
      using __sndr_t = _Sender;
      using __rcvr_t = __walk_receiver<_Env...>::__t;

      template <class _Key>
      using __nested_sender_t = __nested_sender_of<_Algo, _Sender, _Key, _Env...>::__t;

      // A nested sender is asked for its completions in its environment, or in none if
      // there is none, so that a dependent child makes the walk dependent.
      template <class _Key>
      static consteval auto __nested_completions()
      {
        return STDEXEC::get_completion_signatures<
          __nested_sender_t<_Key>,
          __facade::__env_t<_Algo, _Key, _Sender, _Env>...>();
      }

      template <class _Key>
      static constexpr bool __nested_ok =
        sender_in<__nested_sender_t<_Key>, __facade::__env_t<_Algo, _Key, _Sender, _Env>...>;

      // The algorithm's own completions depend on an environment, and there is none.
      static constexpr bool __dependent = sizeof...(_Env) == 0
                                       && _Algo::template __depends_on_env<__decay_t<_Sender>>;
    };

    // Computed only once the children are known to type-check, because the state type may
    // depend on their completions, and once the algorithm has any environment it needs.
    template <class _Ctx>
    using __state_of_t =
      __state_type_t<typename _Ctx::__algo_t, typename _Ctx::__sndr_t, typename _Ctx::__rcvr_t>;

    // Concatenates lists of events, or returns the first error. (Errors are returned only
    // when constexpr exceptions are not available; otherwise they are thrown.) The walk
    // passes a list of events as a value of type __mtype<__mlist<_Events...>>.
    template <class... _Results>
    consteval auto __concat_events(_Results...) noexcept
    {
      using __error_t = __mfind_error<_Results...>;
      if constexpr (__merror<__error_t>)
        return __error_t();
      else
        return __mtype<__mcall<__mconcat<>, __t<_Results>...>>();
    }

    template <class _Ctx, class _Src, class... _Items>
    consteval auto __token_events(__effects<_Items...>);

    template <class _Ctx, class _Key, class _Src, class _Tag, class... _Args>
    consteval auto __sig_events(__mtype<_Tag(_Args...)>)
    {
      using __token_t = decltype(_Ctx::__algo_t::__complete(_Key(),
                                                            __declval<__state_of_t<_Ctx> &>(),
                                                            _Tag(),
                                                            __declval<_Args>()...));
      return __facade::__token_events<_Ctx, __via<_Key, _Tag, _Src>>(__token_t());
    }

    template <class _Ctx, class _Key, class _Src, class... _Sigs>
    consteval auto __sigs_events(completion_signatures<_Sigs...>)
    {
      return __facade::__concat_events(
        __facade::__sig_events<_Ctx, _Key, _Src>(__mtype<_Sigs>())...);
    }

    template <class _Ctx, class _Key, class _Src>
    consteval auto __nested_events()
    {
      auto __sigs = _Ctx::template __nested_completions<_Key>();
      STDEXEC_IF_OK(__sigs)
      {
        return __facade::__sigs_events<_Ctx, _Key, _Src>(__sigs);
      }
    }

    template <class _Ctx, class _Src, class _Item>
    consteval auto __item_events(__mtype<_Item>)
    {
      return __mtype<__mlist<__walk_event<_Item, _Src>>>();
    }

    // Is the operation keyed by _Key on the path to _Src? (Then starting it again from _Src
    // is a loop.)
    template <class _Key, class _Src>
    inline constexpr bool __on_path_v = false;

    template <class _Key, class _Tag, class _Src>
    inline constexpr bool __on_path_v<_Key, __via<_Key, _Tag, _Src>> = true;

    template <class _Key, class _Key2, class _Tag, class _Src>
    inline constexpr bool __on_path_v<_Key, __via<_Key2, _Tag, _Src>> = __on_path_v<_Key, _Src>;

    // (See "Attribute derivation" below.)
    template <class _Ctx, class _Src, class... _QEnv>
    struct __resolve;

    // The place where the operations that are started from _Src start, as the attributes
    // see it in the walk's environment: a behavior, a domain, and a scheduler expression
    template <class _Ctx, class _Src>
    struct __place_of;

    template <class _Algo, class _Sender, class... _Env, class _Src>
    struct __place_of<__ctx<_Algo, _Sender, _Env...>, _Src>
    {
      using __resolved_t = __resolve<__ctx<_Algo, _Sender, _Env...>, _Src, _Env...>;
      using __t          = __mlist<__completion_behavior::__constant_t<__resolved_t::__behavior>,
                                   typename __resolved_t::__domain_t,
                                   typename __resolved_t::__sched_t>;
    };

    // Was the operation keyed by _Key started in the place _Place earlier on the path to
    // _Src?
    template <class _Ctx, class _Key, class _Place, class _Src>
    inline constexpr bool __started_in_v = false;

    template <class _Ctx, class _Key, class _Place, class _Key2, class _Tag, class _Src>
    inline constexpr bool __started_in_v<_Ctx, _Key, _Place, __via<_Key2, _Tag, _Src>> =
      __started_in_v<_Ctx, _Key, _Place, _Src>;

    template <class _Ctx, class _Key, class _Place, class _Tag, class _Src>
    inline constexpr bool __started_in_v<_Ctx, _Key, _Place, __via<_Key, _Tag, _Src>> =
      __same_as<_Place, __t<__place_of<_Ctx, _Src>>> || __started_in_v<_Ctx, _Key, _Place, _Src>;

    // A nested operation that is started again from one of its own completions (directly,
    // or through other operations) is a loop. An iteration starts where the one before it
    // ended, which can be somewhere other than where the first one started, so the
    // iterations can complete in different places. But the attributes of a place are few (a
    // behavior, a domain, and a scheduler expression), and those of the completions of an
    // operation depend only on those of the place where it starts (see __resolve). So the
    // walk follows the loop until an iteration starts in a place where an earlier iteration
    // on the same path started. From then on, the iterations complete as earlier ones did, so
    // starting the operation there adds no events.
    template <class _Ctx, class _Src, class _Key>
    consteval auto __item_events(__mtype<__starts<_Key>>)
    {
      if constexpr (!__on_path_v<_Key, _Src>)
        return __facade::__nested_events<_Ctx, _Key, _Src>();  // NOLINT(bugprone-branch-clone)
      else if constexpr (__started_in_v<_Ctx, _Key, __t<__place_of<_Ctx, _Src>>, _Src>)
        return __mtype<__mlist<>>();
      else
        return __facade::__nested_events<_Ctx, _Key, _Src>();
    }

    template <class _Ctx, class _Src, class... _Items>
    consteval auto __token_events(__effects<_Items...>)
    {
      using __error_t = __mfind_error<_Items...>;
      if constexpr (__merror<__error_t>)  // a handler reported a type error
        return STDEXEC::__throw_compile_time_error_r<__mtype<__mlist<>>>(__error_t());
      else
        return __facade::__concat_events(__facade::__item_events<_Ctx, _Src>(__mtype<_Items>())...);
    }

    template <class _Sigs>
    consteval auto __error_or_nothing(_Sigs __sigs)
    {
      if constexpr (__merror<_Sigs>)
        return __sigs;
      else
        return __mtype<__mlist<>>();
    }

    template <class _Ctx, std::size_t... _Is>
    consteval auto __walk_impl(__indices<_Is...>)
    {
      if constexpr (!(_Ctx::template __nested_ok<__child<_Is>> && ...))
      {
        // Evaluating an ill-typed child's get_completion_signatures throws its error (or
        // returns it), which becomes this sender's error.
        return __facade::__concat_events(
          __facade::__error_or_nothing(_Ctx::template __nested_completions<__child<_Is>>())...);
      }
      else if constexpr (_Ctx::__dependent)
      {
        // Checked after the children, so that a child's type error is still reported when
        // the sender is constructed.
        return STDEXEC::__throw_dependent_sender_error_r<__mtype<__mlist<>>,
                                                         typename _Ctx::__sndr_t>();
      }
      else if constexpr (__merror<__state_of_t<_Ctx>>)  // __get_state reported a type error
      {
        return STDEXEC::__throw_compile_time_error_r<__mtype<__mlist<>>>(__state_of_t<_Ctx>());
      }
      else
      {
        using __children_t =
          __children_ref<__connected_children_v<typename _Ctx::__algo_t, typename _Ctx::__sndr_t>,
                         void>;
        using __start_token_t = decltype(_Ctx::__algo_t::__start(__declval<__state_of_t<_Ctx> &>(),
                                                                 __declval<__children_t &>()));
        return __facade::__token_events<_Ctx, __from_start>(__start_token_t());
      }
    }

    template <class _Algo, class _Sender, class... _Env>
    consteval auto __walk()
    {
      return __facade::__walk_impl<__ctx<_Algo, _Sender, _Env...>>(
        __make_indices<__nbr_children_v<_Sender>>());
    }

    consteval auto __is_ok(auto __result) noexcept -> bool
    {
      return !__merror<decltype(__result)>;
    }

    // Did the walk succeed? This is false, rather than a hard error, when the walk throws.
    // (The same technique as __constant_completion_signatures.)
    template <class _Algo, class _Sender, class... _Env>
    concept __walk_ok = requires {
      typename __mbool<__facade::__is_ok(__facade::__walk<_Algo, _Sender, _Env...>())>;
    } && __facade::__is_ok(__facade::__walk<_Algo, _Sender, _Env...>());

    template <class _Algo, class _Sender, class... _Env>
    using __events_t = __t<decltype(__facade::__walk<_Algo, _Sender, _Env...>())>;

    template <class _Events>
    struct __sigs_of;

    template <class... _Sigs, class... _Srcs>
    struct __sigs_of<__mlist<__walk_event<_Sigs, _Srcs>...>>
    {
      using __t = __mcall<__munique<__qq<completion_signatures>>, _Sigs...>;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Attribute derivation: resolve each provenance to a (behavior, domain, scheduler
    // expression) triple, then join over the events of a channel.
    using __behavior_t = __completion_behavior::__behavior;

    // NOT consteval: Clang 16 rejects the calls to __seq and __join in the static data
    // member initializers below ("cannot take address of consteval function").

    // Run one thing, then (from its completion) another: may complete inline only if
    // both may; may be asynchronous or non-affine if either may.
    constexpr auto __seq(__behavior_t __first, __behavior_t __second) noexcept -> __behavior_t
    {
      constexpr auto __inline_bit = std::uint8_t(__completion_behavior::__inline_);
      auto const     __a          = static_cast<std::uint8_t>(__first);
      auto const     __b          = static_cast<std::uint8_t>(__second);
      return static_cast<__behavior_t>(((__a | __b) & ~__inline_bit) | (__a & __b & __inline_bit));
    }

    // One of several things: the union of the possibilities.
    template <class... _Bs>
    constexpr auto __join(_Bs... __bs) noexcept -> __behavior_t
    {
      return static_cast<__behavior_t>((std::uint8_t(0) | ... | static_cast<std::uint8_t>(__bs)));
    }

    // Domains are compile-time information, so a channel whose completions can happen in
    // several domains reports all of them.
    template <class _Domain>
    struct __leaf_domains
    {
      using __t = __mlist<_Domain>;
    };

    template <class... _Domains>
    struct __leaf_domains<indeterminate_domain<_Domains...>>
    {
      using __t = __mcall<__mconcat<>, typename __leaf_domains<_Domains>::__t...>;
    };

    template <class _Domains>
    struct __domain_of_list
    {
      using __t = void;  // no domains: unknown
    };

    template <class _Domain>
    struct __domain_of_list<__mlist<_Domain>>
    {
      using __t = _Domain;
    };

    // Several domains: their common type, if they have one (when one is a base of the
    // others), and otherwise all of them.
    template <class... _Domains>
    consteval auto __domain_of_several() noexcept
    {
      if constexpr (__minvocable_q<std::common_type_t, _Domains...>)
        return __mtype<std::common_type_t<_Domains...>>();
      else
        return __mtype<indeterminate_domain<_Domains...>>();
    }

    template <class _D0, class _D1, class... _Ds>
    struct __domain_of_list<__mlist<_D0, _D1, _Ds...>>
    {
      using __t = decltype(__facade::__domain_of_several<_D0, _D1, _Ds...>())::__t;
    };

    template <class... _Domains>
    using __domain_join_t = __domain_of_list<
      __mcall<__mconcat<__munique<>>, typename __leaf_domains<_Domains>::__t...>>::__t;

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Sender>
    struct __attrs;

    // The facade sender that a sender is (it may be of a type derived from one)
    template <class _Tag, class _Data, class... _Child>
    auto __as_facade_sender(__sender<_Tag, _Data, _Child...> const &)
      -> __sender<_Tag, _Data, _Child...>;

    template <class _Sender>
    struct __facade_sender_of
    {
      using __t = void;
    };

    template <class _Sender>
      requires requires { __facade::__as_facade_sender(__declval<__decay_t<_Sender> const &>()); }
    struct __facade_sender_of<_Sender>
    {
      using __t = decltype(__facade::__as_facade_sender(__declval<__decay_t<_Sender> const &>()));
    };

    template <class _Sender>
    using __facade_sender_t = __facade_sender_of<_Sender>::__t;

    // A domain may customize a facade sender by giving its data and children to another
    // facade algorithm (that is, by transforming __sender<Tag, Data, Child...> into
    // __sender<Tag2, Data, Child...>). Then the sender's operation is that algorithm's, and
    // so are its completions and attributes. And because the parts are the same, the
    // attributes can be computed from the sender as it is, without transforming it.
    template <class _Sender, class _Transformed>
    struct __retag
    {
      using __t = void;
    };

    template <class _Tag, class _Tag2, class _Data, class... _Child>
      requires(!__same_as<_Tag, _Tag2>)
    struct __retag<__sender<_Tag, _Data, _Child...>, __sender<_Tag2, _Data, _Child...>>
    {
      using __t = __algorithm<_Tag2>;
    };

    // A domain that customizes facade algorithms by re-tagging their senders says so, with a
    // member `static constexpr bool __retags_facade_senders = true;`. Then the facade sees
    // through its customizations: in an environment in which such a domain customizes a
    // sender by re-tagging it, the sender's completion attributes are the re-tagged
    // sender's (see __attrs), as are its completion signatures (which get_completion_signatures
    // computes for the customized sender). Seeing through a customization means asking the
    // domain how it would customize the sender, so the facade asks only the domains that
    // say they re-tag: a domain that decides how to customize a sender by asking the sender
    // for its own attributes (as task_scheduler_domain does for bulk) would be asked again.
    // And asking every domain would cost every attribute query a transformation.
    template <class _Domain>
    concept __retagging_domain = requires { requires bool(_Domain::__retags_facade_senders); };

    // The algorithm of the sender as the domain customizes it in the environment, if that
    // is a re-tagging by a domain that says it re-tags; otherwise void
    template <class _Sender, class... _Env>
    struct __customized_algo
    {
      using __t = void;
    };

    template <class _Sender, class _Env>
      requires(!__same_as<__facade_sender_t<_Sender>, void>)
           && __retagging_domain<
                __call_result_t<get_completion_domain_t<>, env_of_t<_Sender>, _Env const &>>
           && __callable<transform_sender_t, __decay_t<_Sender>, _Env const &>
    struct __customized_algo<_Sender, _Env>
    {
      using __t =
        __retag<__facade_sender_t<_Sender>,
                __decay_t<transform_sender_result_t<__decay_t<_Sender>, _Env const &>>>::__t;
    };

    template <class _Sender, class... _Env>
    using __customized_algo_t = __customized_algo<_Sender, _Env...>::__t;

    // The attributes of the child or nested operation keyed by _Key: those of its sender,
    // as its domain customizes it if that is a re-tagging, unless the algorithm says
    // otherwise with a __nested_attrs hook
    template <class _Algo, class _Key, class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __nested_attrs_of(_Key           __key,
                                     _Sender const &__sndr,
                                     _Env const &...__env) noexcept -> decltype(auto)
    {
      if constexpr (requires { _Algo::__nested_attrs(__key, __sndr, __env...); })
      {
        return _Algo::__nested_attrs(__key, __sndr, __env...);
      }
      else
      {
        using __nested_t = decltype(_Algo::__nested_sender(__key, __sndr, __env...));
        if constexpr (sizeof...(_Env) == 1 && __same_as<__nested_t, __nested_t &>)
        {
          using __algo2_t =
            __customized_algo_t<__nested_t,
                                __facade::__env_t<_Algo, _Key, _Sender, _Env const &>...>;
          if constexpr (!__same_as<__algo2_t, void>)
          {
            using __base_t = __facade_sender_t<__nested_t>;
            return __attrs<__algo2_t, __base_t>{_Algo::__nested_sender(__key, __sndr, __env...)};
          }
          else
          {
            return STDEXEC::get_env(_Algo::__nested_sender(__key, __sndr, __env...));
          }
        }
        else
        {
          return STDEXEC::get_env(_Algo::__nested_sender(__key, __sndr, __env...));
        }
      }
    }

    // Scheduler expressions, evaluated when the attributes are queried
    struct __start_scheduler
    {
      template <class _Algo, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __eval(_Sender const &, _Env const &__env) noexcept
      {
        return STDEXEC::get_start_scheduler(__env);
      }
    };

    template <class _Key, class _Tag>
    struct __completion_scheduler_of
    {
      template <class _Algo, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __eval(_Sender const &__sndr, _Env const &...__env) noexcept
      {
        return STDEXEC::get_completion_scheduler<_Tag>(__facade::__nested_attrs_of<_Algo>(_Key(),
                                                                                          __sndr,
                                                                                          __env...),
                                                       _Algo::__env_for(_Key(), __sndr, __env)...);
      }
    };

    // Attributes that report the same completion scheduler for every kind of completion that
    // they report one for can say so, with a static data member:
    //
    //   static constexpr bool __one_completion_scheduler = true;
    //
    // (A schedule sender's attributes, which report the scheduler on which its operation
    // completes, however it completes, can.) Only the type of the attributes says so, so that
    // attributes that join them with others, which may report other schedulers for some
    // kinds of completion, do not.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Attrs>
    concept __reports_one_completion_scheduler = requires {
      requires bool(_Attrs::__one_completion_scheduler);
    };

    // Where the _Tag completions of a sender with the attributes _Attrs get their completion
    // scheduler from, in the environment _Env: the scheduler expression that its
    // get_completion_scheduler<_Tag> query evaluates, if the sender is a facade sender (see
    // __attrs::__sched_place_t); the attributes, if they report one completion scheduler
    // and report one for _Tag; and otherwise void. Completions that get their scheduler from
    // the same place get the same scheduler.
    template <class _Attrs, class _Tag, class... _Env>
    struct __sched_place_of
    {
      using __t = void;
    };

    template <class _Algo, class _Sender, class _Tag, class... _Env>
    struct __sched_place_of<__attrs<_Algo, _Sender>, _Tag, _Env...>
    {
      using __t = __attrs<_Algo, _Sender>::template __sched_place_t<_Tag, __decay_t<_Env>...>;
    };

    template <class _Attrs, class _Tag, class... _Env>
      requires __reports_one_completion_scheduler<_Attrs>
            && __callable<typename get_completion_scheduler_t<_Tag>::__read_query_t,
                          _Attrs const &,
                          _Env...>
    struct __sched_place_of<_Attrs, _Tag, _Env...>
    {
      using __t = _Attrs;
    };

    template <class _Ctx, class _Src, class... _QEnv>
    struct __resolve;

    template <class _Ctx, class... _QEnv>
    struct __resolve<_Ctx, __from_start, _QEnv...>
    {
      static constexpr __behavior_t __behavior = __behavior_t::__inline_completion;
      static constexpr bool         __at_start = true;

      static constexpr bool __has_start_sched = sizeof...(_QEnv) == 1
                                             && __callable<get_start_scheduler_t, _QEnv const &...>;

      // Agrees with the start scheduler, if there is one (see __check_domain).
      static consteval auto __get_domain() noexcept
      {
        if constexpr (__has_start_sched)
        {
          using __sch_t = __call_result_t<get_start_scheduler_t, _QEnv const &...>;
          return __mtype<
            __call_result_t<get_completion_domain_t<set_value_t>, __sch_t, _QEnv const &...>>();
        }
        else if constexpr (sizeof...(_QEnv) == 1 && __callable<get_domain_t, _QEnv const &...>)
          return __mtype<__call_result_t<get_domain_t, _QEnv const &...>>();
        else if constexpr (sizeof...(_QEnv) == 1)
          return __mtype<default_domain>();
        else
          return __mtype<void>();
      }

      static consteval auto __get_sched() noexcept
      {
        if constexpr (__has_start_sched)
          return __mtype<__start_scheduler>();
        else
          return __mtype<void>();
      }

      using __domain_t = decltype(__get_domain())::__t;
      using __sched_t  = decltype(__get_sched())::__t;
    };

    template <class _Ctx, class _Key, class _Tag, class _Src, class... _QEnv>
    struct __resolve<_Ctx, __via<_Key, _Tag, _Src>, _QEnv...>
    {
      using __algo_t = _Ctx::__algo_t;
      using __sndr_t = _Ctx::__sndr_t;
      using __up_t   = __resolve<_Ctx, _Src, _QEnv...>;
      // A nested sender that is created at run time has only the attributes of its type.
      using __attrs_t = __if_c<__type_only_v<__algo_t, __sndr_t, _Key, _QEnv...>,
                               env_of_t<typename _Ctx::template __nested_sender_t<_Key>>,
                               __decay_t<decltype(__facade::__nested_attrs_of<__algo_t>(
                                 _Key(),
                                 __declval<__decay_t<__sndr_t> const &>(),
                                 __declval<_QEnv const &>()...))>>;

      // (The nested operation's environment is computed from _QEnv, as the walk computes it
      // and as connect does, not from _QEnv const &. An algorithm that joins the environment
      // with another one, as starts_on does, gives a reference a different type, and the
      // nested sender would be walked again in that environment.)
      using __own_behavior_t =
        __call_result_t<__get_completion_behavior_t<_Tag>,
                        __attrs_t const &,
                        __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>;
      static constexpr bool __affine = __completion_behavior::__is_affine(__own_behavior_t());
      static constexpr __behavior_t __behavior = __facade::__seq(__up_t::__behavior,
                                                                 __own_behavior_t::value);

      static constexpr bool __has_own_domain =
        __callable<typename get_completion_domain_t<_Tag>::__read_query_t,
                   __attrs_t const &,
                   __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>;
      static constexpr bool __has_own_sched =
        __callable<typename get_completion_scheduler_t<_Tag>::__read_query_t,
                   __attrs_t const &,
                   __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>;

      // A nested sender created at run time cannot be asked for its scheduler.
      static constexpr bool __type_only = __type_only_v<__algo_t, __sndr_t, _Key, _QEnv...>;
      static constexpr bool __own_sched = __has_own_sched && !__type_only;

      static constexpr bool __has_domain =
        __callable<get_completion_domain_t<_Tag>,
                   __attrs_t const &,
                   __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>;

      // The nested operation's dispatch domain: the domain in which its customizations are
      // found, which is the domain that implements it
      static constexpr bool __has_dispatch_domain =
        sizeof...(_QEnv) == 1
        && __callable<get_completion_domain_t<>,
                      __attrs_t const &,
                      __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>;

      // If the nested operation completes where it started, it completes on the scheduler
      // on which it was started (at _Src), when that is known. Then its scheduler and domain
      // are those of _Src. (So completions that started in the same place agree, even when
      // they come from different operations.)
      static constexpr bool __at_src = __affine && !__same_as<typename __up_t::__sched_t, void>;

      // Does the nested operation complete where the operation started?
      static constexpr bool __at_start = __affine && __up_t::__at_start;

      // Does the environment of the nested operation say where it starts? (An algorithm must
      // not give a nested operation the receiver's start scheduler if it does not start it
      // there.)
      static constexpr bool __env_says_where =
        sizeof...(_QEnv) != 0
        && (__callable<get_start_scheduler_t, __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>>
            && ...);

      // Otherwise, it completes where its own scheduler says, if it has one; or, if it
      // completes where it started, where it was started. But an operation that completes
      // where it starts, in an environment that does not say where, can name a scheduler
      // only relative to wherever it started: inline_scheduler names itself, which the one
      // who asks takes to mean its own start scheduler. That is true only of an operation
      // that was started where the operation started. If it was started somewhere else,
      // nothing says where it completes.
      static constexpr bool __believe_own_sched = __own_sched
                                               && (!__affine || __env_says_where
                                                   || __up_t::__at_start);

      static consteval auto __get_domain() noexcept
      {
        if constexpr (__at_src)
          return __mtype<typename __up_t::__domain_t>();
        else if constexpr (__believe_own_sched && __has_domain)
          return __mtype<__call_result_t<get_completion_domain_t<_Tag>,
                                         __attrs_t const &,
                                         __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>>();
        else if constexpr (__affine)
          return __mtype<typename __up_t::__domain_t>();
        else if constexpr ((__has_own_domain || __has_own_sched) && __has_domain)
          return __mtype<__call_result_t<get_completion_domain_t<_Tag>,
                                         __attrs_t const &,
                                         __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>>();
        // Otherwise, nothing says where the nested operation completes this way. It completes
        // as the domain that implements it completes it, which can be in ways that its
        // sender's attributes do not mention. (A CUDA stream operation adds error completions,
        // for example, which happen on the stream.) So it completes in that domain.
        else if constexpr (__has_dispatch_domain)
          return __mtype<__call_result_t<get_completion_domain_t<>,
                                         __attrs_t const &,
                                         __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>>();
        else if constexpr (sizeof...(_QEnv) == 1)
          return __mtype<default_domain>();
        else
          return __mtype<void>();
      }

      // Where the nested operation's _Tag2 completions get their scheduler from (see
      // __sched_place_of)
      template <class _Tag2>
      using __sched_place_t =
        __sched_place_of<__attrs_t,
                         _Tag2,
                         __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>::__t;

      // The first of the nested operation's kinds of completion whose completions get their
      // scheduler from the same place as its _Tag completions do. Their scheduler names the
      // scheduler of its _Tag completions, so that completions of different kinds that the
      // nested operation completes on the same scheduler are seen to complete on the same
      // scheduler. (The value and error completions of then(schedule(sch), fn) do, for
      // example.)
      static consteval auto __get_sched_tag() noexcept
      {
        if constexpr (__same_as<_Tag, set_value_t>)
          return __mtype<_Tag>();
        else if constexpr (__same_as<__sched_place_t<_Tag>, void>)
          return __mtype<_Tag>();
        else if constexpr (__same_as<__sched_place_t<_Tag>, __sched_place_t<set_value_t>>)
          return __mtype<set_value_t>();
        else if constexpr (__same_as<_Tag, set_stopped_t>
                           && __same_as<__sched_place_t<_Tag>, __sched_place_t<set_error_t>>)
          return __mtype<set_error_t>();
        else
          return __mtype<_Tag>();
      }

      static consteval auto __get_sched() noexcept
      {
        if constexpr (__at_src)
          return __mtype<typename __up_t::__sched_t>();
        else if constexpr (__believe_own_sched)
          return __mtype<
            __completion_scheduler_of<_Key, typename decltype(__get_sched_tag())::__t>>();
        else
          return __mtype<void>();
      }

      using __domain_t = decltype(__get_domain())::__t;
      using __sched_t  = decltype(__get_sched())::__t;
    };

    template <class _Tag, class _Event>
    struct __source_if
    {
      using __t = __mlist<>;
    };

    template <class _Tag, class... _Args, class _Src>
    struct __source_if<_Tag, __walk_event<_Tag(_Args...), _Src>>
    {
      using __t = __mlist<_Src>;
    };

    template <class _Tag, class _Events>
    struct __sources_of;

    template <class _Tag, class... _Events>
    struct __sources_of<_Tag, __mlist<_Events...>>
    {
      using __t = __mcall<__mconcat<__munique<>>, typename __source_if<_Tag, _Events>::__t...>;
    };

    // The sources of a channel's completions. A channel that has none has the source that
    // the algorithm names for it, if any (see __defaults::__empty_channel_source).
    template <class _Algo, class _Tag, class _Sources>
    struct __channel_sources
    {
      using __t = _Sources;
    };

    template <class _Algo, class _Tag>
      requires(!__same_as<typename _Algo::template __empty_channel_source<_Tag>, void>)
    struct __channel_sources<_Algo, _Tag, __mlist<>>
    {
      using __t = __mlist<typename _Algo::template __empty_channel_source<_Tag>>;
    };

    template <class _Scheds>
    struct __the_one_scheduler
    {
      using __t = void;
    };

    template <class _Sched>
    struct __the_one_scheduler<__mlist<_Sched>>
    {
      using __t = _Sched;
    };

    template <class _Ctx, class _Sources, class... _QEnv>
    struct __channel_join;

    template <class _Ctx, class... _Srcs, class... _QEnv>
    struct __channel_join<_Ctx, __mlist<_Srcs...>, _QEnv...>
    {
      static constexpr bool __has_events = sizeof...(_Srcs) != 0;

      static constexpr __behavior_t __behavior = __facade::__join(
        __resolve<_Ctx, _Srcs, _QEnv...>::__behavior...);

      static consteval auto __get_domain() noexcept
      {
        if constexpr ((__same_as<typename __resolve<_Ctx, _Srcs, _QEnv...>::__domain_t, void>
                       || ...))
          return __mtype<void>();
        else
          return __mtype<
            __domain_join_t<typename __resolve<_Ctx, _Srcs, _QEnv...>::__domain_t...>>();
      }

      using __domain_t = decltype(__get_domain())::__t;

      // A completion scheduler is a runtime value, so report one only if every event on
      // this channel gets its scheduler from the same place.
      using __sched_t = __the_one_scheduler<
        __mcall<__munique<>, typename __resolve<_Ctx, _Srcs, _QEnv...>::__sched_t...>>::__t;
    };

    template <class _Ctx, class _Src, class... _QEnv>
    struct __dispatch_domain
    {
      using __t = __resolve<_Ctx, _Src, _QEnv...>::__domain_t;
    };

    template <class _Ctx, class... _QEnv>
    struct __dispatch_domain<_Ctx, void, _QEnv...>
    {
      using __t = void;
    };

    // The domain of a nested operation's completion, if it can complete that way, or if its
    // attributes say where it would (as the standard's SCHED-ATTRS do for continues_on, of a
    // child that never completes with a value)
    template <class _Ctx, class _Key, class _Tag, class _Src, class... _QEnv>
    struct __dispatch_domain<_Ctx, __via<_Key, _Tag, _Src>, _QEnv...>
    {
      using __algo_t     = _Ctx::__algo_t;
      using __sndr_t     = _Ctx::__sndr_t;
      using __resolved_t = __resolve<_Ctx, __via<_Key, _Tag, _Src>, _QEnv...>;

      static consteval auto __get() noexcept
      {
        if constexpr (__sends<_Tag,
                              typename _Ctx::template __nested_sender_t<_Key>,
                              __facade::__env_t<__algo_t, _Key, __sndr_t, _QEnv>...>)
          return __mtype<typename __resolved_t::__domain_t>();
        else if constexpr (__resolved_t::__has_own_domain || __resolved_t::__own_sched)
          return __mtype<typename __resolved_t::__domain_t>();
        else
          return __mtype<void>();
      }

      using __t = decltype(__get())::__t;
    };

    // An object with no attributes
    struct __no_attrs
    {
      void query() const noexcept = delete;
    };

    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Algo, class _Sender>
    struct __attrs
    {
      template <class _Tag, class... _QEnv>
      using __join_t = __channel_join<
        __ctx<_Algo, _Sender, _QEnv...>,
        typename __channel_sources<
          _Algo,
          _Tag,
          typename __sources_of<_Tag, __events_t<_Algo, _Sender, _QEnv...>>::__t>::__t,
        _QEnv...>;

      template <class... _QEnv>
      using __dispatch_domain_t = __dispatch_domain<__ctx<_Algo, _Sender, _QEnv...>,
                                                    typename _Algo::__dispatch_source,
                                                    _QEnv...>::__t;

      using __fwd_attrs_t = env_of_t<decltype(_Algo::__fwd_sender(__declval<_Sender const &>()))>;

      // The domain in which to look for customizations of the algorithm: the dispatch domain,
      // if the algorithm names one, and otherwise the domain of its value completions. Since
      // it decides the customization, it is the sender's own, not the customized sender's.
      template <class... _QEnv>
      static consteval auto __get_customizing_domain() noexcept
      {
        if constexpr (!__same_as<__dispatch_domain_t<_QEnv...>, void>)
          return __mtype<__dispatch_domain_t<_QEnv...>>();
        else if constexpr (__walk_ok<_Algo, _Sender, _QEnv...>)
          return __mtype<typename __join_t<set_value_t, _QEnv...>::__domain_t>();
        else
          return __mtype<void>();
      }

      template <class... _QEnv>
      using __customizing_domain_t = decltype(__get_customizing_domain<_QEnv...>())::__t;

      // The algorithm of the sender as a domain that re-tags customizes it in the
      // environment, or void if none does. (Only the sender's own algorithm asks: a
      // customized sender's attributes are not customized again.) The customizing domain is
      // checked first, so that the attributes answer get_completion_domain<>() before the
      // domain is asked how it customizes the sender. (Otherwise that query would fall back
      // to the value completion domain, which would ask how the domain customizes the
      // sender.)
      template <class... _QEnv>
      static consteval auto __get_customized_algo() noexcept
      {
        if constexpr (sizeof...(_QEnv) != 1
                      || !__same_as<_Algo, __algorithm<typename _Sender::__tag_t>>)
          return __mtype<void>();  // NOLINT(bugprone-branch-clone)
        else if constexpr (!__retagging_domain<__customizing_domain_t<_QEnv...>>)
          return __mtype<void>();
        else
          return __mtype<__customized_algo_t<_Sender, _QEnv...>>();
      }

      template <class... _QEnv>
      using __customized_t = decltype(__get_customized_algo<_QEnv...>())::__t;

      template <class... _QEnv>
      static constexpr bool __customized_v = !__same_as<__customized_t<_QEnv...>, void>;

      // Without an environment, a sender is not customized, so a sender that a domain that
      // re-tags customizes in an environment can answer differently there. And a query
      // function that gets no answer in an environment asks again without one (see
      // get_completion_domain_t::__read_query_t), which would mix the customized sender's
      // answers with the uncustomized sender's. So a sender that such a domain would
      // customize (its customizing domain without an environment is one, and that domain
      // changes it in an empty environment) does not answer the completion queries without
      // an environment. (A sender that the domain has customized already, it leaves as it is.)
      template <class... _QEnv>
      static consteval auto __answers_without_env() noexcept -> bool
      {
        if constexpr (sizeof...(_QEnv) != 0
                      || !__same_as<_Algo, __algorithm<typename _Sender::__tag_t>>)
          return true;  // NOLINT(bugprone-branch-clone)
        else if constexpr (!__retagging_domain<__customizing_domain_t<>>)
          return true;
        else if constexpr (__callable<transform_sender_t, _Sender, env<> const &>)
          return __same_as<__decay_t<__call_result_t<transform_sender_t, _Sender, env<> const &>>,
                           _Sender>;
        else
          return false;
      }

      // Where the sender's _Tag completions get their completion scheduler from: the
      // scheduler expression that the get_completion_scheduler<_Tag> query below evaluates,
      // or void if the query has no answer. (See __sched_place_of.)
      template <class _Tag, class... _QEnv>
      static consteval auto __get_sched_place() noexcept
      {
        if constexpr (sizeof...(_QEnv) > 1)
          return __mtype<void>();  // NOLINT(bugprone-branch-clone)
        else if constexpr (__customized_v<_QEnv...>)
          return __mtype<typename __attrs<__customized_t<_QEnv...>,
                                          _Sender>::template __sched_place_t<_Tag, _QEnv...>>();
        else if constexpr (!__answers_without_env<_QEnv...>()
                           || !__walk_ok<_Algo, _Sender, _QEnv...>)
          return __mtype<void>();
        else
          return __mtype<typename __join_t<_Tag, _QEnv...>::__sched_t>();
      }

      template <class _Tag, class... _QEnv>
      using __sched_place_t = decltype(__get_sched_place<_Tag, _QEnv...>())::__t;

      // Each completion query is available only if the sender type-checks in the query's
      // environment.
      template <class _Tag, class... _QEnv>
        requires(sizeof...(_QEnv) <= 1) && (!__customized_v<_QEnv...>)
             && (__answers_without_env<_QEnv...>())
             && __walk_ok<_Algo, _Sender, _QEnv...> && __join_t<_Tag, _QEnv...>::__has_events
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(__get_completion_behavior_t<_Tag>, _QEnv const &...) const noexcept
      {
        return __completion_behavior::__constant_t<__join_t<_Tag, _QEnv...>::__behavior>();
      }

      // (get_completion_domain<>(), whose _Tag is void, is answered below.)
      template <class _Tag, class... _QEnv>
        requires(sizeof...(_QEnv) <= 1) && (!__same_as<_Tag, void>) && (!__customized_v<_QEnv...>)
             && (__answers_without_env<_QEnv...>()) && __walk_ok<_Algo, _Sender, _QEnv...>
             && (!__same_as<typename __join_t<_Tag, _QEnv...>::__domain_t, void>)
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_domain_t<_Tag>, _QEnv const &...) const noexcept ->
        typename __join_t<_Tag, _QEnv...>::__domain_t
      {
        return {};
      }

      // The domain in which to look for customizations of the algorithm (see
      // __get_customizing_domain). When the algorithm names a dispatch domain, this does not
      // require the sender to type-check in the query's environment, unlike the other
      // completion queries, because a customization can be what makes it type-check (for
      // example, by giving a child a different environment).
      template <class... _QEnv>
        requires(sizeof...(_QEnv) <= 1) && (!__same_as<__customizing_domain_t<_QEnv...>, void>)
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_domain_t<>, _QEnv const &...) const noexcept
        -> __customizing_domain_t<_QEnv...>
      {
        return {};
      }

      template <class _Tag, class... _QEnv>
        requires(sizeof...(_QEnv) <= 1) && (!__customized_v<_QEnv...>)
             && (__answers_without_env<_QEnv...>()) && __walk_ok<_Algo, _Sender, _QEnv...>
             && (!__same_as<typename __join_t<_Tag, _QEnv...>::__sched_t, void>)
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(get_completion_scheduler_t<_Tag>, _QEnv const &...__env) const noexcept
      {
        return __join_t<_Tag, _QEnv...>::__sched_t::template __eval<_Algo>(__sndr_, __env...);
      }

      // In an environment in which a domain that re-tags customizes the sender, the
      // completion queries other than get_completion_domain<>() are answered as the
      // customized sender's are. (Its algorithm computes them from the sender as it is: the
      // parts of a re-tagged sender are the same.)
      template <class _Query, class _QEnv>
        requires __completion_query<_Query> && (!__same_as<_Query, get_completion_domain_t<>>)
              && __customized_v<_QEnv>
              && __queryable_with<__attrs<__customized_t<_QEnv>, _Sender>, _Query, _QEnv const &>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(_Query, _QEnv const &__env) const noexcept -> decltype(auto)
      {
        return __attrs<__customized_t<_QEnv>, _Sender>{__sndr_}.query(_Query(), __env);
      }

      // Other forwarding queries are answered by the attributes the algorithm names. (The
      // attributes' type is a template parameter so that it is checked only on use. The result
      // type is deduced: compilers that substitute into the return type before checking the
      // constraints, as Clang does through version 19, would otherwise ask those attributes a
      // completion query, such as get_completion_domain<>(), in the environment of the query.
      // The child must be asked in the environment that the algorithm gives it.)
      template <class _FwdAttrs = __fwd_attrs_t, __forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>) && __queryable_with<_FwdAttrs, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto query(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<_FwdAttrs, _Query, _Args...>) -> decltype(auto)
      {
        return STDEXEC::get_env(_Algo::__fwd_sender(__sndr_))
          .query(_Query(), static_cast<_Args &&>(__args)...);
      }

      _Sender const &__sndr_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // The attributes of a nested operation that schedules onto a scheduler, for
    // an algorithm's __nested_attrs hook. Where the operation completes is
    // reported by the scheduler (a scheduler is the completion scheduler of its
    // schedule sender), and how it completes by the schedule sender. _Scheduler
    // may be a reference type.
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Scheduler>
    struct __schedule_attrs
    {
      using __scheduler_t = __decay_t<_Scheduler>;

      template <class _Tag, class... _Env>
        requires __callable<get_completion_scheduler_t<_Tag>, __scheduler_t const &, _Env const &...>
      [[nodiscard]] STDEXEC_ATTRIBUTE(host, device) constexpr auto query(
        get_completion_scheduler_t<_Tag>,
        _Env const &...__env) const noexcept
      {
        return STDEXEC::get_completion_scheduler<_Tag>(__sch_, __env...);
      }

      template <class _Tag, class... _Env>
        requires __callable<get_completion_domain_t<_Tag>, __scheduler_t const &, _Env const &...>
      [[nodiscard]] STDEXEC_ATTRIBUTE(host,
                                      device) constexpr auto query(get_completion_domain_t<_Tag>,
                                                                   _Env const &...) const noexcept
        -> __call_result_t<get_completion_domain_t<_Tag>, __scheduler_t const &, _Env const &...>
      {
        return {};
      }

      template <class _Tag, class... _Env>
      [[nodiscard]] STDEXEC_ATTRIBUTE(host, device) constexpr auto query(
        __get_completion_behavior_t<_Tag>,
        _Env const &...) const noexcept
      {
        return STDEXEC::__get_completion_behavior<_Tag,
                                                  schedule_result_t<__scheduler_t const &>,
                                                  _Env...>();
      }

      _Scheduler __sch_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // The default operation state of an algorithm: its receiver and its data
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Receiver, class _Data>
    struct __basic_state
    {
      template <class _CvData>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __basic_state(_Receiver __rcvr, _CvData &&__data)
        noexcept(__nothrow_decay_copyable<_CvData>)
        : __rcvr_(static_cast<_Receiver &&>(__rcvr))
        , __data_(STDEXEC::__allocator_aware_forward(static_cast<_CvData &&>(__data), __rcvr_))
      {}

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Data     __data_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Default implementations of the algorithm hooks
    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __defaults
    {
      // The base class of the receivers that the facade connects the algorithm's children
      // and nested operations to. (A domain's algorithm can use this to make them the kind
      // of receiver that its senders complete differently, as a CUDA stream sender
      // completes a stream receiver on the host.)
      struct __receiver_base
      {};

      // Opt-in: do the algorithm's own completions depend on the receiver's environment?
      // (Dependence on the environment through a child needs no opt-in.) If so, the sender
      // is dependent when there is no environment. _Sender is the __sender type.
      template <class _Sender>
      static constexpr bool __depends_on_env = false;

      // Does the facade connect the children, so that __start can start them? An algorithm
      // that connects (some of) its children itself sets this to false.
      static constexpr bool __connects_children = true;

      // The domain in which customizations of the algorithm are found, given as the
      // provenance whose completion domain it is, or void for the domain of the value
      // completions.
      using __dispatch_source = void;

      // The provenance whose attributes a channel has when nothing completes on it, or void
      // for none. Such attributes are vacuous, but they let an algorithm report, for example,
      // the scheduler on which its value completions would happen, as the standard's
      // SCHED-ATTRS does for continues_on.
      template <class _Tag>
      using __empty_channel_source = void;

      // Connects the child or nested operation keyed by _Key. The facade connects the
      // children with this, and an algorithm that connects operations in its state can too.
      // (A domain's algorithm can use it to adapt a child that would complete the receiver
      // in a way the algorithm cannot handle, as when a CUDA stream operation completes a
      // receiver from a kernel.)
      template <class _Key, class _State, class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __connect(_Key, _State &, _Sender &&__sndr, _Receiver __rcvr)
        noexcept(__nothrow_connectable<_Sender, _Receiver>) -> connect_result_t<_Sender, _Receiver>
      {
        return STDEXEC::connect(static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr));
      }

      // The child whose items are the algorithm's, if it has items: __child<_Idx>, or void.
      // _Sender is the __sender type. If that child is a sequence sender, so is the
      // algorithm's sender. Subscribing it subscribes the child (with __subscribe), whose
      // receiver forwards the child's items to the algorithm's receiver. (Connecting it
      // connects the child, as any other.)
      template <class _Sender>
      using __item_source = void;

      // Subscribes the item source, keyed by _Key, when the algorithm's sender is subscribed.
      // Like exec::subscribe, it transforms the sender first.
      template <class _Key, class _State, class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __subscribe(_Key, _State &, _Sender &&__sndr, _Receiver __rcvr)
        noexcept(__nothrow_callable<transform_sender_t, _Sender, env_of_t<_Receiver> const &>
                 && __nothrow_callable<__subscribe_::__subscribe_t,
                                       transform_sender_result_t<_Sender, env_of_t<_Receiver>>,
                                       _Receiver>)
          -> __subscribe_result_t<transform_sender_result_t<_Sender, env_of_t<_Receiver>>,
                                  _Receiver>
      {
        auto &&__env      = STDEXEC::get_env(__rcvr);
        auto &&__tfx_sndr = STDEXEC::transform_sender(static_cast<_Sender &&>(__sndr), __env);
        return STDEXEC::__subscribe(static_cast<decltype(__tfx_sndr) &&>(__tfx_sndr),
                                    static_cast<_Receiver &&>(__rcvr));
      }

      // The environment of the child or nested operation keyed by _Key, given the sender
      // and the environment of the receiver connected to it. The environment is forwarded,
      // so that the result's type is the one __get_env returns at run time (where the
      // receiver's environment is whatever get_env returns).
      template <class _Key, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __env_for(_Key, _Sender const &, _Env &&__env) noexcept -> _Env
      {
        return static_cast<_Env &&>(__env);
      }

      // The same environment, at run time
      template <class _Key, class _State>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_env(_Key, _State const &__state) noexcept -> decltype(auto)
      {
        return STDEXEC::get_env(__state.__rcvr_);
      }

      // The sender of the child or nested operation keyed by _Key. It is used for its type,
      // and by the attributes for its own attributes. (A nested sender that is created at
      // run time is returned as an __mtype.)
      template <std::size_t _Idx, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __nested_sender(__child<_Idx>,
                                            _Sender const &__sndr,
                                            _Env const &...) noexcept -> auto const &
      {
        return __facade::__get_child<_Idx>(__sndr);
      }

      // The sender whose attributes answer forwarding queries that are not completion
      // queries: the child, if there is exactly one.
      template <class _Sender>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __fwd_sender(_Sender const &__sndr) noexcept -> decltype(auto)
      {
        if constexpr (__nbr_children_v<_Sender> == 1)
          return __facade::__get_child<0>(__sndr);
        else
          return __no_attrs();
      }

      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr)
        noexcept(__nothrow_decay_copyable<__data_of_t<_Sender>>)
          -> __basic_state<_Receiver, __decay_t<__data_of_t<_Sender>>>
      {
        return __basic_state<_Receiver, __decay_t<__data_of_t<_Sender>>>{
          static_cast<_Receiver &&>(__rcvr),
          __facade::__get_data(static_cast<_Sender &&>(__sndr))};
      }

      // The state when the sender is subscribed (see __subscribing_t): the same
      template <class _Sender, class _Receiver>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __get_state(_Sender &&__sndr, _Receiver __rcvr, __subscribing_t)
        noexcept(__nothrow_decay_copyable<__data_of_t<_Sender>>)
          -> __basic_state<_Receiver, __decay_t<__data_of_t<_Sender>>>
      {
        return __defaults::__get_state(static_cast<_Sender &&>(__sndr),
                                       static_cast<_Receiver &&>(__rcvr));
      }

      template <class _State, class _Children>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __start(_State &, _Children &__children) noexcept
      {
        return __children.__start_all();
      }

      template <class _Key, class _State, class _Tag, class... _Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __complete(_Key,
                                       _State &__state,
                                       _Tag    __tag,
                                       _Args &&...__args) noexcept
      {
        return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args &&>(__args)...);
      }
    };

    ////////////////////////////////////////////////////////////////////////////////////////
    // The senders of facade algorithms

    // The attributes of a sender: the ones that the facade derives, unless the algorithm
    // gives them itself with a hook `__get_attrs(sndr)`
    template <class _Algo, class _Sender>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    constexpr auto __attrs_of(_Sender const &__sndr) noexcept
    {
      if constexpr (requires { _Algo::__get_attrs(__sndr); })
        return _Algo::__get_attrs(__sndr);
      else
        return __attrs<_Algo, _Sender>{__sndr};
    }

    // The completion signatures of a sender: the ones that the facade derives, unless the
    // algorithm gives them itself with a hook `__get_completion_signatures<Sender, Env...>()`.
    // If the sender does not type-check, this throws (or returns) the walk's error.
    template <class _Algo, class _Sender, class... _Env>
    consteval auto __completion_signatures_of()
    {
      if constexpr (requires { _Algo::template __get_completion_signatures<_Sender, _Env...>(); })
      {
        return _Algo::template __get_completion_signatures<_Sender, _Env...>();
      }
      else
      {
        auto __events = __facade::__walk<_Algo, _Sender, _Env...>();
        STDEXEC_IF_OK(__events)
        {
          return typename __sigs_of<__t<decltype(__events)>>::__t();
        }
      }
    }

    // Like the senders of the standard algorithms, a facade sender is a product type of its
    // tag, its data, and its children:
    //
    //   auto& [tag, data, ...children] = sndr;
    //
    // An algorithm can give its senders a name of their own with a type derived from this.
    // Such a sender is seen, by the algorithm's hooks, as a __sender.
    template <class _Tag, class _Data, class... _Child>
    struct __sender : __tuple<_Tag, _Data, _Child...>
    {
      // It is a sequence sender if its item source is (see __defaults::__item_source).
      using sender_concept =
        __if_c<__facade::__is_sequence_sender<_Child...>(
                 static_cast<__item_source_t<__algorithm<_Tag>, __sender> const *>(nullptr)),
               __sequence_sender_tag,
               sender_tag>;
      using __tag_t      = _Tag;
      using __data_t     = _Data;
      using __children_t = __mlist<_Child...>;

#  if STDEXEC_HAS_BUILTIN(__type_pack_element)
      template <std::size_t _Idx>
      using __child_t = __type_pack_element<_Idx, _Child...>;
#  else
      template <std::size_t _Idx>
      using __child_t = __m_at_c<_Idx, _Child...>;
#  endif

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<__algorithm<_Tag>, __copy_cvref_t<_Self, __sender>, _Receiver>;

      template <class _Self, class _Receiver>
      using __subscribe_opstate_t =
        __opstate<__algorithm<_Tag>, __copy_cvref_t<_Self, __sender>, _Receiver, __subscribing_t>;

      // The items of a sequence sender, for exec::get_item_types
      template <class _Self, class... _Env>
      using __item_types_from_t =
        __facade::__item_types_from<__algorithm<_Tag>,
                                    __copy_cvref_t<_Self, __sender>,
                                    __item_source_t<__algorithm<_Tag>, __sender>,
                                    _Env...>::__t;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto get_env() const noexcept
      {
        return __facade::__attrs_of<__algorithm<_Tag>>(*this);
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures()
      {
        static_assert(STDEXEC_IS_BASE_OF(__sender, __decay_t<_Self>));
        return __facade::__completion_signatures_of<__algorithm<_Tag>,
                                                    __copy_cvref_t<_Self, __sender>,
                                                    _Env...>();
      }

      template <class _Self, receiver _Receiver>
        requires sender_in<__copy_cvref_t<_Self, __sender>, env_of_t<_Receiver>>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto STDEXEC_EXPLICIT_THIS_BEGIN(connect)(this _Self &&__self, _Receiver __rcvr)
        noexcept(__opstate_t<_Self, _Receiver>::__nothrow_v) -> __opstate_t<_Self, _Receiver>
      {
        return __opstate_t<_Self, _Receiver>(STDEXEC::__c_upcast<__sender>(
                                               static_cast<_Self &&>(__self)),
                                             static_cast<_Receiver &&>(__rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      // A sequence sender is subscribed like this, by exec::subscribe (via STDEXEC::__subscribe)
      template <class _Self, receiver _Receiver>
        requires __sequence_sender_tagged<_Self>
              && sender_in<__copy_cvref_t<_Self, __sender>, env_of_t<_Receiver>>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      static constexpr auto __static_subscribe(_Self &&__self, _Receiver __rcvr)
        noexcept(__subscribe_opstate_t<_Self, _Receiver>::__nothrow_v)
          -> __subscribe_opstate_t<_Self, _Receiver>
      {
        return __subscribe_opstate_t<_Self, _Receiver>(STDEXEC::__c_upcast<__sender>(
                                                         static_cast<_Self &&>(__self)),
                                                       static_cast<_Receiver &&>(__rcvr));
      }
    };

    template <class _Tag, class _Data = __, class... _Child>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sender(_Tag, _Data, _Child...) -> __sender<_Tag, _Data, _Child...>;
  }  // namespace __facade
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
