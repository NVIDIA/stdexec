/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../../stdexec/execution.hpp"

// include these after execution.hpp
#include "../../stdexec/__detail/__continues_on.hpp"
#include "../../stdexec/__detail/__sender_facade.hpp"
#include "../../stdexec/__detail/__senders.hpp"
#include "../../stdexec/__detail/__tuple.hpp"
#include "../sender_for.hpp"
#include "../sequence_senders.hpp"

#include "../../stdexec/__detail/__atomic.hpp"

namespace experimental::execution
{
  template <class _Variant, class _Type, class... _Args>
  concept __variant_emplaceable = requires(_Variant& __var, _Args&&... __args) {
    __var.template emplace<_Type>(static_cast<_Args&&>(__args)...);
  };

  template <class _Variant, class _Type, class... _Args>
  concept __nothrow_variant_emplaceable = requires(_Variant& __var, _Args&&... __args) {
    { __var.template emplace<_Type>(static_cast<_Args&&>(__args)...) } noexcept;
  };

  namespace __ignore_all_values
  {
    using namespace STDEXEC;

    template <class _ResultsStorage>
    struct __result_type
    {
      template <class... _Args>
      void __emplace(_Args&&... __args) noexcept
      {
        int __expected = 0;
        if (__emplaced_.compare_exchange_strong(__expected, 1, __std::memory_order_relaxed))
        {
          __result_.template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
          __emplaced_.store(2, __std::memory_order_release);
        }
      }

      _ResultsStorage    __result_;
      __std::atomic<int> __emplaced_{0};
    };

    template <class _ItemReceiver, class _ResultsStorage>
    struct __item_operation_base
    {
      STDEXEC_ATTRIBUTE(no_unique_address) _ItemReceiver __rcvr_;
      __result_type<_ResultsStorage>* __result_;
    };

    template <class _ItemReceiver, class _ResultsStorage>
    struct __item_receiver
    {
      using receiver_concept   = STDEXEC::receiver_tag;
      using __child_receiver_t = void;  // see STDEXEC::__child_receiver
      __item_operation_base<_ItemReceiver, _ResultsStorage>* __op_;

      template <class... _Args>
      void set_value([[maybe_unused]] _Args&&... __args) noexcept
      {
        // ignore incoming values
        STDEXEC::set_value(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      template <class _Error>
        requires __variant_emplaceable<_ResultsStorage,
                                       __decayed_tuple<set_error_t, _Error>,
                                       set_error_t,
                                       _Error>
              && __callable<STDEXEC::set_stopped_t, _ItemReceiver>
      void set_error(_Error&& __error) noexcept
      {
        // store error and signal stop
        __op_->__result_->__emplace(set_error_t(), static_cast<_Error&&>(__error));
        STDEXEC::set_stopped(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      void set_stopped() noexcept
        requires __variant_emplaceable<_ResultsStorage,
                                       __decayed_tuple<set_stopped_t>,
                                       set_stopped_t>
              && __callable<set_stopped_t, _ItemReceiver>
      {
        // stop without error
        __op_->__result_->__emplace(set_stopped_t());
        STDEXEC::set_stopped(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<_ItemReceiver>
      {
        return STDEXEC::get_env(__op_->__rcvr_);
      }
    };

    template <class _Sender, class _ItemReceiver, class _ResultsStorage>
    struct __item_operation : __item_operation_base<_ItemReceiver, _ResultsStorage>
    {
      using __base_t          = __item_operation_base<_ItemReceiver, _ResultsStorage>;
      using __item_receiver_t = __item_receiver<_ItemReceiver, _ResultsStorage>;

      __item_operation(__result_type<_ResultsStorage>* __parent,
                       _Sender&&                       __sndr,
                       _ItemReceiver                   __rcvr)
        noexcept(__nothrow_decay_copyable<_ItemReceiver>
                 && __nothrow_connectable<_Sender, __item_receiver_t>)
        : __base_t{static_cast<_ItemReceiver&&>(__rcvr), __parent}
        , __op_{STDEXEC::connect(static_cast<_Sender&&>(__sndr), __item_receiver_t{this})}
      {}

      void start() & noexcept
      {
        STDEXEC::start(__op_);
      }

      connect_result_t<_Sender, __item_receiver_t> __op_;
    };

    template <class _Sender, class _ResultsStorage>
    struct __item_sender
    {
      using sender_concept        = STDEXEC::sender_tag;
      using completion_signatures = STDEXEC::completion_signatures<set_value_t(), set_stopped_t()>;

      template <class _Self, class _Receiver>
      using __operation_t =
        __item_operation<__copy_cvref_t<_Self, _Sender>, _Receiver, _ResultsStorage>;

      template <class _Receiver>
      using __item_receiver_t = __item_receiver<_Receiver, _ResultsStorage>;

      template <__decays_to<__item_sender>                  _Self,
                STDEXEC::receiver_of<completion_signatures> _Receiver>
        requires sender_to<__copy_cvref_t<_Self, _Sender>, __item_receiver_t<_Receiver>>
      auto STDEXEC_EXPLICIT_THIS_BEGIN(connect)(this _Self&& __self, _Receiver __rcvr)
        -> __operation_t<_Self, _Receiver>
      {
        return {__self.__parent_,
                static_cast<_Self&&>(__self).__sender_,
                static_cast<_Receiver&&>(__rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _Sender                         __sender_;
      __result_type<_ResultsStorage>* __parent_;
    };

    template <class _Signature>
    using __is_set_value_signature_t =
      __mbool<__same_as<set_value_t, __signature_tag_t<_Signature>>>;

    // Storage for the non-set_value completions
    template <class _Sender, class _Env>
    using __result_variant_t =
      __mapply<__mremove_if<__q1<__is_set_value_signature_t>, __qq<__results_storage>>,
               __sequence_completion_signatures_of_t<_Sender, _Env>>;

    template <class _Sender, class _Receiver>
    struct __state;

    template <class _State>
    struct __receiver;

    // The key of the subscription to the sequence. (It is not the child's key, __child<0>,
    // because the facade would take the child itself to be the operation's sender. The
    // operation subscribes to it.)
    struct __subscription
    {};

    // The sequence, as the operation that subscribes to it sees it: it completes with
    // set_value_t() when it ends, or with its errors or set_stopped_t(). (Its items' results are
    // saved, see __item_receiver.) It is known only by its type.
    template <class _Completions>
    struct __sequence_sender
    {
      using sender_concept        = sender_tag;
      using completion_signatures = _Completions;
    };

    template <class _Sequence, class... _Env>
    using __sequence_sender_t =
      __sequence_sender<decltype(exec::__sequence_completion_signatures<_Sequence, _Env...>())>;

    // ignore_all_values(seq): the operation subscribes to the sequence, and ignores the values
    // of its items. The first item that fails or stops saves its result, and stops the
    // sequence. When the sequence completes with a value, the operation sends the saved result,
    // if there is one, and otherwise completes with set_value_t(). The sequence's other
    // completions are the operation's.
    struct __algo : __facade::__defaults
    {
      // (The operation subscribes to the sequence, which the facade cannot connect.)
      static constexpr bool __connects_children = false;

      using __facade::__defaults::__nested_sender;

      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __nested_sender(__subscription,
                                            _Sender const &,
                                            _Env const &...) noexcept
        -> __mtype<__sequence_sender_t<__facade::__nth_child_of_t<_Sender, 0>, _Env...>>
      {
        return {};
      }

      // (See __facade::__state_type_t.)
      template <class _Sender, class _Receiver>
      using __state_type = __state<_Sender, _Receiver>;

      template <class _Sender, class _Receiver>
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__nothrow_constructible_from<__state<_Sender, _Receiver>, _Sender, _Receiver>)
          -> __state<_Sender, _Receiver>
      {
        return __state<_Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                           static_cast<_Receiver&&>(__rcvr));
      }

      template <class _State, class _Children>
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return __facade::__start_nested<__subscription>(__state.__op_);
      }

      using __facade::__defaults::__complete;

      // The sequence completed with a value: send the saved result, if there is one.
      template <class _State>
      static constexpr auto __complete(__subscription, _State& __state, set_value_t) noexcept
      {
        return __facade::__choose(
          __state.__emplaced_.load(__std::memory_order_acquire) == 0,
          [&]() noexcept { return __facade::__send(__state.__rcvr_, set_value_t()); },
          [&]() noexcept { return __trnsfr::__send_saved(__state.__result_, __state.__rcvr_); });
      }
    };

    // The receiver of the sequence. Its items' operations save their results (see
    // __item_receiver), and its completions go to the algorithm.
    template <class _State>
    struct __receiver
    {
      using receiver_concept   = STDEXEC::receiver_tag;
      using __child_receiver_t = void;  // see STDEXEC::__child_receiver

      constexpr explicit __receiver(_State* __state) noexcept
        : __state_{__state}
      {}

      // nvc++ needs this destructor to be defined to avoid a codegen issue
      STDEXEC_PP_WHEN(STDEXEC_NVHPC(), ~__receiver(){})

      template <sender _Item>
      [[nodiscard]]
      auto set_next(_Item&& __item) & noexcept(__nothrow_decay_copyable<_Item>)
        -> __item_sender<__decay_t<_Item>, typename _State::__variant_t>
      {
        return {static_cast<_Item&&>(__item), __state_};
      }

      void set_value() noexcept
      {
        (void) __algo::__complete(__subscription(), *__state_, set_value_t());
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept
      {
        (void) __algo::__complete(__subscription(),
                                  *__state_,
                                  set_error_t(),
                                  static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept
      {
        (void) __algo::__complete(__subscription(), *__state_, set_stopped_t());
      }

      auto get_env() const noexcept -> env_of_t<typename _State::__receiver_t>
      {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

     private:
      _State* __state_;
    };

    // The part of the state that the sequence's receiver points to: the receiver and the
    // saved result. (The receiver's type does not name the operation's sender, as the state's
    // does: the types of the items' operations name it, and they would grow with the size of
    // the sender, as would the debug info that names them.)
    template <class _Receiver, class _Variant>
    struct __state_base : __result_type<_Variant>
    {
      using __variant_t  = _Variant;
      using __receiver_t = _Receiver;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    template <class _Sender, class _Receiver>
    struct __state
      : __state_base<
          _Receiver,
          __result_variant_t<__facade::__nth_child_of_t<_Sender, 0>, env_of_t<_Receiver>>>
    {
      using __child_t = __facade::__nth_child_of_t<_Sender, 0>;
      using __base_t  = __state_base<_Receiver, __result_variant_t<__child_t, env_of_t<_Receiver>>>;
      using __seq_rcvr_t = __ignore_all_values::__receiver<__base_t>;

      explicit __state(_Sender&& __sndr, _Receiver&& __rcvr)
        noexcept(__nothrow_subscribable<__child_t, __seq_rcvr_t>)
        : __base_t{{}, static_cast<_Receiver&&>(__rcvr)}
        , __op_{__subscribe_child(static_cast<_Sender&&>(__sndr), this)}
      {}

      STDEXEC_IMMOVABLE(__state);

      using __op_t = __facade::__state_op_t<_Receiver, subscribe_result_t, __child_t, __seq_rcvr_t>;

      // Subscribes the child, unless the state is for a walk: then __op_ is a placeholder
      static constexpr auto __subscribe_child(_Sender&& __sndr, __state* __self) -> __op_t
      {
        if constexpr (__same_as<__op_t, __facade::__walk_op>)
          return __op_t();
        else
          return exec::subscribe(__facade::__get_child<0>(static_cast<_Sender&&>(__sndr)),
                                 __seq_rcvr_t{__self});
      }

      __op_t __op_;
    };

    struct ignore_all_values_t
    {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const
      {
        return __facade::__sender{ignore_all_values_t(), __(), static_cast<_Sender&&>(__sndr)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const noexcept
      {
        return __closure(*this);
      }
    };
  }  // namespace __ignore_all_values

  using __ignore_all_values::ignore_all_values_t;
  inline constexpr ignore_all_values_t ignore_all_values{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::ignore_all_values_t> : exec::__ignore_all_values::__algo
  {};
}  // namespace STDEXEC
