/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "../stop_token.hpp"
#include "__atomic.hpp"
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__receivers.hpp"
#include "__scope.hpp"
#include "__scope_concepts.hpp"
#include "__senders.hpp"
#include "__spawn_common.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"
#include "__variant.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace STDEXEC {

  /////////////////////////////////////////////////////////////////////////////
  // [exec.spawn.future]
  namespace __spawn_future {

    template <class _Sig>
    struct __future_sig_fns;

    template <class _Tag, class... _Args>
    struct __future_sig_fns<_Tag(_Args...)> {
      using __tuple = __decayed_tuple<_Tag, _Args...>;

      static constexpr bool __is_nothrow_storable =
        (__nothrow_constructible_from<__decay_t<_Args>, _Args> && ...);

      using __decayed_sig = _Tag(__decay_t<_Args>...);
    };

    // [exec.spawn.future] paragraph 4
    template <class _Sig>
    using __as_tuple = __future_sig_fns<_Sig>::__tuple;

    template <class _Sig>
    using __decayed_sig = __future_sig_fns<_Sig>::__decayed_sig;

    template <class... _Sigs>
    inline constexpr bool __sigs_are_nothrow_storable =
      (__future_sig_fns<_Sigs>::__is_nothrow_storable && ...);

    template <bool _NothrowStorable, class... _Sigs>
    struct __future_variant {
      // this case handles _NothrowStorable == true
      using type =
        __uniqued_variant<std::monostate, __decayed_tuple<set_stopped_t>, __as_tuple<_Sigs>...>;
    };

    template <class... _Sigs>
    struct __future_variant<false, _Sigs...> {
      using type = __uniqued_variant<
        std::monostate,
        __decayed_tuple<set_stopped_t>,
        __decayed_tuple<set_error_t, std::exception_ptr>,
        __as_tuple<_Sigs>...
      >;
    };

    template <class... _Sigs>
    using __future_variant_t =
      // [exec.spawn.future] paragraphs 4.1 and 4.2
      __future_variant<__sigs_are_nothrow_storable<_Sigs...>, _Sigs...>::type;

    template <bool _NothrowStorable, class... _Sigs>
    struct __future_completions {
      // this case handles _NothrowStorable == true
      using type =
        __mcall<__munique<__qq<completion_signatures>>, set_stopped_t(), __decayed_sig<_Sigs>...>;
    };

    template <class... _Sigs>
    struct __future_completions<false, _Sigs...> {
      using type = __mcall<
        __munique<__qq<completion_signatures>>,
        set_stopped_t(),
        set_error_t(std::exception_ptr),
        __decayed_sig<_Sigs>...
      >;
    };

    template <class... _Sigs>
    using __future_completions_t =
      __future_completions<__sigs_are_nothrow_storable<_Sigs...>, _Sigs...>::type;

    // [exec.spawn.future] paragraph 3
    template <class _Completions>
    struct __spawn_future_state_base;

    template <class... _Sigs>
    struct __spawn_future_state_base<completion_signatures<_Sigs...>> {
      using __variant_t = __future_variant_t<_Sigs...>;
      using __completions_t = __future_completions_t<_Sigs...>;

      __variant_t __result_{__no_init};

      __spawn_future_state_base() noexcept {
        __result_.template emplace<std::monostate>();
      }

      virtual void __complete() noexcept = 0;
    };

    // [exec.spawn.future] paragraph 5
    template <class _Completions>
    struct __spawn_future_receiver {
      using receiver_concept = receiver_t;

      __spawn_future_state_base<_Completions>* __state_;

      template <class... _T>
      void set_value(_T&&... __t) && noexcept {
        __set_complete<set_value_t>(static_cast<_T&&>(__t)...);
      }

      template <class _E>
      void set_error(_E&& __e) && noexcept {
        __set_complete<set_error_t>(static_cast<_E&&>(__e));
      }

      void set_stopped() && noexcept {
        __set_complete<set_stopped_t>();
      }

     private:
      template <class _CPO, class... _T>
      void __set_complete(_T&&... __t) noexcept {
        constexpr bool nothrow = (__nothrow_constructible_from<__decay_t<_T>, _T> && ...);

        try {
          __state_->__result_
            .template emplace<__decayed_tuple<_CPO, _T...>>(_CPO{}, static_cast<_T&&>(__t)...);
        } catch (...) {
          if constexpr (!nothrow) {
            using tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
            __state_->__result_.template emplace<tuple_t>(set_error_t{}, std::current_exception());
          }
        }

        __state_->__complete();
      }
    };

    // [exec.spawn.future] paragraph 6
    // ssource-t is inplace_stop_token
    template <class _Sender, class _Env>
    using __future_spawned_sender = decltype(write_env(
      __stop_when(__declval<_Sender>(), inplace_stop_token{}),
      __declval<_Env>()));

    // [exec.spawn.future] paragraph 7
    template <class _Alloc, scope_token _Token, sender _Sender, class _Env>
    struct __spawn_future_state final
      : __spawn_future_state_base<
          // the spec says completion_signatures_of_t<__future_spawned_sender<_Sender, _Env>> but
          // that breaks with an inscrutable error for _Sender = starts_on(sched, just() | then(...))
          //
          // I managed to fix the break by adding an extra _Env to the query, like so:
          //
          //     completion_signatures_of_t<__future_spawned_sender<_Sender, _Env>, _Env>
          //
          // but that's hard to justify--the future-spawned-sender will be connected to a receiver with
          // an empty environment after all. This code works and seems sensible (neither write_env nor
          // __stop_when change the completion signatures of their children, other than write_env
          // modifying the environment for its child, which is exactly what we want).
          completion_signatures_of_t<_Sender, _Env>
        > {
      using __sigs_t =
        // this is "wrong" in the same way as the above
        completion_signatures_of_t<_Sender, _Env>;

      using __receiver_t = __spawn_future_receiver<__sigs_t>;

      using __op_t = connect_result_t<__future_spawned_sender<_Sender, _Env>, __receiver_t>;

      __spawn_future_state(_Alloc __alloc, _Sender&& __sndr, _Token __token, _Env __env)
        : __alloc_(std::move(__alloc))
        , __op_(
            STDEXEC::connect(
              write_env(
                __stop_when(static_cast<_Sender&&>(__sndr), __stopSource_.get_token()),
                std::move(__env)),
              __receiver_t(this)))
        , __assoc_(__token.try_associate()) {
        if (__assoc_) {
          STDEXEC::start(__op_);
        } else {
          STDEXEC::set_stopped(__receiver_t(this));
        }
      }

      void __complete() noexcept override {
        void* receiver = nullptr;
        if (__registeredReceiver_
              .compare_exchange_strong(receiver, this, __std::memory_order_acq_rel)) {
          // we completed before a receiver was registered
          return;
        }

        assert(receiver != nullptr);
        // one of __consume or __abandon must have set __callback_ before winning
        // the race to set __registeredReceiver_
        assert(__callback_ != nullptr);

        // either __consume registered receiver to be completed by us, or __abandon
        // has finished invoking __stopSource.request_stop() and we're about to invoke
        // __destroy through the callback it registered
        //
        // we could tell which we're about to do by comparing `receiver` to `this`; if
        // they're equal then we're about to invoke destroy, otherwise, we're about to
        // undo the type-erasure of receiver and pass it to __do_consume. we don't check
        // on the assumption that it's better to do an unconditional indirect call than
        // a conditional one
        __callback_(this, receiver);
      }

      template <receiver _Rcvr>
      void __consume(_Rcvr& __rcvr) noexcept {
        __callback_ = [](__spawn_future_state* __self, void* __ptr) noexcept {
          auto& __rcvr = *static_cast<_Rcvr*>(__ptr);
          __self->__do_consume(__rcvr);
        };

        void* sentinel = nullptr;

        if (__registeredReceiver_.compare_exchange_strong(
              sentinel, std::addressof(__rcvr), __std::memory_order_acq_rel)) {
          return;
        }

        // if the CAS failed then __complete has already "registered" the sentinel value
        assert(sentinel == this);

        __do_consume(__rcvr);
      }

      void __abandon() noexcept {
        // TODO: is relaxed OK when the loaded value isn't nullptr?
        auto* sentinel = __registeredReceiver_.load(__std::memory_order_relaxed);

        if (sentinel == nullptr) {
          // __complete hasn't happened yet
          __stopSource_.request_stop();
          __callback_ = [](
                          __spawn_future_state* __self,
                          [[maybe_unused]]
                          void* __sentinel) noexcept {
            assert(__sentinel == __self);
            __self->__destroy();
          };

          if (__registeredReceiver_
                .compare_exchange_strong(sentinel, this, __std::memory_order_acq_rel)) {
            // callback registered for later
            return;
          }
        }

        // __complete happened, possibly between the load and CAS

        // the opstate unconditionally calls __abandon, possibly after calling __consume,
        // so the only thing we know about sentinel here is that it's not nullptr; it could
        // be either `this` or the address of the receiver that was passed to __consume
        assert(sentinel != nullptr);

        __destroy();
      }

     private:
      using __assoc_t = std::remove_cvref_t<decltype(__declval<_Token&>().try_associate())>;

      _Alloc __alloc_;
      inplace_stop_source __stopSource_;
      __op_t __op_;
      __assoc_t __assoc_;
      // type-erased receiver; three possible values:
      //   1. `nullptr` means "unset"
      //   2. `this` means either __complete has been invoked or __abandon has been invoked
      //   3. any other value means __consume has "registered" its receiver to be completed
      //      by __complete when it is invoked
      __std::atomic<void*> __registeredReceiver_{nullptr};
      // type-erased completion callback; the void* will receive the address of the
      // receiver if we're completing "for real" or `this` if __complete is responsible
      // for invoking __destroy because __abandon was invoked before __complete was
      void (*__callback_)(__spawn_future_state*, void*) noexcept;

      void __destroy() noexcept {
        [[maybe_unused]]
        auto assoc = std::move(__assoc_);

        {
          using traits =
            std::allocator_traits<_Alloc>::template rebind_traits<__spawn_future_state>;
          typename traits::allocator_type alloc(std::move(__alloc_));
          traits::destroy(alloc, this);
          traits::deallocate(alloc, this, 1);
        }
      }

      void __do_consume(receiver auto& __rcvr) noexcept {
        __visit(
          [&__rcvr](auto&& __tuple) noexcept {
            if constexpr (!__same_as<std::remove_reference_t<decltype(__tuple)>, std::monostate>) {
              __apply(
                [&__rcvr](auto cpo, auto&&... __vals) {
                  cpo(std::move(__rcvr), std::move(__vals)...);
                },
                std::move(__tuple)); // NOLINT(bugprone-move-forwarding-reference)
            }
          },
          std::move(this->__result_));
      }
    };

    struct spawn_future_t {
      template <sender _Sender, scope_token _Token>
      auto operator()(_Sender&& __sndr, _Token&& __tkn) const -> __well_formed_sender auto {
        return (*this)(static_cast<_Sender&&>(__sndr), static_cast<_Token&&>(__tkn), env<>{});
      }

      template <sender _Sender, scope_token _Token, class _Env>
      auto operator()(_Sender&& __sndr, _Token&& __tkn, _Env&& __env) const -> __well_formed_sender
        auto {
        return impl(
          __tkn.wrap(static_cast<_Sender&&>(__sndr)),
          static_cast<_Token&&>(__tkn),
          static_cast<_Env&&>(__env));
      }

     private:
      template <sender _Sender, scope_token _Token, class _Env>
      auto impl(_Sender&& __sndr, _Token&& __tkn, _Env&& __env) const {
        using alloc_t = decltype(__spawn_common::__choose_alloc(__env, get_env(__sndr)));
        using senv_t = decltype(__spawn_common::__choose_senv(__env, get_env(__sndr)));

        using spawn_future_state_t =
          __spawn_future_state<alloc_t, std::remove_cvref_t<_Token>, _Sender, senv_t>;

        using traits = std::allocator_traits<alloc_t>::template rebind_traits<spawn_future_state_t>;
        typename traits::allocator_type alloc(
          __spawn_common::__choose_alloc(__env, get_env(__sndr)));

        auto* op = traits::allocate(alloc, 1);

        __scope_guard __guard{[&]() noexcept { traits::deallocate(alloc, op, 1); }};

        traits::construct(
          alloc,
          op,
          alloc,
          static_cast<_Sender&&>(__sndr),
          static_cast<_Token&&>(__tkn),
          __spawn_common::__choose_senv(__env, get_env(__sndr)));

        __guard.__dismiss();

        struct abandoner {
          void operator()(spawn_future_state_t* __p) const noexcept {
            __p->__abandon();
          }
        };

        return __make_sexpr<spawn_future_t>(std::unique_ptr<spawn_future_state_t, abandoner>(op));
      }
    };

    struct __spawn_future_impl : __sexpr_defaults {
      template <class _Sender>
      using __completions_t =
        __data_of<std::remove_cvref_t<_Sender>>::element_type::__completions_t;

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() //
        -> __completions_t<_Sender> {
        return {};
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver __rcvr) noexcept /* TODO */ {
          auto& [__tag, __future] = __sndr;
          return std::pair{std::move(__future), std::move(__rcvr)};
        };

      static constexpr auto start = [](auto& __state) noexcept {
        auto& [__future, __rcvr] = __state;
        __future->__consume(__rcvr);
      };
    };
  } // namespace __spawn_future

  using __spawn_future::spawn_future_t;

  /// @brief The spawn_future sender adaptor
  /// @hideinitializer
  inline constexpr spawn_future_t spawn_future{};

  template <>
  struct __sexpr_impl<spawn_future_t> : __spawn_future::__spawn_future_impl { };
} // namespace STDEXEC
