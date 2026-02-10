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
      static constexpr bool __is_nothrow_storable = __nothrow_decay_copyable<_Args...>;

      using __decayed_sig = _Tag(__decay_t<_Args>...);
    };

    // [exec.spawn.future] paragraph 4
    template <class _Sig>
    using __as_tuple = __mapply_q<__decayed_tuple, _Sig>;

    template <class _Sig>
    using __decayed_sig = __future_sig_fns<_Sig>::__decayed_sig;

    template <class... _Sigs>
    inline constexpr bool __sigs_are_nothrow_storable =
      (__future_sig_fns<_Sigs>::__is_nothrow_storable && ...);

    template <bool _NothrowStorable, class... _Sigs>
    struct __future_variant {
      // this case handles _NothrowStorable == true
      using type = __uniqued_variant<__decayed_tuple<set_stopped_t>, __as_tuple<_Sigs>...>;
    };

    template <class... _Sigs>
    struct __future_variant<false, _Sigs...> {
      using type = __uniqued_variant<
        __decayed_tuple<set_stopped_t>,
        __decayed_tuple<set_error_t, std::exception_ptr>,
        __as_tuple<_Sigs>...
      >;
    };

    template <class... _Sigs>
    using __future_variant_t =
      // [exec.spawn.future] paragraphs 4.1 and 4.2
      __future_variant<__sigs_are_nothrow_storable<_Sigs...>, _Sigs...>::type;

    struct __try_cancelable {
      explicit __try_cancelable(void (*__try_cancel)(__try_cancelable*) noexcept) noexcept
        : __try_cancel_(__try_cancel) {
      }

      __try_cancelable(__try_cancelable&&) = delete;

      void __try_cancel() noexcept {
        __try_cancel_(this);
      }

     protected:
      ~__try_cancelable() = default;

     private:
      void (*__try_cancel_)(__try_cancelable*) noexcept;
    };

    // this is the stop callback registered by a future (i.e. the result of spawn_future) when it is
    // connected and started; once registered, it ensures that stop requests delivered to the started
    // future are forwarded into the spawned work
    struct __future_stop_callback {
      __try_cancelable* __self_;

      void operator()() noexcept {
        __self_->__try_cancel();
      }
    };

    template <bool _NothrowSyncWork, class... _Sigs>
    struct __future_completions {
      // this case handles _NothrowSyncWork == true
      // this case is applicable when both conditions are true:
      //  - the results of all possible completions can be decay-copied into the decayed-tuple that
      //    stores the results for later consumption by the future; and
      //  - the stop token provided by the future's receiver can no-throw-construct a stop callback
      //    in the future's operation state
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

    template <class _Env>
    inline constexpr bool __stop_callback_is_nothrow_constructible = __nothrow_constructible_from<
      stop_callback_for_t<stop_token_of_t<_Env>, __future_stop_callback>,
      stop_token_of_t<_Env>,
      __future_stop_callback
    >;

    template <class _Env, class... _Sigs>
    using __future_completions_t = __future_completions<
      __stop_callback_is_nothrow_constructible<_Env> && __sigs_are_nothrow_storable<_Sigs...>,
      _Sigs...
    >::type;

    // [exec.spawn.future] paragraph 3
    template <class _Completions>
    struct __spawn_future_state_base;

    template <class... _Sigs>
    struct __spawn_future_state_base<completion_signatures<_Sigs...>> : __try_cancelable {
      using __variant_t = __future_variant_t<_Sigs...>;
      template <class _Env>
      using __completions_t = __future_completions_t<_Env, _Sigs...>;

      __variant_t __result_{__no_init};

      explicit __spawn_future_state_base(
        void (*__try_cancel)(__try_cancelable*) noexcept,
        void (*__complete)(__spawn_future_state_base*) noexcept) noexcept
        : __try_cancelable(__try_cancel)
        , __complete_(__complete) {
      }

      __spawn_future_state_base(__spawn_future_state_base&&) = delete;

      void __complete() noexcept {
        __complete_(this);
      }

     protected:
      ~__spawn_future_state_base() = default;

     private:
      void (*__complete_)(__spawn_future_state_base*) noexcept;
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
        constexpr bool __non_throwing = (__nothrow_constructible_from<__decay_t<_T>, _T> && ...);

        try {
          __state_->__result_
            .template emplace<__decayed_tuple<_CPO, _T...>>(_CPO{}, static_cast<_T&&>(__t)...);
        } catch (...) {
          if constexpr (!__non_throwing) {
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
          // but that's hard to justify--the future-spawned-sender will be connected to a receiver
          // with an empty environment after all. This code works; I don't understand why the extra
          // env type changes the result, but this is a reasonably small change we can make to the
          // spec to bring things into alignment.
          completion_signatures_of_t<__future_spawned_sender<_Sender, _Env>, env<>>
        > {
      using __sigs_t =
        // this is "wrong" in the same way as the above
        completion_signatures_of_t<__future_spawned_sender<_Sender, _Env>, env<>>;

      using __receiver_t = __spawn_future_receiver<__sigs_t>;

      using __op_t = connect_result_t<__future_spawned_sender<_Sender, _Env>, __receiver_t>;

      using __base = __spawn_future_state_base<completion_signatures_of_t<_Sender, _Env>>;

      __spawn_future_state(_Alloc __alloc, _Sender&& __sndr, _Token __token, _Env __env)
        : __base(__do_try_cancel, __do_complete)
        , __alloc_(std::move(__alloc))
        , __op_(
            STDEXEC::connect(
              write_env(
                __stop_when(static_cast<_Sender&&>(__sndr), __stop_source_.get_token()),
                std::move(__env)),
              __receiver_t(this)))
        , __assoc_(__token.try_associate()) {
        if (__assoc_) {
          STDEXEC::start(__op_);
        } else {
          STDEXEC::set_stopped(__receiver_t(this));
        }
      }

      // NOTE: _Rcvr is unconstrained because the thing we pass doesn't satisfy receiver
      template <class _Rcvr>
      void __consume(_Rcvr& __rcvr) noexcept {
        // Write this before synchronizing with the producer, below.
        __callback_ = [](__spawn_future_state* __self, void* __ptr) noexcept {
          auto& __rcvr = *static_cast<_Rcvr*>(__ptr);
          if (__self != nullptr) {
            __self->__do_consume(__rcvr);
          } else {
            STDEXEC::set_stopped(std::move(__rcvr));
          }
        };

        void* __sentinel = nullptr;
        if (__registered_receiver_.compare_exchange_strong(
              __sentinel,
              std::addressof(__rcvr),
              // We need store-release on success to ensure that the future completion of the
              // producer can see the callback we wrote into __callback_, and we need load-acquire
              // on failure in case we're about to observe that the producer has already finished
              // so we can see the result it produced. The success order must be stronger than the
              // failure order so success has to be acquire-release.
              __std::memory_order_acq_rel,
              __std::memory_order_acquire)) {
          // Since our CAS succeeded, we can conclude that we observed a null __registered_receiver_
          // and successfully updated it to point to the receiver. That means the consumer has
          // successfully registered a receiver and it's up to the producer to complete it when the
          // result is ready; alternatively, a stop request may arrive leading __try_cancel to try
          // to complete us eagerly with set_stopped. In either case, __try_cancel and __complete
          // will negotiate how to complete the future and we have nothing left to do.
          return;
        }

        if (__sentinel == (this + 1)) { // NOTE: we didn't update __registered_receiver_
          // __try_cancel ran before both __complete and __consume; now we need to negotiate with
          // __complete to decide whether it finished in time to consume its output.
          //
          // We need acquire-release semantics here. If we succeed in abandoning the operation then
          // the producer will be responsible for invoking __destroy, which means it needs to see
          // the write to __callback_, which requires a store-release. IF we fail to abandon the
          // operation then that means the producer finished in time for us to consume its result,
          // which means we need a load-acquire to consume it properly.
          __sentinel = __registered_receiver_.exchange(this, __std::memory_order_acq_rel);
        }

        if (__sentinel == this) {
          // Either the producer completed before we CAS'd, or it snuck in and completed between
          // our CAS and our exchange; in either case, we ought to consume its result.
          __do_consume(__rcvr);
          __destroy();
        } else {
          STDEXEC_ASSERT(__sentinel == (this + 1));
          // Our exchange observed the same (this + 1) that the CAS did, which means that the producer
          // didn't finish in time; also, by setting __registered_receiver_ to (this), we've marked the
          // operation as "ready for destruction" by the producer so the current object may already be
          // destroyed. We must complete with set_stopped.
          STDEXEC::set_stopped(std::move(__rcvr));
        }
      }

      void __abandon() noexcept {
        // We're about to mark the consuming side as complete, at which point the producing side is
        // free to destroy this object so we can't "optimize" by deferring this stop request
        __stop_source_.request_stop();

        // We need store-release semantics if we happen to be completing the consumer side before
        // the producer side has finished because we need to publish the writes committed in the
        // above call to request_stop for the destructor not to commit a data race. We need
        // load-acquire semantics if we happen to be second to consume the writes done by the
        // producer so the destructor can destroy that data without committing a data race. In
        // combination, we need acquire-release semantics here.
        void* __sentinel = __registered_receiver_.exchange(this, __std::memory_order_acq_rel);

        if (__sentinel == nullptr) {
          // The producer hadn't finished by the time we marked the consumer as done so we've handed
          // over clean-up responsibility and have nothing else to do.
          return;
        } else {
          STDEXEC_ASSERT(__sentinel == this);
          // The producer side completed before we did so we're responsible for clean-up.
          __destroy();
        }
      }

     private:
      using __assoc_t = std::remove_cvref_t<decltype(__declval<_Token&>().try_associate())>;

      _Alloc __alloc_;
      inplace_stop_source __stop_source_;
      __op_t __op_;
      __assoc_t __assoc_;
      // Type-erased receiver. Several possible values:
      //   1. `nullptr` means "unset"
      //   2. `this` means either the producer or consumer is done with the operation and the
      //      other (whichever hasn't completed yet) is responsible for clean-up
      //   3. `this` + 1 means that the future has received a stop request and __try_cancel has
      //      marked the operation so that __complete and __consume can negotiate how to complete
      //   4. any other value means __consume has "registered" its receiver to be completed
      //      by __complete when it is invoked
      __std::atomic<void*> __registered_receiver_{nullptr};
      // Type-erased completion callback.
      //
      // The void* will receive the address of the receiver, which will need to have its
      // type unerased. The __spawn_future_state* will receive either `this`, indicating the
      // callback ought to complete the receive with the value of __result_, which can be
      // retrieved through the self-pointer, or nullptr, indicating that the receiver should
      // be completed with set_stopped because the future received and processed a stop request
      // before the producer could finish.
      void (*__callback_)(__spawn_future_state*, void*) noexcept;

      void __destroy() noexcept {
        [[maybe_unused]]
        auto __assoc = std::move(__assoc_);

        {
          using __traits =
            std::allocator_traits<_Alloc>::template rebind_traits<__spawn_future_state>;
          typename __traits::allocator_type __alloc(std::move(__alloc_));
          __traits::destroy(__alloc, this);
          __traits::deallocate(__alloc, this, 1);
        }
      }

      // NOTE: __rcvr's type is unconstrained because the thing we pass doesn't satisfy receiver
      void __do_consume(auto& __rcvr) noexcept {
        __visit(
          [&__rcvr](auto&& __tuple) noexcept {
            __apply(
              [&__rcvr](auto cpo, auto&&... __vals) {
                cpo(std::move(__rcvr), std::move(__vals)...);
              },
              std::move(__tuple)); // NOLINT(bugprone-move-forwarding-reference)
          },
          std::move(this->__result_));
      }

      static void __do_complete(__base* __base_ptr) noexcept {
        auto* __self = static_cast<__spawn_future_state*>(__base_ptr);

        // Consider: it'd be nice to eagerly destruct __op_ here; to do that, we'd need to store
        //           it in an anonymous union, use a scope-guard in the constructor to ensure it
        //           gets destructed in the event that try_associate() throws, and manually destroy
        //           it here *before* updating __registered_receiver_. I'm not sure it would be to
        //           spec to do that, though.

        // We need acquire-release semantics here to ensure correct synchronization whether we
        // arrived before or after the consumer. In the case we finish first, we need to store-release
        // so the consumer can see what we've written; in the case we finish second, we need to
        // load-acquire so we can see what the consumer has written.
        void* __sentinel = __self->__registered_receiver_
                             .exchange(__self, __std::memory_order_acq_rel);

        if (__sentinel == nullptr) { // NOLINT(bugprone-branch-clone)
          // The producer side has completed first and we've updated __registered_receiver_ with
          // (__self) to mark things as such; the consumer side is responsible for all further
          // actions.
          return;
        } else if (__sentinel == __self) {
          // There are two possible histories here; the consumer side either
          //  1. abandoned the operation without ever starting, or
          //  2. was started but received a stop request and successfully cancelled before we
          //     could produce our value.
          //
          // In either case, the producer is "too late" and there's nothing to do but clean up.
          __self->__destroy();
        } else if (__sentinel == (__self + 1)) {
          // The consumer side has been started and a stop request was received before __consume
          // could be invoked; the producer has also completed before __consume could be invoked
          // and we've left (__self) in place of (__self + 1) so, when __consume gets around to
          // observing __registered_receiver_, it'll see that the producer has finished and complete
          // the receiver with the value we produced. This state is similar to the producer having
          // finished first; by overwriting __registered_receiver_ with (__self), we've erased the
          // fact that the stop request happened to come in before we did.
          return;
        } else {
          // The producer finished after the consumer register a receiver for us to complete. There
          // may be an incoming or outstanding stop request that causes __try_start to race with us
          // but, if so, we don't need to worry about it--invoking __callback_ like __self will
          // eagerly destroy the stop callback, which will either prevent __try_cancel from being
          // invoked, or will block until it returns. In either case, our call to __callback_ has
          // "won" and __try_cancel will no-op.
          __self->__callback_(__self, __sentinel);

          // Having completed the receiver, we are responsible for cleaning up the allocated state.
          __self->__destroy();
        }
      }

      static void __do_try_cancel(__try_cancelable* __base_ptr) noexcept {
        auto* __self = static_cast<__spawn_future_state*>(__base_ptr);

        // Consider: there's a sense in which we only need to invoke request_stop if we've arrived
        //           here before the producer has invoked __complete so I wonder whether it's possible
        //           to avoid invoking request_stop when it's unnecessary. It might be.
        //
        //           We *must* invoke request_stop before returning from the function if our CAS
        //           succeeds (because we arrived before either __complete or __consume), or if the
        //           CAS fails and reports that the consumer finished first and the later exchange
        //           observes that the producer still hasn't finished.
        //
        //           The most obvious risk to deferring a call to request_stop is that the stop source
        //           might be destroyed before our invocation. There's also a risk that we get our
        //           memory ordering wrong; I've learned through analyzing TSAN failures that
        //           request_stop constitutes a store-release that must be load-acquired before the
        //           destructor runs so we need to be careful about synchronizing with the final
        //           owner of this object.
        //
        //           It might be safe to invoke request_stop before the early return when the CAS
        //           succeeds and between a failed CAS and the later exchange. In the first case
        //           (after a successful CAS), the final owner of the object will be the slower of
        //           the producer or consumer, which will complete the receiver before destroying the
        //           operation. Completing the receiver will entail destroying the stop callback that
        //           has invoked us, which will synchronize with us. In the second case (after a
        //           failed CAS and before the subsequent exchange), we know that the consumer has
        //           already finished and it's the producer that will destroy the operation. We'll
        //           synchronize with the producer either by invoking a store-release before
        //           completing the receiver, or the producer will complete the receiver and destroy
        //           the stop callback.
        //
        //           Upon further reflection, the above analysis is flawed: invoking request_stop
        //           after a successful CAS would lead to data races when the consumer successfully
        //           abandons the operation (i.e. it observes the (__self + 1) value and successfully
        //           replaces it with (__self) before completing the receiver with set_stopped). In
        //           that scenario, the consumer thread would be responsible for destroying the stop
        //           callback, not the producer thread, and there would be no other store-release
        //           to publish the writes from request_stop to the producer thread before it runs
        //           the operation's destructor.
        //
        //           Perhaps we can salvage the effort by deferring the request_stop to __complete.
        //           When our CAS succeeds, we know that we've run before either of the producer or
        //           consumer, which means a call to __consume is still in the future. If that call
        //           observes (__self + 1) then we know that both __try_cancel and __consume have run
        //           before __complete, and the consumer is going to try to abandon the producer's
        //           work. It would be sensible for the consumer to issue the stop request before it
        //           performs the exchange(__self, acq_rel) that is its attempt to so signal and the
        //           store-release entailed by that exchange would synchronize with the producer as
        //           required.

        // It feels wasteful to do this when we might not need to but, once we update
        // __registered_receiver_ to (__self + 1), there's a chance the current object
        // will be destroyed so it's not safe to defer this.
        //
        // This operation constitutes a write that must be published to whichever thread invokes
        // __destroy.
        __self->__stop_source_.request_stop();

        // If neither __complete nor __consume has been invoked then mark us as having stop requested.
        // This only succeeds if stop is requested very early in the process so it's quite unlikely.
        void* __sentinel = nullptr;
        if (__self->__registered_receiver_.compare_exchange_strong(
              __sentinel,
              __self + 1,
              // We need to store-release on success so that whichever of the consumer or producer
              // ends up invoking __destroy sees the write we performed in request_stop(); we need
              // load-acquire on failure so we can safely read the value of __callback_ that may have
              // been written by the consumer.
              __std::memory_order_acq_rel,
              __std::memory_order_acquire)) {
          // We succeeded, meaning that __registered_receiver_ was nullptr on entry, which signals
          // that __try_cancel ran before either of __consume or __complete. Leaving (__self + 1)
          // as the sentinel will allow the two forthcoming functions to negotiate how to
          // complete the overall operation.
          return;
        }

        if (__sentinel == __self) {
          // __complete has already finished and left a value behind; we should let __consume
          // consume it.
          //
          // We would normally need to perform a store-release here to ensure the consumer sees
          // the write we performed by invoking request_stop, but we're running inside a stop
          // callback that the consumer will destroy before destroying this object and destroying
          // a stop callback is a synchronizing operation so we're good.
          return;
        }

        // __consume has registered a receiver and __sentinel is its address; the producer may invoke
        // __complete at any moment, which would ordinarily risk destroying the current object out
        // from under us but, because we're running in a stop callback, the completion path will
        // block until we're done before doing that so we're still safe.
        //
        // Do two things:
        //  1. store a copy of __callback_ on the stack so we can use it after the operation is
        //     destroyed, and
        //  2. try to mark the operation as abandoned so that the producer can take clean-up
        //     responsibility.

        const auto __rcvr = __sentinel;
        auto __cb = __self->__callback_;

        // We have already synchronized with the consumer by performing a load-acquire above (and
        // we know it was the *consumer* we synchronized with because __sentinel contains the address
        // of the receiver it registered), but we haven't yet synchronized with the producer.
        //
        // There are two possible futures here:
        //  1. we mark the operation as abandoned before the producer finishes, or
        //  2. the producer finishes before we can abandon the operation.
        //
        // In the first case, we can synchronize with the producer by performing a store-release on
        // __registered_receiver_; when the producer completes, it will load-acquire that value and
        // clean up.
        //
        // In the second case, the producer will complete the registered receiver, which will have
        // the side effect of destroying the stop callback that we're running inside, which is a
        // synchronization point with us.
        //
        // So, we store-release here in case we're in case 1.
        __sentinel = __self->__registered_receiver_.exchange(__self, __std::memory_order_release);

        if (__sentinel == __rcvr) {
          // __registered_receiver_ still contained the value we observed during the CAS, which means
          // the producer still hadn't updated it to contain (__self). This means we succeeded in
          // marking the operation as abandoned and the producer will destroy it when it completes
          // (which could happen at any moment); we need to complete the consumer with set_stopped. We
          // invoke __cb (the copy of __callback_ that we put on the stack) with a null self-pointer
          // to signal that it ought to invoke set_stopped(std::move(*__rcvr)) without touching the
          // operation.
          __cb(nullptr, __rcvr);
        } else {
          STDEXEC_ASSERT(__sentinel == __self);
          // The producer beat us to the punch; it's busy trying to complete and is about to
          // destroy the operation. Bail out.
        }
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
        return __impl(
          __tkn.wrap(static_cast<_Sender&&>(__sndr)),
          static_cast<_Token&&>(__tkn),
          static_cast<_Env&&>(__env));
      }

     private:
      template <sender _Sender, scope_token _Token, class _Env>
      auto __impl(_Sender&& __sndr, _Token&& __tkn, _Env&& __env) const {
        using __alloc_t = decltype(__spawn_common::__choose_alloc(__env, get_env(__sndr)));
        using __senv_t = decltype(__spawn_common::__choose_senv(__env, get_env(__sndr)));

        using __spawn_future_state_t =
          __spawn_future_state<__alloc_t, std::remove_cvref_t<_Token>, _Sender, __senv_t>;

        using __traits =
          std::allocator_traits<__alloc_t>::template rebind_traits<__spawn_future_state_t>;
        typename __traits::allocator_type __alloc(
          __spawn_common::__choose_alloc(__env, get_env(__sndr)));

        auto* __op = __traits::allocate(__alloc, 1);

        __scope_guard __guard{[&]() noexcept { __traits::deallocate(__alloc, __op, 1); }};

        __traits::construct(
          __alloc,
          __op,
          __alloc,
          static_cast<_Sender&&>(__sndr),
          static_cast<_Token&&>(__tkn),
          __spawn_common::__choose_senv(__env, get_env(__sndr)));

        __guard.__dismiss();

        struct __abandoner {
          void operator()(__spawn_future_state_t* __p) const noexcept {
            __p->__abandon();
          }
        };

        return __make_sexpr<spawn_future_t>(
          std::unique_ptr<__spawn_future_state_t, __abandoner>(__op));
      }
    };

    template <class _Sender>
    struct __future_operation_base {
      // __data_of<_Sender> is a unique_ptr specialization
      using __unique_ptr_t = __data_of<std::remove_cvref_t<_Sender>>;

      explicit __future_operation_base(__unique_ptr_t&& __future) noexcept
        : __future_(std::move(__future)) {
      }

     protected:
      __unique_ptr_t __future_;
    };

    template <class _Sender, class _Receiver>
    struct __future_operation : __future_operation_base<_Sender> {
      using __base = __future_operation_base<_Sender>;

      __future_operation(__base::__unique_ptr_t&& __future, _Receiver __rcvr) noexcept
        : __future_operation_base<_Sender>(std::move(__future))
        , __rcvr_(std::move(__rcvr)) {
      }

      ~__future_operation() {
      }

      void __run() noexcept(__stop_callback_is_nothrow_constructible<env_of_t<_Receiver>>) {
        // this might throw
        std::construct_at(
          &__callback_,
          STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_)),
          __future_stop_callback{this->__future_.get()});

        // this is no-throw
        this->__future_.release()->__consume(__inner_rcvr_);
      }

      _Receiver __rcvr_;

     private:
      struct __receiver {
        using receiver_concept = STDEXEC::receiver_t;

        template <class... _Ts>
        void set_value(_Ts&&... __ts) && noexcept {
          __op_->__complete<STDEXEC::set_value_t>(static_cast<_Ts&&>(__ts)...);
        }

        template <class _E>
        void set_error(_E&& __e) && noexcept {
          __op_->__complete<STDEXEC::set_error_t>(static_cast<_E&&>(__e));
        }

        void set_stopped() && noexcept {
          __op_->__complete<STDEXEC::set_stopped_t>();
        }

        __future_operation* __op_;
      };

      __receiver __inner_rcvr_{this};
      // __callback_ is left unconstructed until __run() is called; if the constructor invocation
      // throws then __callback_ is never constructed or destructed at all, otherwise, its destructor
      // is invoked when the receiver contract is completed in __complete, below.
      union {
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>>, __future_stop_callback>
          __callback_;
      };

      template <class _CPO, class... _Ts>
      void __complete(_Ts&&... __ts) noexcept {
        std::destroy_at(std::addressof(__callback_));
        _CPO{}(std::move(__rcvr_), static_cast<_Ts&&>(__ts)...);
      }
    };

    struct __spawn_future_impl : __sexpr_defaults {
      // __data_of<_Sender> is a unique_ptr specialization
      template <class _Sender>
      using __unique_ptr_t = __data_of<std::remove_cvref_t<_Sender>>;

      template <class _Sender>
      using __spawn_future_state_t = __unique_ptr_t<_Sender>::element_type;

      template <class _Sender, class _Env>
      using __completions_t = __spawn_future_state_t<_Sender>::template __completions_t<_Env>;

      template <class _Sender, class _Env>
      static consteval auto get_completion_signatures() //
        -> __completions_t<_Sender, _Env> {
        return {};
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver __rcvr) noexcept /* TODO */
        -> __future_operation<std::remove_cvref_t<_Sender>, _Receiver> {
        auto& [_, __future] = __sndr;
        return {std::move(__future), std::move(__rcvr)};
      };

      static constexpr auto start = [](auto& __state) noexcept {
        constexpr bool __non_throwing = noexcept(__state.__run());

        try {
          __state.__run();
        } catch (...) {
          if constexpr (!__non_throwing) {
            STDEXEC::set_error(std::move(__state.__rcvr_), std::current_exception());
          }
        }
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
