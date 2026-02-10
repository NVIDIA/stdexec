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
#include "__concepts.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"
#include "__sender_concepts.hpp"
#include "__stop_when.hpp"

#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace STDEXEC {
  struct _JOINING_A_COUNTING_SCOPE_NEEDS_A_SCHEDULER_IN_THE_ENVIRONMENT_ { };

  namespace __counting_scopes {
    struct __base_scope;

    // a non-template base class for __join_state
    //
    // here, we implement the intrusive singly-linked list of join operations
    // that have been "registered" with the scope they're joining
    struct __join_state_base {
      // we only ever need one of these pointers at a time
      union {
        // the scope we're joining; necessary during the execution of start()
        // to reach into the scope and ask whether the join operation can
        // complete inline or needs to be "registered" for deferred completion
        __base_scope* __scope_;
        // if we end up registered for deferred execution then this pointer
        // points to the next operation in the list of registered operations
        __join_state_base* __next_;
      };

      explicit __join_state_base(
        __base_scope* __scope,
        void (*__complete_fn)(__join_state_base*) noexcept) noexcept
        : __scope_(__scope)
        , __complete_(__complete_fn) {
      }

      void __complete() noexcept {
        __complete_(this);
      }

     private:
      // the implementation of deferred completion of this operation
      //
      // store a pointer-to-function instead of using a virtual
      // to avoid RTTI overhead
      void (*__complete_)(__join_state_base*) noexcept;
    };

    struct __scope_join_t { };

    /////////////////////////////////////////////////////////////////////////////
    // [exec.counting.scopes.general] paragraph 4
    struct __scope_join_impl : __sexpr_defaults {
      template <class _Env>
      using __scheduler_of_t = __call_result_t<get_scheduler_t, const _Env&>;

      template <class _Env>
      using __sched_sender_of_t = __call_result_t<schedule_t, __scheduler_of_t<_Env>>;

      template <class _Sender, class _Env>
      static consteval auto get_completion_signatures() {
        if constexpr (__callable<get_scheduler_t, const _Env&>) {
          return STDEXEC::get_completion_signatures<__sched_sender_of_t<_Env>, _Env>();
        } else {
          return STDEXEC::__throw_compile_time_error<
            _WHAT_(_JOINING_A_COUNTING_SCOPE_NEEDS_A_SCHEDULER_IN_THE_ENVIRONMENT_),
            _WHY_(_THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_),
            _WHERE_(STDEXEC::_IN_ALGORITHM_, __scope_join_t),
            _WITH_PRETTY_SENDER_<_Sender>,
            _WITH_ENVIRONMENT_(_Env)
          >();
        }
      };

      template <class _Rcvr>
      struct __join_state : __join_state_base {
        struct __rcvr_t {
          using receiver_concept = receiver_t;

          _Rcvr& __rcvr_;

          void set_value() && noexcept {
            STDEXEC::set_value(std::move(__rcvr_));
          }

          template <class _Error>
          void set_error(_Error&& __err) && noexcept {
            STDEXEC::set_error(std::move(__rcvr_), static_cast<_Error&&>(__err));
          };

          void set_stopped() && noexcept {
            STDEXEC::set_stopped(std::move(__rcvr_));
          }

          [[nodiscard]]
          auto get_env() const noexcept -> env_of_t<_Rcvr> {
            return STDEXEC::get_env(__rcvr_);
          }
        };

        using __sched_sender = __sched_sender_of_t<env_of_t<_Rcvr>>;

        using __op_t = connect_result_t<__sched_sender, __rcvr_t>;

        _Rcvr __rcvr_;
        __op_t __op_;

        explicit __join_state(__base_scope* __scope, _Rcvr __rcvr)
          noexcept(__nothrow_callable<connect_t, __sched_sender, __rcvr_t>)
          : __join_state_base(
              __scope,
              [](__join_state_base* __base) noexcept {
                auto* __self = static_cast<__join_state*>(__base);
                // if the schedule-sender is no-throw connectable with __rcvr_ then
                // we could save some storage by deferring connection to this point
                STDEXEC::start(__self->__op_);
              })
          , __rcvr_(std::move(__rcvr))
          , __op_(
              STDEXEC::connect(
                schedule(get_scheduler(STDEXEC::get_env(__rcvr_))),
                __rcvr_t(__rcvr_))) {
        }

        __join_state(__join_state&&) = delete;

        ~__join_state() = default;

        void __complete_inline() noexcept {
          set_value(std::move(__rcvr_));
        }
      };

      static constexpr auto get_state =
        []<class _Sndr, class _Rcvr>(_Sndr&& __sender, _Rcvr __rcvr) noexcept(
          __nothrow_constructible_from<__join_state<_Rcvr>, __base_scope*, _Rcvr>) {
          auto [_, __scope] = __sender;
          return __join_state<_Rcvr>(__scope, std::move(__rcvr));
        };

      static constexpr auto start = [](auto& __state) noexcept {
        if (__state.__scope_->__start_join_sender(__state)) {
          __state.__complete_inline();
        }
      };
    };

    /////////////////////////////////////////////////////////////////////////////
    // [exec.counting.scopes.general] paragraph 5
    //
    // this class template doesn't really *need* to know the concrete type of
    // the scope it's associated with so it's tempting to refactor this into
    // a non-templated struct that operates in terms of a __base_scope*, but
    // keeping it this way ensures that two associations from distinct scopes
    // can't be confused with each other
    template <class _Scope>
    struct __association_t {
      __association_t() = default;

      explicit operator bool() const noexcept {
        // [exec.counting.scopes.general] paragraph 5.2
        return __scope_ != nullptr;
      }

      [[nodiscard]]
      __association_t try_associate() const noexcept {
        // [exec.counting.scopes.general] paragraph 5.3
        if (__scope_) {
          return __scope_->__try_associate();
        } else {
          return __association_t();
        }
      }

     private:
      // the scope needs access to the private constructor
      friend _Scope;

      struct __disassociater {
        void operator()(_Scope* __p) const noexcept {
          // [exec.counting.scopes.general] paragraph 5.4
          __p->__disassociate();
        }
      };

      // [exec.counting.scopes.general] paragraph 5.1
      std::unique_ptr<_Scope, __disassociater> __scope_;

      explicit __association_t(_Scope& __scope) noexcept
        : __scope_(std::addressof(__scope)) {
      }
    };

    struct __base_scope {
      // we represent the (count, state) pair in a single std::size_t by allocating
      // the lower three bits to state, leaving all the rest of the bits for count;
      // the result is that we can count up to MAX_SIZE_T >> 3 outstanding ops.
      //
      // The extra parens around the function name are to work around windows.h defining
      // a function-like macro named max.
      static constexpr std::size_t max_associations = (std::numeric_limits<std::size_t>::max)()
                                                   >> 3;

      __base_scope() = default;

      // [exec.simple.counting.ctor] paragraph 1
      // Postconditions: count is 0 and state is unused.
      //
      // NOTE: we rely on the __bits_ initializer to meet the postconditions
      __base_scope(__base_scope&&) = delete;

      ~__base_scope() {
        // [exec.simple.counting.ctor] paragraph 2
        // Effects: If state is not one of joined, unused, or unused-and-closed, invokes
        // terminate. Otherwise, has no effects.
        //
        // NOTE: we check the termination conditions in __destructible()
        //
        // Regarding memory ordering, there are three cases to consider here:
        //  1. we're about to terminate, in which case memory ordering is irrelevant;
        //  2. the scope is unused, which means there were never any associated operations
        //     to synchronize with, which again means the ordering doesn't matter; or
        //  3. the scope was used and has been joined.
        //
        // In the third case, any threads that completed a join-sender have synchronized
        // with all the now-completed associated operations but the scope may be destroyed
        // on yet another thread so we ought to execute a load-acquire to ensure the
        // current thread has properly synchronized.
        auto __bits = __bits_.load(__std::memory_order_acquire);
        if (!__destructible(__bits)) {
          std::terminate();
        }
      }

      void close() noexcept {
        // [exec.simple.counting.mem] paragraph 2
        // Effects: If state is
        //  (2.1) -- unused, then changes state to unused-and-closed
        //  (2.2) -- open, then changes state to closed
        //  (2.3) -- open-and-joining, then changes state to closed-and-joining
        //  (2.4) -- otherwise, no effect.
        //
        // NOTE: Given our choice of representation, the implementation is simply
        //       "set the __closed bit" so we don't have to check anything before
        //       updating our state.
        //
        // we need store-release semantics to ensure this closure happens-before
        // any subsequent calls to __try_associate that must fail as a result of
        // this closure; we don't use acquire-release semantics because the caller
        // is *sending* a signal, not receiving one
        __bits_.fetch_or(__closed, __std::memory_order_release);
      }

      bool __try_associate() noexcept {
        constexpr auto __make_new_bits = [](std::size_t __bits) noexcept {
          // [exec.simple.counting.mem] paragraph 5 and 5.3 say there is no
          // effect if state is closed or if count is equal to max_associations
          // so we should not be calculating a new (count, state) pair if either
          // of those conditions hold.
          STDEXEC_ASSERT(!__is_closed(__bits) && __count(__bits) < max_associations);

          // [exec.simple.counting.mem] paragraph 5
          // Effects: .... Otherwise, if state is
          //  (5.1) -- unused, then increments count and changes state to open;
          //  (5.2) -- open or open-and-joining, then increments count;
          //
          // NOTE: we represent "open" with __join_needed, and "open-and-joining"
          //       with (__join_needed | __join_running) so we can implement the
          //       update to state by simply ensuring the __join_needed bit is
          //       set; incrementing count is done in the obvious way.
          return __make_bits(__count(__bits) + 1ul, __state(__bits) | __join_needed);
        };

        // we might be about to observe that the scope has been closed; we should
        // establish that the closure happened-before this attempt to associate
        // so this needs to be a load-acquire
        auto __old_bits = __bits_.load(__std::memory_order_acquire);
        std::size_t __new_bits; //intentionally uninitialized

        do {
          // [exec.simple.counting.mem] paragraph 5
          // Effects: if count is equal to max_associations then no effects.
          // Otherwise, if state is
          //  ...
          //  (5.3) -- otherwise, no effect.
          //
          // NOTE: Paragraph 5.3 applies when state is closed, closed-and-joining,
          //       or joined, all of which can be detected by checking the closed bit
          if (__is_closed(__old_bits) || __count(__old_bits) == max_associations) {
            // Paragraph 6 (the Returns clause) says we return assoc-t() "otherwise",
            // which applies when count is not incremented, i.e. right here.
            return false;
          }

          __new_bits = __make_new_bits(__old_bits);
        } while (!__bits_.compare_exchange_weak(
          __old_bits,
          __new_bits,
          // on success we only need store-relaxed because we're "just" incrementing
          // a reference count but on failure we need load-acquire to synchronize
          // with the thread that closed the scope if we happen to observe that; it's
          // UB for the on-failure ordering to be weaker than the on-success ordering
          // so we have to use acquire for both.
          __std::memory_order_acquire));

        // [exec.simple.counting.mem] paragraph 6
        // Returns: If count was incremented, an object of type assoc-t that is engaged
        // and associated with *this, and assoc-t() otherwise.
        //
        // NOTE: we only break out of the while loop and execute this return statement
        //       if the CAS succeeded, the side effect of which is to increment count,
        //       so we must return "an object of type assoc-t that is engaged and
        //       associated with *this" here.
        return true;
      }

      void __disassociate() noexcept {
        // NOTE: The spec says, "Decrements count. If count is zero after
        //       decrementing and state is [joining] then changes state to
        //       joined...", which could be transliterated to code like so:
        //
        //         auto __old_count = __count(__bits_.fetch_sub(1 << 3));
        //
        //         if (__old_count == 1)
        //           __bits_.store(__closed);
        //
        //       but that would introduce a race condition: if the scope is
        //       open then another sender could be associated with the scope
        //       between the fetch_sub and the store, leading to an invalid
        //       state. Instead, we use a CAS loop to atomically update the
        //       count and state simultaneously.

        constexpr auto __make_new_bits = [](std::size_t __bits) noexcept {
          // [exec.simple.counting.mem] paragraph 8
          // Effects: Decrements count. ...
          const auto __new_count = __count(__bits) - 1ul;
          // ... If count is zero after decrementing and state is open-and-joining
          // or closed-and-joining, changes state to joined...
          //
          // NOTE: We can check for both open-and-joining and closed-and-joining by
          //       checking the joining bit; it doesn't matter whether the scope is
          //       open or closed, only whether a join-sender is pending or not.
          const auto __new_state =
            (__new_count == 0ul && __is_joining(__bits) ? __closed : __state(__bits));

          STDEXEC_ASSERT(__new_count < __count(__bits));

          return __make_bits(__new_count, __new_state);
        };

        // relaxed is sufficient here because the CAS loop we're about to run won't
        // complete until we've synchronized with acquire-release semantics
        auto __old_bits = __bits_.load(__std::memory_order_relaxed);
        std::size_t __new_bits; // intentionally uninitialized

        // [exec.simple.counting.mem] paragraph 7
        // Preconditions: count > 0
        STDEXEC_ASSERT(__count(__old_bits) > 0ul);

        do {
          __new_bits = __make_new_bits(__old_bits);
        } while (!__bits_.compare_exchange_weak(
          __old_bits,
          __new_bits,
          // on success, we need store-release semantics to publish the consequences
          // of the just-finished operation to other scope users, and we also need
          // load-acquire semantics in case we're the last associated operation to
          // complete and thus initiate the tear-down of the scope
          __std::memory_order_acq_rel,
          // on failure, we're going to immediately try to synchronize again so we
          // can get away with relaxed semantics
          __std::memory_order_relaxed));

        if (__is_joined(__new_bits)) {
          // [exec.simple.counting.mem] paragraph 8 continued...
          // [state has been updated to joined so] call complete() on all objects
          // registered with *this
          __complete_registered_join_operations();
        } else {
          STDEXEC_ASSERT(!__is_joining(__new_bits) || __count(__new_bits) > 0ul);
        }
      }

      bool __start_join_sender(__counting_scopes::__join_state_base& __join_op) noexcept {
        // relaxed is sufficient because the CAS loop below will continue until
        // we've synchronized
        auto __old_bits = __bits_.load(__std::memory_order_relaxed);

        do {
          // [exec.simple.counting.mem] para (9.1)
          // unused, unused-and-closed, or joined -> joined
          //
          // NOTE: there's a spec bug; we need to move to the joined
          //       state and return true when count is zero, regardless
          //       of state
          if (__count(__old_bits) == 0ul) {
            const auto __new_bits = __make_bits(__count(__old_bits), __closed);

            STDEXEC_ASSERT(__is_joined(__new_bits));

            // try to make it joined
            if (__bits_.compare_exchange_weak(
                  __old_bits,
                  __new_bits,
                  // on success, we need to publish to future callers of try_associate
                  // that the scope is closed and consume from all the now-completed
                  // associated operations any updates they made so we need
                  // acquire-release semantics
                  __std::memory_order_acq_rel,
                  // on failure, relaxed is fine because we'll loop back and try
                  // again to synchronize
                  __std::memory_order_relaxed)) {
              return true;
            }
          }
          // [exec.simple.counting.mem] para (9.2)
          // open or open-and-joining -> open-and-joining
          // [exec.simple.counting.mem] para (9.3)
          // closed or closed-and-joining -> closed-and-joining
          else {
            STDEXEC_ASSERT(__is_join_needed(__old_bits));

            // try to make it {open|closed}-and-joining
            const auto __new_bits = __old_bits | __join_running;

            STDEXEC_ASSERT(__is_joining(__new_bits));

            if (__bits_.compare_exchange_weak(
                  __old_bits,
                  __new_bits,
                  // on success, relaxed is sufficient because __register will further synchronize;
                  // it's fine for the joining thread to synchronize with associated operations on
                  // __registered_join_ops_ and not on __bits because __disassociate decrements the
                  // outstanding operation count with acquire-release semantics and then the last
                  // decrementer dequeues the list of registered joiners with acquire-release
                  // semantics, establishing that all decrements strongly happen-before the
                  // completion of any join operation.
                  //
                  // on failure, relaxed is sufficient because we'll loop around and try again
                  __std::memory_order_relaxed)) {
              return !__register(__join_op);
            }
          }
        } while (true);
      }

     private:
      static constexpr std::size_t __closed{1ul};
      static constexpr std::size_t __join_needed{2ul};
      static constexpr std::size_t __join_running{4ul};

      // Storage for the (count, state) pair referenced in the spec; we store
      // the spec'd "state" in the lower three bits and the spec'd "count" in
      // all the rest of the bits; the states named in the spec map like so:
      //   unused             = all bits zero
      //   open               = any count |                  __join_needed
      //   closed             = any count |                  __join_needed | __closed
      //   open-and-joining   = any count | __join_running | __join_needed
      //   closed-and-joining = any count | __join_running | __join_needed | __closed
      //   unused-and-closed  = zero      |                                  __closed
      //   joined             = zero      |                                  __closed
      //
      // INVARIANT: __bits_ = (count << 3) | (state & 7)
      __std::atomic<std::size_t> __bits_{0ul};

      // An intrusive singly-linked list of join-sender operation states;
      // elements of the list have been "registered" to be completed when the
      // last outstanding associated operation completes. The value can be
      // the possibly-null pointer to the head of the list (where nullptr
      // means the list is empty) or `this`, which is the sentinel value that
      // indicates that the last associated operation has been disassociated
      // and any previously-registered join operations have been (or are about
      // to be) completed.
      __std::atomic<void*> __registered_join_ops_{nullptr};

      // returns true in the unused and unused-and-closed states; since the bit
      // pattern for joined matches the unused-and-closed bit pattern, returns
      // true in the joined state, too
      constexpr static bool __is_unused(std::size_t __bits) noexcept {
        return (__bits & __join_needed) == 0ul;
      }

      // returns true in the unused, open, and open-and-joining states
      constexpr static bool __is_open(std::size_t __bits) noexcept {
        return (__bits & __closed) == 0ul;
      }

      // returns true in the closed, closed-and-joining, unused-and-closed, and
      // joined states
      constexpr static bool __is_closed(std::size_t __bits) noexcept {
        return !__is_open(__bits);
      }

      // returns true in the joined state, which shares a representation with the
      // unused-and-closed state; for the scope to be fully joined, the number of
      // outstanding associated operations must be zero, so we check for exact
      // equality with __closed rather than using it as a bit mask
      constexpr static bool __is_joined(std::size_t __bits) noexcept {
        return __bits == __closed;
      }

      // returns true in the open-and-joining and closed-and-joining states
      constexpr static bool __is_joining(std::size_t __bits) noexcept {
        return (__bits & __join_running) != 0ul;
      }

      // returns false in the unused, unused-and-closed, and joined states;
      // return true in all other states
      constexpr static bool __is_join_needed(std::size_t __bits) noexcept {
        return !__is_unused(__bits);
      }

      // returns true if the destructor is safe to run
      constexpr static bool __destructible(std::size_t __bits) noexcept {
        // acceptable terminal states are __bits == 0 and __bits == __closed; we
        // can check for both at once by expecting __bits with the __closed bit
        // cleared to be equal to zero
        return (__bits & ~__closed) == 0ul;
      }

      // extracts from __bits what the spec calls count, which is the number of
      // outstanding operations associated with this scope
      constexpr static std::size_t __count(std::size_t __bits) noexcept {
        return __bits >> 3;
      }

      // extracts from __bits what the spec calls state, which determines where
      // this scope is in its lifecycle and which operations are valid
      constexpr static std::size_t __state(std::size_t __bits) noexcept {
        return __bits & 7ul;
      }

      // composes ___new_count and __new_state into the packed representation we store
      // in __bits
      constexpr static std::size_t
        __make_bits(std::size_t ___new_count, std::size_t __new_state) noexcept {
        // no high bits set
        STDEXEC_ASSERT(__count(___new_count << 3) == ___new_count);

        // no high bits set
        STDEXEC_ASSERT(__state(__new_state) == __new_state);

        return (___new_count << 3) | __state(__new_state);
      }

      bool __register(__join_state_base& __join_op) noexcept {
        // we need acquire semantics in case the join operation being started is about to
        // observe that the last decrement has already happened; in that case, we need to
        // establish that all of the now-completed associated operations happen-before the
        // completion of this join operation
        auto* __head = __registered_join_ops_.load(__std::memory_order_acquire);

        do {
          if (__head == this) {
            // __registered_join_ops_ == this when the list has been cleared
            return false;
          }

          // make __join_op's __next_ point to the current head; note that, on the first
          // iteration of this loop, this assignment is the first write to __next_ that
          // establishes a non-indeterminate value for the variable
          __join_op.__next_ = static_cast<__join_state_base*>(__head);
        } while (
          // try to make the head point to __join_op
          !__registered_join_ops_.compare_exchange_weak(
            __head,
            std::addressof(__join_op),
            // on success, we need at least release semantics to ensure that the final
            // disassociation can see the full join operation when it dequeues the list
            // of registered operations with acquire semantics; however, the on-success
            // ordering has to be at least as strong as the on-failure ordering, for
            // which we need acquire semantics, so on-success has to be acquire-release.
            __std::memory_order_acq_rel,
            // on failure, we need acquire semantics in case we're about to observe the
            // sentinel value and return early without trying again
            __std::memory_order_acquire));

        return true;
      }

      __join_state_base* __dequeue_registered_join_operations() noexcept {
        // leave __registered_join_ops_ pointing at *this, which we use as a sentinel;
        // any join operations that start after this exchange will observe the
        // sentinel, conclude that the scope has already been joined, and thus
        // complete inline
        //
        // we need acquire semantics to establish that this dequeue operation
        // happens-after all the now-completed associated operations, and we
        // need release semantics to ensure that any future join-senders that
        // observe the sentinel value perform that observation with a happens-after
        // relationship with the current update
        void* __waiting_ops = __registered_join_ops_.exchange(this, __std::memory_order_acq_rel);

        // at this point, __waiting_ops had better be either nullptr or the address
        // of a join operation waiting to be completed; otherwise, the upcoming
        // static_cast is UB
        STDEXEC_ASSERT(__waiting_ops != this);

        return static_cast<__join_state_base*>(__waiting_ops);
      }

      void __complete_registered_join_operations() noexcept {
        for (auto* __ops = __dequeue_registered_join_operations(); __ops != nullptr; /*nothing*/) {
          // completing an async operation is likely to lead to the end of that
          // operation's lifetime so make sure we advance the cursor before we
          // invoke __complete on the current list element; otherwise, it may
          // be UB to access __next_
          auto* __op = std::exchange(__ops, __ops->__next_);
          __op->__complete();
        }
      }
    };
  } // namespace __counting_scopes

  template <>
  struct __sexpr_impl<__counting_scopes::__scope_join_t> : __counting_scopes::__scope_join_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.simple.counting]
  class simple_counting_scope : private __counting_scopes::__base_scope {
   public:
    using __assoc_t = __counting_scopes::__association_t<simple_counting_scope>;

    /////////////////////////////////////////////////////////////////////////////
    // [exec.simple.counting.token]
    struct token {
      template <sender _Sender>
      [[nodiscard]]
      _Sender&& wrap(_Sender&& __snd) const noexcept {
        // [exec.simple.counting.token] paragraph 1
        return static_cast<_Sender&&>(__snd);
      }

      [[nodiscard]]
      __assoc_t try_associate() const noexcept {
        // [exec.simple.counting.token] paragraph 2
        return __scope_->__try_associate();
      }

     private:
      friend class simple_counting_scope;

      simple_counting_scope* __scope_;

      explicit token(simple_counting_scope* __scope) noexcept
        : __scope_(__scope) {
      }
    };

    using __counting_scopes::__base_scope::max_associations;

    [[nodiscard]]
    token get_token() noexcept {
      // [exec.simple.counting.mem] paragraph 1
      return token{this};
    }

    // [exec.simple.counting.mem] paragraphs 2 and 3
    using __counting_scopes::__base_scope::close;

    [[nodiscard]]
    sender auto join() noexcept {
      // [exec.simple.counting.mem] paragraph 4
      return __make_sexpr<__counting_scopes::__scope_join_t>(this);
    }

   private:
    // needs access to call __try_associate
    friend __assoc_t;
    // needs access to call the public members of __base_scope
    friend __counting_scopes::__scope_join_impl;

    __assoc_t __try_associate() noexcept {
      // [exec.simple.counting.mem] paragraphs 5 and 6
      if (__counting_scopes::__base_scope::__try_associate()) {
        return __assoc_t{*this};
      } else {
        return __assoc_t{};
      }
    }
  };

  /////////////////////////////////////////////////////////////////////////////
  // [exec.scope.counting]
  class counting_scope : private __counting_scopes::__base_scope {
   public:
    using __assoc_t = __counting_scopes::__association_t<counting_scope>;

    struct token {
      template <sender _Sender>
      [[nodiscard]]
      sender auto wrap(_Sender&& __snd) const
        noexcept(__nothrow_constructible_from<std::remove_cvref_t<_Sender>, _Sender>) {
        // [exec.scope.counting] paragraph 7
        return __stop_when(static_cast<_Sender&&>(__snd), __scope_->__stop_source_.get_token());
      }

      [[nodiscard]]
      __assoc_t try_associate() const noexcept {
        // [exec.scope.counting] paragraph 8
        return __scope_->__try_associate();
      }

     private:
      friend class counting_scope;

      counting_scope* __scope_;

      explicit token(counting_scope* __scope) noexcept
        : __scope_(__scope) {
      }
    };

    using __counting_scopes::__base_scope::max_associations;

    [[nodiscard]]
    token get_token() noexcept {
      // [exec.scope.counting] paragraph 2
      return token{this};
    }

    using __counting_scopes::__base_scope::close;

    [[nodiscard]]
    sender auto join() noexcept {
      return __make_sexpr<__counting_scopes::__scope_join_t>(this);
    }

    void request_stop() noexcept {
      // [exec.scope.counting] paragraphs 3 and 4
      __stop_source_.request_stop();
    }

   private:
    // needs access to call __try_associate
    friend __assoc_t;
    // needs access to call the public members of __base_scope
    friend __counting_scopes::__scope_join_impl;

    inplace_stop_source __stop_source_;

    __assoc_t __try_associate() noexcept {
      // [exec.scope.counting] paragraphs 5 and 6
      if (__counting_scopes::__base_scope::__try_associate()) {
        return __assoc_t{*this};
      } else {
        return __assoc_t{};
      }
    }
  };
} // namespace STDEXEC
