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

#include "../stdexec/__detail/__execution_fwd.hpp"

#include "../stdexec/__detail/__env.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__scope.hpp"
#include "../stdexec/__detail/__submit.hpp"
#include "../stdexec/__detail/__transform_sender.hpp"

#include <memory>

namespace experimental::execution
{
  /////////////////////////////////////////////////////////////////////////////
  namespace __start_detached
  {
    struct __submit_receiver
    {
      using receiver_concept = STDEXEC::receiver_tag;

      template <class... _As>
      constexpr void set_value(_As&&...) noexcept
      {}

      template <class _Error>
      [[noreturn]]
      void set_error(_Error&&) noexcept
      {
        // A detached operation failed. There is noplace for the error to go.
        // This is unrecoverable, so we terminate.
        std::terminate();
      }

      constexpr void set_stopped() noexcept {}

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> STDEXEC::__root_env
      {
        return {};
      }
    };

    template <class _Env>
    struct __op_base : STDEXEC::__immovable
    {
      constexpr explicit __op_base(_Env __env) noexcept(STDEXEC::__nothrow_move_constructible<_Env>)
        : __env_(static_cast<_Env&&>(__env))
      {}

      virtual ~__op_base() = default;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env __env_;
    };

    // The start_detached receiver deletes the operation state.
    template <class _Env>
    struct __receiver
    {
      using receiver_concept = STDEXEC::receiver_tag;

      template <class... _As>
      constexpr void set_value(_As&&...) noexcept
      {
        delete __op_;  // NB: invalidates *this
      }

      template <class _Error>
      [[noreturn]]
      void set_error(_Error&&) noexcept
      {
        // A detached operation failed. There is noplace for the error to go.
        // This is unrecoverable, so we terminate.
        std::terminate();
      }

      constexpr void set_stopped() noexcept
      {
        delete __op_;  // NB: invalidates *this
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> _Env const &
      {
        return __op_->__env_;
      }

      __op_base<_Env>* __op_;
    };

    template <class _Sender, class _Env>
    struct __operation : __op_base<_Env>
    {
      constexpr explicit __operation(STDEXEC::connect_t, _Sender&& __sndr, _Env __env)
        : __op_base<_Env>(static_cast<_Env&&>(__env))
        , __op_data_(static_cast<_Sender&&>(__sndr), __receiver<_Env>{this})
      {}

      constexpr explicit __operation(_Sender&& __sndr, _Env __env)
        : __operation(STDEXEC::connect, static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env))
      {
        // If the operation completes synchronously, then the following line will cause
        // the destruction of *this, which is not a problem because we used a delegating
        // constructor, so *this is considered fully constructed.
        __op_data_.submit(static_cast<_Sender&&>(__sndr), __receiver<_Env>{this});
      }

      static constexpr void operator delete(__operation* __self, std::destroying_delete_t) noexcept
      {
        auto __alloc       = STDEXEC::__with_default(STDEXEC::get_allocator,
                                               std::allocator<__operation>())(__self->__env_);
        using __alloc_t    = decltype(__alloc);
        using __op_alloc_t = std::allocator_traits<__alloc_t>::template rebind_alloc<__operation>;
        __op_alloc_t __op_alloc{__alloc};
        std::allocator_traits<__op_alloc_t>::destroy(__op_alloc, __self);
        std::allocator_traits<__op_alloc_t>::deallocate(__op_alloc, __self, 1);
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      STDEXEC::submit_result<_Sender, __receiver<_Env>> __op_data_;
    };

    template <class _Sender, class _Env>
    concept __use_submit =
      STDEXEC::__submittable<_Sender, __submit_receiver>
      && STDEXEC::__same_as<_Env, STDEXEC::__root_env>
      && STDEXEC::__same_as<void, STDEXEC::__submit_result_t<_Sender, __submit_receiver>>;
  }  // namespace __start_detached

  //! @brief A sender consumer that eagerly starts a sender and forgets it.
  //!
  //! @c start_detached connects its argument sender to a built-in receiver
  //! and starts the resulting operation immediately, allocating the
  //! operation state on the heap (using an allocator from the optional
  //! environment) so it can outlive the call. The completion of the sender
  //! deallocates the operation state and discards the result; nothing is
  //! returned to the caller.
  //!
  //! Use @c start_detached for top-level fire-and-forget work that has no
  //! caller waiting on its result and no enclosing async scope — for
  //! example, kicking off a logging or telemetry pipeline from @c main(),
  //! or a one-shot background task at program startup. For fire-and-forget
  //! work that should be *tracked* by a scope (so the scope can be joined
  //! at shutdown), prefer @c stdexec::spawn. For top-level *waiting*, use
  //! @c stdexec::sync_wait.
  //!
  //! @note @c start_detached is an @b stdexec @b extension. It is not part
  //!       of the C++26 working draft. The standardized way to spawn
  //!       fire-and-forget work is @c stdexec::spawn (see [exec.spawn]),
  //!       which requires an async scope to take ownership of the
  //!       operation.
  //!
  //! @code{.cpp}
  //! exec::start_detached(stdexec::just(42) | stdexec::then([](int x) {
  //!   std::println("background work produced {}", x);
  //! }));
  //! @endcode
  //!
  //! **Completion behavior.**
  //!
  //! The sender must complete via @c set_value or @c set_stopped — both
  //! are accepted and the result is discarded. The sender *must not*
  //! complete via @c set_error: there is no caller to deliver the error
  //! to, and an error completion is therefore considered a contract
  //! violation. The implementation enforces this with a static assertion
  //! when possible; if the sender's completion signatures include
  //! @c set_error_t the program is ill-formed.
  //!
  //! **Allocator support.**
  //!
  //! The two-argument overload accepts an environment from which an
  //! allocator can be queried (via @c stdexec::get_allocator). That
  //! allocator is used to allocate the operation state, so callers can
  //! avoid the default global @c new for hot paths.
  //!
  //! **Cancellation.**
  //!
  //! @c start_detached does not arrange for cancellation of the spawned
  //! work. If the operation observes a stop token via the environment,
  //! it can self-cancel; otherwise the work runs to natural completion.
  //!
  //! @see stdexec::sync_wait  — top-level synchronous wait that returns the result
  //! @see stdexec::spawn      — fire-and-forget into a scope (standardized in C++26)
  //! @see stdexec::spawn_future — spawn into a scope and observe via a sender
  struct start_detached_t
  {
    template <class _Sender, class _Env>
    using __compl_domain_t =
      STDEXEC::__mcall<STDEXEC::__mwith_default_q<STDEXEC::__completion_domain_of_t,
                                                  STDEXEC::indeterminate_domain<>>,
                       STDEXEC::set_value_t,
                       _Sender,
                       STDEXEC::__as_root_env_t<_Env>>;

    //! @brief Eagerly start @c __sndr; allocate its operation state on the
    //!        heap with the default allocator.
    //!
    //! @tparam _Sender A sender type with no @c set_error_t completions.
    //! @param __sndr   The sender to launch. Forwarded into the heap-allocated
    //!                 operation state.
    //!
    //! @pre @c __sndr must not be able to complete with @c set_error.
    template <STDEXEC::sender_in<STDEXEC::__root_env> _Sender>
      requires STDEXEC::__callable<STDEXEC::apply_sender_t,
                                   __compl_domain_t<_Sender, STDEXEC::__root_env>,
                                   start_detached_t,
                                   _Sender>
    void operator()(_Sender&& __sndr) const
    {
      using __domain_t = __compl_domain_t<_Sender, STDEXEC::__root_env>;
      STDEXEC::apply_sender(__domain_t{}, *this, static_cast<_Sender&&>(__sndr));
    }

    //! @brief Eagerly start @c __sndr; use the allocator from @c __env to
    //!        allocate the operation state.
    //!
    //! @tparam _Env    An environment type. Queried with
    //!                 @c stdexec::get_allocator for an allocator;
    //!                 falls back to @c std::allocator if absent.
    //! @tparam _Sender A sender type with no @c set_error_t completions.
    //!
    //! @param __sndr   The sender to launch.
    //! @param __env    The environment used both for the allocator query
    //!                 and for the receiver's environment.
    //!
    //! @pre @c __sndr must not be able to complete with @c set_error.
    template <class _Env, STDEXEC::sender_in<STDEXEC::__as_root_env_t<_Env>> _Sender>
      requires STDEXEC::__callable<STDEXEC::apply_sender_t,
                                   __compl_domain_t<_Sender, STDEXEC::__as_root_env_t<_Env>>,
                                   start_detached_t,
                                   _Sender,
                                   STDEXEC::__as_root_env_t<_Env>>
    void operator()(_Sender&& __sndr, _Env&& __env) const
    {
      auto __env2      = STDEXEC::__as_root_env(static_cast<_Env&&>(__env));
      using __domain_t = __compl_domain_t<_Sender, STDEXEC::__as_root_env_t<_Env>>;
      STDEXEC::apply_sender(__domain_t{}, *this, static_cast<_Sender&&>(__sndr), __env2);
    }

    // Below is the default implementation for `start_detached`.
    template <class _CvSender, class _Env = STDEXEC::__root_env>
      requires STDEXEC::sender_in<_CvSender, STDEXEC::__as_root_env_t<_Env>>
    void apply_sender(_CvSender&& __sndr, _Env&& __env = {}) const noexcept(false)
    {
      using __opstate_t = __start_detached::__operation<_CvSender, STDEXEC::__decay_t<_Env>>;

#if !STDEXEC_APPLE_CLANG()  // There seems to be a codegen bug in apple clang that causes
                            // `start_detached` to segfault when the code path below is
                            // taken.
      // BUGBUG NOT TO SPEC: the use of the non-standard `submit` algorithm here is a
      // conforming extension.
      if constexpr (__start_detached::__use_submit<_CvSender, _Env>)
      {
        // If submit(sndr, rcvr) returns void, then no state needs to be kept alive
        // for the operation. We can just call submit and return.
        STDEXEC::__submit::__submit(static_cast<_CvSender&&>(__sndr),
                                    __start_detached::__submit_receiver{});
      }
      else
#endif
      {
        // Use the provided allocator if any to allocate the operation state.
        auto __alloc       = STDEXEC::__with_default(STDEXEC::get_allocator,
                                               std::allocator<__opstate_t>())(__env);
        using __alloc_t    = decltype(__alloc);
        using __op_alloc_t = std::allocator_traits<__alloc_t>::template rebind_alloc<__opstate_t>;
        // We use the allocator to allocate the op state and also to construct it.
        __op_alloc_t           __op_alloc{__alloc};
        __opstate_t*           __op = std::allocator_traits<__op_alloc_t>::allocate(__op_alloc, 1);
        STDEXEC::__scope_guard __g{
          [__op, &__op_alloc]() noexcept
          { std::allocator_traits<__op_alloc_t>::deallocate(__op_alloc, __op, 1); }};
        // This can potentially throw. If it does, the scope guard will deallocate the
        // storage automatically.
        std::allocator_traits<__op_alloc_t>::construct(__op_alloc,
                                                       __op,
                                                       static_cast<_CvSender&&>(__sndr),
                                                       static_cast<_Env&&>(__env));
        // The operation state is now constructed, dismiss the scope guard.
        __g.__dismiss();
        // The operation has now started and is responsible for deleting itself when it
        // completes.
      }
    }
  };

  //! @brief The customization point object for the @c start_detached sender consumer.
  //!
  //! @c start_detached is an instance of @ref start_detached_t. See
  //! @ref start_detached_t for the full description and a usage example.
  //!
  //! @hideinitializer
  inline constexpr start_detached_t start_detached{};
}  // namespace experimental::execution

namespace exec = experimental::execution;
