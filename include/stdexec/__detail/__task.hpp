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

#include "__affine_on.hpp"
#include "__as_awaitable.hpp"
#include "__config.hpp"
#include "__inline_scheduler.hpp"
#include "__manual_lifetime.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__schedulers.hpp"
#include "__task_scheduler.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

namespace STDEXEC {
#if !STDEXEC_NO_STD_COROUTINES()
  namespace __task {
    ////////////////////////////////////////////////////////////////////////////////
    // A base class for task::promise_type so it can be specialized when _Ty is void:
    template <class _Ty>
    struct __promise_base {
      template <class _Value = _Ty>
      constexpr void return_value(_Value&& __value) {
        __result_.emplace(static_cast<_Value&&>(__value));
      }

      __optional<_Ty> __result_{};
    };

    template <>
    struct __promise_base<void> {
      constexpr void return_void() {
      }
    };

    template <class _StopSource>
    struct __on_stopped {
      void operator()() noexcept {
        __source_.request_stop();
      }
      _StopSource& __source_;
    };

    template <class _StopSource>
    __on_stopped(_StopSource&) -> __on_stopped<_StopSource>;

    constexpr size_t __divmod(size_t __total_size, size_t __chunk_size) noexcept {
      return (__total_size / __chunk_size) + (__total_size % __chunk_size != 0);
    }

    struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__) __memblock {
      std::byte __storage_[__STDCPP_DEFAULT_NEW_ALIGNMENT__];
    };

    struct __any_alloc_base {
      virtual void __deallocate_(void* __ptr, size_t __bytes) noexcept = 0;
    };

    template <class _PAlloc>
    struct __any_alloc final : __any_alloc_base {
      using value_type = std::allocator_traits<_PAlloc>::value_type;
      static_assert(__same_as<value_type, __memblock>);

      explicit __any_alloc(_PAlloc __alloc)
        : __alloc_(std::move(__alloc)) {
      }

      void __deallocate_(void* __ptr, size_t __bytes) noexcept final {
        // __bytes here is the same as __bytes passed to promise_type::operator new. We
        // overallocated to store the allocator in the blocks immediately following the
        // promise object. We now use that allocator to deallocate the entire block of
        // memory:
        size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__memblock));
        void* const __alloc_loc = static_cast<__memblock*>(__ptr) + __promise_blocks;
        // the number of blocks needed to store an object of type __palloc_t:
        static constexpr size_t __alloc_blocks =
          __task::__divmod(sizeof(__task::__any_alloc<_PAlloc>), sizeof(__task::__memblock));

        // Quick sanity check to make sure the allocator is where we expect it to be.
        STDEXEC_ASSERT(__alloc_loc == static_cast<void*>(this));

        // Move the allocator out of the block before deallocating, in case the allocator
        // is stateful and its destructor does something interesting:
        auto __alloc = std::move(__alloc_);
        // Destroy self:
        std::destroy_at(this);
        // Deallocate the entire block of memory:
        std::allocator_traits<_PAlloc>::deallocate(
          __alloc, static_cast<__memblock*>(__ptr), __promise_blocks + __alloc_blocks);
      }

      _PAlloc __alloc_;
    };

    template <class _Env>
    using __allocator_type = _Env::allocator_type;

    template <class _Env>
    using __scheduler_type = _Env::scheduler_type;

    template <class _Env>
    using __stop_source_type = _Env::stop_source_type;

    template <class _Env, class _Rcvr>
    using __environment_type = _Env::template env_type<env_of_t<_Rcvr>>;

    template <class _Env>
    using __error_types = __error_types_t<
      typename _Env::error_types,
      __munique<__qq<completion_signatures>>,
      __mcompose<__qf<set_error_t>, __q1<__decay_t>>
    >;

    template <class _Rcvr, class _Alloc>
    concept __has_allocator_compatible_with = requires(_Rcvr& __rcvr) {
      _Alloc(get_allocator(get_env(__rcvr)));
    } || std::default_initializable<_Alloc>;
  } // namespace __task

  ////////////////////////////////////////////////////////////////////////////////
  // STDEXEC::with_error
  template <class _Error>
  struct with_error {
    _Error __error_;
  };

  template <class _Error>
  with_error(_Error) -> with_error<_Error>;

  ////////////////////////////////////////////////////////////////////////////////
  // STDEXEC::task
  template <class _Ty, class _Env = env<>>
  class [[nodiscard]] task {
    struct __promise;
    template <class _Rcvr>
    struct __opstate;
   public:
    using sender_concept = sender_t;
    using promise_type = __promise;

    using allocator_type =
      __minvoke_or_q<__task::__allocator_type, std::allocator<std::byte>, _Env>;
    using scheduler_type = __minvoke_or_q<__task::__scheduler_type, task_scheduler, _Env>;
    using stop_source_type = __minvoke_or_q<__task::__stop_source_type, inplace_stop_source, _Env>;
    using stop_token_type = decltype(__declval<stop_source_type>().get_token());
    using error_types = __minvoke_or_q<__task::__error_types, __eptr_completion, _Env>;

    constexpr task(task&& __that) noexcept
      : __coro_(std::exchange(__that.__coro_, {})) {
    }

    constexpr ~task() {
      if (__coro_)
        __coro_.destroy();
    }

    template <receiver _Rcvr>
    constexpr auto connect(_Rcvr rcvr) && -> __opstate<_Rcvr> {
      STDEXEC_ASSERT(__coro_);
      static_assert(__task::__has_allocator_compatible_with<_Rcvr, allocator_type>);
      return __opstate<_Rcvr>(std::exchange(__coro_, {}), static_cast<_Rcvr&&>(rcvr));
    }

    template <class>
    static consteval auto get_completion_signatures() noexcept {
      return __completions_t{};
    }

   private:
    using __on_stopped_t = __task::__on_stopped<stop_source_type>;

    using __error_variant_t =
      __error_types_t<error_types, __mbind_front_q<__variant, __monostate>, __q1<__decay_t>>;

    using __completions_t = __concat_completion_signatures_t<
      completion_signatures<__detail::__single_value_sig_t<_Ty>, set_stopped_t()>,
      error_types
    >;

    template <class _Rcvr>
    using __stop_callback_t = stop_callback_for_t<stop_token_of_t<env_of_t<_Rcvr>>, __on_stopped_t>;

    template <class _Rcvr>
    static constexpr bool __needs_stop_callback =
      __not_same_as<stop_token_type, stop_token_of_t<env_of_t<_Rcvr>>>;

    struct __opstate_base {
      constexpr explicit __opstate_base(scheduler_type __sched) noexcept
        : __sch_(std::move(__sched)) {
        // Initialize the errors variant to monostate, the "no error" state:
        __errors_.template emplace<0>();
      }

      virtual void __completed() noexcept = 0;
      virtual void __canceled() noexcept = 0;
      virtual auto __get_allocator() noexcept -> allocator_type = 0;

      scheduler_type __sch_;
      __error_variant_t __errors_{__no_init};
    };

    constexpr explicit task(__std::coroutine_handle<promise_type> __coro) noexcept
      : __coro_(std::move(__coro)) {
    }

    __std::coroutine_handle<promise_type> __coro_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // task<T,E>::__opstate
  template <class _Ty, class _Env>
  template <class _Rcvr>
  struct STDEXEC_ATTRIBUTE(empty_bases) task<_Ty, _Env>::__opstate
    : __opstate_base
    , __if_c<__needs_stop_callback<_Rcvr>, __manual_lifetime<__stop_callback_t<_Rcvr>>, __empty> {
   public:
    using operation_state_concept = operation_state_t;

    explicit __opstate(__std::coroutine_handle<promise_type> __coro, _Rcvr&& __rcvr) noexcept
      : __opstate_base(__mk_sched(__rcvr))
      , __coro_(std::move(__coro))
      , __rcvr_(static_cast<_Rcvr&&>(__rcvr))
      , __own_env_(__mk_own_env(__rcvr_))
      , __env_(__mk_env(__rcvr_, __own_env_)) {
      // Set the promise's state pointer to this operation state, so it can call back into
      // it when the coroutine completes or is stopped.
      __coro_.promise().__state_ = this;
      // Initialize the promise's stop source if translation is needed between the
      // receiver's stop token and the task's stop token:
      if constexpr (__needs_stop_callback<_Rcvr>) {
        __coro_.promise().__stop_.template emplace<0>();
      } else {
        __coro_.promise().__stop_.template emplace<1>(get_stop_token(get_env(__rcvr_)));
      }
    }

    ~__opstate() {
      if (__coro_)
        __coro_.destroy();
    }

    void start() & noexcept {
      if constexpr (__needs_stop_callback<_Rcvr>) {
        // If the receiver's stop token is different from the task's stop token, then we need
        // to set up a callback to request a stop on the task's stop source when the receiver's
        // stop token is triggered:
        __stop_callback().__construct(
          get_stop_token(get_env(__rcvr_)),
          __on_stopped_t{__var::__get<0>(__coro_.promise().__stop_)});
      }
      __coro_.resume();
    }

   private:
    using __own_env_t = __minvoke_or_q<__task::__environment_type, env<>, _Env, _Rcvr>;

    static auto __mk_own_env(const _Rcvr& __rcvr) noexcept -> __own_env_t {
      if constexpr (__std::constructible_from<__own_env_t, env_of_t<_Rcvr>>) {
        return __own_env_t(get_env(__rcvr));
      } else {
        return __own_env_t{};
      }
    }

    static auto __mk_env(const _Rcvr& __rcvr, const __own_env_t& __own_env) noexcept -> _Env {
      if constexpr (__std::constructible_from<_Env, const __own_env_t&>) {
        return _Env(__own_env);
      } else if constexpr (__std::constructible_from<_Env, env_of_t<_Rcvr>>) {
        return _Env(get_env(__rcvr));
      } else {
        return _Env{};
      }
    }

    static auto __mk_sched(const _Rcvr& __rcvr) noexcept -> scheduler_type {
      if constexpr (requires { scheduler_type(get_scheduler(get_env(__rcvr))); }) {
        return scheduler_type(get_scheduler(get_env(__rcvr)));
      } else {
        return scheduler_type{};
      }
    }

    auto __stop_callback() noexcept -> __manual_lifetime<__stop_callback_t<_Rcvr>>&
      requires __needs_stop_callback<_Rcvr>
    {
      return *this;
    }

    void __completed() noexcept final {
      if constexpr (__needs_stop_callback<_Rcvr>) {
        // If we set up a stop callback on the receiver's stop token, then we need to
        // disable it when the operation completes:
        __stop_callback().__destroy();
      }

      std::printf("opstate completed, &__errors_ = %p\n", static_cast<void*>(&this->__errors_));

      if (this->__errors_.index() != 0) {
        std::exchange(__coro_, {}).destroy();
        __visit(STDEXEC::set_error, std::move(this->__errors_), static_cast<_Rcvr&&>(__rcvr_));
      } else if constexpr (__same_as<_Ty, void>) {
        std::exchange(__coro_, {}).destroy();
        STDEXEC::set_value(static_cast<_Rcvr&&>(__rcvr_));
      } else {
        STDEXEC_TRY {
          // Move the result out of the promise before destroying the coroutine.
          _Ty __result = static_cast<_Ty&&>(*__coro_.promise().__result_);
          std::exchange(__coro_, {}).destroy();
          STDEXEC::set_value(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ty&&>(__result));
        }
        STDEXEC_CATCH_ALL {
          if constexpr (!__nothrow_move_constructible<_Ty>) {
            std::exchange(__coro_, {}).destroy();
            STDEXEC::set_error(static_cast<_Rcvr&&>(__rcvr_), std::current_exception());
          }
        }
      }
    }

    void __canceled() noexcept final {
      if constexpr (__needs_stop_callback<_Rcvr>) {
        __stop_callback().__destroy();
      }

      std::exchange(__coro_, {}).destroy();
      STDEXEC::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    auto __get_allocator() noexcept -> allocator_type final {
      if constexpr (requires { allocator_type(get_allocator(get_env(__rcvr_))); }) {
        return allocator_type(get_allocator(get_env(__rcvr_)));
      } else {
        return allocator_type{};
      }
    }

    __std::coroutine_handle<promise_type> __coro_;
    _Rcvr __rcvr_;
    __own_env_t __own_env_;
    _Env __env_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // task<T,E>::promise_type
  template <class _Ty, class _Env>
  struct task<_Ty, _Env>::__promise : __task::__promise_base<_Ty> {
    __promise() noexcept = default;

    [[nodiscard]]
    auto get_return_object() noexcept -> task {
      return task(__std::coroutine_handle<__promise>::from_promise(*this));
    }

    static constexpr __std::suspend_always initial_suspend() noexcept {
      return {};
    }

    auto final_suspend() noexcept {
      return __completed_awaitable{};
    }

    void unhandled_exception() {
      if constexpr (!__mapply<__mcontains<std::exception_ptr>, __error_variant_t>::value) {
        STDEXEC::__die(
          "A task threw an exception but does not have std::exception_ptr in its error_types. "
          "Either add std::exception_ptr to the task's error_types or ensure that all code called "
          "by the task is noexcept.");
      } else {
        __state_->__errors_.template emplace<std::exception_ptr>(std::current_exception());
      }
    }

    auto unhandled_stopped() noexcept -> __std::coroutine_handle<> {
      __state_->__canceled();
      return std::noop_coroutine();
    }

    template <class _Error>
    constexpr auto
      yield_value(with_error<_Error> __error) noexcept(__nothrow_decay_copyable<_Error>) {
      if constexpr (__mapply<__mcontains<__decay_t<_Error>>, __error_variant_t>::value) {
        __state_->__errors_.template emplace<__decay_t<_Error>>(std::move(__error).__error_);
      } else {
        static_assert(__mnever<_Error>, "Error type not in task's error_types");
      }
      return __completed_awaitable{};
    }

    template <sender _Sender>
    constexpr auto await_transform(_Sender&& __sndr) noexcept {
      if constexpr (__same_as<scheduler_type, STDEXEC::inline_scheduler>) {
        return STDEXEC::as_awaitable(static_cast<_Sender&&>(__sndr), *this);
      } else {
        return STDEXEC::as_awaitable(
          STDEXEC::affine_on(static_cast<_Sender&&>(__sndr), __state_->__sch_), *this);
      }
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept {
      return __env{this};
    }

    template <class _Alloc, class... _Args>
    void* operator new(size_t __bytes, std::allocator_arg_t, _Alloc __alloc, _Args&&...) {
      using __palloc_t = std::allocator_traits<_Alloc>::template rebind_alloc<__task::__memblock>;
      using __pointer_t = std::allocator_traits<__palloc_t>::pointer;
      static_assert(std::is_pointer_v<__pointer_t>, "Allocator pointer type must be a raw pointer");

      // the number of blocks needed to store an object of type __palloc_t:
      static constexpr size_t __alloc_blocks =
        __task::__divmod(sizeof(__task::__any_alloc<__palloc_t>), sizeof(__task::__memblock));
      size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__task::__memblock));

      __palloc_t __palloc(__alloc);
      __pointer_t const __ptr =
        std::allocator_traits<__palloc_t>::allocate(__palloc, __promise_blocks + __alloc_blocks);

      // construct the allocator in the blocks immediately following the promise object:
      void* const __alloc_loc = __ptr + __promise_blocks;
      std::construct_at(
        static_cast<__task::__any_alloc<__palloc_t>*>(__alloc_loc), std::move(__palloc));
      return __ptr;
    }

    void* operator new(size_t __bytes) {
      return operator new(__bytes, std::allocator_arg, std::allocator<std::byte>{});
    }

    void operator delete(void* __ptr, size_t __bytes) noexcept {
      size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__task::__memblock));
      void* const __alloc_loc = static_cast<__task::__memblock*>(__ptr) + __promise_blocks;
      auto* __alloc = static_cast<__task::__any_alloc_base*>(__alloc_loc);
      __alloc->__deallocate_(__ptr, __bytes);
    }

   private:
    template <class>
    friend struct __opstate;

    struct __completed_awaitable {
      static constexpr bool await_ready() noexcept {
        return false;
      }

      static constexpr void await_suspend(__std::coroutine_handle<__promise> __coro) noexcept {
        __promise& __self = __coro.promise();
        __self.__state_->__completed();
      }

      static constexpr void await_resume() noexcept {
      }
    };

    struct __env {
      [[nodiscard]]
      constexpr auto query(get_scheduler_t) const noexcept -> scheduler_type {
        return __promise_->__state_->__sch_;
      }

      [[nodiscard]]
      constexpr auto query(get_allocator_t) const noexcept -> allocator_type {
        return __promise_->__state_->__get_allocator();
      }

      [[nodiscard]]
      constexpr auto query(get_stop_token_t) const noexcept -> stop_token_type {
        if (__promise_->__stop_.index() == 0) {
          return __promise_->__stop_.template get<0>().get_token();
        } else {
          return __promise_->__stop_.template get<1>();
        }
      }

      __promise const * __promise_;
    };

    __variant<stop_source_type, stop_token_type> __stop_{__no_init};
    __opstate_base* __state_ = nullptr;
  };
#endif // !STDEXEC_NO_STD_COROUTINES()
} // namespace STDEXEC
