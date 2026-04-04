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

#include "../stop_token.hpp"
#include "__affine_on.hpp"
#include "__as_awaitable.hpp"
#include "__config.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__schedulers.hpp"
#include "__task_scheduler.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmismatched-new-delete")

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  namespace __task
  {
    ////////////////////////////////////////////////////////////////////////////////
    // A base class for task::promise_type so it can be specialized when _Ty is void:
    template <class _Ty>
    struct __promise_base
    {
      template <class _Value = _Ty>
      constexpr void return_value(_Value&& __value)
      {
        __result_.emplace(static_cast<_Value&&>(__value));
      }

      __optional<_Ty> __result_{};
    };

    template <>
    struct __promise_base<void>
    {
      constexpr void return_void() {}
    };

    constexpr size_t __divmod(size_t __total_size, size_t __chunk_size) noexcept
    {
      return (__total_size / __chunk_size) + (__total_size % __chunk_size != 0);
    }

    struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__) __memblock
    {
      std::byte __storage_[__STDCPP_DEFAULT_NEW_ALIGNMENT__];
    };

    struct __any_alloc_base
    {
      virtual void __deallocate_(void* __ptr, size_t __bytes) noexcept = 0;
    };

    template <class _PAlloc>
    struct __any_alloc final : __any_alloc_base
    {
      using value_type = std::allocator_traits<_PAlloc>::value_type;
      static_assert(__same_as<value_type, __memblock>);

      explicit __any_alloc(_PAlloc __alloc)
        : __alloc_(std::move(__alloc))
      {}

      void __deallocate_(void* __ptr, size_t __bytes) noexcept final
      {
        // __bytes here is the same as __bytes passed to promise_type::operator new. We
        // overallocated to store the allocator in the blocks immediately following the
        // promise object. We now use that allocator to deallocate the entire block of
        // memory:
        size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__memblock));
        [[maybe_unused]]
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
        std::allocator_traits<_PAlloc>::deallocate(__alloc,
                                                   static_cast<__memblock*>(__ptr),
                                                   __promise_blocks + __alloc_blocks);
      }

      _PAlloc __alloc_;
    };

    template <class _Env>
    using __allocator_type = _Env::allocator_type;

    template <class _Env>
    using __scheduler_type = _Env::scheduler_type;

    template <class _Env>
    using __stop_source_type = _Env::stop_source_type;

    template <class _Env>
    using __error_types = _Env::error_types;

    template <class _Env, class _EnvProvider>
    using __environment_type = _Env::template env_type<env_of_t<_EnvProvider>>;

    template <class _EnvProvider, class _Alloc>
    concept __has_allocator_compatible_with = requires(_EnvProvider const & __has_env) {
      _Alloc(STDEXEC::get_allocator(STDEXEC::get_env(__has_env)));
    };

    template <class _EnvProvider, class _Scheduler, class... _Alloc>
    concept __has_scheduler_compatible_with = requires(_EnvProvider const & __has_env,
                                                       _Alloc const &... __alloc) {
      _Scheduler(STDEXEC::get_scheduler(STDEXEC::get_env(__has_env)), __alloc...);
    };

    template <class _StopSource>
    using __stop_source_token_t = decltype(__declval<_StopSource>().get_token());

    template <class _StopToken, class _StopSource>
    struct __stop_callback_box
    {
      void __register_callback(__ignore, __ignore) noexcept {}
      void __reset_callback() noexcept {}
    };

    template <class _StopToken, class _StopSource>
      requires __not_same_as<__stop_source_token_t<_StopSource>, _StopToken>
    struct __stop_callback_box<_StopToken, _StopSource>
    {
      using __stop_variant_t  = __variant<_StopSource, __stop_source_token_t<_StopSource>>;
      using __callback_fn_t   = __forward_stop_request<_StopSource>;
      using __stop_callback_t = stop_callback_for_t<_StopToken, __callback_fn_t>;

      constexpr __stop_callback_box() {}

      constexpr ~__stop_callback_box() {}

      void __register_callback(auto const & __has_env, __stop_variant_t& __stop)
        noexcept(__nothrow_constructible_from<__stop_callback_t, _StopToken, _StopSource&>)
      {
        std::construct_at(&__cb_, get_stop_token(get_env(__has_env)), __var::__get<0>(__stop));
      }

      void __reset_callback() noexcept
      {
        std::destroy_at(&__cb_);
      }

      union
      {
        __stop_callback_t __cb_;
      };
    };

    template <class _EnvProvider, class _StopSource>
    using __stop_callback_box_t =
      __stop_callback_box<stop_token_of_t<env_of_t<_EnvProvider>>, _StopSource>;

    inline constexpr auto __throw_error = __overload{
      []([[maybe_unused]] auto&& __error) { STDEXEC_THROW((decltype(__error)&&) __error); },
      []([[maybe_unused]] std::error_code __ec) { STDEXEC_THROW(std::system_error(__ec)); },
      []([[maybe_unused]] std::exception_ptr __eptr) { std::rethrow_exception(__eptr); }};

  }  // namespace __task

  ////////////////////////////////////////////////////////////////////////////////
  // STDEXEC::with_error
  template <class _Error>
  struct with_error
  {
    using type = __decay_t<_Error>;
    type error;
  };

  template <class _Error>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE with_error(_Error) -> with_error<_Error>;

  ////////////////////////////////////////////////////////////////////////////////
  // STDEXEC::task
  template <class _Ty, class _Env = env<>>
  class [[nodiscard]] task
  {
    struct __promise;
    template <class _Rcvr>
    struct __opstate;
    template <class _EnvProvider>
    using __own_env_t = __minvoke_or_q<__task::__environment_type, env<>, _Env, _EnvProvider>;
   public:
    using sender_concept = sender_t;
    using promise_type   = __promise;

    using allocator_type =
      __minvoke_or_q<__task::__allocator_type, std::allocator<std::byte>, _Env>;
    using scheduler_type   = __minvoke_or_q<__task::__scheduler_type, task_scheduler, _Env>;
    using stop_source_type = __minvoke_or_q<__task::__stop_source_type, inplace_stop_source, _Env>;
    using stop_token_type  = decltype(__declval<stop_source_type>().get_token());
    using error_types      = __minvoke_or_q<__task::__error_types, __eptr_completion_t, _Env>;

    constexpr task(task&& __that) noexcept
      : __coro_(std::exchange(__that.__coro_, {}))
    {}

    constexpr ~task()
    {
      if (__coro_)
        __coro_.destroy();
    }

    template <receiver _Rcvr>
    constexpr auto connect(_Rcvr rcvr) && -> __opstate<_Rcvr>
    {
      static_assert(__task::__has_allocator_compatible_with<_Rcvr, allocator_type>
                    || std::default_initializable<allocator_type>);
      STDEXEC_ASSERT(__coro_);
      return __opstate<_Rcvr>(static_cast<task&&>(*this), static_cast<_Rcvr&&>(rcvr));
    }

    template <class>
    static consteval auto get_completion_signatures() noexcept
    {
      return __completions_t{};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept
    {
      return __attrs{};
    }

#  if !STDEXEC_GCC()
    // This transforms a task into an __awaiter that can perform symmetric transfer when
    // co_awaited. It is disabled on GCC due to
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94794, which causes unbounded stack
    // growth.
    template <class _ParentPromise>
    constexpr auto as_awaitable(_ParentPromise& __parent) && noexcept
    {
      return __awaiter<_ParentPromise>(static_cast<task&&>(*this), __parent);
    }
#  endif

   private:
    using __on_stopped_t   = __forward_stop_request<stop_source_type>;
    using __stop_variant_t = __variant<stop_source_type, stop_token_type>;

    template <class _EnvProvider>
    using __stop_callback_t =
      stop_callback_for_t<stop_token_of_t<env_of_t<_EnvProvider>>, __on_stopped_t>;

    template <class _EnvProvider>
    using __stop_callback_box_t = __task::__stop_callback_box_t<_EnvProvider, stop_source_type>;

    template <class _EnvProvider>
    static constexpr bool __needs_stop_callback =
      __not_same_as<stop_token_type, stop_token_of_t<env_of_t<_EnvProvider>>>;

    template <class _EnvProvider>
    static constexpr bool __nothrow_callback_registration = noexcept(
      __declval<__stop_callback_box_t<_EnvProvider>&>()
        .__register_callback(__declval<_EnvProvider&>(), __declval<__stop_variant_t&>()));

    using __error_variant_t = __error_types_t<error_types, __q<__variant>, __q1<__decay_t>>;

    using __completions_t = __concat_completion_signatures_t<
      completion_signatures<__single_value_sig_t<_Ty>, set_stopped_t()>,
      error_types>;

    static constexpr void __sink(task) noexcept {}

    template <class _EnvProvider>
    [[nodiscard]]
    static auto __mk_alloc(_EnvProvider const & __has_env) noexcept -> allocator_type
    {
      if constexpr (__task::__has_allocator_compatible_with<_EnvProvider, allocator_type>)
      {
        return allocator_type(get_allocator(STDEXEC::get_env(__has_env)));
      }
      else
      {
        return allocator_type{};
      }
    }

    template <class _EnvProvider>
    [[nodiscard]]
    static auto __mk_sched(_EnvProvider const & __has_env, allocator_type const & __alloc) noexcept
      -> scheduler_type
    {
      // NOT TO SPEC: try constructing the scheduler with the allocator if possible.
      if constexpr (__task::__has_scheduler_compatible_with<_EnvProvider,
                                                            scheduler_type,
                                                            allocator_type>)
      {
        return scheduler_type(get_scheduler(STDEXEC::get_env(__has_env)), __alloc);
      }
      else if constexpr (__task::__has_scheduler_compatible_with<_EnvProvider, scheduler_type>)
      {
        return scheduler_type(get_scheduler(STDEXEC::get_env(__has_env)));
      }
      else
      {
        return scheduler_type{};
      }
    }

    template <class _EnvProvider>
    [[nodiscard]]
    static auto __mk_own_env(_EnvProvider const & __has_env) noexcept
    {
      if constexpr (__std::constructible_from<__own_env_t<_EnvProvider>, env_of_t<_EnvProvider>>)
      {
        return __own_env_t<_EnvProvider>(STDEXEC::get_env(__has_env));
      }
      else
      {
        return __own_env_t<_EnvProvider>{};
      }
    }

    template <class _EnvProvider>
    [[nodiscard]]
    static auto
    __mk_env(_EnvProvider const & __has_env, __own_env_t<_EnvProvider> const & __own_env) noexcept
      -> _Env
    {
      if constexpr (__std::constructible_from<_Env, __own_env_t<_EnvProvider> const &>)
      {
        return _Env(__own_env);
      }
      else if constexpr (__std::constructible_from<_Env, env_of_t<_EnvProvider>>)
      {
        return _Env(STDEXEC::get_env(__has_env));
      }
      else
      {
        return _Env{};
      }
    }

    struct __opstate_base : private allocator_type
    {
      template <class _EnvProvider>
      constexpr explicit __opstate_base(task&& __task, _EnvProvider const & __has_env) noexcept
        : allocator_type(__mk_alloc(__has_env))
        , __sch_(__mk_sched(__has_env, __get_allocator()))
        , __task_(static_cast<task&&>(__task))
      {
        auto& __promise = __task_.__coro_.promise();
        // Set the promise's state pointer to this operation state, so it can call back into
        // it when the coroutine completes or is stopped.
        __promise.__state_ = this;

        // Initialize the promise's stop source if translation is needed between the
        // receiver's stop token and the task's stop token:
        if constexpr (__needs_stop_callback<_EnvProvider>)
        {
          __promise.__stop_.template emplace<0>();
        }
        else
        {
          __promise.__stop_.template emplace<1>(get_stop_token(STDEXEC::get_env(__has_env)));
        }
      }

      STDEXEC_IMMOVABLE(__opstate_base);

      virtual auto __completed() noexcept -> __std::coroutine_handle<> = 0;
      virtual auto __canceled() noexcept -> __std::coroutine_handle<>  = 0;

      [[nodiscard]]
      constexpr auto __get_allocator() const noexcept -> allocator_type const &
      {
        return static_cast<allocator_type const &>(*this);
      }

      constexpr auto __handle() const noexcept -> __std::coroutine_handle<promise_type>
      {
        return __task_.__coro_;
      }

      scheduler_type    __sch_;
      task              __task_;
      __error_variant_t __errors_{__no_init};
    };

    template <class _ParentPromise>
    struct STDEXEC_ATTRIBUTE(empty_bases) __awaiter final
      : __opstate_base
      , __stop_callback_box_t<_ParentPromise>
    {
      constexpr explicit __awaiter(task&& __task, _ParentPromise& __parent) noexcept
        : __opstate_base(static_cast<task&&>(__task), __parent)
        , __own_env_(__mk_own_env(__parent))
        , __env_(__mk_env(__parent, __own_env_))
        , __parent_(__parent)
      {}

      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      constexpr auto await_suspend(__std::coroutine_handle<_ParentPromise> __h)
        noexcept(__nothrow_callback_registration<_ParentPromise>) -> __std::coroutine_handle<>
      {
        auto& __task_promise = this->__handle().promise();
        // If the following throws, the coroutine is immediately resumed and the exception
        // is rethrown at the suspension point.
        this->__register_callback(__h.promise(), __task_promise.__stop_);
        __task_promise.__state_ = this;
        __continuation_         = __h;
        return this->__handle();
      }

      constexpr auto await_resume() -> _Ty
      {
        // Destroy the coroutine after moving the result/error out of it
        auto __task = std::move(this->__task_);
        if (!this->__errors_.__is_valueless())
        {
          __visit(__task::__throw_error, std::move(this->__errors_));
        }
        else if constexpr (__same_as<_Ty, void>)
        {
          return;
        }
        else
        {
          return static_cast<_Ty&&>(*__task.__coro_.promise().__result_);
        }
        __std::unreachable();
      }

      [[nodiscard]]
      auto __completed() noexcept -> __std::coroutine_handle<> final
      {
        this->__reset_callback();
        return __continuation_;
      }

      [[nodiscard]]
      auto __canceled() noexcept -> __std::coroutine_handle<> final
      {
        this->__reset_callback();
        return __parent_.unhandled_stopped();
      }

      // STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __own_env_t<_ParentPromise> __own_env_;
      // STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Env                      __env_;
      __std::coroutine_handle<> __continuation_;
      _ParentPromise&           __parent_;
    };

    struct __attrs
    {
      template <class _Tag, class... _OtherEnv>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<_Tag>, _OtherEnv&&...) const noexcept
      {
        using __attrs_t = env_of_t<schedule_result_t<scheduler_type>>;

        if constexpr (__completes_inline<set_value_t, __attrs_t, _OtherEnv...>)
        {
          return __completion_behavior::__unknown;
        }
        else
        {
          return __completion_behavior::__asynchronous_affine
               | __completion_behavior::__inline_completion;
        }
      }
    };

    constexpr explicit task(__std::coroutine_handle<promise_type> __coro) noexcept
      : __coro_(std::move(__coro))
    {}

    __std::coroutine_handle<promise_type> __coro_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // task<T,E>::__opstate
  template <class _Ty, class _Env>
  template <class _Rcvr>
  struct STDEXEC_ATTRIBUTE(empty_bases) task<_Ty, _Env>::__opstate final
    : __opstate_base
    , __stop_callback_box_t<_Rcvr>
  {
   public:
    using operation_state_concept = operation_state_t;

    explicit __opstate(task&& __task, _Rcvr&& __rcvr) noexcept
      : __opstate_base(static_cast<task&&>(__task), __rcvr)
      , __own_env_(__mk_own_env(__rcvr))
      , __env_(__mk_env(__rcvr, __own_env_))
      , __rcvr_(static_cast<_Rcvr&&>(__rcvr))
    {}

    void start() & noexcept
    {
      STDEXEC_TRY
      {
        // Register a stop callback if needed
        this->__register_callback(__rcvr_, this->__handle().promise().__stop_);
        this->__handle().resume();
      }
      STDEXEC_CATCH_ALL
      {
        if constexpr (__nothrow_callback_registration<_Rcvr>)
        {
          // no-op
        }
        else if constexpr (__mapply<__mcontains<set_error_t(std::exception_ptr)>,
                                    error_types>::value)
        {
          STDEXEC::set_error(static_cast<_Rcvr&&>(__rcvr_), std::current_exception());
        }
        else
        {
          STDEXEC::__die("Starting the task failed due to an exception being thrown while "
                         "registering a stop callback, but the task's error_types does not "
                         "include std::exception_ptr, so the exception cannot be propagated.");
        }
      }
    }

   private:
    auto __completed() noexcept -> __std::coroutine_handle<> final
    {
      STDEXEC_TRY
      {
        this->__reset_callback();

        if (!this->__errors_.__is_valueless())
        {
          // Move the errors out of the promise before destroying the coroutine.
          auto __errors = std::move(this->__errors_);
          __sink(static_cast<task&&>(this->__task_));
          __visit(STDEXEC::set_error, std::move(__errors), static_cast<_Rcvr&&>(__rcvr_));
        }
        else if constexpr (__same_as<_Ty, void>)
        {
          __sink(static_cast<task&&>(this->__task_));
          STDEXEC::set_value(static_cast<_Rcvr&&>(__rcvr_));
        }
        else
        {
          // Move the result out of the promise before destroying the coroutine.
          _Ty __result = static_cast<_Ty&&>(*this->__handle().promise().__result_);
          __sink(static_cast<task&&>(this->__task_));
          STDEXEC::set_value(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ty&&>(__result));
        }
      }
      STDEXEC_CATCH_ALL
      {
        if constexpr (!__nothrow_move_constructible<_Ty>
                      || !__nothrow_move_constructible<__error_variant_t>)
        {
          __sink(static_cast<task&&>(this->__task_));
          STDEXEC::set_error(static_cast<_Rcvr&&>(__rcvr_), std::current_exception());
        }
      }
      return std::noop_coroutine();
    }

    auto __canceled() noexcept -> __std::coroutine_handle<> final
    {
      this->__reset_callback();
      __sink(static_cast<task&&>(this->__task_));
      STDEXEC::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
      return std::noop_coroutine();
    }

    // STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
    __own_env_t<_Rcvr> __own_env_;
    // STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
    _Env  __env_;
    _Rcvr __rcvr_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // task<T,E>::promise_type
  template <class _Ty, class _Env>
  struct task<_Ty, _Env>::__promise : __task::__promise_base<_Ty>
  {
    __promise() noexcept = default;

    [[nodiscard]]
    auto get_return_object() noexcept -> task
    {
      return task(__std::coroutine_handle<__promise>::from_promise(*this));
    }

    static constexpr __std::suspend_always initial_suspend() noexcept
    {
      return {};
    }

    auto final_suspend() noexcept
    {
      return __completed_awaiter{};
    }

    void unhandled_exception()
    {
      if constexpr (!__mapply<__mcontains<std::exception_ptr>, __error_variant_t>::value)
      {
        STDEXEC::__die("A task threw an exception but does not have std::exception_ptr in its "
                       "error_types. Either add std::exception_ptr to the task's error_types or "
                       "ensure that all code called by the task is noexcept.");
      }
      else
      {
        __state_->__errors_.template emplace<std::exception_ptr>(std::current_exception());
      }
    }

    [[nodiscard]]
    auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
    {
      return __state_->__canceled();
    }

    template <class _Error>
    constexpr auto yield_value(with_error<_Error> __error)  //
      noexcept(__nothrow_decay_copyable<_Error>)
    {
      if constexpr (__mapply<__mcontains<__decay_t<_Error>>, __error_variant_t>::value)
      {
        __state_->__errors_.template emplace<__decay_t<_Error>>(std::move(__error).error);
      }
      else
      {
        static_assert(__mnever<_Error>, "Error type not in task's error_types");
      }
      return __completed_awaiter{};
    }

    template <sender _Sender>
    constexpr auto await_transform(_Sender&& __sndr) noexcept
    {
      using __schedule_sndr_t = schedule_result_t<scheduler_type>;
      if constexpr (__completes_where_it_starts<set_value_t, env_of_t<__schedule_sndr_t>, __env>)
      {
        return STDEXEC::as_awaitable(static_cast<_Sender&&>(__sndr), *this);
      }
      else
      {
        return STDEXEC::as_awaitable(STDEXEC::affine_on(static_cast<_Sender&&>(__sndr)), *this);
      }
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept
    {
      return __env{this};
    }

    // When no allocator passed to the coroutine:
    static void* operator new(size_t __bytes)
    {
      return __promise::operator new(__bytes, std::allocator_arg, std::allocator<std::byte>{});
    }

    static void operator delete(void* __ptr, size_t __bytes) noexcept
    {
      size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__task::__memblock));
      void* const  __alloc_loc      = static_cast<__task::__memblock*>(__ptr) + __promise_blocks;
      auto*        __alloc          = static_cast<__task::__any_alloc_base*>(__alloc_loc);
      __alloc->__deallocate_(__ptr, __bytes);
    }

    // When an allocator is passed to the coroutine:
    template <class _Alloc, class... _Args>
    static void* operator new(size_t __bytes, std::allocator_arg_t, _Alloc __alloc, _Args&&...)
    {
      using __palloc_t  = std::allocator_traits<_Alloc>::template rebind_alloc<__task::__memblock>;
      using __pointer_t = std::allocator_traits<__palloc_t>::pointer;
      static_assert(std::is_pointer_v<__pointer_t>, "Allocator pointer type must be a raw pointer");

      // the number of blocks needed to store an object of type __palloc_t:
      static constexpr size_t __alloc_blocks =
        __task::__divmod(sizeof(__task::__any_alloc<__palloc_t>), sizeof(__task::__memblock));
      size_t const __promise_blocks = __task::__divmod(__bytes, sizeof(__task::__memblock));

      __palloc_t  __palloc(__alloc);
      auto* const __ptr = std::allocator_traits<__palloc_t>::allocate(__palloc,
                                                                      __promise_blocks
                                                                        + __alloc_blocks);

      // construct the allocator in the blocks immediately following the promise object:
      void* const __alloc_loc = __ptr + __promise_blocks;
      std::construct_at(static_cast<__task::__any_alloc<__palloc_t>*>(__alloc_loc),
                        std::move(__palloc));
      return __ptr;
    }

    template <class _Alloc, class... _Args>
    static void
    operator delete(void* __ptr, size_t __bytes, std::allocator_arg_t, _Alloc, _Args&&...) noexcept
    {
      __promise::operator delete(__ptr, __bytes);
    }

   private:
    template <class>
    friend struct __awaiter;
    template <class>
    friend struct __opstate;
    friend struct __opstate_base;

    struct __completed_awaiter
    {
      static constexpr bool await_ready() noexcept
      {
        return false;
      }

      static constexpr auto await_suspend(__std::coroutine_handle<__promise> __coro) noexcept  //
        -> __std::coroutine_handle<>
      {
        return __coro.promise().__state_->__completed();
      }

      static constexpr void await_resume() noexcept {}
    };

    struct __env
    {
      [[nodiscard]]
      constexpr auto query(get_scheduler_t) const noexcept -> scheduler_type
      {
        return __promise_->__state_->__sch_;
      }

      [[nodiscard]]
      constexpr auto query(get_allocator_t) const noexcept -> allocator_type
      {
        return __promise_->__state_->__get_allocator();
      }

      [[nodiscard]]
      constexpr auto query(get_stop_token_t) const noexcept -> stop_token_type
      {
        if (__promise_->__stop_.index() == 0)
        {
          return __var::__get<0>(__promise_->__stop_).get_token();
        }
        else
        {
          return __var::__get<1>(__promise_->__stop_);
        }
      }

      __promise const * __promise_;
    };

    __stop_variant_t __stop_{__no_init};
    __opstate_base*  __state_ = nullptr;
  };
#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
