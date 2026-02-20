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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__bulk.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__inline_scheduler.hpp"
#include "__meta.hpp"
#include "__parallel_scheduler_backend.hpp"
#include "__schedulers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__typeinfo.hpp"
#include "__variant.hpp"  // IWYU pragma: keep for __variant

#include <cstddef>

#include <exception>
#include <memory>
#include <span>
#include <utility>

namespace STDEXEC
{
  class task_scheduler;
  struct task_scheduler_domain;

  namespace __task
  {
    // The concrete type-erased sender returned by task_scheduler::schedule()
    struct __sender;

    template <class _Sndr>
    struct __bulk_sender;

    template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
    struct __bulk_state;

    template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
    struct __bulk_receiver;

    struct __task_scheduler_backend : system_context_replaceability::parallel_scheduler_backend
    {
      [[nodiscard]]
      virtual constexpr auto query(get_forward_progress_guarantee_t) const noexcept  //
        -> forward_progress_guarantee                                                      = 0;
      virtual constexpr auto __equal_to(void const * __other, __type_index __type) -> bool = 0;
    };

    using __backend_ptr_t = std::shared_ptr<__task_scheduler_backend>;

    //////////////////////////////////////////////////////////////////////////////////////
    // __env
    struct __unstoppable_env : __detail::__proxy_env
    {
      using __detail::__proxy_env::query;

      [[nodiscard]]
      auto query(get_stop_token_t) const noexcept -> never_stop_token
      {
        return never_stop_token{};
      }
    };

    template <bool _Unstoppable>
    using __env = __if_c<_Unstoppable, __unstoppable_env, __detail::__proxy_env>;
  }  // namespace __task

  struct _CANNOT_DISPATCH_BULK_ALGORITHM_TO_TASK_SCHEDULER_BECAUSE_THERE_IS_NO_TASK_SCHEDULER_IN_THE_ENVIRONMENT;
  struct _ADD_A_CONTINUES_ON_TRANSITION_TO_THE_TASK_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  struct task_scheduler_domain : default_domain
  {
    template <class _Sndr, class _Env, class _BulkTag = tag_of_t<_Sndr>>
      requires __one_of<_BulkTag, bulk_chunked_t, bulk_unchunked_t>
    [[nodiscard]]
    static constexpr auto transform_sender(set_value_t, _Sndr&& __sndr, _Env const & __env)
    {
      using __sched_t = __call_result_or_t<get_completion_scheduler_t<set_value_t>,
                                           __not_a_scheduler<>,
                                           env_of_t<_Sndr>,
                                           _Env const &>;

      if constexpr (!__same_as<__sched_t, task_scheduler>)
      {
        return __not_a_sender<
          _WHERE_(_IN_ALGORITHM_, _BulkTag),
          _WHAT_(
            _CANNOT_DISPATCH_BULK_ALGORITHM_TO_TASK_SCHEDULER_BECAUSE_THERE_IS_NO_TASK_SCHEDULER_IN_THE_ENVIRONMENT),
          _TO_FIX_THIS_ERROR_(
            _ADD_A_CONTINUES_ON_TRANSITION_TO_THE_TASK_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
          _WITH_PRETTY_SENDER_<_Sndr>,
          _WITH_ENVIRONMENT_(_Env)>{};
      }
      else
      {
        auto __sch = get_completion_scheduler<set_value_t>(get_env(__sndr), __env);
        return __task::__bulk_sender<_Sndr>{static_cast<_Sndr&&>(__sndr), std::move(__sch)};
      }
    }
  };

  //! @brief A type-erased scheduler.
  //!
  //! The `task_scheduler` struct is implemented in terms of a backend type derived from
  //! @c parallel_scheduler_backend, providing a type-erased interface for scheduling tasks.
  //! It exposes query functions to retrieve the completion scheduler and domain.
  //!
  //! @see parallel_scheduler_backend
  class task_scheduler
  {
    template <class _Sch, class _Alloc>
    class __backend_for;

   public:
    using scheduler_concept = scheduler_t;

    template <__not_same_as<task_scheduler> _Sch, class _Alloc = std::allocator<std::byte>>
      requires __infallible_scheduler<_Sch, __task::__env<true>>
    explicit task_scheduler(_Sch __sch, _Alloc __alloc = {})
      : __backend_(
          std::allocate_shared<__backend_for<_Sch, _Alloc>>(__alloc, std::move(__sch), __alloc))
    {}

    [[nodiscard]]
    STDEXEC_CONSTEXPR_CXX23 auto schedule() const noexcept -> __task::__sender;

    [[nodiscard]]
    bool operator==(task_scheduler const & __rhs) const noexcept = default;

    template <__not_same_as<task_scheduler> _Sch>
      requires __infallible_scheduler<_Sch, __task::__env<true>>
    [[nodiscard]]
    auto operator==(_Sch const & __other) const noexcept -> bool
    {
      return __backend_->__equal_to(std::addressof(__other), __mtypeid<_Sch>);
    }

    [[nodiscard]]
    auto query(get_forward_progress_guarantee_t) const noexcept  //
      -> forward_progress_guarantee
    {
      return __backend_->query(get_forward_progress_guarantee);
    }

    [[nodiscard]]
    constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept  //
      -> task_scheduler const &
    {
      return *this;
    }

    [[nodiscard]]
    constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
    {
      return task_scheduler_domain{};
    }

   private:
    template <class>
    friend struct __task::__bulk_sender;
    friend struct __task::__sender;

    __task::__backend_ptr_t __backend_;
  };

  namespace __task
  {
    //! @brief A type-erased opstate returned when connecting the result of
    //! task_scheduler::schedule() to a receiver.
    template <class _Rcvr>
    class __any_opstate
    {
     public:
      using operation_state_concept = operation_state_t;

      STDEXEC_CONSTEXPR_CXX23 explicit __any_opstate(__backend_ptr_t __backend, _Rcvr __rcvr)
        : __rcvr_proxy_(std::move(__rcvr))
        , __backend_(std::move(__backend))
      {}

      void start() noexcept
      {
        STDEXEC_TRY
        {
          __backend_->schedule(__rcvr_proxy_, std::span{__storage_});
        }
        STDEXEC_CATCH_ALL
        {
          __rcvr_proxy_.set_error(std::current_exception());
        }
      }

     private:
      __detail::__receiver_proxy<_Rcvr, true> __rcvr_proxy_;
      __backend_ptr_t                         __backend_;
      std::byte                               __storage_[8 * sizeof(void*)];
    };

    //! @brief A type-erased sender returned by task_scheduler::schedule().
    struct __sender
    {
      using sender_concept = sender_t;

      STDEXEC_CONSTEXPR_CXX23 explicit __sender(task_scheduler __sch)
        : __attrs_{std::move(__sch)}
      {}

      template <class _Rcvr>
      [[nodiscard]]
      constexpr auto connect(_Rcvr __rcvr) const noexcept -> __any_opstate<_Rcvr>
      {
        return __any_opstate<_Rcvr>(get_completion_scheduler<set_value_t>(__attrs_).__backend_,
                                    std::move(__rcvr));
      }

      template <class _Self, class _Env>
      [[nodiscard]]
      static consteval auto get_completion_signatures() noexcept
      {
        if constexpr (unstoppable_token<stop_token_of_t<_Env>>)
        {
          return completion_signatures<set_value_t()>{};
        }
        else
        {
          return completion_signatures<set_value_t(), set_stopped_t()>{};
        }
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __sched_attrs<task_scheduler> const &
      {
        return __attrs_;
      }

     private:
      __sched_attrs<task_scheduler> __attrs_;
    };

    //! @brief A receiver used to connect the predecessor of a bulk operation launched by a
    //! task_scheduler. Its set_value member stores the predecessor's values in the bulk
    //! operation state and then starts the bulk operation.
    template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
    struct __bulk_receiver
    {
      using receiver_concept = receiver_t;

      template <class... _As>
      void set_value(_As&&... __as) noexcept
      {
        STDEXEC_TRY
        {
          // Store the predecessor's values in the bulk operation state.
          using __values_t = __decayed_tuple<_As...>;
          __state_->__values_.template emplace<__values_t>(static_cast<_As&&>(__as)...);

          // Start the bulk operation.
          if constexpr (__same_as<_BulkTag, bulk_chunked_t>)
          {
            __state_->__backend_->schedule_bulk_chunked(__state_->__shape_,
                                                        *__state_,
                                                        std::span{__state_->__storage_});
          }
          else
          {
            __state_->__backend_->schedule_bulk_unchunked(__state_->__shape_,
                                                          *__state_,
                                                          std::span{__state_->__storage_});
          }
        }
        STDEXEC_CATCH_ALL
        {
          STDEXEC::set_error(std::move(__state_->__rcvr_), std::current_exception());
        }
      }

      template <class _Error>
      constexpr void set_error(_Error&& __err) noexcept
      {
        STDEXEC::set_error(std::move(__state_->__rcvr_), static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(std::move(__state_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Rcvr>
      {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, _Values>* __state_;
    };

    //! Returns a visitor (callable) used to invoke the bulk (unchunked) function with the
    //! predecessor's values, which are stored in a variant in the bulk operation state.
    template <bool _Parallelize, class _Fn>
    [[nodiscard]]
    constexpr auto __get_execute_bulk_fn(bulk_unchunked_t,
                                         _Fn&   __fn,
                                         size_t __shape,
                                         size_t __begin,
                                         size_t) noexcept
    {
      return [=, &__fn](auto& __args)
      {
        constexpr bool __valid_args = !__same_as<decltype(__args), __monostate&>;
        // runtime assert that we never take this path without valid args from the predecessor:
        STDEXEC_ASSERT(__valid_args);

        if constexpr (__valid_args)
        {
          // If we are not parallelizing, we need to run all the iterations sequentially.
          size_t const __increments = _Parallelize ? 1 : __shape;
          // Precompose the function with the arguments so we don't have to do it every iteration.
          auto __precomposed_fn = __apply(
            [&](auto&... __as)
            {
              return [&](size_t __i) -> void
              {
                __fn(__i, __as...);
              };
            },
            __args);
          for (size_t __i = __begin; __i < __begin + __increments; ++__i)
          {
            __precomposed_fn(__i);
          }
        }
      };
    }

    template <bool _Parallelize, class _Fn>
    struct __apply_bulk_execute
    {
      template <class... _As>
      constexpr void
      operator()(_As&... __as) const noexcept(__nothrow_callable<_Fn&, size_t, _As&...>)
      {
        if constexpr (_Parallelize)
        {
          __fn_(__begin_, __end_, __as...);
        }
        else
        {
          // If we are not parallelizing, we need to pass the entire range to the functor.
          __fn_(size_t(0), __shape_, __as...);
        }
      }

      size_t __begin_, __end_, __shape_;
      _Fn&   __fn_;
    };

    //! Returns a visitor (callable) used to invoke the bulk (chunked) function with the
    //! predecessor's values, which are stored in a variant in the bulk operation state.
    template <bool _Parallelize, class _Fn>
    [[nodiscard]]
    constexpr auto __get_execute_bulk_fn(bulk_chunked_t,
                                         _Fn&   __fn,
                                         size_t __shape,
                                         size_t __begin,
                                         size_t __end) noexcept
    {
      return [=, &__fn](auto& __args)
      {
        constexpr bool __valid_args = !__same_as<decltype(__args), __monostate&>;
        STDEXEC_ASSERT(__valid_args);

        if constexpr (__valid_args)
        {
          __apply(__apply_bulk_execute<_Parallelize, _Fn>{__begin, __end, __shape, __fn}, __args);
        }
      };
    }

    //! Stores the state for a bulk operation launched by a task_scheduler. A type-erased
    //! reference to this object is passed to either the task_scheduler's
    //! schedule_bulk_chunked or schedule_bulk_unchunked methods, which is expected to call
    //! execute(begin, end) on it to run the bulk operation. After the bulk operation is
    //! complete, set_value is called, which forwards the predecessor's values to the
    //! downstream receiver.
    template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
    struct __bulk_state
      : __detail::__receiver_proxy_base<_Rcvr,
                                        system_context_replaceability::bulk_item_receiver_proxy,
                                        true>
    {
      STDEXEC_CONSTEXPR_CXX23 explicit __bulk_state(_Rcvr           __rcvr,
                                                    size_t          __shape,
                                                    _Fn             __fn,
                                                    __backend_ptr_t __backend)
        : __bulk_state::__receiver_proxy_base(std::move(__rcvr))
        , __fn_(std::move(__fn))
        , __shape_(__shape)
        , __backend_(std::move(__backend))
      {}

      constexpr void set_value() noexcept final
      {
        // Send the stored values to the downstream receiver.
        __visit(
          [this](auto& __tupl)
          {
            constexpr bool __valid_args = __not_same_as<decltype(__tupl), __monostate&>;
            // runtime assert that we never take this path without valid args from the predecessor:
            STDEXEC_ASSERT(__valid_args);

            if constexpr (__valid_args)
            {
              __apply(STDEXEC::set_value, std::move(__tupl), std::move(this->__rcvr_));
            }
          },
          __values_);
      }

      //! Actually runs the bulk operation over the specified range.
      constexpr void execute(size_t __begin, size_t __end) noexcept final
      {
        STDEXEC_TRY
        {
          using __policy_t = std::remove_cvref_t<decltype(__declval<_Policy>().__get())>;
          constexpr bool __parallelize =
            std::same_as<__policy_t, STDEXEC::parallel_policy>
            || std::same_as<__policy_t, STDEXEC::parallel_unsequenced_policy>;
          __visit(__task::__get_execute_bulk_fn<__parallelize>(_BulkTag(),
                                                               __fn_,
                                                               __shape_,
                                                               __begin,
                                                               __end),
                  __values_);
        }
        STDEXEC_CATCH_ALL
        {
          STDEXEC::set_error(std::move(this->__rcvr_), std::current_exception());
        }
      }

     private:
      template <class, class, class, class, class>
      friend struct __bulk_receiver;

      _Fn             __fn_;
      size_t          __shape_;
      _Values         __values_{__no_init};
      __backend_ptr_t __backend_;
      std::byte       __storage_[8 * sizeof(void*)];
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // Operation state for task scheduler bulk operations
    template <class _BulkTag, class _Policy, class _Sndr, class _Fn, class _Rcvr>
    struct __bulk_opstate
    {
      using operation_state_concept = operation_state_t;

      STDEXEC_CONSTEXPR_CXX23 explicit __bulk_opstate(_Sndr&&         __sndr,
                                                      size_t          __shape,
                                                      _Fn             __fn,
                                                      _Rcvr           __rcvr,
                                                      __backend_ptr_t __backend)
        : __state_{std::move(__rcvr), __shape, std::move(__fn), std::move(__backend)}
        , __opstate1_(STDEXEC::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_}))
      {}

      constexpr void start() noexcept
      {
        STDEXEC::start(__opstate1_);
      }

     private:
      using __values_t   = value_types_of_t<_Sndr,
                                            __fwd_env_t<env_of_t<_Rcvr>>,
                                            __decayed_tuple,
                                            __mbind_front_q<__variant, __monostate>::__f>;
      using __rcvr_t     = __bulk_receiver<_BulkTag, _Policy, _Fn, _Rcvr, __values_t>;
      using __opstate1_t = connect_result_t<_Sndr, __rcvr_t>;

      __bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, __values_t> __state_;
      __opstate1_t                                            __opstate1_;
    };

    template <class _Sndr>
    struct __bulk_sender
    {
      using sender_concept = sender_t;

      STDEXEC_CONSTEXPR_CXX23 explicit __bulk_sender(_Sndr __sndr, task_scheduler __sch)
        : __sndr_(std::move(__sndr))
        , __attrs_{std::move(__sch)}
      {}

      template <class _Rcvr>
      constexpr auto connect(_Rcvr __rcvr) &&
      {
        auto& [__tag, __data, __child] = __sndr_;
        auto& [__pol, __shape, __fn]   = __data;
        return __bulk_opstate<decltype(__tag),
                              decltype(__pol),
                              decltype(__child),
                              decltype(__fn),
                              _Rcvr>{std::move(__child),
                                     static_cast<size_t>(__shape),
                                     std::move(__fn),
                                     std::move(__rcvr),
                                     std::move(__attrs_.__sched_.__backend_)};
      }

      template <__same_as<__bulk_sender> _Self, class _Env>  // accept only rvalues
      [[nodiscard]]
      static consteval auto get_completion_signatures()
      {
        // This calls get_completion_signatures on the wrapped bulk_[un]chunked sender. We
        // call it directly instead of using STDEXEC::get_completion_signatures to avoid
        // another trip through transform_sender, which would lead to infinite recursion.
        auto __completions = __decay_t<_Sndr>::template get_completion_signatures<_Sndr, _Env>();
        return STDEXEC::__transform_completion_signatures(__completions,
                                                          __decay_arguments<set_value_t>(),
                                                          {},
                                                          {},
                                                          __eptr_completion());
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __sched_attrs<task_scheduler> const &
      {
        return __attrs_;
      }

     private:
      _Sndr                         __sndr_;
      __sched_attrs<task_scheduler> __attrs_;
    };

    //! Function called by the `bulk_chunked` operation; calls `execute` on the bulk_item_receiver_proxy.
    struct __bulk_chunked_fn
    {
      constexpr void operator()(size_t __begin, size_t __end) noexcept
      {
        __rcvr_.execute(__begin, __end);
      }

      system_context_replaceability::bulk_item_receiver_proxy& __rcvr_;
    };

    //! Function called by the `bulk_unchunked` operation; calls `execute` on the bulk_item_receiver_proxy.
    struct __bulk_unchunked_fn
    {
      constexpr void operator()(size_t __idx) noexcept
      {
        __rcvr_.execute(__idx, __idx + 1);
      }

      system_context_replaceability::bulk_item_receiver_proxy& __rcvr_;
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // __emplace_into
    template <class _Ty, class _Alloc, class... _Args>
    constexpr auto
    __emplace_into(std::span<std::byte> __storage, _Alloc& __alloc, _Args&&... __args) -> _Ty&
    {
      using __traits_t = std::allocator_traits<_Alloc>::template rebind_traits<_Ty>;
      using __alloc_t  = std::allocator_traits<_Alloc>::template rebind_alloc<_Ty>;
      __alloc_t __alloc_copy{__alloc};

      bool const __in_situ = __storage.size() >= sizeof(_Ty);
      auto*      __ptr     = __in_situ ? reinterpret_cast<_Ty*>(__storage.data())
                                       : __traits_t::allocate(__alloc_copy, 1);
      __traits_t::construct(__alloc_copy, __ptr, static_cast<_Args&&>(__args)...);
      return *std::launder(__ptr);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // __opstate
    template <class _Alloc, class _Sndr, class _Env>
    class __opstate : _Alloc
    {
     public:
      using allocator_type = _Alloc;
      using receiver_proxy = system_context_replaceability::receiver_proxy;
      using __receiver_t   = __detail::__proxy_receiver<receiver_proxy, _Env>;

      constexpr explicit __opstate(_Alloc          __alloc,
                                   _Sndr           __sndr,
                                   receiver_proxy& __rcvr,
                                   bool            __in_situ)
        : _Alloc(std::move(__alloc))
        , __opstate_(STDEXEC::connect(std::move(__sndr),
                                      __receiver_t{__rcvr, this, __destroy_pfn_[__in_situ]}))
      {}
      __opstate(__opstate&&) = delete;

      constexpr void start() noexcept
      {
        STDEXEC::start(__opstate_);
      }

      [[nodiscard]]
      constexpr auto query(get_allocator_t) const noexcept -> _Alloc const &
      {
        return *this;
      }

     private:
      template <bool _InSitu>
      static constexpr void __delete_(void* __ptr) noexcept
      {
        using __traits_t        = std::allocator_traits<_Alloc>::template rebind_traits<__opstate>;
        using __alloc_t         = std::allocator_traits<_Alloc>::template rebind_alloc<__opstate>;
        auto*     __opstate_ptr = static_cast<__opstate*>(__ptr);
        __alloc_t __alloc_copy{get_allocator(*__opstate_ptr)};

        __traits_t::destroy(__alloc_copy, __opstate_ptr);
        if constexpr (!_InSitu)
        {
          __traits_t::deallocate(__alloc_copy, __opstate_ptr, 1);
        }
      }

      using __destroy_fn_t = void(void*) noexcept;
      static constexpr __destroy_fn_t* __destroy_pfn_[2]{__delete_<false>, __delete_<true>};

      using __child_opstate_t = connect_result_t<
        _Sndr,
        __detail::__proxy_receiver<system_context_replaceability::receiver_proxy, _Env>>;
      __child_opstate_t __opstate_;
    };
  }  // namespace __task

  [[nodiscard]]
  inline STDEXEC_CONSTEXPR_CXX23 auto task_scheduler::schedule() const noexcept  //
    -> __task::__sender
  {
    return __task::__sender{*this};
  }

  template <class _Sch, class _Alloc>
  class task_scheduler::__backend_for
    : public __task::__task_scheduler_backend
    , _Alloc
  {
    template <class _RcvrProxy, class _Env>
    friend struct __detail::__proxy_receiver;

    template <class _Env, class _RcvrProxy, class _Sndr>
    constexpr void
    __schedule(_RcvrProxy& __rcvr_proxy, _Sndr&& __sndr, std::span<std::byte> __storage) noexcept
    {
      STDEXEC_TRY
      {
        using __opstate_t = connect_result_t<_Sndr, __detail::__proxy_receiver<_RcvrProxy, _Env>>;
        bool const __in_situ = __storage.size() >= sizeof(__opstate_t);
        _Alloc&    __alloc   = *this;
        auto&      __opstate = __task::__emplace_into<__task::__opstate<_Alloc, _Sndr, _Env>>(
          __storage,
          __alloc,
          __alloc,
          static_cast<_Sndr&&>(__sndr),
          __rcvr_proxy,
          __in_situ);
        STDEXEC::start(__opstate);
      }
      STDEXEC_CATCH_ALL
      {
        __rcvr_proxy.set_error(std::current_exception());
      }
    }

   public:
    constexpr explicit __backend_for(_Sch __sch, _Alloc __alloc)
      : _Alloc(std::move(__alloc))
      , __sch_(std::move(__sch))
    {}

    constexpr void schedule(system_context_replaceability::receiver_proxy& __rcvr_proxy,
                            std::span<std::byte>                           __storage) noexcept final
    {
      // Check whether the receiver's stop token is unstoppable. If so, we can connect the
      // schedule sender with a receiver that doesn't propagate stop requests, which may
      // prevent stopped signals from being sent.
      auto const __token = __rcvr_proxy.template try_query<inplace_stop_token>(get_stop_token);
      bool const __unstoppable = (!__token.has_value() || !(*__token).stop_possible());
      if (__unstoppable)
      {
        __schedule<__task::__env<true>>(__rcvr_proxy, STDEXEC::schedule(__sch_), __storage);
      }
      else
      {
        __schedule<__task::__env<false>>(__rcvr_proxy, STDEXEC::schedule(__sch_), __storage);
      }
    }

    constexpr void
    schedule_bulk_chunked(size_t                                                   __size,
                          system_context_replaceability::bulk_item_receiver_proxy& __rcvr_proxy,
                          std::span<std::byte> __storage) noexcept final
    {
      auto __sndr = STDEXEC::bulk_chunked(STDEXEC::schedule(__sch_),
                                          par,
                                          __size,
                                          __task::__bulk_chunked_fn{__rcvr_proxy});
      __schedule<__task::__env<true>>(__rcvr_proxy, std::move(__sndr), __storage);
    }

    constexpr void
    schedule_bulk_unchunked(size_t                                                   __size,
                            system_context_replaceability::bulk_item_receiver_proxy& __rcvr_proxy,
                            std::span<std::byte> __storage) noexcept final
    {
      auto __sndr = STDEXEC::bulk_unchunked(STDEXEC::schedule(__sch_),
                                            par,
                                            __size,
                                            __task::__bulk_unchunked_fn{__rcvr_proxy});
      __schedule<__task::__env<true>>(__rcvr_proxy, std::move(__sndr), __storage);
    }

    [[nodiscard]]
    constexpr auto
    query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee final
    {
      return get_forward_progress_guarantee(__sch_);
    }

    [[nodiscard]]
    constexpr bool __equal_to(void const * __other, __type_index __type) final
    {
      if (__type == __mtypeid<_Sch>)
      {
        _Sch const & __other_sch = *static_cast<_Sch const *>(__other);
        return __sch_ == __other_sch;
      }
      return false;
    }

   private:
    _Sch __sch_;
  };

  namespace __detail
  {
    // Implementation of the get_scheduler_t query for __receiver_proxy_base and
    // __proxy_env from __parallel_scheduler_backend.hpp.
    template <class _Rcvr, class _Proxy, bool _Infallible>
    constexpr void
    __receiver_proxy_base<_Rcvr, _Proxy, _Infallible>::__query(get_scheduler_t,
                                                               __type_index __value_type,
                                                               void*        __dest) const noexcept
    {
      if (__value_type == __mtypeid<task_scheduler>)
      {
        auto& __val = *static_cast<std::optional<task_scheduler>*>(__dest);
        if constexpr (__callable<get_scheduler_t, env_of_t<_Rcvr>>)
        {
          __val.emplace(get_scheduler(get_env(__rcvr_)));
        }
        else
        {
          __val.emplace(inline_scheduler{});
        }
      }
    }

    [[nodiscard]]
    inline auto __proxy_env::query(get_scheduler_t) const noexcept -> task_scheduler
    {
      auto __sched = __rcvr_.template try_query<task_scheduler>(get_scheduler);
      return __sched ? *__sched : task_scheduler(inline_scheduler{});
    }
  }  // namespace __detail
}  // namespace STDEXEC
