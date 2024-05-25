/*
 * Copyright (c) 2024 Kirk Shoop
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

#include "stdexec/__detail/__execution_fwd.hpp"
#include "stdexec/__detail/__transform_completion_signatures.hpp"

#include "stdexec/concepts.hpp"
#include "stdexec/functional.hpp"

#include "__detail/__decl_receiver.hpp"
#include "__detail/__tuple_reverse.hpp"

#include "async_object.hpp"

namespace exec {

//
// implementation of async_using
//

namespace __async_using {

template <class _Sigs>
using __variant_for_t = stdexec::__compl_sigs::__maybe_for_all_sigs<
  _Sigs,
  stdexec::__q<stdexec::__decayed_tuple>,
  stdexec::__nullable_variant_t>;

template <class... _Tys>
using __omit_set_value_t = stdexec::completion_signatures<>;

template <class _Sender, class _Env>
using __non_value_completion_signatures_t = stdexec::make_completion_signatures<
  _Sender,
  _Env,
  stdexec::completion_signatures<>,
  __omit_set_value_t>;

template <class _ResultId, class _ErrorCompletionFilterId, class _ReceiverId>
struct __destructed {
  using _Result = stdexec::__t<_ResultId>;
  using _ErrorCompletionFilter = stdexec::__t<_ErrorCompletionFilterId>;
  using _Receiver = stdexec::__t<_ReceiverId>;

  struct __t {
    using __id = __destructed;

    using receiver_concept = stdexec::receiver_t;

    _Result* __result_;
    _Receiver* __rcvr_;

    void set_value() && noexcept {
      STDEXEC_ASSERT(!__result_->valueless_by_exception());
      std::visit(
        [__rcvr = this->__rcvr_]<class _Tup>(_Tup& __tupl) noexcept -> void {
          if constexpr (stdexec::same_as<_Tup, std::monostate>) {
            std::terminate(); // reaching this indicates a bug
          } else {
            stdexec::__apply(
              [&]<class... _Args>(auto __tag, _Args&... __args) noexcept -> void {
                // use calculation of the completion_signatures in __sender
                // to filter out set_error_t(std::exception_ptr) when it cannot occur
                if constexpr (stdexec::same_as<
                  stdexec::completion_signatures<decltype(__tag)(_Args...)>, 
                  _ErrorCompletionFilter>) {
                  std::terminate();
                } else {
                  __tag(std::move(*__rcvr), (_Args&&) __args...);
                }
              },
              __tupl);
          }
        },
        *__result_);
    }

    template<class _Error>
    void set_error(_Error&&) && noexcept = delete;

    void set_stopped() && noexcept {
      STDEXEC_ASSERT(!__result_->valueless_by_exception());
      std::visit(
        [__rcvr = this->__rcvr_]<class _Tup>(_Tup& __tupl) noexcept -> void {
          if constexpr (stdexec::same_as<_Tup, std::monostate>) {
            std::terminate(); // reaching this indicates a bug
          } else {
            stdexec::__apply(
              [&]<class... _Args>(auto __tag, _Args&... __args) noexcept -> void {
                // use calculation of the completion_signatures in __sender
                // to filter out set_error_t(std::exception_ptr) when it cannot occur
                if constexpr (stdexec::same_as<
                  stdexec::completion_signatures<decltype(__tag)(_Args...)>, 
                  _ErrorCompletionFilter>) {
                  std::terminate();
                } else {
                  __tag(std::move(*__rcvr), (_Args&&) __args...);
                }
              },
              __tupl);
          }
        },
        *__result_);
    }

    stdexec::env_of_t<_Receiver> get_env() && noexcept {
      return stdexec::get_env(*__rcvr_);
    }
  };
};

template <class _ResultId, class _DestructStateId, class _ReceiverId>
struct __outside {
  using _Result = stdexec::__t<_ResultId>;
  using _DestructState = stdexec::__t<_DestructStateId>;
  using _Receiver = stdexec::__t<_ReceiverId>;

  struct __t {
    using __id = __outside;

    using receiver_concept = stdexec::receiver_t;

    _Result* __result_;
    _DestructState* __destruct_state_; 
    _Receiver* __rcvr_;

    template <class... _An>
    void set_value(_An&&... __an) && noexcept {
      using __async_result = stdexec::__decayed_tuple<stdexec::set_value_t, _An...>;
      __result_->template emplace<__async_result>(stdexec::set_value, (_An&&)__an...);
      stdexec::start(*__destruct_state_);
    }

    template <class _Error>
    void set_error(_Error&& __err) && noexcept {
      using __async_result = stdexec::__decayed_tuple<stdexec::set_error_t, _Error>;
      __result_->template emplace<__async_result>(stdexec::set_error, (_Error&&) __err);
      stdexec::start(*__destruct_state_);
    }

    void set_stopped() && noexcept {
      using __async_result = stdexec::__decayed_tuple<stdexec::set_stopped_t>;
      __result_->template emplace<__async_result>(stdexec::set_stopped);
      stdexec::start(*__destruct_state_);
    }

    stdexec::env_of_t<_Receiver> get_env() const& noexcept {
      return stdexec::get_env(*__rcvr_);
    }
  };
};

template <class _ResultId, class _InnerFnId, class _InsideStateId, class _DestructStateId, class _ErrorCompletionFilterId, class _ReceiverId, class... _FynId>
struct __constructed {
  using _Result = stdexec::__t<_ResultId>;
  using _InnerFn = stdexec::__t<_InnerFnId>;
  using _InsideState = stdexec::__t<_InsideStateId>;
  using _DestructState = stdexec::__t<_DestructStateId>;
  using _ErrorCompletionFilter = stdexec::__t<_ErrorCompletionFilterId>;
  using _Receiver = stdexec::__t<_ReceiverId>;

  struct __t {
    using __id = __constructed;

    using receiver_concept = stdexec::receiver_t;

    using __fyn_t = stdexec::__decayed_tuple<stdexec::__t<_FynId>...>;
    using __stgn_t = stdexec::__decayed_tuple<typename stdexec::__t<_FynId>::storage...>;

    using __inside = stdexec::__call_result_t<_InnerFn, typename stdexec::__t<_FynId>::handle...>;

    using __destructed_t = stdexec::__t<__destructed<_ResultId, _ErrorCompletionFilterId, _ReceiverId>>;
    using __outside_t = stdexec::__t<__outside<_ResultId, _DestructStateId, _ReceiverId>>;

    __fyn_t* __fyn_;
    __stgn_t* __stgn_;
    _Result* __result_;
    _InnerFn* __inner_;
    std::optional<_InsideState>* __inside_state_; 
    std::optional<_DestructState>* __destruct_state_; 
    _Receiver* __rcvr_;

    template<class _O>
    using __destruction_n = stdexec::__call_result_t<async_destruct_t, _O&, typename _O::storage&>;
    template<class... _Dn>
    using __destruct_all = stdexec::__call_result_t<stdexec::when_all_t, _Dn...>;
    using __destruction = exec::__apply_reverse<__destruct_all, __destruction_n<stdexec::__t<_FynId>>...>;
    using __destruct_state = stdexec::connect_result_t<__destruction, __destructed_t>;

    void __make_destruct() noexcept {
      auto __destruct = [&, this](){
        return stdexec::connect(
          stdexec::__apply(
            [&](auto&&... __fy_){ 
              return stdexec::__apply(
                [&](auto&... __stg_){ 
                  return stdexec::__apply(
                    [&](auto&&... __d_){ 
                      return stdexec::when_all(__d_...);
                    }, exec::__tuple_reverse(std::make_tuple(async_destruct(__fy_, __stg_)...)));
                }, *__stgn_);
            }, *__fyn_), __destructed_t{__result_, __rcvr_});
      };
      __destruct_state_->emplace(stdexec::__conv{__destruct});
    }

    void set_value(typename stdexec::__t<_FynId>::handle... __o) && noexcept {
      // launch nested function
      auto inside = [&, this] {
        __make_destruct();
        auto inner = (*__inner_)(typename stdexec::__t<_FynId>::handle{__o}...);
        return stdexec::connect(std::move(inner), __outside_t{__result_, &__destruct_state_->value(), __rcvr_});
      };
      if constexpr (
        stdexec::__nothrow_callable<_InnerFn, typename stdexec::__t<_FynId>::handle...> && 
        stdexec::__nothrow_callable<stdexec::connect_t, __inside, __outside_t>) {
        __inside_state_->emplace(stdexec::__conv{inside});
      } else {
        try {
          __inside_state_->emplace(stdexec::__conv{inside});
        } catch (...) {
          using __async_result = stdexec::__decayed_tuple<stdexec::set_error_t, std::exception_ptr>;
          __result_->template emplace<__async_result>(stdexec::set_error, std::current_exception());
          __make_destruct();
          stdexec::start(__destruct_state_->value());
          return;
        }
      }
      stdexec::start(__inside_state_->value());
    }

    template <class _Error>
    void set_error(_Error&& __err) && noexcept {
      using __async_result = stdexec::__decayed_tuple<stdexec::set_error_t, _Error>;
      __result_->template emplace<__async_result>(stdexec::set_error, (_Error&&) __err);
      __make_destruct();
      stdexec::start(__destruct_state_->value());
    }

    void set_stopped() && noexcept {
      using __async_result = stdexec::__decayed_tuple<stdexec::set_stopped_t>;
      __result_->template emplace<__async_result>(stdexec::set_stopped);
      __make_destruct();
      stdexec::start(__destruct_state_->value());
    }

    stdexec::env_of_t<_Receiver> get_env() const& noexcept {
      return stdexec::get_env(*__rcvr_);
    }
  };
};

// async-using operation state. 
// constructs all the async-objects into reserved storage
// destructs all the async-objects in the reserved storage
template <class _InnerFnId, class _ReceiverId, class _ErrorCompletionFilterId, class... _FynId>
struct __operation {
  using _InnerFn = stdexec::__t<_InnerFnId>;
  using _Receiver = stdexec::__t<_ReceiverId>;
  using _ErrorCompletionFilter = stdexec::__t<_ErrorCompletionFilterId>;
  using fyn_t = stdexec::__decayed_tuple<stdexec::__t<_FynId>...>;
  using stgn_t = stdexec::__decayed_tuple<typename stdexec::__t<_FynId>::storage...>;

  struct __t {
    using __id = __operation;

    template<class _O>
    using __construction_n = stdexec::__call_result_t<async_construct_t, _O&, typename _O::storage&>;
    using __construction = stdexec::__call_result_t<stdexec::when_all_t, __construction_n<stdexec::__t<_FynId>>...>;

    using __inside = stdexec::__call_result_t<_InnerFn, typename stdexec::__t<_FynId>::handle...>;
    using __result_t = __async_using::__variant_for_t<
      stdexec::__concat_completion_signatures_t<
        __async_using::__non_value_completion_signatures_t<__construction, stdexec::env_of_t<_Receiver>>,
        stdexec::completion_signatures_of_t<__inside, stdexec::env_of_t<_Receiver>>,
        // always reserve storage for exception_ptr so that the actual 
        // completion-signatures can be calculated
        stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>>>;

    using __destructed_t = stdexec::__t<__destructed<stdexec::__id<__result_t>, _ErrorCompletionFilterId, _ReceiverId>>;
    template<class _O>
    using __destruction_n = stdexec::__call_result_t<async_destruct_t, _O&, typename _O::storage&>;
    template<class... _Dn>
    using __destruct_all = stdexec::__call_result_t<stdexec::when_all_t, _Dn...>;
    using __destruction = exec::__apply_reverse<__destruct_all, __destruction_n<stdexec::__t<_FynId>>...>;
    using __destruct_state = stdexec::connect_result_t<__destruction, __destructed_t>;

    using __outside_t = stdexec::__t<__outside<stdexec::__id<__result_t>, stdexec::__id<__destruct_state>, _ReceiverId>>;
    using __inside_state = stdexec::connect_result_t<__inside, __outside_t>;

    using __constructed_t = stdexec::__t<__constructed<
      stdexec::__id<__result_t>, 
      _InnerFnId, 
      stdexec::__id<__inside_state>, 
      stdexec::__id<__destruct_state>, 
      _ErrorCompletionFilterId, 
      _ReceiverId, 
      _FynId...>>;
    using __construct_state = stdexec::connect_result_t<__construction, __constructed_t>;

    STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __rcvr_;
    STDEXEC_ATTRIBUTE((no_unique_address)) _InnerFn __inner_;
    STDEXEC_ATTRIBUTE((no_unique_address)) fyn_t __fyn_;

    STDEXEC_ATTRIBUTE((no_unique_address)) stgn_t __stgn_;
    STDEXEC_ATTRIBUTE((no_unique_address)) __result_t __result_;
    STDEXEC_ATTRIBUTE((no_unique_address)) __construct_state __construct_state_;
    STDEXEC_ATTRIBUTE((no_unique_address)) std::optional<__destruct_state> __destruct_state_;
    STDEXEC_ATTRIBUTE((no_unique_address)) std::optional<__inside_state> __inside_state_;

    __t(_Receiver __r_, _InnerFn __i_, fyn_t __fy_) : 
      __rcvr_(std::move(__r_)), __inner_(std::move(__i_)), 
      __fyn_(std::move(__fy_)), __construct_state_(
        stdexec::connect(
          stdexec::__apply(
            [this](auto&&... __fy_){ 
              return stdexec::__apply(
                [&](auto&&... __stg_){ 
                  return stdexec::when_all(async_construct(__fy_, __stg_)...); 
                }, __stgn_);
            }, __fyn_), __constructed_t{&__fyn_, &__stgn_, &__result_, &__inner_, &__inside_state_, &__destruct_state_, &__rcvr_})) {
    }

    void start() noexcept;  
  };
};

template<class _InnerFnId, class... _FynId>
struct __sender {
  using _InnerFn = stdexec::__t<_InnerFnId>;

  struct __t {
    using __id = __sender;

    using __fyn_t = stdexec::__decayed_tuple<stdexec::__t<_FynId>...>;

    _InnerFn __inner_;
    __fyn_t __fyn_;
    explicit __t(_InnerFn __i, __fyn_t __fy_) : __inner_(std::move(__i)), __fyn_(std::move(__fy_)) {}

    using sender_concept = stdexec::sender_t;

    template<class _O>
    using __construction_n = stdexec::__call_result_t<async_construct_t, _O&, typename _O::storage&>;
    using __construction = stdexec::__call_result_t<stdexec::when_all_t, __construction_n<stdexec::__t<_FynId>>...>;

    using __inside = stdexec::__call_result_t<_InnerFn, typename stdexec::__t<_FynId>::handle...>;

    using __exception_completion = stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>;

    template <class _Receiver>
    using __result_t = __async_using::__variant_for_t<
      stdexec::__concat_completion_signatures_t<
        __async_using::__non_value_completion_signatures_t<__construction, stdexec::env_of_t<_Receiver>>,
        stdexec::completion_signatures_of_t<__inside, stdexec::env_of_t<_Receiver>>,
        // always reserve *storage* for exception_ptr so that the actual 
        // completion-signatures can be calculated
        __exception_completion>>;

    //
    // calculate the completion_signatures using a __decl_receiver<_Env>
    // and an empty error completion filter
    //

    template <class _Receiver>
    using __destructed_t = stdexec::__t<__destructed<
      stdexec::__id<__result_t<_Receiver>>, 
      // do not filter out any completion in result_t 
      // so that the actual completion-signatures can be calculated
      stdexec::__id<stdexec::completion_signatures<>>, 
      stdexec::__id<_Receiver>>>;
    template<class _O>
    using __destruction_n = stdexec::__call_result_t<async_destruct_t, _O&, typename _O::storage&>;
    using __destruction = stdexec::__call_result_t<stdexec::when_all_t, __destruction_n<stdexec::__t<_FynId>>...>;
    template <class _Receiver>
    using __destruct_state = stdexec::connect_result_t<__destruction, __destructed_t<_Receiver>>;

    template <class _Receiver>
    using __outside_t = stdexec::__t<__outside<
      stdexec::__id<__result_t<_Receiver>>, 
      stdexec::__id<__destruct_state<_Receiver>>, 
      stdexec::__id<_Receiver>>>;

    template <class _Env>
    using __fake_rcvr = stdexec::__t<exec::__decl_receiver<_Env>>;

    // calculate if using InnerFn can throw 
    template<class _Receiver>
    static constexpr bool __inner_nothrow = 
      stdexec::__nothrow_callable<_InnerFn, typename stdexec::__t<_FynId>::handle...> &&
      stdexec::__nothrow_callable<stdexec::connect_t, __inside, __outside_t<_Receiver>>;

    template <class _Env>
    STDEXEC_ATTRIBUTE((always_inline))                          //
    auto get_completion_signatures(_Env&& __env) const noexcept //
      -> stdexec::__concat_completion_signatures_t<
          // add completions of sender returned from InnerFn  
          stdexec::completion_signatures_of_t<__inside, _Env>,
          // add non-set_value completions of all the async-constructors  
          __async_using::__non_value_completion_signatures_t<__construction, _Env>,
          // add std::exception_ptr if using InnerFn can throw 
          stdexec::__if_c<
            __inner_nothrow<__fake_rcvr<_Env>>,
            stdexec::completion_signatures<>,
            __exception_completion>> {
      return {};
    }

  private:
    //
    // produce the actual operation once the receiver is connected
    //

    // calculate the filter to use when applying result_t to the _Receiver
    template<class _Receiver>
    using __error_completion_filter = stdexec::__if_c<
      __inner_nothrow<_Receiver>,
      __exception_completion,
      stdexec::completion_signatures<>>;

    template <class _Receiver>
    using __operation = stdexec::__t<__operation<
      _InnerFnId, stdexec::__id<std::remove_cvref_t<_Receiver>>, 
      // apply the filter to use when applying result_t to the _Receiver
      stdexec::__id<__error_completion_filter<_Receiver>>, 
      _FynId...>>;

    using connect_t = stdexec::connect_t;
    template <stdexec::receiver _Receiver>
      requires stdexec::receiver_of<_Receiver, stdexec::completion_signatures_of_t<__t, stdexec::env_of_t<_Receiver>>>
    STDEXEC_MEMFN_DECL(auto connect)(this const __t& __self, _Receiver __rcvr) -> __operation<_Receiver> {
      return {(_Receiver&&) __rcvr, __self.__inner_, __self.__fyn_};
    }
  };
};
template<class _InnerFn, class... _Fyn>
using __sender_t = stdexec::__t<__sender<stdexec::__id<std::remove_cvref_t<_InnerFn>>, stdexec::__id<std::remove_cvref_t<_Fyn>>...>>;

template <class _InnerFnId, class _ReceiverId, class _ErrorCompletionFilterId, class... _FynId>
inline void __operation<_InnerFnId, _ReceiverId, _ErrorCompletionFilterId, _FynId...>::__t::start() noexcept {
  stdexec::start(__construct_state_);
}

} // namespace __async_using

// async_using is an algorithm that creates a set of async-objects
// and provides handles to the constructed objects to a given async-function
struct async_using_t {
  template<class _InnerFn, class... _Fyn>
  using sender_t = __async_using::__sender_t<_InnerFn, _Fyn...>;

  template<class _InnerFn, class... _Fyn>
  sender_t<_InnerFn, _Fyn...> operator()(_InnerFn&& __inner, _Fyn&&... __fyn) const {
    using __fyn_t = typename sender_t<_InnerFn, _Fyn...>::__fyn_t;
    return sender_t<_InnerFn, _Fyn...>{(_InnerFn&&)__inner, __fyn_t{(_Fyn&&)__fyn...}};
  }
};
constexpr inline static async_using_t async_using{};

} // namespace exec
