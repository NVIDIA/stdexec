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

    template <stdexec::same_as<stdexec::set_value_t> _Tag>
    friend void tag_invoke(_Tag, __t&& __rcvr) noexcept {
      STDEXEC_ASSERT(!__rcvr.__result_->valueless_by_exception());
      std::visit(
        [__rcvr = __rcvr]<class _Tup>(_Tup& __tupl) noexcept -> void {
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
                  __tag(std::move(*__rcvr.__rcvr_), (_Args&&) __args...);
                }
              },
              __tupl);
          }
        },
        *__rcvr.__result_);
    }

    template <stdexec::same_as<stdexec::set_stopped_t> _Tag>
    friend void tag_invoke(_Tag __d, __t&& __rcvr) noexcept {
      STDEXEC_ASSERT(!__rcvr.__result_->valueless_by_exception());
      std::visit(
        [__rcvr = __rcvr]<class _Tup>(_Tup& __tupl) noexcept -> void {
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
                  __tag(std::move(*__rcvr.__rcvr_), (_Args&&) __args...);
                }
              },
              __tupl);
          }
        },
        *__rcvr.__result_);
    }

    friend stdexec::env_of_t<_Receiver> tag_invoke(stdexec::get_env_t, const __t& __rcvr) noexcept {
      return stdexec::get_env(*__rcvr.__rcvr_);
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

    template <stdexec::same_as<stdexec::set_value_t> _Tag, class... _An>
    friend void tag_invoke(_Tag, __t&& __rcvr, _An&&... __an) noexcept {
      using __async_result = stdexec::__decayed_tuple<_Tag, _An...>;
      __rcvr.__result_->template emplace<__async_result>(_Tag(), (_An&&)__an...);
      stdexec::start(*__rcvr.__destruct_state_);
    }

    template <stdexec::same_as<stdexec::set_error_t> _Tag, class _Error>
    friend void tag_invoke(_Tag, __t&& __rcvr, _Error&& __err) noexcept {
      using __async_result = stdexec::__decayed_tuple<_Tag, _Error>;
      __rcvr.__result_->template emplace<__async_result>(_Tag(), (_Error&&) __err);
      stdexec::start(*__rcvr.__destruct_state_);
    }

    template <stdexec::same_as<stdexec::set_stopped_t> _Tag>
    friend void tag_invoke(_Tag __d, __t&& __rcvr) noexcept {
      using __async_result = stdexec::__decayed_tuple<_Tag>;
      __rcvr.__result_->template emplace<__async_result>(_Tag());
      stdexec::start(*__rcvr.__destruct_state_);
    }

    friend stdexec::env_of_t<_Receiver> tag_invoke(stdexec::get_env_t, const __t& __rcvr) noexcept {
      return stdexec::get_env(*__rcvr.__rcvr_);
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
    _DestructState* __destruct_state_; 
    _Receiver* __rcvr_;

    template<class _O>
    using __destruction_n = stdexec::__call_result_t<async_destruct_t, _O&, typename _O::storage&>;
    using __destruction = stdexec::__call_result_t<stdexec::when_all_t, __destruction_n<stdexec::__t<_FynId>>...>;

    template <stdexec::same_as<stdexec::set_value_t> _Tag>
    friend void tag_invoke(_Tag, __t&& __rcvr, typename stdexec::__t<_FynId>::handle... __o) noexcept {
      // launch nested function
      auto inside = [&] {
        auto inner = (*__rcvr.__inner_)(typename stdexec::__t<_FynId>::handle{__o}...);
        return stdexec::connect(std::move(inner), __outside_t{__rcvr.__result_, __rcvr.__destruct_state_, __rcvr.__rcvr_});
      };
      if constexpr (
        stdexec::__nothrow_callable<_InnerFn, typename stdexec::__t<_FynId>::handle...> && 
        stdexec::__nothrow_callable<stdexec::connect_t, __inside, __outside_t>) {
        __rcvr.__inside_state_->emplace(stdexec::__conv{inside});
      } else {
        try {
          __rcvr.__inside_state_->emplace(stdexec::__conv{inside});
        } catch (...) {
          using __async_result = stdexec::__decayed_tuple<stdexec::set_error_t, std::exception_ptr>;
          __rcvr.__result_->template emplace<__async_result>(stdexec::set_error, std::current_exception());
          stdexec::start(*__rcvr.__destruct_state_);
          return;
        }
      }
      stdexec::start(__rcvr.__inside_state_->value());
    }

    template <stdexec::same_as<stdexec::set_error_t> _Tag, class _Error>
    friend void tag_invoke(_Tag, __t&& __rcvr, _Error&& __err) noexcept {
      using __async_result = stdexec::__decayed_tuple<_Tag, _Error>;
      __rcvr.__result_->template emplace<__async_result>(_Tag(), (_Error&&) __err);
      stdexec::start(*__rcvr.__destruct_state_);
    }

    template <stdexec::same_as<stdexec::set_stopped_t> _Tag>
    friend void tag_invoke(_Tag __d, __t&& __rcvr) noexcept {
      using __async_result = stdexec::__decayed_tuple<_Tag>;
      __rcvr.__result_->template emplace<__async_result>(_Tag());
      stdexec::start(*__rcvr.__destruct_state_);
    }

    friend stdexec::env_of_t<_Receiver> tag_invoke(stdexec::get_env_t, const __t& __rcvr) noexcept {
      return stdexec::get_env(*__rcvr.__rcvr_);
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
    STDEXEC_ATTRIBUTE((no_unique_address)) __destruct_state __destruct_state_;
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
            }, __fyn_), __constructed_t{&__fyn_, &__stgn_, &__result_, &__inner_, &__inside_state_, &__destruct_state_, &__rcvr_})),
      __destruct_state_(
        stdexec::connect(
          stdexec::__apply(
            [&](auto&&... __fy_){ 
              return stdexec::__apply(
                [&](auto&... __stg_){ 
                  return stdexec::__apply(
                    [&](auto&&... __d_){ 
                      return stdexec::when_all(__d_...);
                    }, exec::__tuple_reverse(std::make_tuple(async_destruct(__fy_, __stg_)...)));
                }, __stgn_);
            }, __fyn_), __destructed_t{&__result_, &__rcvr_})) {
    }

    friend void tag_invoke(stdexec::start_t, __t& __self) noexcept {
      __self.__start_();
    }

    void __start_() noexcept;
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

    template < stdexec::same_as<stdexec::get_completion_signatures_t> _Tag, stdexec::__decays_to<__t> _Self, class _Env>
    STDEXEC_ATTRIBUTE((always_inline))                                  //
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) noexcept //
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

    template <class _Receiver>
    friend __operation<_Receiver> tag_invoke(
      stdexec::connect_t,
      const __t& __self,
      _Receiver __rcvr) {
      return __self.__connect_((_Receiver&&) __rcvr);
    }
    template <class _Receiver>
    __operation<_Receiver> __connect_(_Receiver&& __rcvr) const {
      return {(_Receiver&&) __rcvr, __inner_, __fyn_};
    }
  };
};
template<class _InnerFn, class... _Fyn>
using __sender_t = stdexec::__t<__sender<stdexec::__id<std::remove_cvref_t<_InnerFn>>, stdexec::__id<std::remove_cvref_t<_Fyn>>...>>;

template <class _InnerFnId, class _ReceiverId, class _ErrorCompletionFilterId, class... _FynId>
inline void __operation<_InnerFnId, _ReceiverId, _ErrorCompletionFilterId, _FynId...>::__t::__start_() noexcept {
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
