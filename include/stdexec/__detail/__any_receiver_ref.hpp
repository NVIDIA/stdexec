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

#include "__execution_fwd.hpp"

#include "__completion_signatures.hpp"
#include "__receivers.hpp"

#include <functional>

namespace stdexec::__any_ {
  template <class _Sig>
  struct __rcvr_vfun;

  template <class _Tag, class... _Args>
  struct __rcvr_vfun<_Tag(_Args...)> {
    void (*__complete_)(void*, _Args...) noexcept;

    void operator()(void* __obj, _Tag, _Args... __args) const noexcept {
      __complete_(__obj, static_cast<_Args&&>(__args)...);
    }
  };

  template <class _GetReceiver = std::identity, class _Obj, class _Tag, class... _Args>
  constexpr auto __rcvr_vfun_fn(_Obj*, _Tag (*)(_Args...)) noexcept {
    return +[](void* __ptr, _Args... __args) noexcept {
      _Obj* __obj = static_cast<_Obj*>(__ptr);
      _Tag()(std::move(_GetReceiver()(*__obj)), static_cast<_Args&&>(__args)...);
    };
  }

  template <class _Sigs, class _Env>
  struct __receiver_vtable_for;

  template <class _Env, class... _Sigs>
  struct __receiver_vtable_for<completion_signatures<_Sigs...>, _Env> : __rcvr_vfun<_Sigs>... {
    _Env (*__do_get_env)(const void* __op_state) noexcept;

    template <class _OpState, class _GetEnv>
    static auto __s_get_env(const void* __ptr) noexcept -> _Env {
      auto* __op_state = static_cast<const _OpState*>(__ptr);
      return _GetEnv()(*__op_state);
    }

    template <class _OpState, class _GetEnv, class _GetReceiver = std::identity>
    explicit constexpr __receiver_vtable_for(_OpState* __op, _GetEnv, _GetReceiver = {}) noexcept
      : __rcvr_vfun<_Sigs>{__rcvr_vfun_fn<_GetReceiver>(__op, static_cast<_Sigs*>(nullptr))}...
      , __do_get_env{&__s_get_env<_OpState, _GetEnv>} {
    }

    template <class _Tag, class... _Args>
      requires __one_of<_Tag(_Args...), _Sigs...>
    void operator()(void* __obj, _Tag, _Args&&... __args) const noexcept {
      const __rcvr_vfun<_Tag(_Args...)>& __vfun = *this;
      __vfun(__obj, _Tag{}, static_cast<_Args&&>(__args)...);
    }

    auto __get_env(const void* __op_state) const noexcept -> _Env {
      return __do_get_env(__op_state);
    }
  };

  template <class _OpState, class _GetEnv, class _GetReceiver, class _Env, class _Sigs>
  inline constexpr __receiver_vtable_for<_Sigs, _Env>
    __receiver_vtable_for_v{static_cast<_OpState*>(nullptr), _GetEnv{}, _GetReceiver{}};

  template <class _Sigs, class _Env = env<>>
  class __receiver_ref {
   public:
    using receiver_concept = receiver_t;
    using __t = __receiver_ref;
    using __id = __receiver_ref;

    template <class _OpState, class _GetEnv, class _GetReceiver = std::identity>
    __receiver_ref(_OpState& __op_state, _GetEnv, _GetReceiver = {}) noexcept
      : __vtable_{&__any_::__receiver_vtable_for_v<_OpState, _GetEnv, _GetReceiver, _Env, _Sigs>}
      , __op_state_{&__op_state} {
    }

    auto get_env() const noexcept -> decltype(auto) {
      return __vtable_->__get_env(__op_state_);
    }

    template <class... _As>
      requires __callable<__receiver_vtable_for<_Sigs, _Env>, void*, set_value_t, _As...>
    void set_value(_As&&... __as) noexcept {
      (*__vtable_)(__op_state_, set_value_t(), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
      requires __callable<__receiver_vtable_for<_Sigs, _Env>, void*, set_error_t, _Error>
    void set_error(_Error&& __err) noexcept {
      (*__vtable_)(__op_state_, set_error_t(), static_cast<_Error&&>(__err));
    }

    void set_stopped() noexcept
      requires __callable<__receiver_vtable_for<_Sigs, _Env>, void*, set_stopped_t>
    {
      (*__vtable_)(__op_state_, set_stopped_t());
    }

   private:
    const __receiver_vtable_for<_Sigs, _Env>* __vtable_;
    void* __op_state_;
  };
} // namespace stdexec::__any_
