/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../../stdexec/execution.hpp"
#include "../../materialize.hpp"

namespace exec { namespace __shared {
  using namespace stdexec;
  template <class _ValuesVariant>
  struct __value_state {
    _ValuesVariant __values_{};
  };

  template <class _ValuesVariant, class _RcvrId>
  struct __value_operation {
    struct __t {
      using __id = __value_operation;
      using _Rcvr = stdexec::__t<_RcvrId>;

      STDEXEC_NO_UNIQUE_ADDRESS _Rcvr __rcvr_;
      __value_state<_ValuesVariant>* __state_;

      friend void tag_invoke(start_t, __t& __self) noexcept {
        std::visit(
          [&__self]<class _Tuple>(_Tuple&& __tup) noexcept {
            if constexpr (!same_as<std::monostate, __decay_t<_Tuple>>) {
              std::apply(
                [&__self]<class... _Args>(_Args&&... __args) noexcept {
                  stdexec::set_value(
                    static_cast<_Rcvr&&>(__self.__rcvr_), static_cast<_Args&&>(__args)...);
                },
                static_cast<_Tuple&&>(__tup));
            }
          },
          static_cast<_ValuesVariant&&>(__self.__state_->__values_));
      }
    };
  };

  template <class... _Args>
  using __to_set_value_impl = set_value_t(_Args&&...);

  template <class _Tuple>
  using __to_set_value = __mapply<__q<__to_set_value_impl>, _Tuple>;

  template <class _ValuesVariant>
  struct __value_sender {
    struct __t {
      using __id = __value_sender;
      __value_state<_ValuesVariant>* __state_;

      using _ValuesWithoutMonostate = __mapply<__pop_front<>, _ValuesVariant>;

      using completion_signatures = __mapply<
        __transform<__q<__to_set_value>, __q<stdexec::completion_signatures>>,
        _ValuesWithoutMonostate>;

      template <class _Rcvr>
      using __value_operation_t =
        stdexec::__t<__value_operation<_ValuesVariant, stdexec::__id<__decay_t<_Rcvr>>>>;

      template <__decays_to<__t> _Self, receiver_of<completion_signatures> _Rcvr>
      friend __value_operation_t<_Rcvr>
        tag_invoke(connect_t, _Self&& __self, _Rcvr&& __rcvr) noexcept {
        return __value_operation_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __self.__state_};
      }
    };
  };

  template <class _Sndr>
  using __demat_t = decltype(exec::dematerialize(__declval<_Sndr>()));

  template <class _Sndr>
  using __mat_t = decltype(exec::materialize(__declval<_Sndr>()));

  template <class _BaseReceiverId>
  struct __receiver_ref {
    struct __t {
      using __id = __receiver_ref;
      using _BaseReceiver = stdexec::__t<_BaseReceiverId>;
      _BaseReceiver* __rcvr_;

      template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
      friend env_of_t<_BaseReceiver> tag_invoke(_GetEnv, const _Self& __self) noexcept {
        return stdexec::get_env(*__self.__rcvr_);
      }

      template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class... _Args>
        requires __callable<set_value_t, _BaseReceiver&&, _Args...>
      friend void tag_invoke(_SetValue, _Self&& __self, _Args&&... __args) noexcept {
        return _SetValue{}(
          static_cast<_BaseReceiver&&>(*__self.__rcvr_), static_cast<_Args&&>(__args)...);
      }

      template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        requires __callable<set_error_t, _BaseReceiver&&, _Error>
      friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
        return _SetError{}(
          static_cast<_BaseReceiver&&>(*__self.__rcvr_), static_cast<_Error&&>(__error));
      }

      template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
      friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
        return _SetStopped{}(static_cast<_BaseReceiver&&>(*__self.__rcvr_));
      }
    };
  };
}}