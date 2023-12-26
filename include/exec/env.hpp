/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(1302)

namespace exec {
  template <class _Tag, class _Value = void>
  using with_t = stdexec::__with<_Tag, _Value>;

  namespace __detail {
    struct __with_t {
      template <class _Tag, class _Value>
      with_t<_Tag, stdexec::__decay_t<_Value>> operator()(_Tag, _Value&& __val) const {
        return stdexec::__mkprop((_Value&&) __val, _Tag());
      }

      template <class _Tag>
      with_t<_Tag> operator()(_Tag) const {
        return stdexec::__mkprop(_Tag());
      }
    };
  } // namespace __detail

  inline constexpr __detail::__with_t with{};

  inline constexpr stdexec::__env::__make_env_t make_env{};

  template <class... _Ts>
  using make_env_t = stdexec::__make_env_t<_Ts...>;

  namespace __read_with_default {
    using namespace stdexec;

    struct read_with_default_t;

    template <class _Tag, class _DefaultId, class _ReceiverId>
    struct __operation : __immovable {
      using _Default = __t<_DefaultId>;
      using _Receiver = __t<_ReceiverId>;

      STDEXEC_ATTRIBUTE((no_unique_address)) _Default __default_;
      _Receiver __rcvr_;

      friend void tag_invoke(start_t, __operation& __self) noexcept {
        try {
          if constexpr (__callable<_Tag, env_of_t<_Receiver>>) {
            const auto& __env = get_env(__self.__rcvr_);
            set_value(std::move(__self.__rcvr_), _Tag{}(__env));
          } else {
            set_value(std::move(__self.__rcvr_), std::move(__self.__default_));
          }
        } catch (...) {
          set_error(std::move(__self.__rcvr_), std::current_exception());
        }
      }
    };

    template <class _Tag, class _DefaultId>
    struct __sender {
      using _Default = __t<_DefaultId>;
      using sender_concept = stdexec::sender_t;
      STDEXEC_ATTRIBUTE((no_unique_address)) _Default __default_;

      template <class _Env>
      using __value_t =
        __minvoke<__with_default<__mbind_back_q<__call_result_t, _Env>, _Default>, _Tag>;
      template <class _Env>
      using __default_t = __if_c<__callable<_Tag, _Env>, __ignore, _Default>;
      template <class _Env>
      using __completions_t =
        completion_signatures< set_value_t(__value_t<_Env>), set_error_t(std::exception_ptr)>;

      template <__decays_to<__sender> _Self, class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __operation<_Tag, __x<__default_t<env_of_t<_Receiver>>>, __x<_Receiver>> {
        return {{}, ((_Self&&) __self).__default_, (_Receiver&&) __rcvr};
      }

      template <class _Env>
      friend auto tag_invoke(get_completion_signatures_t, __sender, _Env&&)
        -> __completions_t<_Env> {
        return {};
      }
    };

    struct __read_with_default_t {
      template <class _Tag, class _Default>
      constexpr auto operator()(_Tag, _Default&& __default) const
        -> __sender<_Tag, __x<__decay_t<_Default>>> {
        return {(_Default&&) __default};
      }
    };
  } // namespace __read_with_default

  inline constexpr __read_with_default::__read_with_default_t read_with_default{};

  inline constexpr stdexec::__write_::__write_t write{};
} // namespace exec

STDEXEC_PRAGMA_POP()
