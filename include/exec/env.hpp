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

#include "../stdexec/execution.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(1302)

namespace exec {
  template <class _Tag, class _Value>
  using with_t = stdexec::prop<_Tag, _Value>;

  namespace __envs {
    struct __with_t {
      template <class _Tag, class _Value>
      constexpr auto operator()(_Tag, _Value&& __val) const {
        return stdexec::prop{_Tag(), static_cast<_Value&&>(__val)};
      }
    };

    struct __without_t {
      template <class _Env, class _Tag>
      constexpr auto operator()(_Env&& __env, _Tag) const -> decltype(auto) {
        return stdexec::__env::__without(static_cast<_Env&&>(__env), _Tag());
      }
    };

    // For making an environment from key/value pairs and optionally
    // another environment.
    struct __make_env_t {
      template <
        stdexec::__nothrow_move_constructible _Base,
        stdexec::__nothrow_move_constructible _Env
      >
      constexpr auto operator()(_Base&& __base, _Env&& __env) const noexcept
        -> stdexec::__join_env_t<_Env, _Base> {
        return stdexec::__env::__join(static_cast<_Env&&>(__env), static_cast<_Base&&>(__base));
      }

      template <stdexec::__nothrow_move_constructible _Env>
      constexpr auto operator()(_Env&& __env) const noexcept -> _Env {
        return static_cast<_Env&&>(__env);
      }
    };
  } // namespace __envs

  inline constexpr __envs::__with_t with{};
  inline constexpr __envs::__without_t without{};
  inline constexpr __envs::__make_env_t make_env{};

  template <class... _Ts>
  using make_env_t = stdexec::__result_of<make_env, _Ts...>;

  namespace __read_with_default {
    using namespace stdexec;

    struct read_with_default_t;

    template <class _Tag, class _DefaultId, class _ReceiverId>
    struct __operation {
      using _Default = stdexec::__t<_DefaultId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __immovable {
        using __id = __operation;

        STDEXEC_ATTRIBUTE(no_unique_address) _Default __default_;
        _Receiver __rcvr_;

        constexpr void start() & noexcept {
          STDEXEC_TRY {
            if constexpr (__callable<_Tag, env_of_t<_Receiver>>) {
              const auto& __env = get_env(__rcvr_);
              stdexec::set_value(std::move(__rcvr_), _Tag{}(__env));
            } else {
              stdexec::set_value(std::move(__rcvr_), std::move(__default_));
            }
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(std::move(__rcvr_), std::current_exception());
          }
        }
      };
    };

    template <class _Tag, class _Default, class _Receiver>
    using __operation_t = __t<__operation<_Tag, __id<_Default>, __id<_Receiver>>>;

    template <class _Tag, class _Default>
    struct __sender {
      using __id = __sender;
      using __t = __sender;
      using sender_concept = stdexec::sender_t;
      STDEXEC_ATTRIBUTE(no_unique_address) _Default __default_;

      template <class _Env>
      using __value_t =
        __minvoke<__with_default<__mbind_back_q<__call_result_t, _Env>, _Default>, _Tag>;
      template <class _Env>
      using __default_t = __if_c<__callable<_Tag, _Env>, __ignore, _Default>;

      template <class _Env>
      using __completions_t =
        completion_signatures<set_value_t(__value_t<_Env>), set_error_t(std::exception_ptr)>;

      template <__decays_to<__sender> _Self, class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      static constexpr auto connect(_Self&& __self, _Receiver __rcvr)
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __operation_t<_Tag, __default_t<env_of_t<_Receiver>>, _Receiver> {
        return {{}, static_cast<_Self&&>(__self).__default_, static_cast<_Receiver&&>(__rcvr)};
      }

      template <class _Env>
      constexpr auto get_completion_signatures(_Env&&) -> __completions_t<_Env> {
        return {};
      }
    };

    struct __read_with_default_t {
      template <class _Tag, class _Default>
      constexpr auto
        operator()(_Tag, _Default&& __default) const -> __sender<_Tag, __decay_t<_Default>> {
        return {static_cast<_Default&&>(__default)};
      }
    };
  } // namespace __read_with_default

  inline constexpr __read_with_default::__read_with_default_t read_with_default{};

  [[deprecated("exec::write has been renamed to stdexec::write_env")]]
  inline constexpr stdexec::__write::write_env_t write{};
  [[deprecated("write_env has been moved to the stdexec:: namespace")]]
  inline constexpr stdexec::__write::write_env_t write_env{};

  namespace __write_attrs {
    using namespace stdexec;

    template <class _SenderId, class _Attrs>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using sender_concept = sender_t;
        using __id = __sender;
        _Sender __sndr_;
        _Attrs __attrs_;

        constexpr auto get_env() const noexcept -> __join_env_t<const _Attrs&, env_of_t<_Sender>> {
          return stdexec::__env::__join(__attrs_, stdexec::get_env(__sndr_));
        }

        template <__decays_to<__t> _Self, class... _Env>
        static constexpr auto get_completion_signatures(_Self&&, _Env&&...)
          -> completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...> {
          return {};
        }

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_in<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>
        static constexpr auto connect(_Self&& __self, _Receiver __rcvr)
          -> connect_result_t<__copy_cvref_t<_Self, _Sender>, _Receiver> {
          return stdexec::connect(std::forward<_Self>(__self).__sndr_, std::move(__rcvr));
        }
      };
    };

    struct __write_attrs_t {
      template <class _Sender, class _Attrs>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto
        operator()(_Sender snd, _Attrs __attrs_) const -> __write_attrs::__sender<_Sender, _Attrs> {
        return __t<__write_attrs::__sender<__id<_Sender>, _Attrs>>{
          static_cast<_Sender&&>(snd), static_cast<_Attrs&&>(__attrs_)};
      }

      template <class _Attrs>
      struct __closure {
        _Attrs __attrs_;

        template <class _Sender>
        STDEXEC_ATTRIBUTE(host, device)
        friend constexpr auto operator|(_Sender __sndr_, __closure _clsr) {
          return __t<__write_attrs::__sender<__id<_Sender>, _Attrs>>{
            static_cast<_Sender&&>(__sndr_), static_cast<_Attrs&&>(_clsr.__attrs_)};
        }
      };

      template <class _Attrs>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Attrs __attrs_) const {
        return __closure<_Attrs>{static_cast<_Attrs&&>(__attrs_)};
      }
    };
  } // namespace __write_attrs

  inline constexpr __write_attrs::__write_attrs_t write_attrs{};

} // namespace exec

STDEXEC_PRAGMA_POP()
