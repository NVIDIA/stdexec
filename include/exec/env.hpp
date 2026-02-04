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
  using with_t = STDEXEC::prop<_Tag, _Value>;

  namespace __envs {
    struct __with_t {
      template <class _Tag, class _Value>
      constexpr auto operator()(_Tag, _Value&& __val) const {
        return STDEXEC::prop{_Tag(), static_cast<_Value&&>(__val)};
      }
    };

    template <class _Env, class _Query>
    struct __without : STDEXEC::__env::__env_base_t<_Env> {
      static_assert(STDEXEC::__nothrow_move_constructible<_Env>);

      using STDEXEC::__env::__env_base_t<_Env>::query;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto query(_Query) const noexcept = delete;
    };

    struct __without_t {
      template <class _Env, class _Query>
      constexpr auto operator()(_Env&& __env, _Query) const noexcept {
        if constexpr (STDEXEC::__queryable_with<_Env, _Query>) {
          return __without<_Env, _Query>{static_cast<_Env&&>(__env)};
        } else {
          return static_cast<_Env&&>(__env);
        }
      }
    };

    // For making an environment from key/value pairs and optionally
    // another environment.
    struct __make_env_t {
      template <
        STDEXEC::__nothrow_move_constructible _Base,
        STDEXEC::__nothrow_move_constructible _Env
      >
      constexpr auto operator()(_Base&& __base, _Env&& __env) const noexcept
        -> STDEXEC::__join_env_t<_Env, _Base> {
        return STDEXEC::__env::__join(static_cast<_Env&&>(__env), static_cast<_Base&&>(__base));
      }

      template <STDEXEC::__nothrow_move_constructible _Env>
      constexpr auto operator()(_Env&& __env) const noexcept -> _Env {
        return static_cast<_Env&&>(__env);
      }
    };
  } // namespace __envs

  inline constexpr __envs::__with_t with{};
  inline constexpr __envs::__without_t without{};
  inline constexpr __envs::__make_env_t make_env{};

  template <class... _Ts>
  using make_env_t = STDEXEC::__result_of<make_env, _Ts...>;

  template <class _Fun>
  struct env_from {
    static_assert(STDEXEC::__nothrow_move_constructible<_Fun>);

    template <class _Query, class... _Args>
      requires STDEXEC::__callable<const _Fun&, _Query, _Args...>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    auto query(_Query, _Args&&... __args) const
      noexcept(STDEXEC::__nothrow_callable<const _Fun&, _Query, _Args...>)
        -> STDEXEC::__call_result_t<const _Fun&, _Query, _Args...> {
      return __fun_(_Query(), static_cast<_Args&&>(__args)...);
    }

    STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_;
  };

  template <class _Fun>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE env_from(_Fun) -> env_from<_Fun>;

  namespace __read_with_default {
    using namespace STDEXEC;

    struct read_with_default_t;

    template <class _Tag, class _Default, class _Receiver>
    struct __opstate : __immovable {
      constexpr void start() & noexcept {
        STDEXEC_TRY {
          if constexpr (__callable<_Tag, env_of_t<_Receiver>>) {
            const auto& __env = get_env(__rcvr_);
            STDEXEC::set_value(std::move(__rcvr_), _Tag{}(__env));
          } else {
            STDEXEC::set_value(std::move(__rcvr_), std::move(__default_));
          }
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(std::move(__rcvr_), std::current_exception());
        }
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Default __default_;
      _Receiver __rcvr_;
    };

    template <class _Tag, class _Default>
    struct __sender {
      using sender_concept = STDEXEC::sender_t;

      template <class _Env>
      using __value_t =
        __minvoke<__mwith_default<__mbind_back_q<__call_result_t, _Env>, _Default>, _Tag>;
      template <class _Env>
      using __default_t = __if_c<__callable<_Tag, _Env>, __ignore, _Default>;

      template <class _Env>
      using __completions_t =
        completion_signatures<set_value_t(__value_t<_Env>), set_error_t(std::exception_ptr)>;

      template <__decays_to<__sender> _Self, class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __opstate<_Tag, __default_t<env_of_t<_Receiver>>, _Receiver> {
        return {{}, static_cast<_Self&&>(__self).__default_, static_cast<_Receiver&&>(__rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <class, class _Env>
      static consteval auto get_completion_signatures() -> __completions_t<_Env> {
        return {};
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      _Default __default_;
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

  [[deprecated("exec::write has been renamed to STDEXEC::write_env")]]
  inline constexpr STDEXEC::__write::write_env_t write{};
  [[deprecated("write_env has been moved to the STDEXEC:: namespace")]]
  inline constexpr STDEXEC::__write::write_env_t write_env{};

  namespace __write_attrs {
    using namespace STDEXEC;

    template <class _Sender, class _Attrs>
    struct __sender {
      using sender_concept = sender_t;

      constexpr auto get_env() const noexcept -> __join_env_t<const _Attrs&, env_of_t<_Sender>> {
        return STDEXEC::__env::__join(__attrs_, STDEXEC::get_env(__sndr_));
      }

      template <__decays_to<__sender> _Self, class... _Env>
      static consteval auto get_completion_signatures()
        -> __completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...> {
        return {};
      }

      template <__decays_to<__sender> _Self, class _Receiver>
        requires sender_in<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>
      constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> connect_result_t<__copy_cvref_t<_Self, _Sender>, _Receiver> {
        return STDEXEC::connect(std::forward<_Self>(__self).__sndr_, std::move(__rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _Sender __sndr_;
      _Attrs __attrs_;
    };

    struct __write_attrs_t {
      template <class _Sender, class _Attrs>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Sender snd, _Attrs __attrs_) const -> __sender<_Sender, _Attrs> {
        return __sender<_Sender, _Attrs>{
          static_cast<_Sender&&>(snd), static_cast<_Attrs&&>(__attrs_)};
      }

      template <class _Attrs>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Attrs __attrs_) const {
        return STDEXEC::__closure(*this, static_cast<_Attrs&&>(__attrs_));
      }
    };
  } // namespace __write_attrs

  inline constexpr __write_attrs::__write_attrs_t write_attrs{};

} // namespace exec

STDEXEC_PRAGMA_POP()
