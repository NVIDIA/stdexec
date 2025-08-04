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

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__receivers.hpp"
#include "__senders_core.hpp"
#include "__transform_completion_signatures.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <optional>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  namespace __sao {
    struct _SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_ONE_ARGUMENT_;

    struct stopped_as_optional_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<stopped_as_optional_t>(__(), static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept -> __binder_back<stopped_as_optional_t> {
        return {{}, {}, {}};
      }
    };

    struct __stopped_as_optional_impl : __sexpr_defaults {
      template <class... _Tys>
        requires(sizeof...(_Tys) == 1)
      using __set_value_t = completion_signatures<set_value_t(std::optional<__decay_t<_Tys>>...)>;

      template <class _Ty>
      using __set_error_t = completion_signatures<set_error_t(_Ty)>;

      static constexpr auto get_completion_signatures =
        []<class _Self, class... _Env>(_Self&&, _Env&&...) noexcept
        requires __mvalid<__completion_signatures_of_t, __child_of<_Self>, _Env...>
      {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        using _Completions = __completion_signatures_of_t<__child_of<_Self>, _Env...>;
        if constexpr (!__valid_completion_signatures<_Completions>) {
          return _Completions();
        } else if constexpr (__single_value_sender<__child_of<_Self>, _Env...>) {
          return transform_completion_signatures<
            _Completions,
            completion_signatures<set_error_t(std::exception_ptr)>,
            __set_value_t,
            __set_error_t,
            completion_signatures<>
          >();
        } else {
          return _ERROR_<
            _WHAT_<>(_SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_ONE_ARGUMENT_),
            _IN_ALGORITHM_(stopped_as_optional_t),
            _WITH_SENDER_<__child_of<_Self>>
          >();
        }
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&&, _Receiver&) noexcept {
          static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
          using _Value = __decay_t<__single_sender_value_t<__child_of<_Self>, env_of_t<_Receiver>>>;
          return __mtype<_Value>();
        };

      static constexpr auto complete =
        []<class _State, class _Receiver, class _Tag, class... _Args>(
          __ignore,
          _State&,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (__same_as<_Tag, set_value_t>) {
          STDEXEC_TRY {
            static_assert(constructible_from<__t<_State>, _Args...>);
            stdexec::set_value(
              static_cast<_Receiver&&>(__rcvr),
              std::optional<__t<_State>>{static_cast<_Args&&>(__args)...});
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
          }
        } else if constexpr (__same_as<_Tag, set_error_t>) {
          stdexec::set_error(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        } else {
          stdexec::set_value(
            static_cast<_Receiver&&>(__rcvr), std::optional<__t<_State>>{std::nullopt});
        }
      };
    };
  } // namespace __sao

  using __sao::stopped_as_optional_t;
  inline constexpr stopped_as_optional_t stopped_as_optional{};

  template <>
  struct __sexpr_impl<stopped_as_optional_t> : __sao::__stopped_as_optional_impl { };
} // namespace stdexec
