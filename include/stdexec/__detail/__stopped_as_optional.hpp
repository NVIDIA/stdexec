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
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__receivers.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <optional>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  namespace __sao {
    struct _SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_ONE_ARGUMENT_;

    struct stopped_as_optional_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender&& __sndr) const -> __well_formed_sender auto {
        return __make_sexpr<stopped_as_optional_t>(__(), static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept {
        return __closure(*this);
      }
    };

    template <class _Receiver, class _Value>
    struct __state {
      using __receiver_t = _Receiver;
      using __value_t = _Value;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    struct __stopped_as_optional_impl : __sexpr_defaults {
      template <class... _Tys>
        requires(sizeof...(_Tys) == 1)
      using __set_value_t = completion_signatures<set_value_t(std::optional<__decay_t<_Tys>>...)>;

      template <class _Ty>
      using __set_error_t = completion_signatures<set_error_t(_Ty)>;

      template <class _Self, class... _Env>
      static constexpr auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        STDEXEC_COMPLSIGS_LET(
          __completions, STDEXEC::get_completion_signatures<__child_of<_Self>, _Env...>()) {
          using _Completions = decltype(__completions);
          if constexpr (__single_value_sender<__child_of<_Self>, _Env...>) {
            return transform_completion_signatures<
              _Completions,
              completion_signatures<set_error_t(std::exception_ptr)>,
              __set_value_t,
              __set_error_t,
              completion_signatures<>
            >();
          } else {
            return STDEXEC::__throw_compile_time_error<
              _WHAT_(_SENDER_MUST_HAVE_EXACTLY_ONE_VALUE_COMPLETION_WITH_ONE_ARGUMENT_),
              _WHERE_(_IN_ALGORITHM_, stopped_as_optional_t),
              _WITH_PRETTY_SENDER_<__child_of<_Self>>
            >();
          }
        }
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&&, _Receiver&& __rcvr) noexcept
        requires sender_in<__child_of<_Self>, env_of_t<_Receiver>>
      {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        using _Value = __decay_t<__single_sender_value_t<__child_of<_Self>, env_of_t<_Receiver>>>;
        return __state<_Receiver, _Value>{static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto complete = []<class _State, class _Tag, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        using __value_t = _State::__value_t;
        if constexpr (__same_as<_Tag, set_value_t>) {
          STDEXEC_TRY {
            static_assert(__std::constructible_from<__value_t, _Args...>);
            STDEXEC::set_value(
              static_cast<_State&&>(__state).__rcvr_,
              std::optional<__value_t>{static_cast<_Args&&>(__args)...});
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
          }
        } else if constexpr (__same_as<_Tag, set_error_t>) {
          STDEXEC::set_error(
            static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        } else {
          STDEXEC::set_value(
            static_cast<_State&&>(__state).__rcvr_, std::optional<__value_t>{std::nullopt});
        }
      };
    };
  } // namespace __sao

  using __sao::stopped_as_optional_t;
  inline constexpr stopped_as_optional_t stopped_as_optional{};

  template <>
  struct __sexpr_impl<stopped_as_optional_t> : __sao::__stopped_as_optional_impl { };
} // namespace STDEXEC
