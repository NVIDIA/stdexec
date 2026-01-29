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
#include "../stop_token.hpp"
#include "__basic_sender.hpp"
#include "__completion_behavior.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__optional.hpp"
#include "__queries.hpp"
#include "__receivers.hpp"
#include "__schedulers.hpp"
#include "__submit.hpp" // IWYU pragma: keep

#include <exception>

namespace STDEXEC {
  namespace __read {
    struct _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_;

    template <
      class _Receiver,
      class _Query,
      class _Ty = __call_result_t<_Query, env_of_t<_Receiver>>
    >
    struct __opstate {
      constexpr void start() noexcept {
        constexpr bool _Nothrow = __nothrow_callable<_Query, env_of_t<_Receiver>>;
        auto __query_fn = [&]() noexcept(_Nothrow) -> _Ty&& {
          auto& __result = __result_.__emplace_from(_Query(), STDEXEC::get_env(__rcvr_));
          return static_cast<_Ty&&>(__result);
        };
        STDEXEC::__set_value_from(static_cast<_Receiver&&>(__rcvr_), __query_fn);
      }

      _Receiver __rcvr_;
      __optional<_Ty> __result_;
    };

    template <class _Receiver, class _Query, class _Ty>
      requires __same_as<_Ty, _Ty&&>
    struct __opstate<_Receiver, _Query, _Ty> {
      constexpr void start() noexcept {
        // The query returns a reference type; pass it straight through to the receiver.
        STDEXEC::__set_value_from(
          static_cast<_Receiver&&>(__rcvr_), _Query(), STDEXEC::get_env(__rcvr_));
      }

      _Receiver __rcvr_;
    };

    template <class _Query>
    struct __attrs {
      template <class _Env>
        requires __callable<_Query, _Env>
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(get_completion_behavior_t<set_value_t>, const _Env&) const noexcept {
        return completion_behavior::inline_completion;
      }

      template <class _Env>
        requires __callable<_Query, _Env> && (!__nothrow_callable<_Query, _Env>)
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(get_completion_behavior_t<set_error_t>, const _Env&) const noexcept {
        return completion_behavior::inline_completion;
      }
    };

    struct read_env_t {
      template <class _Query>
      constexpr auto operator()(_Query) const noexcept {
        return __make_sexpr<read_env_t>(_Query());
      }
    };

    struct __read_env_impl : __sexpr_defaults {
      static constexpr auto get_attrs = []<class _Query>(__ignore, _Query) noexcept {
        return __attrs<_Query>{};
      };

      template <class _Self, class _Env>
      static consteval auto get_completion_signatures() {
        using __query_t = __data_of<_Self>;
        if constexpr (__callable<__query_t, _Env>) {
          using __result_t = __call_result_t<__query_t, _Env>;
          if constexpr (__nothrow_callable<__query_t, _Env>) {
            return completion_signatures<set_value_t(__result_t)>();
          } else {
            return completion_signatures<set_value_t(__result_t), set_error_t(std::exception_ptr)>();
          }
        } else {
          return STDEXEC::__throw_compile_time_error<
            _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_VALUE_FOR_THE_GIVEN_QUERY_,
            _WHERE_(_IN_ALGORITHM_, read_env_t),
            _WITH_QUERY_(__query_t),
            _WITH_ENVIRONMENT_(_Env)
          >();
        }
      };

      static constexpr auto connect =
        []<class _Self, class _Receiver>(const _Self&, _Receiver&& __rcvr) noexcept {
          using __query_t = __data_of<_Self>;
          return __opstate<_Receiver, __query_t>{static_cast<_Receiver&&>(__rcvr)};
        };

      static constexpr auto submit =
        []<class _Sender, class _Receiver>(const _Sender&, _Receiver&& __rcvr) noexcept
        requires std::is_reference_v<__call_result_t<__data_of<_Sender>, env_of_t<_Receiver>>>
      {
        static_assert(sender_expr_for<_Sender, read_env_t>);
        using __query_t = __data_of<_Sender>;
        STDEXEC::__set_value_from(
          static_cast<_Receiver&&>(__rcvr), __query_t(), STDEXEC::get_env(__rcvr));
      };
    };
  } // namespace __read

  using __read::read_env_t;
  [[deprecated("read has been renamed to read_env")]]
  inline constexpr read_env_t read{};
  inline constexpr read_env_t read_env{};

  template <>
  struct __sexpr_impl<__read::read_env_t> : __read::__read_env_impl { };

  namespace __queries {
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto get_scheduler_t::operator()() const noexcept {
      return read_env(get_scheduler);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto get_delegation_scheduler_t::operator()() const noexcept {
      return read_env(get_delegation_scheduler);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto get_allocator_t::operator()() const noexcept {
      return read_env(get_allocator);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto get_stop_token_t::operator()() const noexcept {
      return read_env(get_stop_token);
    }
  } // namespace __queries
} // namespace STDEXEC
