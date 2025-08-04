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
#include "__env.hpp"
#include "__optional.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__submit.hpp" // IWYU pragma: keep

#include <exception>

namespace stdexec {
  namespace __read {
    template <class _Tag, class _ReceiverId>
    using __result_t = __call_result_t<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    template <class _Tag, class _ReceiverId>
    concept __nothrow_t = __nothrow_callable<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    inline constexpr __mstring __query_failed_diag =
      "The current execution environment doesn't have a value for the given query."_mstr;

    template <class _Tag, class _Env>
    using __query_failed_error = __mexception<
      _NOT_CALLABLE_<"In stdexec::read_env()..."_mstr, __query_failed_diag>,
      _WITH_QUERY_<_Tag>,
      _WITH_ENVIRONMENT_<_Env>
    >;

    template <class _Tag, class _Env>
      requires __callable<_Tag, _Env>
    using __completions_t = __if_c<
      __nothrow_callable<_Tag, _Env>,
      completion_signatures<set_value_t(__call_result_t<_Tag, _Env>)>,
      completion_signatures<set_value_t(__call_result_t<_Tag, _Env>), set_error_t(std::exception_ptr)>
    >;

    template <class _Tag, class _Ty>
    struct __state {
      using __query = _Tag;
      using __result = _Ty;
      __optional<_Ty> __result_;
    };

    template <class _Tag, class _Ty>
      requires __same_as<_Ty, _Ty&&>
    struct __state<_Tag, _Ty> {
      using __query = _Tag;
      using __result = _Ty;
    };

    struct read_env_t {
      template <class _Tag>
      constexpr auto operator()(_Tag) const noexcept {
        return __make_sexpr<read_env_t>(_Tag());
      }
    };

    struct __read_env_impl : __sexpr_defaults {
      template <class _Tag, class _Env>
      using __completions_t =
        __minvoke<__mtry_catch_q<__read::__completions_t, __q<__query_failed_error>>, _Tag, _Env>;

      static constexpr auto get_attrs =
        [](__ignore) noexcept -> cprop<__is_scheduler_affine_t, true> {
        return {};
      };

      static constexpr auto get_completion_signatures =
        []<class _Self, class _Env>(const _Self&, _Env&&) noexcept
        -> __completions_t<__data_of<_Self>, _Env> {
        return {};
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(const _Self&, _Receiver&) noexcept {
          using __query = __data_of<_Self>;
          using __result = __call_result_t<__query, env_of_t<_Receiver>>;
          return __state<__query, __result>();
        };

      static constexpr auto start =
        []<class _State, class _Receiver>(_State& __state, _Receiver& __rcvr) noexcept -> void {
        using __query = typename _State::__query;
        using __result = typename _State::__result;
        if constexpr (__same_as<__result, __result&&>) {
          // The query returns a reference type; pass it straight through to the receiver.
          stdexec::__set_value_invoke(
            static_cast<_Receiver&&>(__rcvr), __query(), stdexec::get_env(__rcvr));
        } else {
          constexpr bool _Nothrow = __nothrow_callable<__query, env_of_t<_Receiver>>;
          auto __query_fn = [&]() noexcept(_Nothrow) -> __result&& {
            __state.__result_.__emplace_from(
              [&]() noexcept(_Nothrow) { return __query()(stdexec::get_env(__rcvr)); });
            return static_cast<__result&&>(*__state.__result_);
          };
          stdexec::__set_value_invoke(static_cast<_Receiver&&>(__rcvr), __query_fn);
        }
      };

      static constexpr auto submit =
        []<class _Sender, class _Receiver>(const _Sender&, _Receiver __rcvr) noexcept
        requires std::is_reference_v<__call_result_t<__data_of<_Sender>, env_of_t<_Receiver>>>
      {
        static_assert(sender_expr_for<_Sender, read_env_t>);
        using __query = __data_of<_Sender>;
        stdexec::__set_value_invoke(
          static_cast<_Receiver&&>(__rcvr), __query(), stdexec::get_env(__rcvr));
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
    template <class _Tag>
    inline auto get_scheduler_t::operator()() const noexcept {
      return read_env(get_scheduler);
    }

    template <class _Tag>
    inline auto get_delegation_scheduler_t::operator()() const noexcept {
      return read_env(get_delegation_scheduler);
    }

    template <class _Tag>
    inline auto get_allocator_t::operator()() const noexcept {
      return read_env(get_allocator);
    }

    template <class _Tag>
    inline auto get_stop_token_t::operator()() const noexcept {
      return read_env(get_stop_token);
    }
  } // namespace __queries
} // namespace stdexec
