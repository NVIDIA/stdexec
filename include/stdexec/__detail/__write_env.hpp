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
#include "__env.hpp"
#include "__queries.hpp"
#include "__sender_adaptor_closure.hpp"

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write {
    struct write_env_t {
      template <sender _Sender, class _Env>
      constexpr auto operator()(_Sender&& __sndr, _Env __env) const {
        return __make_sexpr<write_env_t>(
          static_cast<_Env&&>(__env), static_cast<_Sender&&>(__sndr));
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Env __env) const {
        return __closure(*this, static_cast<_Env&&>(__env));
      }
    };

    struct __write_env_impl : __sexpr_defaults {
      static constexpr auto get_attrs =
        []<class _Child>(__ignore, __ignore, const _Child& __child) noexcept {
          return __sync_attrs{__child};
        };

      static constexpr auto get_env = []<class _State>(__ignore, const _State& __state) noexcept
        -> decltype(__env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_))) {
        return __env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_));
      };

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, write_env_t>);
        return STDEXEC::get_completion_signatures<
          __child_of<_Self>,
          __minvoke_q<__join_env_t, const __decay_t<__data_of<_Self>>&, _Env>...
        >();
      }
    };
  } // namespace __write

  using __write::write_env_t;
  inline constexpr write_env_t write_env{};

  template <>
  struct __sexpr_impl<write_env_t> : __write::__write_env_impl { };
} // namespace STDEXEC
