/*
 * Copyright (c) 2024 NVIDIA Corporation
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
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__basic_sender.hpp"

namespace exec {
  namespace __into_tuple {
    using namespace stdexec;

    template <
      __mstring _Where = "In into_tuple: "_mstr,
      __mstring _What = "The input sender must have at exactly one possible value completion"_mstr
    >
    struct _INVALID_ARGUMENT_TO_INTO_TUPLE_ { };

    template <class _Sender, class _Env>
    using __too_many_completions_error = __mexception<
      _INVALID_ARGUMENT_TO_INTO_TUPLE_<>,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_Env>
    >;

    template <class _Sender, class... _Env>
    using __try_result_tuple_t = __value_types_t<
      __completion_signatures_of_t<_Sender, _Env...>,
      __q<__decayed_std_tuple>,
      __q<__msingle>
    >;

    template <class _Sender, class... _Env>
    using __result_tuple_t = __minvoke<
      __mtry_catch_q<__try_result_tuple_t, __q<__too_many_completions_error>>,
      _Sender,
      _Env...
    >;

    template <class _Tuple>
    using __tuple_completions_t =
      stdexec::completion_signatures<set_error_t(std::exception_ptr), set_value_t(_Tuple)>;

    template <class _Sender, class... _Env>
    using __completions_t = transform_completion_signatures<
      __completion_signatures_of_t<_Sender, _Env...>,
      __meval<__tuple_completions_t, __result_tuple_t<_Sender, _Env...>>,
      __mconst<stdexec::completion_signatures<>>::__f
    >;

    struct __into_tuple_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures =
        []<class _Sender, class... _Env>(_Sender &&, _Env &&...) noexcept {
          return __completions_t<__child_of<_Sender>, _Env...>{};
        };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender &&, _Receiver &) {
          return __mtype<__result_tuple_t<__child_of<_Sender>, env_of_t<_Receiver>>>();
        };

      static constexpr auto complete =
        []<class _State, class _Receiver, class _Tag, class... _Args>(
          __ignore,
          _State,
          _Receiver &__rcvr,
          _Tag,
          _Args &&...__args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          using __tuple_t = __t<_State>;
          STDEXEC_TRY {
            set_value(
              static_cast<_Receiver &&>(__rcvr), __tuple_t{static_cast<_Args &&>(__args)...});
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(static_cast<_Receiver &&>(__rcvr), std::current_exception());
          }
        } else {
          _Tag()(static_cast<_Receiver &&>(__rcvr), static_cast<_Args &&>(__args)...);
        }
      };
    };

    struct into_tuple_t {
      template <sender _Sender>
      auto operator()(_Sender &&__sndr) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<into_tuple_t>({}, static_cast<_Sender &&>(__sndr)));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const noexcept -> __binder_back<into_tuple_t> {
        return {{}, {}, {}};
      }
    };
  } // namespace __into_tuple

  using __into_tuple::into_tuple_t;
  inline constexpr into_tuple_t into_tuple{};
} // namespace exec

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::__into_tuple::into_tuple_t> : exec::__into_tuple::__into_tuple_impl { };
} // namespace stdexec
