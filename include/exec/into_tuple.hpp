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

#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/execution.hpp"

namespace exec {
  namespace __into_tuple {
    using namespace STDEXEC;

    struct into_tuple_t;
    struct _THE_INPUT_SENDER_MUST_HAVE_AT_EXACTLY_ONE_POSSIBLE_VALUE_COMPLETION_ { };

    template <class _Sender, class _Env>
    using __too_many_completions_error = __mexception<
      _WHAT_(_INVALID_ARGUMENT_),
      _WHY_(_THE_INPUT_SENDER_MUST_HAVE_AT_EXACTLY_ONE_POSSIBLE_VALUE_COMPLETION_),
      _WHERE_(_IN_ALGORITHM_, into_tuple_t),
      _WITH_PRETTY_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_(_Env)
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
      STDEXEC::completion_signatures<set_error_t(std::exception_ptr), set_value_t(_Tuple)>;

    template <class _Sender, class... _Env>
    using __completions_t = transform_completion_signatures<
      __completion_signatures_of_t<_Sender, _Env...>,
      __minvoke_q<__tuple_completions_t, __result_tuple_t<_Sender, _Env...>>,
      __mconst<STDEXEC::completion_signatures<>>::__f
    >;

    struct __into_tuple_impl : __sexpr_defaults {
      template <class _Receiver, class _Tuple>
      struct __state {
        using __tuple_t = _Tuple;
        STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
        _Receiver __rcvr_;
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: port this to use constant evaluation
        return __completions_t<__child_of<_Sender>, _Env...>{};
      }

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender &&, _Receiver &&__rcvr) {
          using __tuple_t = __result_tuple_t<__child_of<_Sender>, env_of_t<_Receiver>>;
          return __state<_Receiver, __tuple_t>{static_cast<_Receiver &&>(__rcvr)};
        };

      static constexpr auto complete = []<class _State, class _Tag, class... _Args>(
                                         __ignore,
                                         _State &__state,
                                         _Tag,
                                         _Args &&...__args) noexcept -> void {
        if constexpr (__std::same_as<_Tag, set_value_t>) {
          using __tuple_t = _State::__tuple_t;
          STDEXEC_TRY {
            set_value(
              static_cast<_State &&>(__state).__rcvr_, __tuple_t{static_cast<_Args &&>(__args)...});
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(static_cast<_State &&>(__state).__rcvr_, std::current_exception());
          }
        } else {
          _Tag()(static_cast<_State &&>(__state).__rcvr_, static_cast<_Args &&>(__args)...);
        }
      };
    };

    struct into_tuple_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr) const {
        return __make_sexpr<into_tuple_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const noexcept {
        return __closure(*this);
      }
    };
  } // namespace __into_tuple

  using __into_tuple::into_tuple_t;
  inline constexpr into_tuple_t into_tuple{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::__into_tuple::into_tuple_t> : exec::__into_tuple::__into_tuple_impl { };
} // namespace STDEXEC
