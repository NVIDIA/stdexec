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
#include "__concepts.hpp"
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp" // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"
#include "__utility.hpp"

#include <exception>
#include <tuple>
#include <variant> // IWYU pragma: keep

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  namespace __into_variant {
    template <class _Sender, class _Env>
      requires sender_in<_Sender, _Env>
    using __into_variant_result_t = value_types_of_t<_Sender, _Env>;

    template <class _Sender, class... _Env>
    using __variant_t = __value_types_t<
      __completion_signatures_of_t<_Sender, _Env...>,
      __qq<__decayed_std_tuple>,
      __qq<__std_variant>
    >;

    template <class _Variant>
    using __variant_completions =
      completion_signatures<set_value_t(_Variant), set_error_t(std::exception_ptr)>;

    template <class _Sender, class... _Env>
    using __completions = transform_completion_signatures<
      __completion_signatures_of_t<_Sender, _Env...>,
      __minvoke_q<__variant_completions, __variant_t<_Sender, _Env...>>,
      __mconst<completion_signatures<>>::__f
    >;

    struct into_variant_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender&& __sndr) const -> __well_formed_sender auto {
        return __make_sexpr<into_variant_t>(__(), static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept {
        return __closure(*this);
      }
    };

    template <class _Receiver, class _Variant>
    struct __state {
      using __variant_t = _Variant;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    struct __into_variant_impl : __sexpr_defaults {
      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&&, _Receiver&& __rcvr) noexcept {
          using __variant_t = value_types_of_t<__child_of<_Self>, env_of_t<_Receiver>>;
          return __state<_Receiver, __variant_t>{static_cast<_Receiver&&>(__rcvr)};
        };

      static constexpr auto complete = []<class _State, class _Tag, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        if constexpr (__same_as<_Tag, set_value_t>) {
          using __variant_t = _State::__variant_t;
          STDEXEC_TRY {
            set_value(
              static_cast<_State&&>(__state).__rcvr_,
              __variant_t{std::tuple<_Args&&...>{static_cast<_Args&&>(__args)...}});
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
          }
        } else {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, into_variant_t>);
        return __completions<__child_of<_Self>, _Env...>{};
      };
    };
  } // namespace __into_variant

  using __into_variant::into_variant_t;
  inline constexpr into_variant_t into_variant{};

  template <>
  struct __sexpr_impl<into_variant_t> : __into_variant::__into_variant_impl { };
} // namespace STDEXEC
