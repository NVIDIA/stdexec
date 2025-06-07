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

#include "__basic_sender.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__meta.hpp"
#include "__senders_core.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__transform_completion_signatures.hpp"
#include "__transform_sender.hpp"
#include "__senders.hpp" // IWYU pragma: keep for __well_formed_sender

// include these after __execution_fwd.hpp
namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_error]
  namespace __upon_error {
    inline constexpr __mstring __upon_error_context =
      "In stdexec::upon_error(Sender, Function)..."_mstr;
    using __on_not_callable = __callable_error<__upon_error_context>;

    template <class _Fun, class _CvrefSender, class... _Env>
    using __completion_signatures_t = transform_completion_signatures<
      __completion_signatures_of_t<_CvrefSender, _Env...>,
      __with_error_invoke_t<__on_not_callable, set_error_t, _Fun, _CvrefSender, _Env...>,
      __sigs::__default_set_value,
      __mbind_front<__mtry_catch_q<__set_value_invoke_t, __on_not_callable>, _Fun>::template __f
    >;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct upon_error_t {
      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<upon_error_t>(static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr)));
      }

      template <__movable_value _Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Fun __fun) const -> __binder_back<upon_error_t, _Fun> {
        return {{static_cast<_Fun&&>(__fun)}, {}, {}};
      }
    };

    struct __upon_error_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures =
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> __completion_signatures_t<__decay_t<__data_of<_Sender>>, __child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, upon_error_t>);
        return {};
      };

      static constexpr auto complete =
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (__same_as<_Tag, set_error_t>) {
          stdexec::__set_value_invoke(
            static_cast<_Receiver&&>(__rcvr),
            static_cast<_State&&>(__state),
            static_cast<_Args&&>(__args)...);
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };
  } // namespace __upon_error

  using __upon_error::upon_error_t;
  inline constexpr upon_error_t upon_error{};

  template <>
  struct __sexpr_impl<upon_error_t> : __upon_error::__upon_error_impl { };
} // namespace stdexec
