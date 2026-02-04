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
#include "__completion_signatures_of.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp" // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"

// include these after __execution_fwd.hpp
namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_stopped]
  namespace __upon_stopped {
    struct upon_stopped_t;
    using __on_not_callable = __mbind_front_q<__callable_error_t, upon_stopped_t>;

    template <class _Fun, class _CvSender, class... _Env>
    using __completion_signatures_t = transform_completion_signatures<
      __completion_signatures_of_t<_CvSender, _Env...>,
      __with_error_invoke_t<__on_not_callable, set_stopped_t, _Fun, _CvSender, _Env...>,
      __cmplsigs::__default_set_value,
      __cmplsigs::__default_set_error,
      __set_value_from_t<_Fun>
    >;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct upon_stopped_t {
      template <sender _Sender, __movable_value _Fun>
        requires __callable<_Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto {
        return __make_sexpr<upon_stopped_t>(
          static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr));
      }

      template <__movable_value _Fun>
        requires __callable<_Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Fun __fun) const noexcept(__nothrow_move_constructible<_Fun>) {
        return __closure(*this, static_cast<_Fun&&>(__fun));
      }
    };

    struct __upon_stopped_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() //
        -> __completion_signatures_t<__decay_t<__data_of<_Sender>>, __child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, upon_stopped_t>);
        // TODO: update this to use constant evaluation:
        return {};
      };

      static constexpr auto complete = []<class _Tag, class _State, class... _Args>(
                                         __ignore,
                                         _State& __state,
                                         _Tag,
                                         _Args&&... __args) noexcept -> void {
        if constexpr (__same_as<_Tag, set_stopped_t>) {
          STDEXEC::__set_value_from(
            static_cast<_State&&>(__state).__rcvr_,
            static_cast<_State&&>(__state).__data_,
            static_cast<_Args&&>(__args)...);
        } else {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };
  } // namespace __upon_stopped

  using __upon_stopped::upon_stopped_t;
  inline constexpr upon_stopped_t upon_stopped{};

  template <>
  struct __sexpr_impl<upon_stopped_t> : __upon_stopped::__upon_stopped_impl { };

} // namespace STDEXEC
