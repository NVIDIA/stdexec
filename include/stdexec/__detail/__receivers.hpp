/*
 * Copyright (c) 2022-2024 NVIDIA Corporation
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

#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__tag_invoke.hpp"

#include <exception>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __rcvrs {
    struct set_value_t {
      template <class _Fn, class... _Args>
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class... _As>
        requires tag_invocable<set_value_t, _Receiver, _As...>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      void
        operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(
          set_value_t{}, static_cast<_Receiver&&>(__rcvr), static_cast<_As&&>(__as)...);
      }
    };

    struct set_error_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 1)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class _Error>
        requires tag_invocable<set_error_t, _Receiver, _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      void
        operator()(_Receiver&& __rcvr, _Error&& __err) const noexcept {
        static_assert(nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void) tag_invoke(
          set_error_t{}, static_cast<_Receiver&&>(__rcvr), static_cast<_Error&&>(__err));
      }
    };

    struct set_stopped_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 0)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver>
        requires tag_invocable<set_stopped_t, _Receiver>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      void
        operator()(_Receiver&& __rcvr) const noexcept {
        static_assert(nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) tag_invoke(set_stopped_t{}, static_cast<_Receiver&&>(__rcvr));
      }
    };
  } // namespace __rcvrs

  using __rcvrs::set_value_t;
  using __rcvrs::set_error_t;
  using __rcvrs::set_stopped_t;
  inline constexpr set_value_t set_value{};
  inline constexpr set_error_t set_error{};
  inline constexpr set_stopped_t set_stopped{};

  inline constexpr struct __try_call_t {
    template <class _Receiver, class _Fun, class... _Args>
      requires __callable<_Fun, _Args...>
    void operator()(_Receiver&& __rcvr, _Fun __fun, _Args&&... __args) const noexcept {
      if constexpr (__nothrow_callable<_Fun, _Args...>) {
        static_cast<_Fun&&>(__fun)(static_cast<_Args&&>(__args)...);
      } else {
        try {
          static_cast<_Fun&&>(__fun)(static_cast<_Args&&>(__args)...);
        } catch (...) {
          set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
        }
      }
    }
  } __try_call{};

  struct receiver_t {
    using receiver_concept = receiver_t; // NOT TO SPEC
  };

  namespace __detail {
    template <class _Receiver>
    concept __enable_receiver =                                            //
      (STDEXEC_NVHPC(requires { typename _Receiver::receiver_concept; }&&) //
       derived_from<typename _Receiver::receiver_concept, receiver_t>)
      || requires { typename _Receiver::is_receiver; } // back-compat, NOT TO SPEC
      || STDEXEC_IS_BASE_OF(receiver_t, _Receiver);    // NOT TO SPEC, for receiver_adaptor
  }                                                    // namespace __detail

  template <class _Receiver>
  inline constexpr bool enable_receiver = __detail::__enable_receiver<_Receiver>; // NOT TO SPEC

  template <class _Receiver>
  concept receiver = enable_receiver<__decay_t<_Receiver>> &&     //
                     environment_provider<__cref_t<_Receiver>> && //
                     move_constructible<__decay_t<_Receiver>> &&  //
                     constructible_from<__decay_t<_Receiver>, _Receiver>;

  template <class _Receiver, class _Completions>
  concept receiver_of =    //
    receiver<_Receiver> && //
    requires(_Completions* __completions) {
      { stdexec::__try_completions<__decay_t<_Receiver>>(__completions) } -> __ok;
    };

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of<_Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;
} // namespace stdexec
