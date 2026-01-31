/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/execution.hpp"

namespace exec {
  namespace __create {
    using namespace STDEXEC;

    struct __void {
      template <class _Fun>
      void emplace(_Fun&& __fun) noexcept(__nothrow_callable<_Fun>) {
        static_cast<_Fun&&>(__fun)();
      }
    };

    template <class _Receiver, class _Args>
    struct __context {
      STDEXEC_ATTRIBUTE(no_unique_address) _Receiver receiver;
      STDEXEC_ATTRIBUTE(no_unique_address) _Args args;
    };

    template <class _Receiver, class _Fun, class _Args>
    struct __opstate : STDEXEC::__immovable {
      using __context_t = __context<_Receiver, _Args>;
      using __result_t = __call_result_t<_Fun, __context_t&>;
      using __state_t = __if_c<__std::same_as<__result_t, void>, __void, std::optional<__result_t>>;

      void start() & noexcept {
        __state_
          .emplace(__emplace_from{[&]() noexcept { return static_cast<_Fun&&>(__fun_)(__ctx_); }});
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __context_t __ctx_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Fun __fun_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __state_t __state_{};
    };

    template <class _Fun, class _Args, class... _Sigs>
    struct __sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = STDEXEC::completion_signatures<_Sigs...>;

      template <__decays_to<__sender> _Self, receiver_of<completion_signatures> _Receiver>
        requires __callable<_Fun, __context<_Receiver, _Args>&>
              && __std::constructible_from<_Fun, __copy_cvref_t<_Self, _Fun>>
              && __std::constructible_from<_Args, __copy_cvref_t<_Self, _Args>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> __opstate<_Receiver, _Fun, _Args> {
        static_assert(__nothrow_callable<_Fun, __context<_Receiver, _Args>&>);
        return {
          {},
          {static_cast<_Receiver&&>(__rcvr), static_cast<_Self&&>(__self).__args_},
          static_cast<_Self&&>(__self).__fun_
        };
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _Fun __fun_;
      _Args __args_;
    };

    template <__completion_signature... _Sigs>
    struct __create_t {
      template <class _Fun, class... _Args>
        requires __std::move_constructible<_Fun>
              && __std::constructible_from<__decayed_std_tuple<_Args...>, _Args...>
      auto operator()(_Fun __fun, _Args&&... __args) const
        -> __sender<_Fun, __decayed_std_tuple<_Args...>, _Sigs...> {
        return {static_cast<_Fun&&>(__fun), {static_cast<_Args&&>(__args)...}};
      }
    };
  } // namespace __create

  template <class... _Sigs>
  extern const STDEXEC::__undefined<_Sigs...> create;

  template <STDEXEC::__completion_signature... _Sigs>
  inline constexpr __create::__create_t<_Sigs...> create<_Sigs...>{};

  template <STDEXEC::__completion_signature... _Sigs>
  inline constexpr __create::__create_t<_Sigs...>
    create<STDEXEC::completion_signatures<_Sigs...>>{};
} // namespace exec
