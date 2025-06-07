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
    using namespace stdexec;

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

    template <class _ReceiverId, class _Fun, class _ArgsId>
    struct __operation {
      using _Context = __context<stdexec::__t<_ReceiverId>, stdexec::__t<_ArgsId>>;
      using _Result = __call_result_t<_Fun, _Context&>;
      using _State = __if_c<same_as<_Result, void>, __void, std::optional<_Result>>;

      struct __t : stdexec::__immovable {
        using __id = __operation;

        STDEXEC_ATTRIBUTE(no_unique_address) _Context __ctx_;
        STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_;
        STDEXEC_ATTRIBUTE(no_unique_address) _State __state_ { };

        void start() & noexcept {
          __state_.emplace(
            __emplace_from{[&]() noexcept { return static_cast<_Fun&&>(__fun_)(__ctx_); }});
        }
      };
    };

    template <class _Fun, class _ArgsId, class... _Sigs>
    struct __sender {
      using _Args = stdexec::__t<_ArgsId>;

      struct __t {
        using __id = __sender;
        using sender_concept = stdexec::sender_t;
        using completion_signatures = stdexec::completion_signatures<_Sigs...>;

        _Fun __fun_;
        _Args __args_;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _Receiver>
          requires __callable<_Fun, __context<_Receiver, _Args>&>
                && constructible_from<_Fun, __copy_cvref_t<_Self, _Fun>>
                && constructible_from<_Args, __copy_cvref_t<_Self, _Args>>
        static auto connect(_Self&& __self, _Receiver __rcvr)
          -> stdexec::__t<__operation<stdexec::__id<_Receiver>, _Fun, _ArgsId>> {
          static_assert(__nothrow_callable<_Fun, __context<_Receiver, _Args>&>);
          return {
            {},
            {static_cast<_Receiver&&>(__rcvr), static_cast<_Self&&>(__self).__args_},
            static_cast<_Self&&>(__self).__fun_
          };
        }
      };
    };

    template <__completion_signature... _Sigs>
    struct __create_t {
      template <class _Fun, class... _Args>
        requires move_constructible<_Fun>
              && constructible_from<__decayed_std_tuple<_Args...>, _Args...>
      auto operator()(_Fun __fun, _Args&&... __args) const
        -> __t<__sender<_Fun, __id<__decayed_std_tuple<_Args...>>, _Sigs...>> {
        return {static_cast<_Fun&&>(__fun), {static_cast<_Args&&>(__args)...}};
      }
    };
  } // namespace __create

  template <class... _Sigs>
  extern const stdexec::__mfront<void, _Sigs...> create;

  template <stdexec::__completion_signature... _Sigs>
  inline constexpr __create::__create_t<_Sigs...> create<_Sigs...>{};

  template <stdexec::__completion_signature... _Sigs>
  inline constexpr __create::__create_t<_Sigs...>
    create<stdexec::completion_signatures<_Sigs...>>{};
} // namespace exec
