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
#include "../stdexec/concepts.hpp"
#include "../stdexec/execution.hpp"

namespace exec {
  namespace __create {
    using namespace stdexec;

    struct __void {
      template <class _Fn>
        void emplace(_Fn&& __fn) noexcept(__nothrow_callable<_Fn>) {
          ((_Fn&&) __fn)();
        }
    };

    template <class _Receiver, class _Args>
      struct __context {
        [[no_unique_address]] _Receiver receiver;
        [[no_unique_address]] _Args args;
      };

    template <class _ReceiverId, class _FnId, class _ArgsId>
      struct __operation {
        using _Fn = __t<_FnId>;
        using _Context = __context<__t<_ReceiverId>, __t<_ArgsId>>;
        using _Result = __call_result_t<_Fn, _Context&>;
        using _State = __if_c<same_as<_Result, void>, __void, std::optional<_Result>>;

        [[no_unique_address]] _Context __ctx_;
        [[no_unique_address]] _Fn __fn_;
        [[no_unique_address]] _State __state_{};

        friend void tag_invoke(start_t, __operation& __self) noexcept {
          __self.__state_.emplace(__conv{
            [&]() noexcept {
              return ((_Fn&&) __self.__fn_)(__self.__ctx_);
            }
          });
        }
      };

    template <class _Sigs, class _FnId, class _ArgsId>
      struct __sender {
        using _Fn = __t<_FnId>;
        using _Args = __t<_ArgsId>;
        using completion_signatures = _Sigs;

        _Fn __fn_;
        _Args __args_;

        template <__decays_to<__sender> _Self, receiver_of<_Sigs> _Receiver>
          requires __callable<_Fn, __context<_Receiver, _Args>&> &&
            constructible_from<_Fn, __member_t<_Self, _Fn>> &&
            constructible_from<_Args, __member_t<_Self, _Args>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation<__x<decay_t<_Receiver>>, _FnId, _ArgsId> {
          static_assert(__nothrow_callable<_Fn, __context<_Receiver, _Args>&>);
          return {{(_Receiver&&) __rcvr, ((_Self&&) __self).__args_}, ((_Self&&) __self).__fn_};
        }
      };

    template <__completion_signature... _Sigs>
      struct __create_t {
        using __compl_sigs = completion_signatures<_Sigs...>;

        template <class _Fn, class... _Args>
            requires move_constructible<_Fn> &&
              constructible_from<__decayed_tuple<_Args...>, _Args...>
          auto operator()(_Fn __fn, _Args&&... __args) const
            -> __sender<__compl_sigs, __x<_Fn>, __x<__decayed_tuple<_Args...>>> {
            return {(_Fn&&) __fn, {(_Args&&) __args...}};
          }
      };
  } // namespace __create

  template <stdexec::__completion_signature... _Sigs>
    inline constexpr __create::__create_t<_Sigs...> create {};
}
