/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <numeric>
#include <ranges>

namespace exec {

  namespace __reduce {

    template <class _ReceiverId, class _InitT, class _RedOp>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __data {
        _Receiver __rcvr_;
        STDEXEC_NO_UNIQUE_ADDRESS _InitT __init_;
        STDEXEC_NO_UNIQUE_ADDRESS _RedOp __redop_;
      };

      struct __t {
        using __id = __receiver;
        __data* __op_;

        template <
          stdexec::__same_as<stdexec::set_value_t> _Tag,
          class _Range,
          class _Value = stdexec::range_value_t<_Range>>
          requires stdexec::invocable<_RedOp, _InitT, _Value>
                && stdexec::__receiver_of_invoke_result<_Receiver, _RedOp, _InitT, _Value>
        friend void tag_invoke(_Tag, __t&& __self, _Range&& __range) noexcept {
          auto result = std::reduce(
            std::ranges::begin(__range), std::ranges::end(__range), __self.__op_->__init_, __self.__op_->__redop_);

          stdexec::set_value((_Receiver&&) __self.__op_->__rcvr_, std::move(result));
        }

        template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> _Tag, class... _As>
          requires stdexec::__callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          __tag((_Receiver&&) __self.__op_->__rcvr_, (_As&&) __as...);
        }

        friend auto tag_invoke(stdexec::get_env_t, const __t& __self) noexcept
          -> stdexec::env_of_t<const _Receiver&> {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _InitT, class _RedOp>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_id = __receiver<_ReceiverId, _InitT, _RedOp>;
      using __receiver_t = stdexec::__t<__receiver_id>;

      struct __t : stdexec::__immovable {
        using __id = __operation;
        typename __receiver_id::__data __data_;
        stdexec::connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver __rcvr, _InitT __init, _RedOp __redop) noexcept(
          stdexec::__nothrow_decay_copyable<_Receiver> && stdexec::__nothrow_decay_copyable<_RedOp>
          && stdexec::__nothrow_connectable<_Sender, __receiver_t>)
          : __data_{(_Receiver&&) __rcvr, (_InitT&&) __init, (_RedOp&&) __redop}
          , __op_(stdexec::connect((_Sender&&) __sndr, __receiver_t{&__data_})) {
        }

        friend void tag_invoke(stdexec::start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _SenderId, class _InitT, class _RedOp>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;
      template <class _Receiver>
      using __receiver = stdexec::__t<__receiver<stdexec::__id<_Receiver>, _InitT, _RedOp>>;
      template <class _Self, class _Receiver>
      using __operation = stdexec::__t<
        __operation<stdexec::__copy_cvref_t<_Self, _Sender>, stdexec::__id<_Receiver>, _InitT, _RedOp>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _InitT __init_;
        STDEXEC_NO_UNIQUE_ADDRESS _RedOp __redop_;

        template <stdexec::__decays_to<__t> _Self, stdexec::receiver _Receiver>
          requires stdexec::sender_to<stdexec::__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(stdexec::connect_t, _Self&& __self, _Receiver __rcvr) noexcept(
          stdexec::__nothrow_constructible_from<
            __operation<_Self, _Receiver>,
            stdexec::__copy_cvref_t<_Self, _Sender>,
            _Receiver&&,
            stdexec::__copy_cvref_t<_Self, _InitT>,
            stdexec::__copy_cvref_t<_Self, _RedOp>>) -> __operation<_Self, _Receiver> {
          return {
            ((_Self&&) __self).__sndr_,
            (_Receiver&&) __rcvr,
            ((_Self&&) __self).__init_,
            ((_Self&&) __self).__redop_};
        }

        template <stdexec::__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env&&)
          -> stdexec::dependent_completion_signatures<_Env>;

        template <stdexec::__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env&&)
          -> stdexec::completion_signatures<stdexec::set_value_t(_InitT)>
          requires true;

        friend auto tag_invoke(stdexec::get_env_t, const __t& __self) noexcept
          -> stdexec::env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    struct reduce_t {
      template <stdexec::sender _Sender, class _InitT, class _RedOp>
      using __sender =
        stdexec::__t<__sender<stdexec::__id<stdexec::__decay_t<_Sender>>, _InitT, _RedOp>>;

      template <stdexec::sender _Sender, class _InitT, stdexec::__movable_value _RedOp>
        requires stdexec::__tag_invocable_with_completion_scheduler<
          reduce_t,
          stdexec::set_value_t,
          _Sender,
          _InitT,
          _RedOp>
      stdexec::sender auto operator()(_Sender&& __sndr, _InitT __init, _RedOp __redop) const noexcept(
        stdexec::nothrow_tag_invocable<
          reduce_t,
          stdexec::__completion_scheduler_for<_Sender, stdexec::set_value_t>,
          _Sender,
          _InitT,
          _RedOp>) {
        auto __sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(__sndr));
        return tag_invoke(
          reduce_t{}, std::move(__sched), (_Sender&&) __sndr, (_InitT&&) __init, (_RedOp&&) __redop);
      }

      template <stdexec::sender _Sender, class _InitT, stdexec::__movable_value _RedOp>
        requires(!stdexec::__tag_invocable_with_completion_scheduler<
                  reduce_t,
                  stdexec::set_value_t,
                  _Sender,
                  _InitT,
                  _RedOp>)
             && stdexec::tag_invocable<reduce_t, _Sender, _InitT, _RedOp>
      stdexec::sender auto operator()(_Sender&& __sndr, _InitT __init, _RedOp __redop) const
        noexcept(stdexec::nothrow_tag_invocable<reduce_t, _Sender, _InitT, _RedOp>) {
        return tag_invoke(reduce_t{}, (_Sender&&) __sndr, (_InitT&&) __init, (_RedOp&&) __redop);
      }

      template <stdexec::sender _Sender, class _InitT, stdexec::__movable_value _RedOp>
        requires(!stdexec::__tag_invocable_with_completion_scheduler<
                  reduce_t,
                  stdexec::set_value_t,
                  _Sender,
                  _InitT,
                  _RedOp>)
             && (!stdexec::tag_invocable<reduce_t, _Sender, _InitT, _RedOp>)
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE __sender<_Sender, _InitT, _RedOp>
        operator()(_Sender&& __sndr, _InitT __init, _RedOp __redop) const {
        return __sender<_Sender, _InitT, _RedOp>{(_Sender&&) __sndr, __init, (_RedOp&&) __redop};
      }

      template <class _InitT, class _RedOp = std::plus<>>
      stdexec::__binder_back<reduce_t, _InitT, _RedOp> operator()(_InitT __init, _RedOp __redop = {}) const {
        return {
          {},
          {},
          {(_InitT&&) __init, (_RedOp&&) __redop}
        };
      }
    };

  }

  using __reduce::reduce_t;
  inline constexpr reduce_t reduce{};
}
