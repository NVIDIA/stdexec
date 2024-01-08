/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/__detail/__config.hpp"

#if STDEXEC_HAS_STD_RANGES()

#include "../sequence_senders.hpp"
#include "../__detail/__basic_sequence.hpp"

#include "../env.hpp"
#include "../trampoline_scheduler.hpp"

#include <ranges>

namespace exec {
  namespace __iterate {
    using namespace stdexec;

    template <class _Iterator, class _Sentinel>
    struct __operation_base {
      STDEXEC_ATTRIBUTE((no_unique_address)) _Iterator __iterator_;
      STDEXEC_ATTRIBUTE((no_unique_address)) _Sentinel __sentinel_;
    };

    template <class _Range>
    using __operation_base_t =
      __operation_base<std::ranges::iterator_t<_Range>, std::ranges::sentinel_t<_Range>>;

    template <class _Iterator, class _Sentinel, class _ItemRcvr>
    struct __item_operation {
      struct __t {
        using __id = __item_operation;
        STDEXEC_ATTRIBUTE((no_unique_address)) _ItemRcvr __rcvr_;
        __operation_base<_Iterator, _Sentinel>* __parent_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::set_value(
            static_cast<_ItemRcvr&&>(__self.__rcvr_), *__self.__parent_->__iterator_++);
        }
      };
    };

    template <class _Iterator, class _Sentinel>
    struct __sender {
      struct __t {
        using __id = __sender;
        using sender_concept = stdexec::sender_t;
        using completion_signatures =
          stdexec::completion_signatures<set_value_t(std::iter_reference_t<_Iterator>)>;
        __operation_base<_Iterator, _Sentinel>* __parent_;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _ItemRcvr>
        friend auto tag_invoke(connect_t, _Self&& __self, _ItemRcvr __rcvr) //
          noexcept(__nothrow_decay_copyable<_ItemRcvr>)
            -> stdexec::__t<__item_operation<_Iterator, _Sentinel, _ItemRcvr>> {
          return {static_cast<_ItemRcvr&&>(__rcvr), __self.__parent_};
        }
      };
    };

    template <class _Range>
    using __sender_t =
      stdexec::__t<__sender<std::ranges::iterator_t<_Range>, std::ranges::sentinel_t<_Range>>>;

    template <class _Range, class _Receiver>
    struct __operation {
      struct __t;
    };

    template <class _Range, class _ReceiverId>
    struct __next_receiver {
      struct __t {
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __id = __next_receiver;
        using receiver_concept = stdexec::receiver_t;
        stdexec::__t<__operation<_Range, _ReceiverId>>* __op_;

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          __self.__op_->__start_next();
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          __set_value_unless_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, _Self&& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Range, class _ReceiverId>
    struct __operation<_Range, _ReceiverId>::__t : __operation_base_t<_Range> {
      using _Receiver = stdexec::__t<_ReceiverId>;
      _Receiver __rcvr_;

      using _ItemSender = decltype(stdexec::on(
        std::declval<trampoline_scheduler&>(),
        std::declval<__sender_t<_Range>>()));

      using __next_receiver_t = stdexec::__t<__next_receiver<_Range, _ReceiverId>>;

      std::optional<connect_result_t<next_sender_of_t<_Receiver, _ItemSender>, __next_receiver_t>>
        __op_{};
      trampoline_scheduler __scheduler_{};

      void __start_next() noexcept {
        if (this->__iterator_ == this->__sentinel_) {
          stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
        } else {
          try {
            stdexec::start(__op_.emplace(__conv{[&] {
              return stdexec::connect(
                exec::set_next(__rcvr_, stdexec::on(__scheduler_, __sender_t<_Range>{this})),
                __next_receiver_t{this});
            }}));
          } catch (...) {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
          }
        }
      }

      template <same_as<__t> _Self>
      friend void tag_invoke(start_t, _Self& __self) noexcept {
        __self.__start_next();
      }
    };

    template <class _Receiver>
    struct __subscribe_fn {
      using _ReceiverId = __id<_Receiver>;
      _Receiver __rcvr_;
      template <class _Range>
      using __operation_t = __t<__operation<__decay_t<_Range>, _ReceiverId>>;

      template <class _Range>
      auto operator()(__ignore, _Range&& __range) //
        noexcept(__nothrow_decay_copyable<_Receiver>) -> __operation_t<_Range> {
        return {
          {std::ranges::begin(__range), std::ranges::end(__range)},
          static_cast<_Receiver&&>(__rcvr_)
        };
      }
    };

    struct iterate_t {
      template <std::ranges::forward_range _Range>
        requires __decay_copyable<_Range>
      auto operator()(_Range&& __range) const {
        return make_sequence_expr<iterate_t>(__decay_t<_Range>{static_cast<_Range&&>(__range)});
      }

      using __completion_sigs =
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;


      template <class _Sequence>
      using _ItemSender = decltype(stdexec::on(
        __declval<trampoline_scheduler&>(),
        __declval<__sender_t<__data_of<_Sequence>>>()));

      template <class _Sequence, class _Receiver>
      using _NextReceiver = stdexec::__t<__next_receiver<__data_of<_Sequence>, __id<_Receiver>>>;

      template <class _Sequence, class _Receiver>
      using _NextSender = next_sender_of_t<_Receiver, _ItemSender<_Sequence>>;

      template <
        sender_expr_for<iterate_t> _SeqExpr,
        sequence_receiver_of<item_types<_ItemSender<_SeqExpr>>> _Receiver>
        requires sender_to<_NextSender<_SeqExpr, _Receiver>, _NextReceiver<_SeqExpr, _Receiver>>
      static auto subscribe(_SeqExpr&& __seq, _Receiver __rcvr) noexcept(
        __nothrow_callable<__sexpr_apply_t, _SeqExpr, __subscribe_fn<_Receiver>>)
        -> __call_result_t<__sexpr_apply_t, _SeqExpr, __subscribe_fn<_Receiver>> {
        return __sexpr_apply(static_cast<_SeqExpr&&>(__seq), __subscribe_fn<_Receiver>{__rcvr});
      }

      static auto get_completion_signatures(__ignore, __ignore) noexcept
        -> completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()> {
        return {};
      }

      template <sender_expr_for<iterate_t> _Sequence>
      static auto get_item_types(_Sequence&&, __ignore) noexcept
        -> item_types<_ItemSender<_Sequence>> {
        return {};
      }

      static empty_env get_env(__ignore) noexcept {
        return {};
      }
    };
  }

  using __iterate::iterate_t;
  inline constexpr iterate_t iterate;
}

#endif // STDEXEC_HAS_STD_RANGES()