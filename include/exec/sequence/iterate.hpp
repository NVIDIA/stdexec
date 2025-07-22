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

#  include "../../stdexec/concepts.hpp"
#  include "../../stdexec/execution.hpp"
#  include "../sequence_senders.hpp"
#  include "../__detail/__basic_sequence.hpp"

#  include "../trampoline_scheduler.hpp"
#  include "../sequence.hpp"

#  include <exception>
#  include <ranges>

namespace exec {
  namespace __iterate {
    using namespace stdexec;

    template <class _Iterator, class _Sentinel>
    struct __operation_base {
      STDEXEC_ATTRIBUTE(no_unique_address) _Iterator __iterator_;
      STDEXEC_ATTRIBUTE(no_unique_address) _Sentinel __sentinel_;
    };

    template <class _Range>
    using __operation_base_t =
      __operation_base<std::ranges::iterator_t<_Range>, std::ranges::sentinel_t<_Range>>;

    template <class _Iterator, class _Sentinel, class _ItemRcvr>
    struct __item_operation {
      struct __t {
        using __id = __item_operation;
        STDEXEC_ATTRIBUTE(no_unique_address) _ItemRcvr __rcvr_;
        __operation_base<_Iterator, _Sentinel>* __parent_;

        void start() & noexcept {
          stdexec::set_value(static_cast<_ItemRcvr&&>(__rcvr_), *__parent_->__iterator_++);
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

        template <receiver_of<completion_signatures> _ItemRcvr>
        auto connect(_ItemRcvr __rcvr) const & noexcept(__nothrow_decay_copyable<_ItemRcvr>)
          -> stdexec::__t<__item_operation<_Iterator, _Sentinel, _ItemRcvr>> {
          return {static_cast<_ItemRcvr&&>(__rcvr), __parent_};
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

        void set_value() noexcept {
          __op_->__start_next();
        }

        void set_stopped() noexcept {
          __set_value_unless_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__op_->__rcvr_);
        }
      };
    };

    template <class _Range, class _ReceiverId>
    struct __operation<_Range, _ReceiverId>::__t : __operation_base_t<_Range> {
      using _Receiver = stdexec::__t<_ReceiverId>;
      _Receiver __rcvr_;

      using __item_sender_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler&>, __sender_t<_Range>>;
      using __next_receiver_t = stdexec::__t<__next_receiver<_Range, _ReceiverId>>;

      std::optional<
        connect_result_t<next_sender_of_t<_Receiver, __item_sender_t>, __next_receiver_t>
      >
        __op_{};
      trampoline_scheduler __scheduler_{};

      void __start_next() noexcept {
        if (this->__iterator_ == this->__sentinel_) {
          stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
        } else {

          STDEXEC_TRY {
            stdexec::start(__op_.emplace(__emplace_from{[&] {
              return stdexec::connect(
                exec::set_next(
                  __rcvr_,
                  exec::sequence(stdexec::schedule(__scheduler_), __sender_t<_Range>{this})),
                __next_receiver_t{this});
            }}));
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
          }
        }
      }

      void start() & noexcept {
        __start_next();
      }
    };

    template <class _Receiver>
    struct __subscribe_fn {
      using _ReceiverId = __id<_Receiver>;
      _Receiver __rcvr_;

      template <class _Range>
      using __operation_t = __t<__operation<__decay_t<_Range>, _ReceiverId>>;

      template <class _Range>
      auto operator()(__ignore, _Range&& __range) noexcept(__nothrow_move_constructible<_Receiver>)
        -> __operation_t<_Range> {
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
      using __item_sender_t = __result_of<
        exec::sequence,
        schedule_result_t<trampoline_scheduler&>,
        __sender_t<__data_of<_Sequence>>
      >;

      template <class _Sequence, class _Receiver>
      using _NextReceiver = stdexec::__t<__next_receiver<__data_of<_Sequence>, __id<_Receiver>>>;

      template <class _Sequence, class _Receiver>
      using _NextSender = next_sender_of_t<_Receiver, __item_sender_t<_Sequence>>;

      template <
        sender_expr_for<iterate_t> _SeqExpr,
        sequence_receiver_of<item_types<__item_sender_t<_SeqExpr>>> _Receiver
      >
        requires sender_to<_NextSender<_SeqExpr, _Receiver>, _NextReceiver<_SeqExpr, _Receiver>>
      static auto subscribe(_SeqExpr&& __seq, _Receiver __rcvr)
        noexcept(__nothrow_callable<__sexpr_apply_t, _SeqExpr, __subscribe_fn<_Receiver>>)
          -> __call_result_t<__sexpr_apply_t, _SeqExpr, __subscribe_fn<_Receiver>> {
        return __sexpr_apply(static_cast<_SeqExpr&&>(__seq), __subscribe_fn<_Receiver>{__rcvr});
      }

      static auto get_completion_signatures(__ignore, __ignore = {}) noexcept
        -> completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()> {
        return {};
      }

      template <sender_expr_for<iterate_t> _Sequence>
      static auto get_item_types(_Sequence&&, __ignore) noexcept //
        -> item_types<__item_sender_t<_Sequence>> {
        return {};
      }

      static auto get_env(__ignore) noexcept -> env<> {
        return {};
      }
    };
  } // namespace __iterate

  using __iterate::iterate_t;
  inline constexpr iterate_t iterate;
} // namespace exec

#endif // STDEXEC_HAS_STD_RANGES()