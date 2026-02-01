/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2026 NVIDIA Corporation
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

#  include "../../stdexec/__detail/__optional.hpp"
#  include "../../stdexec/execution.hpp"

#  include "../__detail/__basic_sequence.hpp"
#  include "../sequence.hpp"
#  include "../sequence_senders.hpp"
#  include "../trampoline_scheduler.hpp"

#  include <exception>
#  include <ranges>

namespace exec {
  namespace __iterate {
    using namespace STDEXEC;

    template <class _Iterator, class _Sentinel>
    struct __operation_base_base {
      STDEXEC_ATTRIBUTE(no_unique_address) _Iterator __iterator_;
      STDEXEC_ATTRIBUTE(no_unique_address) _Sentinel __sentinel_;
      trampoline_scheduler __scheduler_{};
    };

    template <class _Iterator, class _Sentinel, class _Receiver>
    struct __operation_base : __operation_base_base<_Iterator, _Sentinel> {
      constexpr explicit __operation_base(_Iterator __it, _Sentinel __sen, _Receiver&& __rcvr)
        : __operation_base_base<_Iterator, _Sentinel>{
            static_cast<_Iterator&&>(__it),
            static_cast<_Sentinel&&>(__sen)}
        , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
      }

      virtual ~__operation_base() = default;
      virtual constexpr void __start_next() noexcept = 0;

      _Receiver __rcvr_;
    };

    template <class _Iterator, class _Sentinel, class _ItemRcvr>
    struct __item_operation {
      constexpr void start() noexcept {
        STDEXEC::set_value(static_cast<_ItemRcvr&&>(__rcvr_), *__parent_->__iterator_++);
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _ItemRcvr __rcvr_;
      __operation_base_base<_Iterator, _Sentinel>* __parent_;
    };

    template <class _Iterator, class _Sentinel>
    struct __sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        STDEXEC::completion_signatures<set_value_t(std::iter_reference_t<_Iterator>)>;

      template <receiver_of<completion_signatures> _ItemRcvr>
      constexpr auto connect(_ItemRcvr __rcvr) const & noexcept(__nothrow_decay_copyable<_ItemRcvr>)
        -> __item_operation<_Iterator, _Sentinel, _ItemRcvr> {
        return {static_cast<_ItemRcvr&&>(__rcvr), __parent_};
      }

      __operation_base_base<_Iterator, _Sentinel>* __parent_;
    };

    template <class _Iterator, class _Sentinel, class _Receiver>
    struct __next_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      constexpr void set_value() noexcept {
        __op_->__start_next();
      }

      constexpr void set_stopped() noexcept {
        __set_value_unless_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__op_->__rcvr_);
      }

      __operation_base<_Iterator, _Sentinel, _Receiver>* __op_;
    };

    template <class _Iterator, class _Sentinel, class _Receiver>
    struct __operation final : __operation_base<_Iterator, _Sentinel, _Receiver> {
      using __item_sender_t = __result_of<
        exec::sequence,
        schedule_result_t<trampoline_scheduler>,
        __sender<_Iterator, _Sentinel>
      >;
      using __next_receiver_t = __next_receiver<_Iterator, _Sentinel, _Receiver>;
      using __next_sender_t = next_sender_of_t<_Receiver, __item_sender_t>;

      constexpr explicit __operation(_Iterator __iterator, _Sentinel __sentinel, _Receiver&& __rcvr)
        : __operation_base<_Iterator, _Sentinel, _Receiver>{
            static_cast<_Iterator&&>(__iterator),
            static_cast<_Sentinel&&>(__sentinel),
            static_cast<_Receiver&&>(__rcvr)} {
      }

      constexpr void __start_next() noexcept final {
        if (this->__iterator_ == this->__sentinel_) {
          STDEXEC::set_value(static_cast<_Receiver&&>(this->__rcvr_));
        } else {
          STDEXEC_TRY {
            STDEXEC::start(__op_.__emplace_from(
              STDEXEC::connect,
              exec::set_next(
                this->__rcvr_,
                exec::sequence(
                  STDEXEC::schedule(this->__scheduler_), __sender<_Iterator, _Sentinel>{this})),
              __next_receiver_t{this}));
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(static_cast<_Receiver&&>(this->__rcvr_), std::current_exception());
          }
        }
      }

      constexpr void start() & noexcept {
        __start_next();
      }

      __optional<connect_result_t<__next_sender_t, __next_receiver_t>> __op_{};
    };

    template <class _Receiver>
    struct __subscribe_fn {
      template <class _Range>
      constexpr auto operator()(__ignore, _Range&& __range) noexcept {
        return __operation{
          std::ranges::begin(static_cast<_Range&&>(__range)),
          std::ranges::end(static_cast<_Range&&>(__range)),
          static_cast<_Receiver&&>(__rcvr_)};
      }

      _Receiver __rcvr_;
    };

    struct iterate_t {
      template <std::ranges::forward_range _Range>
        requires __decay_copyable<_Range>
      constexpr auto operator()(_Range&& __range) const -> __well_formed_sequence_sender auto {
        return make_sequence_expr<iterate_t>(__decay_t<_Range>{static_cast<_Range&&>(__range)});
      }

      template <class _Range>
      using __sender_t = __sender<std::ranges::iterator_t<_Range>, std::ranges::sentinel_t<_Range>>;

      using __completion_sigs =
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;

      template <class _Sequence>
      using __item_sender_t = __result_of<
        exec::sequence,
        schedule_result_t<trampoline_scheduler&>,
        __sender_t<__data_of<_Sequence>>
      >;

      template <class _Sequence, class _Receiver>
      using _NextReceiver = __next_receiver<
        std::ranges::iterator_t<__data_of<_Sequence>>,
        std::ranges::sentinel_t<__data_of<_Sequence>>,
        _Receiver
      >;

      template <class _Sequence, class _Receiver>
      using _NextSender = next_sender_of_t<_Receiver, __item_sender_t<_Sequence>>;

      template <
        sender_expr_for<iterate_t> _SeqExpr,
        sequence_receiver_of<item_types<__item_sender_t<_SeqExpr>>> _Receiver
      >
        requires sender_to<_NextSender<_SeqExpr, _Receiver>, _NextReceiver<_SeqExpr, _Receiver>>
      static constexpr auto subscribe(_SeqExpr&& __seq, _Receiver __rcvr)
        noexcept(__nothrow_applicable<__subscribe_fn<_Receiver>, _SeqExpr>)
          -> __apply_result_t<__subscribe_fn<_Receiver>, _SeqExpr> {
        return __apply(__subscribe_fn<_Receiver>{__rcvr}, static_cast<_SeqExpr&&>(__seq));
      }

      template <class, class...>
      static consteval auto get_completion_signatures() noexcept {
        return completion_signatures<
          set_value_t(),
          set_error_t(std::exception_ptr),
          set_stopped_t()
        >();
      }

      template <sender_expr_for<iterate_t> _Sequence, class... _Env>
      static consteval auto get_item_types() noexcept {
        return item_types<__item_sender_t<_Sequence>>();
      }

      [[nodiscard]]
      static constexpr auto get_env(__ignore) noexcept -> env<> {
        return {};
      }
    };
  } // namespace __iterate

  using __iterate::iterate_t;
  inline constexpr iterate_t iterate{};
} // namespace exec

#endif // STDEXEC_HAS_STD_RANGES()
