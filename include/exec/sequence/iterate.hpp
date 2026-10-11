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

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)
import std;
import stdexec;
#else
#  include "../../stdexec/__detail/__execution_fwd.hpp"

#  include "../../stdexec/__detail/__concepts.hpp"
#  include "../../stdexec/__detail/__connect.hpp"
#  include "../../stdexec/__detail/__env.hpp"
#  include "../../stdexec/__detail/__operation_states.hpp"
#  include "../../stdexec/__detail/__optional.hpp"
#  include "../../stdexec/__detail/__receivers.hpp"
#  include "../../stdexec/__detail/__schedulers.hpp"
#  include "../../stdexec/__detail/__sender_concepts.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#    include <ranges>
#  endif

#  include "../detail/basic_sequence.hpp"
#  include "../sender_for.hpp"
#  include "../sequence.hpp"
#  include "../sequence_senders.hpp"
#  include "../trampoline_scheduler.hpp"

namespace experimental::execution
{
  namespace __iterate
  {
    using namespace STDEXEC;

    // The operation state owns the range it iterates, mirroring what the
    // default `get-state` does for a sender's data: the range is copied out of
    // an lvalue sender and moved out of an rvalue one. The range member is
    // declared first so that the iterators are initialized from the stored
    // copy and destroyed before it; the child item operations stored in the
    // derived class are destroyed before anything in this base class.
    template <class _Range>
    struct __operation_base_base
    {
      constexpr explicit __operation_base_base(_Range __range)
        noexcept(__nothrow_move_constructible<_Range> && noexcept(std::ranges::begin(__range_))
                 && noexcept(std::ranges::end(__range_))
                 && __nothrow_move_constructible<std::ranges::iterator_t<_Range>,
                                                 std::ranges::sentinel_t<_Range>>)
        : __range_{static_cast<_Range&&>(__range)}
        , __iterator_{std::ranges::begin(__range_)}
        , __sentinel_{std::ranges::end(__range_)}
      {}

      STDEXEC_ATTRIBUTE(no_unique_address) _Range __range_;
      STDEXEC_ATTRIBUTE(no_unique_address) std::ranges::iterator_t<_Range> __iterator_;
      STDEXEC_ATTRIBUTE(no_unique_address) std::ranges::sentinel_t<_Range> __sentinel_;
      trampoline_scheduler __scheduler_{};
    };

    template <class _Range, class _Receiver>
    struct __operation_base : __operation_base_base<_Range>
    {
      constexpr explicit __operation_base(_Range __range, _Receiver&& __rcvr)
        noexcept(noexcept(__operation_base_base<_Range>{static_cast<_Range&&>(__range)})
                 && __nothrow_move_constructible<_Receiver>)
        : __operation_base_base<_Range>{static_cast<_Range&&>(__range)}
        , __rcvr_{static_cast<_Receiver&&>(__rcvr)}
      {}

      virtual ~__operation_base()                    = default;
      virtual constexpr void __start_next() noexcept = 0;

      _Receiver __rcvr_;
    };

    template <class _Range, class _ItemRcvr>
    struct __item_operation
    {
      constexpr void start() noexcept
      {
        STDEXEC::set_value(static_cast<_ItemRcvr&&>(__rcvr_), *__parent_->__iterator_++);
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _ItemRcvr                      __rcvr_;
      __operation_base_base<_Range>* __parent_;
    };

    template <class _Range>
    struct __sender
    {
      using sender_concept = STDEXEC::sender_tag;
      using completion_signatures = STDEXEC::completion_signatures<set_value_t(
        std::iter_reference_t<std::ranges::iterator_t<_Range>>)>;

      template <receiver_of<completion_signatures> _ItemRcvr>
      constexpr auto connect(_ItemRcvr __rcvr) const & noexcept(__nothrow_decay_copyable<_ItemRcvr>)
        -> __item_operation<_Range, _ItemRcvr>
      {
        return {static_cast<_ItemRcvr&&>(__rcvr), __parent_};
      }

      __operation_base_base<_Range>* __parent_;
    };

    template <class _Range, class _Receiver>
    struct __next_receiver
    {
      using receiver_concept = STDEXEC::receiver_tag;

      constexpr void set_value() noexcept
      {
        __op_->__start_next();
      }

      constexpr void set_stopped() noexcept
      {
        __set_value_unless_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__op_->__rcvr_);
      }

      __operation_base<_Range, _Receiver>* __op_;
    };

    template <class _Range, class _Receiver>
    struct __operation final : __operation_base<_Range, _Receiver>
    {
      using __item_sender_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler>, __sender<_Range>>;
      using __next_receiver_t = __next_receiver<_Range, _Receiver>;
      using __next_sender_t   = next_sender_of_t<_Receiver, __item_sender_t>;

      constexpr explicit __operation(_Range __range, _Receiver&& __rcvr)
        noexcept(noexcept(__operation_base_base<_Range>{static_cast<_Range&&>(__range)})
                 && __nothrow_move_constructible<_Receiver>)
        : __operation_base<_Range, _Receiver>{static_cast<_Range&&>(__range),
                                              static_cast<_Receiver&&>(__rcvr)}
      {}

      constexpr void __start_next() noexcept final
      {
        if (this->__iterator_ == this->__sentinel_)
        {
          STDEXEC::set_value(static_cast<_Receiver&&>(this->__rcvr_));
        }
        else
        {
          STDEXEC_TRY
          {
            STDEXEC::start(__op_.__emplace_from(
              STDEXEC::connect,
              exec::set_next(this->__rcvr_,
                             exec::sequence(STDEXEC::schedule(this->__scheduler_),
                                            __sender<_Range>{this})),
              __next_receiver_t{this}));
          }
          STDEXEC_CATCH_ALL
          {
            STDEXEC::set_error(static_cast<_Receiver&&>(this->__rcvr_), std::current_exception());
          }
        }
      }

      constexpr void start() & noexcept
      {
        __start_next();
      }

      __optional<connect_result_t<__next_sender_t, __next_receiver_t>> __op_{};
    };

    template <class _Receiver>
    struct __subscribe_fn
    {
      template <class _Range>
      constexpr auto operator()(__ignore, _Range&& __range) noexcept(
        __nothrow_decay_copyable<_Range>
        && noexcept(__operation<__decay_t<_Range>, _Receiver>{static_cast<_Range&&>(__range),
                                                              static_cast<_Receiver&&>(__rcvr_)}))
      {
        // The operation state owns a decayed copy of the range so that its
        // iterators never point into the sender, which may be a temporary.
        return __operation<__decay_t<_Range>, _Receiver>{static_cast<_Range&&>(__range),
                                                         static_cast<_Receiver&&>(__rcvr_)};
      }

      _Receiver& __rcvr_;
    };

    STDEXEC_MODULE_EXPORT
    struct iterate_t
    {
      template <std::ranges::forward_range _Range>
        requires __decay_copyable<_Range>
      constexpr auto operator()(_Range&& __range) const -> __well_formed_sequence_sender auto
      {
        return make_sequence_expr<iterate_t>(__decay_t<_Range>{static_cast<_Range&&>(__range)});
      }

      template <class _Range>
      using __sender_t = __sender<_Range>;

      using __completion_sigs =
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;

      // The operation state iterates a decayed, non-const copy of the
      // sequence's range (see `__subscribe_fn` above), so the item types are
      // computed from that copy rather than from the sender's data, whose
      // cvref qualification can differ.
      template <class _Sequence>
      using __item_sender_t = __result_of<exec::sequence,
                                          schedule_result_t<trampoline_scheduler&>,
                                          __sender_t<__decay_t<__data_of<_Sequence>>>>;

      template <class _Sequence, class _Receiver>
      using _NextReceiver = __next_receiver<__decay_t<__data_of<_Sequence>>, _Receiver>;

      template <class _Sequence, class _Receiver>
      using _NextSender = next_sender_of_t<_Receiver, __item_sender_t<_Sequence>>;

      template <class _SeqExpr,
                sequence_receiver_of<item_types<__item_sender_t<_SeqExpr>>> _Receiver>
        requires sender_to<_NextSender<_SeqExpr, _Receiver>, _NextReceiver<_SeqExpr, _Receiver>>
      static constexpr auto subscribe(_SeqExpr&& __seq, _Receiver __rcvr)
        noexcept(__nothrow_applicable<__subscribe_fn<_Receiver>, _SeqExpr>)
          -> __apply_result_t<__subscribe_fn<_Receiver>, _SeqExpr>
      {
        static_assert(sender_for<_SeqExpr, iterate_t>);
        return __apply(__subscribe_fn<_Receiver>{__rcvr}, static_cast<_SeqExpr&&>(__seq));
      }

      template <class, class...>
      static consteval auto get_completion_signatures() noexcept
      {
        return completion_signatures<set_value_t(),
                                     set_error_t(std::exception_ptr),
                                     set_stopped_t()>();
      }

      template <class _Sequence, class... _Env>
      static consteval auto get_item_types() noexcept
      {
        static_assert(sender_for<_Sequence, iterate_t>);
        return item_types<__item_sender_t<_Sequence>>();
      }

      [[nodiscard]]
      static constexpr auto get_env(__ignore) noexcept -> env<>
      {
        return {};
      }
    };
  }  // namespace __iterate

  STDEXEC_MODULE_EXPORT
  using __iterate::iterate_t;
  inline constexpr iterate_t iterate{};
}  // namespace experimental::execution

namespace exec = experimental::execution;
#endif
