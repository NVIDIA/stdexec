/*
 * Copyright (c) 2022-2023 NVIDIA Corporation
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

#include <type_traits>

#include <cuda/std/type_traits>

#include <range/v3/range/concepts.hpp>
#include <range/v3/range/access.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/range/traits.hpp>

#include <cub/device/device_reduce.cuh>

#include "../../stdexec/execution.hpp"

#include "common.cuh"

namespace nvexec {
namespace STDEXEC_STREAM_DETAIL_NS {

namespace reduce_ {

template <class BoundRange, class BoundFun, class... As>
  struct reduce_result_;

// Range and function are bound.
template <ranges::range BoundRange, std::execution::__movable_value BoundFun, class... Tail>
    requires ((sizeof...(Tail) == 0) && is_bound<BoundRange> && is_bound<BoundFun>)
  struct reduce_result_<BoundRange, BoundFun, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<BoundFun,
                                   ranges::range_value_t<BoundRange>,
                                   ranges::range_value_t<BoundRange>>
    >;
  };

// Range is bound, function is the default.
template <ranges::range BoundRange, class... Tail>
    requires ((sizeof...(Tail) == 0) && is_bound<BoundRange>)
  struct reduce_result_<BoundRange, unbound, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<cub::Sum,
                                   ranges::range_value_t<BoundRange>,
                                   ranges::range_value_t<BoundRange>>
    >;
  };

// Range is bound, predecessor sends function.
template <ranges::range BoundRange, std::execution::__movable_value Fun, class... Tail>
    requires ((sizeof...(Tail) == 0) && is_bound<BoundRange>)
  struct reduce_result_<BoundRange, unbound, Fun, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<Fun,
                                   ranges::range_value_t<BoundRange>,
                                   ranges::range_value_t<BoundRange>>
    >;
  };

// Predecessor sends range, function is bound.
template <std::execution::__movable_value BoundFun, ranges::range Range, class... Tail>
    requires ((sizeof...(Tail) == 0) && is_bound<BoundFun>)
  struct reduce_result_<unbound, BoundFun, Range, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<BoundFun,
                                   ranges::range_value_t<Range>,
                                   ranges::range_value_t<Range>>
    >;
  };

// Predecessor sends range, function is the default.
template <ranges::range Range, class... Tail>
    requires (sizeof...(Tail) == 0)
  struct reduce_result_<unbound, unbound, Range, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<cub::Sum,
                                   ranges::range_value_t<Range>,
                                   ranges::range_value_t<Range>>
    >;
  };

// Predecessor sends range and function.
template <ranges::range Range, std::execution::__movable_value Fun, class... Tail>
    requires (sizeof...(Tail) == 0)
  struct reduce_result_<unbound, unbound, Range, Fun, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<Fun,
                                   ranges::range_value_t<Range>,
                                   ranges::range_value_t<Range>>
    >;
  };

template <class BoundRange, class BoundFun, class... As>
  using reduce_result_t = std::__t<reduce_result_<BoundRange, BoundFun, As...>>;

template <class SenderId, class ReceiverId, class BoundRange, class BoundFun>
    requires ((ranges::range<BoundRange> || !is_bound<BoundRange>) && std::execution::__movable_value<BoundFun>)
  class receiver_t : public receiver_base_t {
    using Sender = std::__t<SenderId>;
    using Receiver = std::__t<ReceiverId>;

    template <class... As>
      using result_size_for_t = std::__index<sizeof(reduce_result_t<BoundRange, BoundFun, As...>)>;

    template <class... Sizes>
      struct max_in_pack {
        static constexpr std::size_t value = std::max({std::size_t{}, std::__v<Sizes>...});
      };

    struct max_result_size {
      static constexpr std::size_t value =
        std::__v<
          std::execution::__gather_sigs_t<
            std::execution::set_value_t,
            Sender,
            std::execution::env_of_t<Receiver>,
            std::__q<result_size_for_t>,
            std::__q<max_in_pack>>>;
    };

    [[no_unique_address]] BoundRange rng_;
    [[no_unique_address]] BoundFun fun_;
    operation_state_base_t<ReceiverId>& op_state_;

  public:

    constexpr static std::size_t memory_allocation_size = max_result_size::value;

    // Range and function are bound.
    friend void tag_invoke(std::execution::set_value_t, receiver_t&& self) noexcept
      requires (is_bound<BoundRange> && is_bound<BoundFun>) {
        using Result = reduce_result_t<BoundRange, BoundFun>;
        self.launch<Result>(self.rng_, self.fun_);
      }

    // Range is bound, function is the default.
    friend void tag_invoke(std::execution::set_value_t, receiver_t&& self) noexcept
      requires (is_bound<BoundRange> && !is_bound<BoundFun>) {
        using Result = reduce_result_t<BoundRange, BoundFun>;
        self.launch<Result>(self.rng_, cub::Sum{});
      }

    // Range is bound, predecessor sends function.
    template <std::execution::__movable_value Fun>
        requires (is_bound<BoundRange>)
      friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, Fun&& fun) noexcept {
        static_assert(!is_bound<BoundFun>);
        using Result = reduce_result_t<BoundRange, BoundFun, Fun>;
        self.launch<Result>(self.rng_, (Fun&&) fun);
      }

    // Predecessor sends range, function is bound.
    template <ranges::range Range>
        requires (is_bound<BoundFun>)
      friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, Range&& rng) noexcept {
        static_assert(!is_bound<BoundRange>);
        using Result = reduce_result_t<BoundRange, BoundFun, Range>;
        self.launch<Result>((Range&&) rng, self.fun_);
      }

    // Predecessor sends range, function is the default.
    template <ranges::range Range>
        requires (!is_bound<BoundFun>)
      friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, Range&& rng) noexcept {
        static_assert(!is_bound<BoundRange>);
        using Result = reduce_result_t<BoundRange, BoundFun, Range>;
        self.launch<Result>((Range&&) rng, cub::Sum{});
      }

    // Predecessor sends range and function.
    template <ranges::range Range, std::execution::__movable_value Fun>
      friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, Range&& rng, Fun&& fun) noexcept {
        static_assert(!is_bound<BoundRange> && !is_bound<BoundFun>);
        using Result = reduce_result_t<BoundRange, BoundFun, Range, Fun>;
        self.launch<Result>((Range&&) rng, (Fun&&) fun);
      }

    template <class Result, ranges::range Range, std::execution::__movable_value Fun = cub::Sum>
      void launch(Range&& rng, Fun&& fun = {}) noexcept {
        cudaStream_t stream = op_state_.stream_;

        Result *d_out = reinterpret_cast<Result*>(op_state_.temp_storage_);

        void *d_temp_storage{};
        std::size_t temp_storage_size{};

        auto first = ranges::begin(rng);
        auto last = ranges::end(rng);

        std::size_t num_items = ranges::size(rng);

        THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, first,
                                                      d_out, num_items, fun, Result{},
                                                      stream));
        THROW_ON_CUDA_ERROR(cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));

        THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, first,
                                                      d_out, num_items, fun, Result{},
                                                      stream));
        THROW_ON_CUDA_ERROR(cudaFreeAsync(d_temp_storage, stream));

        op_state_.propagate_completion_signal(std::execution::set_value, *d_out);
      }

    template <std::__one_of<std::execution::set_error_t,
                            std::execution::set_stopped_t> Tag,
              class... As _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        );
      }

    friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const receiver_t& self) {
      return std::execution::get_env(self.op_state_.receiver_);
    }

    receiver_t(operation_state_base_t<ReceiverId>& op_state, BoundRange rng, BoundFun fun)
      : rng_((BoundRange&&) rng)
      , fun_((BoundFun&&) fun)
      , op_state_(op_state)
    {}
  };

} // namespace reduce_

template <class SenderId, class BoundRangeId, class BoundFunId>
  struct reduce_sender_t : gpu_sender_base_t {
    using Sender = std::__t<SenderId>;
    using BoundRange = std::__t<BoundRangeId>;
    using BoundFun = std::__t<BoundFunId>;

    Sender sndr_;
    [[no_unique_address]] BoundRange rng_;
    [[no_unique_address]] BoundFun fun_;

    template <class Receiver>
      using receiver_t = reduce_::receiver_t<SenderId, std::__x<Receiver>, BoundRange, BoundFun>;

    template <class... As>
      using set_value_t =
        std::execution::completion_signatures<std::execution::set_value_t(
          std::add_lvalue_reference_t<
            reduce_::reduce_result_t<BoundRange, BoundFun, As...>
          >
        )>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::make_completion_signatures<
          std::__member_t<Self, Sender>,
          Env,
          std::execution::completion_signatures<std::execution::set_error_t(cudaError_t)>,
          set_value_t
          >;

    template <std::__decays_to<reduce_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<std::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<std::__member_t<Self, Sender>>(
          ((Self&&)self).sndr_,
          (Receiver&&)rcvr,
          [&](operation_state_base_t<std::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
            return receiver_t<Receiver>(stream_provider, self.rng_, self.fun_);
          });
    }

    template <std::__decays_to<reduce_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <std::__decays_to<reduce_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires std::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const reduce_sender_t& self, As&&... as)
      noexcept(std::__nothrow_callable<Tag, const Sender&, As...>)
      -> std::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

struct reduce_t {
  template <class Sender, class BoundRange, class BoundFun>
    using sender_t = reduce_sender_t<
      std::__x<::cuda::std::remove_cvref_t<Sender>>,
      std::__x<::cuda::std::remove_cvref_t<BoundRange>>,
      std::__x<::cuda::std::remove_cvref_t<BoundFun>>
    >;

  // Range and function are bound.
  template <std::execution::sender Sender, ranges::range Range, std::execution::__movable_value Fun>
    sender_t<Sender, Range, Fun> operator()(Sender&& sndr, Range&& rng, Fun&& fun) const {
      return {{}, (Sender&&) sndr, (Range&&) rng, (Fun&&) fun};
    }

  // Range is bound, predecessor sends function or it's the default.
  template <std::execution::sender Sender, ranges::range Range>
    sender_t<Sender, Range, unbound> operator()(Sender&& sndr, Range&& rng) const {
      return {{}, (Sender&&) sndr, (Range&&) rng, {}};
    }

  // Predecessor sends range, function is bound.
  // FIXME: How do we constrain `Fun` without making this overload ambiguous
  // with the above range is bound, function is not overload?
  template <std::execution::sender Sender, class Fun>
    sender_t<Sender, unbound, Fun> operator()(Sender&& sndr, Fun&& fun) const {
      return {{}, (Sender&&) sndr, {}, (Fun&&) fun};
    }

  // Predecessor sends range, predecessor sends function or it's the default.
  template <std::execution::sender Sender>
    sender_t<Sender, unbound, unbound> operator()(Sender&& sndr) const {
      return {{}, (Sender&&) sndr, {}, {}};
    }

  // Range and function are bound.
  template <ranges::range Range, std::execution::__movable_value Fun>
    std::execution::__binder_back<reduce_t, Range, Fun> operator()(Range&& rng, Fun&& fun) const {
      return {{}, {}, {(Range&&) rng, (Fun&&) fun}};
    }

  // Range is bound, predecessor sends function or it's the default.
  template <ranges::range Range>
    std::execution::__binder_back<reduce_t, Range> operator()(Range&& rng) const {
      return {{}, {}, (Range&&) rng};
    }

  // Predecessor sends range, function is bound.
  // FIXME: How do we constrain `Fun` without making this overload ambiguous
  // with the above range is bound, function is not overload?
  template <class Fun>
    std::execution::__binder_back<reduce_t, Fun> operator()(Fun&& fun) const {
      return {{}, {}, (Fun&&) fun};
    }

  // Predecessor sends range, predecessor sends function or it's the default.
  std::execution::__binder_back<reduce_t> operator()() const {
    return {{}, {}};
  }
};

} // namespace STDEXEC_STREAM_DETAIL_NS

inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};

} // namespace nvexec

