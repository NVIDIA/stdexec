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

template <class Fun, ranges::range Range, class... Tail>
    requires (sizeof...(Tail) == 0)
  using reduce_result_t = ::cuda::std::decay_t<
    ::cuda::std::invoke_result_t<Fun,
                                 ranges::range_value_t<Range>,
                                 ranges::range_value_t<Range>>
  >;

template <class SenderId, class ReceiverId, class Fun>
  class receiver_t : public receiver_base_t {
    using Sender = std::__t<SenderId>;
    using Receiver = std::__t<ReceiverId>;

    template <ranges::range... Range>
        requires (sizeof...(Range) == 1)
      struct result_size_for {
        using __t = std::__index<sizeof(reduce_result_t<Fun, Range...>)>;
      };

    template <ranges::range... Range>
        requires (sizeof...(Range) == 1)
      using result_size_for_t = std::__t<result_size_for<Range...>>;

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

    Fun f_;
    operation_state_base_t<ReceiverId> &op_state_;

  public:

    constexpr static std::size_t memory_allocation_size = max_result_size::value;

    template <ranges::range Range>
    friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, Range&& rng) noexcept {
      cudaStream_t stream = self.op_state_.stream_;

      using value_t = reduce_result_t<Fun, Range>;
      value_t *d_out = reinterpret_cast<value_t*>(self.op_state_.temp_storage_);

      void *d_temp_storage{};
      std::size_t temp_storage_size{};

      auto first = ranges::begin(rng);
      auto last = ranges::end(rng);

      std::size_t num_items = ranges::size(rng);

      THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, first,
                                                    d_out, num_items, self.f_, value_t{},
                                                    stream));
      THROW_ON_CUDA_ERROR(cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));

      THROW_ON_CUDA_ERROR(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_size, first,
                                                    d_out, num_items, self.f_, value_t{},
                                                    stream));
      THROW_ON_CUDA_ERROR(cudaFreeAsync(d_temp_storage, stream));

      self.op_state_.propagate_completion_signal(std::execution::set_value, *d_out);
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

    receiver_t(Fun fun, operation_state_base_t<ReceiverId> &op_state)
      : f_((Fun&&) fun)
      , op_state_(op_state)
    {}
  };

} // namespace reduce_

template <class SenderId, class FunId>
  struct reduce_sender_t : gpu_sender_base_t {
    using Sender = std::__t<SenderId>;
    using Fun = std::__t<FunId>;

    Sender sndr_;
    Fun fun_;

    template <class Receiver>
      using receiver_t = reduce_::receiver_t<
          SenderId,
          std::__x<Receiver>,
          Fun>;

    template <ranges::range... Range>
        requires (sizeof...(Range) == 1)
      using set_value_t =
        std::execution::completion_signatures<std::execution::set_value_t(
          std::add_lvalue_reference_t<
            reduce_::reduce_result_t<Fun, Range>
          >...
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
            return receiver_t<Receiver>(self.fun_, stream_provider);
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
  template <class Sender, class Fun>
    using __sender =
      reduce_sender_t<
        std::__x<std::remove_cvref_t<Sender>>,
        std::__x<std::remove_cvref_t<Fun>>>;

  template <std::execution::sender Sender, std::execution::__movable_value Fun>
    __sender<Sender, Fun> operator()(Sender&& __sndr, Fun __fun) const {
      return __sender<Sender, Fun>{{}, (Sender&&) __sndr, (Fun&&) __fun};
    }

  template <class Fun = cub::Sum>
    std::execution::__binder_back<reduce_t, Fun> operator()(Fun __fun={}) const {
      return {{}, {}, {(Fun&&) __fun}};
    }
};

} // namespace STDEXEC_STREAM_DETAIL_NS

inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};

} // namespace nvexec

