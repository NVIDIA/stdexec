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
template <ranges::range BoundRange, stdexec::__movable_value BoundFun, class... Tail>
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
template <ranges::range BoundRange, stdexec::__movable_value Fun, class... Tail>
    requires ((sizeof...(Tail) == 0) && is_bound<BoundRange>)
  struct reduce_result_<BoundRange, unbound, Fun, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<Fun,
                                   ranges::range_value_t<BoundRange>,
                                   ranges::range_value_t<BoundRange>>
    >;
  };

// Predecessor sends range, function is bound.
template <stdexec::__movable_value BoundFun, ranges::range Range, class... Tail>
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
template <ranges::range Range, stdexec::__movable_value Fun, class... Tail>
    requires (sizeof...(Tail) == 0)
  struct reduce_result_<unbound, unbound, Range, Fun, Tail...> {
    using __t = ::cuda::std::decay_t<
      ::cuda::std::invoke_result_t<Fun,
                                   ranges::range_value_t<Range>,
                                   ranges::range_value_t<Range>>
    >;
  };

template <class BoundRange, class BoundFun, class... As>
  using reduce_result_t = stdexec::__t<reduce_result_<BoundRange, BoundFun, As...>>;

template <class SenderId, class ReceiverId, class BoundRange, class BoundFun>
    requires ((ranges::range<BoundRange> || !is_bound<BoundRange>) && stdexec::__movable_value<BoundFun>)
  struct receiver_t {
    class __t : public stream_receiver_base {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      template <class... As>
        struct result_size_for {
          using __t = stdexec::__msize_t<sizeof(reduce_result_t<BoundRange, BoundFun, As...>)>;
        };

      template <class... Sizes>
        struct max_in_pack {
          static constexpr std::size_t value = std::max({std::size_t{}, stdexec::__v<Sizes>...});
        };

      struct max_result_size {
        template <class... _As>
          using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

        static constexpr std::size_t value =
          stdexec::__v<
            stdexec::__gather_completions_for<
              stdexec::set_value_t,
              Sender,
              stdexec::env_of_t<Receiver>,
              stdexec::__q<result_size_for_t>,
              stdexec::__q<max_in_pack>>>;
      };

      operation_state_base_t<ReceiverId>& op_state_;
      [[no_unique_address]] BoundRange rng_;
      [[no_unique_address]] BoundFun fun_;

    public:
      using __id = receiver_t;

      constexpr static std::size_t memory_allocation_size = max_result_size::value;

      // Range and function are bound.
      friend void tag_invoke(stdexec::set_value_t, __t&& self) noexcept
        requires (is_bound<BoundRange> && is_bound<BoundFun>) {
          using Result = reduce_result_t<BoundRange, BoundFun>;
          self.template launch<Result>(self.rng_, self.fun_);
        }

      // Range is bound, function is the default.
      friend void tag_invoke(stdexec::set_value_t, __t&& self) noexcept
        requires (is_bound<BoundRange> && !is_bound<BoundFun>) {
          using Result = reduce_result_t<BoundRange, BoundFun>;
          self.template launch<Result>(self.rng_, cub::Sum{});
        }

      // Range is bound, predecessor sends function.
      template <stdexec::__movable_value Fun>
          requires (is_bound<BoundRange>)
        friend void tag_invoke(stdexec::set_value_t, __t&& self, Fun&& fun) noexcept {
          static_assert(!is_bound<BoundFun>);
          using Result = reduce_result_t<BoundRange, BoundFun, Fun>;
          self.template launch<Result>(self.rng_, (Fun&&) fun);
        }

      // Predecessor sends range, function is bound.
      template <ranges::range Range>
          requires (is_bound<BoundFun>)
        friend void tag_invoke(stdexec::set_value_t, __t&& self, Range&& rng) noexcept {
          static_assert(!is_bound<BoundRange>);
          using Result = reduce_result_t<BoundRange, BoundFun, Range>;
          self.template launch<Result>((Range&&) rng, self.fun_);
        }

      // Predecessor sends range, function is the default.
      template <ranges::range Range>
          requires (!is_bound<BoundFun>)
        friend void tag_invoke(stdexec::set_value_t, __t&& self, Range&& rng) noexcept {
          static_assert(!is_bound<BoundRange>);
          using Result = reduce_result_t<BoundRange, BoundFun, Range>;
          self.template launch<Result>((Range&&) rng, cub::Sum{});
        }

      // Predecessor sends range and function.
      template <ranges::range Range, stdexec::__movable_value Fun>
        friend void tag_invoke(stdexec::set_value_t, __t&& self, Range&& rng, Fun&& fun) noexcept {
          static_assert(!is_bound<BoundRange> && !is_bound<BoundFun>);
          using Result = reduce_result_t<BoundRange, BoundFun, Range, Fun>;
          self.template launch<Result>((Range&&) rng, (Fun&&) fun);
        }

      template <class Result, ranges::range Range, stdexec::__movable_value Fun = cub::Sum>
        void launch(Range&& rng, Fun&& fun = {}) noexcept {
          cudaStream_t stream = op_state_.get_stream();

          Result *d_out = reinterpret_cast<Result*>(op_state_.temp_storage_);

          void *d_temp_storage{};
          std::size_t temp_storage_size{};

          auto first = ranges::begin(rng);

          std::size_t num_items = ranges::size(rng);

          std::cout << "size: " << num_items << std::endl;

          std::abort();

          cudaError_t status;

          do {
            if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                  d_temp_storage, temp_storage_size,
                  first, d_out, num_items,
                  fun, Result{},
                  stream));
                status != cudaSuccess) {
              break;
            }

            if (status = STDEXEC_DBG_ERR(
                  cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
                status != cudaSuccess) {
              break;
            }

            if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                  d_temp_storage, temp_storage_size,
                  first, d_out, num_items,
                  fun, Result{},
                  stream));
                status != cudaSuccess) {
              break;
            }

            status = STDEXEC_DBG_ERR(cudaFreeAsync(d_temp_storage, stream));
          } while (false);

          if (status == cudaSuccess) {
            op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
          } else {
            op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

      template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag,
                class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        }

      friend stdexec::env_of_t<Receiver> tag_invoke(stdexec::get_env_t, const __t& self) {
        return stdexec::get_env(self.op_state_.receiver_);
      }

      __t(operation_state_base_t<ReceiverId>& op_state, BoundRange rng, BoundFun fun)
        : op_state_(op_state)
        , rng_((BoundRange&&) rng)
        , fun_((BoundFun&&) fun)
      {}
    };
  };

} // namespace reduce_

template <class SenderId, class BoundRange, class BoundFun>
  struct reduce_sender_t {
    struct __t : stream_sender_base {
      using __id = reduce_sender_t;

      using Sender = stdexec::__t<SenderId>;

      Sender sndr_;
      [[no_unique_address]] BoundRange rng_;
      [[no_unique_address]] BoundFun fun_;

      template <class Receiver>
        using receiver_t =
          stdexec::__t<reduce_::receiver_t<SenderId, stdexec::__id<Receiver>, BoundRange, BoundFun>>;

      template <class... As>
        using set_value_t =
          stdexec::completion_signatures<stdexec::set_value_t(
            std::add_lvalue_reference_t<
              reduce_::reduce_result_t<BoundRange, BoundFun, As...>
            >
          )>;

      template <class Self, class Env>
        using completion_signatures =
          stdexec::make_completion_signatures<
            stdexec::__copy_cvref_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
            set_value_t
            >;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> stream_op_state_t<stdexec::__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
          return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
            ((Self&&)self).sndr_,
            (Receiver&&)rcvr,
            [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider) -> receiver_t<Receiver> {
              return receiver_t<Receiver>(stream_provider, self.rng_, self.fun_);
            });
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env> requires true;

      friend auto tag_invoke(stdexec::get_env_t, const __t& self)
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }
    };
  };

struct reduce_t {
  template <class Sender, class BoundRange, class BoundFun>
    using sender_t = stdexec::__t<reduce_sender_t<
      stdexec::__id<::cuda::std::decay_t<Sender>>,
      ::cuda::std::remove_cvref_t<BoundRange>,
      ::cuda::std::remove_cvref_t<BoundFun>
    >>;

  // Range and function are bound.
  template <stdexec::sender Sender, ranges::range Range, stdexec::__movable_value Fun>
    sender_t<Sender, Range, Fun> operator()(Sender&& sndr, Range&& rng, Fun&& fun) const {
      return {{}, (Sender&&) sndr, (Range&&) rng, (Fun&&) fun};
    }

  // Range is bound, predecessor sends function or it's the default.
  template <stdexec::sender Sender, ranges::range Range>
    sender_t<Sender, Range, unbound> operator()(Sender&& sndr, Range&& rng) const {
      return {{}, (Sender&&) sndr, (Range&&) rng, {}};
    }

  // Predecessor sends range, function is bound.
  // FIXME: How do we constrain `Fun` without making this overload ambiguous
  // with the above range is bound, function is not overload?
  template <stdexec::sender Sender, class Fun>
    sender_t<Sender, unbound, Fun> operator()(Sender&& sndr, Fun&& fun) const {
      return {{}, (Sender&&) sndr, {}, (Fun&&) fun};
    }

  // Predecessor sends range, predecessor sends function or it's the default.
  template <stdexec::sender Sender>
    sender_t<Sender, unbound, unbound> operator()(Sender&& sndr) const {
      return {{}, (Sender&&) sndr, {}, {}};
    }

  // Range and function are bound.
  template <ranges::range Range, stdexec::__movable_value Fun>
    stdexec::__binder_back<reduce_t, Range, Fun> operator()(Range&& rng, Fun&& fun) const {
      return {{}, {}, {(Range&&) rng, (Fun&&) fun}};
    }

  // Range is bound, predecessor sends function or it's the default.
  template <ranges::range Range>
    stdexec::__binder_back<reduce_t, Range> operator()(Range&& rng) const {
      return {{}, {}, (Range&&) rng};
    }

  // Predecessor sends range, function is bound.
  // FIXME: How do we constrain `Fun` without making this overload ambiguous
  // with the above range is bound, function is not overload?
  template <class Fun>
    stdexec::__binder_back<reduce_t, Fun> operator()(Fun&& fun) const {
      return {{}, {}, (Fun&&) fun};
    }

  // Predecessor sends range, predecessor sends function or it's the default.
  stdexec::__binder_back<reduce_t> operator()() const {
    return {{}, {}};
  }
};

} // namespace STDEXEC_STREAM_DETAIL_NS

inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};

} // namespace nvexec

