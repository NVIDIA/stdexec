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

#include "../../stdexec/execution.hpp"
#include <type_traits>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "common.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <typename Range>
    auto __begin(Range&& range) {
      return begin(range);
    }

    namespace reduce_ {
      struct range_value_ {
        template <class RT, class... Range>
        auto operator()(RT&& range, Range...) {
          return *begin(range);
        }
      };

      template <class... Range>
      using range_value_t = ::cuda::std::invoke_result_t<range_value_, Range...>;

      template <class SenderId, class ReceiverId, class Fun>
      struct receiver_t {
        class __t : public stream_receiver_base {
          using Sender = stdexec::__t<SenderId>;
          using Receiver = stdexec::__t<ReceiverId>;

          template <class... Range>
          struct result_size_for {
            using __t = stdexec::__msize_t< sizeof(
              ::cuda::std::decay_t<
                ::cuda::std::
                  invoke_result_t< Fun, range_value_t<Range...>, range_value_t<Range...> > >)>;
          };

          template <class... Sizes>
          struct max_in_pack {
            static constexpr std::size_t value = std::max({std::size_t{}, stdexec::__v<Sizes>...});
          };

          struct max_result_size {
            template <class... _As>
            using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

            static constexpr std::size_t value = //
              stdexec::__v< stdexec::__gather_completions_for<
                stdexec::set_value_t,
                Sender,
                stdexec::env_of_t<Receiver>,
                stdexec::__q<result_size_for_t>,
                stdexec::__q<max_in_pack>>>;
          };

          Fun f_;
          operation_state_base_t<ReceiverId>& op_state_;

         public:
          using __id = receiver_t;

          constexpr static std::size_t memory_allocation_size = max_result_size::value;

          template <class Range>
          friend void tag_invoke(stdexec::set_value_t, __t&& self, Range&& range) noexcept {
            cudaStream_t stream = self.op_state_.get_stream();

            using Result = //
              ::cuda::std::decay_t< ::cuda::std::invoke_result_t<
                Fun,
                decltype(*begin(std::declval<Range>())),
                decltype(*begin(std::declval<Range>()))>>;

            using value_t = Result;
            value_t* d_out = static_cast<value_t*>(self.op_state_.temp_storage_);

            void* d_temp_storage{};
            std::size_t temp_storage_size{};

            auto first = begin(range);
            auto last = end(range);

            std::size_t num_items = std::distance(first, last);

            cudaError_t status;

            do {
              if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                    d_temp_storage,
                    temp_storage_size,
                    first,
                    d_out,
                    num_items,
                    self.f_,
                    value_t{},
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
                    d_temp_storage,
                    temp_storage_size,
                    first,
                    d_out,
                    num_items,
                    self.f_,
                    value_t{},
                    stream));
                  status != cudaSuccess) {
                break;
              }

              status = STDEXEC_DBG_ERR(cudaFreeAsync(d_temp_storage, stream));
            } while (false);

            if (status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }

          template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
          friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
            self.op_state_.propagate_completion_signal(tag, (As&&) as...);
          }

          friend stdexec::env_of_t<Receiver> tag_invoke(stdexec::get_env_t, const __t& self) {
            return stdexec::get_env(self.op_state_.receiver_);
          }

          __t(Fun fun, operation_state_base_t<ReceiverId>& op_state)
            : f_((Fun&&) fun)
            , op_state_(op_state) {
          }
        };
      };
    }

    template <class SenderId, class Fun>
    struct reduce_sender_t {
      using Sender = stdexec::__t<SenderId>;

      struct __t : stream_sender_base {
        using __id = reduce_sender_t;

        Sender sndr_;
        Fun fun_;

        template <class Receiver>
        using receiver_t =
          stdexec::__t< reduce_::receiver_t< SenderId, stdexec::__id<Receiver>, Fun>>;

        template <class... Range>
          requires(sizeof...(Range) == 1)
        using set_value_t = stdexec::completion_signatures<stdexec::set_value_t(
          std::add_lvalue_reference_t< ::cuda::std::decay_t< ::cuda::std::invoke_result_t<
            Fun,
            decltype(*__begin(std::declval<Range>())),
            decltype(*__begin(std::declval<Range>()))> > >...)>;

        template <class Self, class Env>
        using completion_signatures = //
          stdexec::make_completion_signatures<
            stdexec::__copy_cvref_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
            set_value_t >;

        template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
          requires stdexec::
            receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
          friend auto
          tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> stream_op_state_t<
            stdexec::__copy_cvref_t<Self, Sender>,
            receiver_t<Receiver>,
            Receiver> {
          return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
            ((Self&&) self).sndr_,
            (Receiver&&) rcvr,
            [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
              -> receiver_t<Receiver> { return receiver_t<Receiver>(self.fun_, stream_provider); });
        }

        template <stdexec::__decays_to<__t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
          -> stdexec::dependent_completion_signatures<Env>;

        template <stdexec::__decays_to<__t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
          -> completion_signatures<Self, Env>
          requires true;

        friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
          noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
            -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
          return stdexec::get_env(self.sndr_);
        }
      };
    };

    struct reduce_t {
      template <class Sender, class Fun>
      using __sender =
        stdexec::__t<reduce_sender_t<stdexec::__id<stdexec::__decay_t<Sender>>, Fun>>;

      template <stdexec::sender Sender, stdexec::__movable_value Fun>
      __sender<Sender, Fun> operator()(Sender&& __sndr, Fun __fun) const {
        return __sender<Sender, Fun>{{}, (Sender&&) __sndr, (Fun&&) __fun};
      }

      template <class Fun = cub::Sum>
      stdexec::__binder_back<reduce_t, Fun> operator()(Fun __fun = {}) const {
        return {{}, {}, {(Fun&&) __fun}};
      }
    };
  }

  inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};
}
