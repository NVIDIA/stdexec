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
#include <ranges>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "common.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS::__algo_base {
    template<class Range, class Fun>
    using binary_invoke_result_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<
                                        Fun,
                                        ::std::ranges::range_value_t<Range>,
                                        ::std::ranges::range_value_t<Range>>>;

    template <class SenderId, class ReceiverId, class Payload, class DerivedReceiver>
    struct receiver_t {
        struct __t : public stream_receiver_base {
            using Sender = stdexec::__t<SenderId>;
            using Receiver = stdexec::__t<ReceiverId>;

            template <class... Range>
            struct result_size_for {
            using __t = stdexec::__msize_t< sizeof(typename DerivedReceiver::result_t<Range...>)>;
            };

            template <class... Sizes>
            struct max_in_pack {
            static constexpr ::std::size_t value = ::std::max({::std::size_t{}, stdexec::__v<Sizes>...});
            };

            struct max_result_size {
            template <class... _As>
            using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

            static constexpr ::std::size_t value = //
                stdexec::__v< stdexec::__gather_completions_for<
                stdexec::set_value_t,
                Sender,
                stdexec::env_of_t<Receiver>,
                stdexec::__q<result_size_for_t>,
                stdexec::__q<max_in_pack>>>;
            };

            operation_state_base_t<ReceiverId>& op_state_;
            Payload payload_;

            public:
            using __id = receiver_t;

            constexpr static ::std::size_t memory_allocation_size = max_result_size::value;

            template <stdexec::same_as<stdexec::set_value_t> _Tag, class Range>
            friend void tag_invoke(_Tag, __t&& self, Range&& range) noexcept {
                DerivedReceiver::set_value_impl((__t&&) self, (Range&&) range);
            }

            template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
            friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
            self.op_state_.propagate_completion_signal(Tag(), (As&&) as...);
            }

            friend stdexec::env_of_t<Receiver> tag_invoke(stdexec::get_env_t, const __t& self) {
            return stdexec::get_env(self.op_state_.receiver_);
            }

            __t(Payload payload, operation_state_base_t<ReceiverId>& op_state) : op_state_(op_state) , payload_((Payload&&) payload) {}
        };
    };
}