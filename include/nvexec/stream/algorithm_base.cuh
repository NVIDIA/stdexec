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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/__detail/__ranges.hpp"
#include "../../stdexec/execution.hpp"
#include <cstddef>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "common.cuh"

namespace nvexec::_strm::__algo_range_init_fun {
  template <class Range, class InitT, class Fun>
  using binary_invoke_result_t = ::cuda::std::decay_t<
    ::cuda::std::invoke_result_t<Fun, STDEXEC::ranges::range_reference_t<Range>, InitT>
  >;

  template <class Sender, class Receiver, class InitT, class Fun, class DerivedReceiver>
  struct receiver : public stream_receiver_base {
    struct result_size_for {
      template <class... Range>
      using __f = __msize_t<sizeof(
        typename __minvoke_q<__mfirst, DerivedReceiver, Range...>::template result_t<Range...>)>;
    };

    _strm::opstate_base<Receiver>& opstate_;
    STDEXEC_ATTRIBUTE(no_unique_address) InitT init_;
    STDEXEC_ATTRIBUTE(no_unique_address) Fun fun_;

   public:
    static constexpr std::size_t memory_allocation_size() noexcept {
      return __gather_completions_of_t<
        set_value_t,
        Sender,
        env_of_t<Receiver>,
        result_size_for,
        maxsize
      >::value;
    }

    template <class Range>
    void set_value(Range&& range) noexcept {
      DerivedReceiver::set_value_impl(static_cast<receiver&&>(*this), static_cast<Range&&>(range));
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
    }

    void set_stopped() noexcept {
      opstate_.propagate_completion_signal(set_stopped_t());
    }

    [[nodiscard]]
    auto get_env() const noexcept -> env_of_t<Receiver> {
      return STDEXEC::get_env(opstate_.rcvr_);
    }

    receiver(InitT init, Fun fun, _strm::opstate_base<Receiver>& opstate)
      : opstate_(opstate)
      , init_(static_cast<InitT&&>(init))
      , fun_(static_cast<Fun&&>(fun)) {
    }
  };

  template <class Sender, class InitT, class Fun, class DerivedSender>
  struct sender : stream_sender_base {
    // in the following two type aliases, __mfirst is used to make DerivedSender,
    // which is incomplete in this context, dependent to avoid hard errors.
    template <class Receiver>
    using receiver_t = __minvoke_q<__mfirst, DerivedSender, Receiver>::template receiver_t<Receiver>;

    template <class Range>
    using _set_value_t = __minvoke_q<__mfirst, DerivedSender, Range>::template _set_value_t<Range>;

    template <class Self, class... Env>
    using _completions_t = STDEXEC::transform_completion_signatures<
      __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
      completion_signatures<set_error_t(cudaError_t)>,
      __mtry_q<_set_value_t>::template __f
    >;

    template <__decays_to_derived_from<sender> Self, STDEXEC::receiver Receiver>
      requires receiver_of<Receiver, _completions_t<Self, env_of_t<Receiver>>>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> stream_opstate_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
      return stream_opstate<__copy_cvref_t<Self, Sender>>(
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Receiver> {
          return receiver_t<Receiver>{
            {self.init_, self.fun_, stream_provider}
          };
        });
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to_derived_from<sender> Self, class... Env>
    static consteval auto get_completion_signatures() -> _completions_t<Self, Env...> {
      return {};
    }

    auto get_env() const noexcept -> env_of_t<const Sender&> {
      return STDEXEC::get_env(sndr_);
    }

    Sender sndr_;
    STDEXEC_ATTRIBUTE(no_unique_address) InitT init_;
    STDEXEC_ATTRIBUTE(no_unique_address) Fun fun_;
  };
} // namespace nvexec::_strm::__algo_range_init_fun

namespace STDEXEC::__detail {
  template <class Sender, class InitT, class Fun, class DerivedSender>
  extern __declfn_t<nvexec::_strm::__algo_range_init_fun::sender<
    __demangle_t<Sender>,
    InitT,
    Fun,
    __demangle_t<DerivedSender>
  >>
    __demangle_v<nvexec::_strm::__algo_range_init_fun::sender<Sender, InitT, Fun, DerivedSender>>;
} // namespace STDEXEC::__detail
