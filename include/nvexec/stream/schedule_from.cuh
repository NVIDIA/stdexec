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

#include "../../stdexec/execution.hpp"

#include <utility>

#include "adaptor.cuh"
#include "common.cuh"

namespace nv::execution
{
  namespace _strm
  {
    // schedule_from as a facade algorithm. The stream_domain customizes schedule_from(sndr)
    // for a sender that completes on a stream scheduler by re-tagging it with this tag (see
    // transform_sender_for<schedule_from_t> below).
    struct schedule_from_t : stream_tag_base
    {};

    namespace _schfr
    {
      // schedule_from(sndr): moves the child's completions off the device. The child is
      // connected to a receiver that stores its completion on the device and enqueues a task,
      // which completes the receiver on the host once the stream has reached it. (A stream
      // sender is connected that way too: completing a stream receiver on the host would not
      // wait for its work.)
      struct algo : stream_adaptor<algo>
      {
        template <class Key, class State, class Sender, class Receiver>
        static auto __connect(Key, State& state, Sender&& sndr, Receiver rcvr)
          -> enqueue_opstate<Sender, Receiver>
        {
          return enqueue_opstate<Sender, Receiver>(static_cast<Sender&&>(sndr),
                                                   static_cast<Receiver&&>(rcvr),
                                                   state);
        }

        // A stream that could not be created is reported on the host.
        template <class State, class Children>
        static auto __start(State& state, Children& children) noexcept
        {
          cudaError_t status = state.stream_provider_.status_;
          return STDEXEC::__facade::__choose(
            status == cudaSuccess,
            [&]() noexcept { return children.__start_all(); },
            [&]() noexcept
            { return STDEXEC::__facade::__send(state.rcvr_, set_error_t(), status); });
        }

        // The child's completion, which the task sends from the host
        template <class State, class Tag, class... Args>
        static auto
        __complete(STDEXEC::__facade::__child<0>, State& state, Tag tag, Args&&... args) noexcept
        {
          return STDEXEC::__facade::__send(state.rcvr_,
                                           tag,
                                           __decay_t<Args>(static_cast<Args&&>(args))...);
        }

        using stream_adaptor<algo>::__complete;
      };
    }  // namespace _schfr

    template <>
    struct transform_sender_for<STDEXEC::schedule_from_t>
    {
      template <class Env, class Data, class Sender>
      constexpr auto operator()(Env const &, __ignore, Data&& data, Sender&& sndr) const
      {
        if constexpr (stream_completing_sender<Sender, Env>)
        {
          return STDEXEC::__facade::__sender{_strm::schedule_from_t(),
                                             static_cast<Data&&>(data),
                                             static_cast<Sender&&>(sndr)};
        }
        else
        {
          // The child does not complete on one stream scheduler (as when_all, whose children
          // may complete on several), so there is no stream to wait for. Such a sender
          // completes a receiver that is not a stream receiver on the host (when_all does,
          // once its children's streams are done), so schedule_from has nothing to do.
          return STDEXEC::__facade::__sender{STDEXEC::schedule_from_t(),
                                             static_cast<Data&&>(data),
                                             static_cast<Sender&&>(sndr)};
        }
      }
    };
  }  // namespace _strm
}  // namespace nv::execution

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::schedule_from_t>
  : nv::execution::_strm::_schfr::algo
{};

namespace nvexec = nv::execution;
