/*
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

#include <nvtx3/nvToolsExt.h>

#include "../stdexec/execution.hpp"
#include <type_traits>

#include "stream/common.cuh"

namespace nvexec {

  namespace STDEXEC_STREAM_DETAIL_NS { namespace nvtx {

    enum class kind {
      push,
      pop
    };

    template <kind Kind, class ReceiverId>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = typename operation_state_base_t<ReceiverId>::env_t;

        operation_state_base_t<ReceiverId>& op_state_;
        std::string name_;

       public:
        using __id = receiver_t;

        template <__completion_tag Tag, class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          if constexpr (Kind == kind::push) {
            nvtxRangePushA(self.name_.c_str());
          } else {
            nvtxRangePop();
          }

          self.op_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend Env tag_invoke(get_env_t, const __t& self) noexcept {
          return self.op_state_.make_env();
        }

        explicit __t(operation_state_base_t<ReceiverId>& op_state, std::string name)
          : op_state_(op_state)
          , name_(name) {
        }
      };
    };

    template <kind Kind, class SenderId>
    struct nvtx_sender_t {
      using Sender = stdexec::__t<SenderId>;

      struct __t : stream_sender_base {
        using __id = nvtx_sender_t;
        Sender sndr_;
        std::string name_;

        template <class Receiver>
        using receiver_t = stdexec::__t<receiver_t<Kind, stdexec::__id<Receiver>>>;

        template <class Self, class Env>
        using _completion_signatures_t = //
          __try_make_completion_signatures<__copy_cvref_t<Self, Sender>, Env>;

        template <__decays_to<__t> Self, receiver Receiver>
          requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
        friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr)
          -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
          return stream_op_state< __copy_cvref_t<Self, Sender>>(
            ((Self&&) self).sndr_,
            (Receiver&&) rcvr,
            [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
              -> receiver_t<Receiver> {
              return receiver_t<Receiver>(stream_provider, std::move(self.name_));
            });
        }

        template <__decays_to<__t> Self, class Env>
        friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
          -> _completion_signatures_t<Self, Env> {
          return {};
        }

        friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
          return get_env(self.sndr_);
        }
      };
    };

    template <kind Kind, stdexec::sender Sender>
    using nvtx_sender_th =
      stdexec::__t<nvtx_sender_t<Kind, stdexec::__id<stdexec::__decay_t<Sender>>>>;

    struct push_t {
      template <stdexec::sender Sender>
      nvtx_sender_th<kind::push, Sender> operator()(Sender&& sndr, std::string&& name) const {
        return nvtx_sender_th<kind::push, Sender>{{}, (Sender&&) sndr, std::move(name)};
      }

      stdexec::__binder_back<push_t, std::string> operator()(std::string name) const {
        return {{}, {}, std::move(name)};
      }
    };

    struct pop_t {
      template <stdexec::sender Sender>
      nvtx_sender_th<kind::pop, Sender> operator()(Sender&& sndr) const {
        return nvtx_sender_th<kind::pop, Sender>{{}, (Sender&&) sndr, {}};
      }

      stdexec::__binder_back<pop_t> operator()() const {
        return {{}, {}};
      }
    };

    inline constexpr push_t push{};
    inline constexpr pop_t pop{};

    struct scoped_t {
      template <stdexec::sender Sender, stdexec::__sender_adaptor_closure Closure>
      auto operator()(Sender&& __sndr, std::string&& name, Closure closure) const noexcept {
        return (Sender&&) __sndr | push(std::move(name)) | closure | pop();
      }

      template <stdexec::__sender_adaptor_closure Closure>
      auto operator()(std::string name, Closure closure) const
        -> stdexec::__binder_back<scoped_t, std::string, Closure> {
        return {
          {},
          {},
          {std::move(name), (Closure&&) closure}
        };
      }
    };

    inline constexpr scoped_t scoped{};

  }} // STDEXEC_STREAM_DETAIL_NS

  namespace nvtx {
  using STDEXEC_STREAM_DETAIL_NS::nvtx::push;
  using STDEXEC_STREAM_DETAIL_NS::nvtx::pop;
  using STDEXEC_STREAM_DETAIL_NS::nvtx::scoped;
  }

} // namespace nvexec
