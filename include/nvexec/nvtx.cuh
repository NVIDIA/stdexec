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

// clang-format Language: Cpp

#pragma once

#include <nvtx3/nvToolsExt.h>

#include "../stdexec/execution.hpp"
#include <string>
#include <utility>

#include "stream/common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nvexec {

  namespace _strm::nvtx {

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

        template <class Tag, class... As>
        void _complete(Tag tag, As&&... as) noexcept {
          if constexpr (Kind == kind::push) {
            nvtxRangePushA(name_.c_str());
          } else {
            nvtxRangePop();
          }

          op_state_.propagate_completion_signal(tag, static_cast<As&&>(as)...);
        }

       public:
        using __id = receiver_t;

        template <class... _Args>
        void set_value(_Args&&... __args) noexcept {
          _complete(stdexec::set_value, static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
        void set_error(_Error&& __error) noexcept {
          _complete(stdexec::set_error, static_cast<_Error&&>(__error));
        }

        void set_stopped() noexcept {
          _complete(stdexec::set_stopped);
        }

        auto get_env() const noexcept -> Env {
          return op_state_.make_env();
        }

        explicit __t(operation_state_base_t<ReceiverId>& op_state, std::string name)
          : op_state_(op_state)
          , name_(std::move(name)) {
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
        using _completion_signatures_t =
          __try_make_completion_signatures<__copy_cvref_t<Self, Sender>, Env>;

        template <__decays_to<__t> Self, receiver Receiver>
          requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
        static auto connect(Self&& self, Receiver rcvr)
          -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
          return stream_op_state<__copy_cvref_t<Self, Sender>>(
            static_cast<Self&&>(self).sndr_,
            static_cast<Receiver&&>(rcvr),
            [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
              -> receiver_t<Receiver> {
              return receiver_t<Receiver>(stream_provider, std::move(self.name_));
            });
        }

        template <__decays_to<__t> Self, class Env>
        static auto
          get_completion_signatures(Self&&, Env&&) -> _completion_signatures_t<Self, Env> {
          return {};
        }

        auto get_env() const noexcept -> stream_sender_attrs<Sender> {
          return {&sndr_};
        }
      };
    };

    template <kind Kind, stdexec::sender Sender>
    using nvtx_sender_th =
      stdexec::__t<nvtx_sender_t<Kind, stdexec::__id<stdexec::__decay_t<Sender>>>>;

    struct push_t {
      template <stdexec::sender Sender>
      auto
        operator()(Sender&& sndr, std::string&& name) const -> nvtx_sender_th<kind::push, Sender> {
        return nvtx_sender_th<kind::push, Sender>{{}, static_cast<Sender&&>(sndr), std::move(name)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name) const -> stdexec::__binder_back<push_t, std::string> {
        return {{std::move(name)}, {}, {}};
      }
    };

    struct pop_t {
      template <stdexec::sender Sender>
      auto operator()(Sender&& sndr) const -> nvtx_sender_th<kind::pop, Sender> {
        return nvtx_sender_th<kind::pop, Sender>{{}, static_cast<Sender&&>(sndr), {}};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept -> stdexec::__binder_back<pop_t> {
        return {{}, {}, {}};
      }
    };

    inline constexpr push_t push{};
    inline constexpr pop_t pop{};

    struct scoped_t {
      template <stdexec::sender Sender, stdexec::__sender_adaptor_closure Closure>
      auto operator()(Sender&& __sndr, std::string&& name, Closure closure) const noexcept {
        return static_cast<Sender&&>(__sndr) | push(std::move(name)) | closure | pop();
      }

      template <stdexec::__sender_adaptor_closure Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name, Closure closure) const
        -> stdexec::__binder_back<scoped_t, std::string, Closure> {
        return {
          {std::move(name), static_cast<Closure&&>(closure)},
          {},
          {}
        };
      }
    };

    inline constexpr scoped_t scoped{};

  } // namespace _strm::nvtx

  namespace nvtx {
    using _strm::nvtx::push;
    using _strm::nvtx::pop;
    using _strm::nvtx::scoped;
  } // namespace nvtx

} // namespace nvexec

STDEXEC_PRAGMA_POP()
