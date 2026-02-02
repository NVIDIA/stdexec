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

    template <kind Kind, class Receiver>
    struct receiver : public stream_receiver_base {
      using env_t = _strm::opstate_base<Receiver>::env_t;

      _strm::opstate_base<Receiver>& opstate_;
      std::string name_;

      template <class Tag, class... Args>
      void _complete(Tag tag, Args&&... args) noexcept {
        if constexpr (Kind == kind::push) {
          nvtxRangePushA(name_.c_str());
        } else {
          nvtxRangePop();
        }

        opstate_.propagate_completion_signal(tag, static_cast<Args&&>(args)...);
      }

     public:
      template <class... Args>
      void set_value(Args&&... args) noexcept {
        _complete(STDEXEC::set_value, static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __error) noexcept {
        _complete(STDEXEC::set_error, static_cast<Error&&>(__error));
      }

      void set_stopped() noexcept {
        _complete(STDEXEC::set_stopped);
      }

      auto get_env() const noexcept -> env_t {
        return opstate_.make_env();
      }

      explicit receiver(_strm::opstate_base<Receiver>& opstate, std::string name)
        : opstate_(opstate)
        , name_(std::move(name)) {
      }
    };

    template <kind Kind, class Sender>
    struct nvtx_sender : stream_sender_base {
      Sender sndr_;
      std::string name_;

      template <class Receiver>
      using receiver_t = receiver<Kind, Receiver>;

      template <class Self, class Env>
      using _completions_t = __try_make_completion_signatures<__copy_cvref_t<Self, Sender>, Env>;

      template <__decays_to<nvtx_sender> Self, STDEXEC::receiver Receiver>
        requires receiver_of<Receiver, _completions_t<Self, env_of_t<Receiver>>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        -> stream_opstate_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_opstate<__copy_cvref_t<Self, Sender>>(
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr),
          [&](_strm::opstate_base<Receiver>& stream_provider) -> receiver_t<Receiver> {
            return receiver_t<Receiver>(stream_provider, std::move(self.name_));
          });
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<nvtx_sender> Self, class Env>
      static consteval auto get_completion_signatures() -> _completions_t<Self, Env> {
        return {};
      }

      auto get_env() const noexcept -> stream_sender_attrs<Sender> {
        return {&sndr_};
      }
    };

    template <kind Kind, STDEXEC::sender Sender>
    using nvtx_sender_t = nvtx_sender<Kind, STDEXEC::__decay_t<Sender>>;

    struct push_t {
      template <STDEXEC::sender Sender>
      auto
        operator()(Sender&& sndr, std::string&& name) const -> nvtx_sender_t<kind::push, Sender> {
        return nvtx_sender_t<kind::push, Sender>{{}, static_cast<Sender&&>(sndr), std::move(name)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name) const {
        return STDEXEC::__closure(*this, std::move(name));
      }
    };

    struct pop_t {
      template <STDEXEC::sender Sender>
      auto operator()(Sender&& sndr) const -> nvtx_sender_t<kind::pop, Sender> {
        return nvtx_sender_t<kind::pop, Sender>{{}, static_cast<Sender&&>(sndr), {}};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept {
        return STDEXEC::__closure(*this);
      }
    };

    inline constexpr push_t push{};
    inline constexpr pop_t pop{};

    struct scoped_t {
      template <STDEXEC::sender Sender, STDEXEC::__sender_adaptor_closure Closure>
      auto operator()(Sender&& __sndr, std::string&& name, Closure closure) const noexcept {
        return static_cast<Sender&&>(__sndr) | push(std::move(name)) | closure | pop();
      }

      template <STDEXEC::__sender_adaptor_closure Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(std::string name, Closure closure) const {
        return STDEXEC::__closure(*this, std::move(name), static_cast<Closure&&>(closure));
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

namespace STDEXEC::__detail {
  template <nvexec::_strm::nvtx::kind Kind, class Sender>
  extern __declfn_t<nvexec::_strm::nvtx::nvtx_sender<Kind, __demangle_t<Sender>>>
    __demangle_v<nvexec::_strm::nvtx::nvtx_sender<Kind, Sender>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
