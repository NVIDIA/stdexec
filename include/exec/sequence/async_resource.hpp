/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "./let_each.hpp"
#include "./ignore_all.hpp"
#include "./zip.hpp"

namespace exec {
  namespace async_resource {
    namespace __run {
      using namespace stdexec;

      struct open_t {
        template <class _Resource>
          requires tag_invocable<open_t, _Resource>
                && sender<tag_invoke_result_t<open_t, _Resource>>
        auto operator()(_Resource&& __resource) const
          noexcept(nothrow_tag_invocable<open_t, _Resource>)
            -> tag_invoke_result_t<open_t, _Resource> {
          return tag_invoke(*this, static_cast<_Resource&&>(__resource));
        }
      };

      struct close_t {
        template <class _Resource>
          requires tag_invocable<close_t, _Resource>
                && sender<tag_invoke_result_t<close_t, _Resource>>
        auto operator()(_Resource&& __resource) const
          noexcept(nothrow_tag_invocable<close_t, _Resource>)
            -> tag_invoke_result_t<close_t, _Resource> {
          return tag_invoke(*this, static_cast<_Resource&&>(__resource));
        }
      };

      template <class _Resource>
      struct __sender {
        struct __t {
          using __id = __sender;
          using __open_sender_t = __call_result_t<open_t, _Resource&>;

          _Resource* __resource;

          template <class _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, const Env&)
            -> make_completion_signatures<
              __copy_cvref_t<_Self, __open_sender_t>,
              _Env,
              completion_signatures<set_stopped()>>;
        };
      };

      template <class _Resource>
      struct resource_facade {
        [[no_unique_address]] _Resource __resource;
      };

      struct run_t {
        template <class _Resource>
          requires tag_invocable<run_t, _Resource> && sender<tag_invoke_result_t<run_t, _Resource>>
        auto operator()(_Resource&& __resource) const
          noexcept(nothrow_tag_invocable<run_t, _Resource>)
            -> tag_invoke_result_t<run_t, _Resource> {
          return tag_invoke(*this, static_cast<_Resource&&>(__resource));
        }
      };
    }

    using __run::run_t;
    using __run::open_t;
    using __run::close_t;

    inline constexpr run_t run{};
    inline constexpr open_t open{};
    inline constexpr close_t close{};
  }

  struct use_resource_t {
    template <class _SenderFactory, class... _Resources>
    auto operator()(_SenderFactory&& __fn, _Resources&&... __resources) const {
      return ignore_all(let_value_each(
        zip(static_cast<_Resources&&>(__resources)...), static_cast<_SenderFactory&&>(__fn)));
    }
  };
}
