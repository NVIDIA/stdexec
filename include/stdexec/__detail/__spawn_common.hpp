/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"
#include "__env.hpp"
#include "__queries.hpp"

#include <memory>

//////////////////////////////////////////////////////////////////////////////////////////////////
// [exec.spawn] paragraph 9
// [exec.spawn.future] paragraph 15
//
// spawn and spawn_future both have to choose an allocator and an injected environment, and they
// do it in the same way; this namespace provides a couple of functions for making those choices
namespace STDEXEC::__spawn_common {
  struct __choose_alloc_fn {
    template <class _Env, class _Attrs>
    auto operator()(const _Env& __env, const _Attrs& __attrs) const {
      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      if constexpr (__callable<get_allocator_t, _Env>) {
        //   (9/15.1) -- if the expression get_allocator(env) is well-formed, then alloc is
        //               the result of get_allocator(env)
        return get_allocator(__env);
      } else if constexpr (__callable<get_allocator_t, _Attrs>) {
        //   (9/15.2) -- otherwise, if the expression get_allocator(get_env(new_sender)) is
        //               well-formed, then alloc is the result of get_allocator(get_env(new_sender))
        return get_allocator(__attrs);
      } else {
        //   (9/15.3) -- otherwise, alloc is allocator<void>()
        return std::allocator<void>{};
      }
    }
  };

  inline constexpr __choose_alloc_fn __choose_alloc{};

  struct __choose_senv_fn {
    template <class _Env, class _Attrs>
    auto operator()(_Env&& __env, _Attrs&& __attrs) const {
      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      if constexpr (__callable<get_allocator_t, _Env&>) {
        //   (9/15.1) -- if the expression get_allocator(env) is well-formed, then ...
        //               senv is the expression env;
        return static_cast<_Env&&>(__env);
      } else if constexpr (__callable<get_allocator_t, _Attrs&>) {
        //   (9/15.2) -- otherwise, if the expression get_allocator(get_env(new_sender)) is
        //               well-formed, then ... senv is the expression
        //               JOIN-ENV(prop(get_allocator, alloc), env);
        return __env::__join(
          prop(get_allocator, get_allocator(__attrs)), static_cast<_Env&&>(__env));
      } else {
        //   (9/15.3) -- otherwise, ... senv is the expression env.
        return static_cast<_Env&&>(__env);
      }
    }
  };

  inline constexpr __choose_senv_fn __choose_senv{};
} // namespace STDEXEC::__spawn_common
