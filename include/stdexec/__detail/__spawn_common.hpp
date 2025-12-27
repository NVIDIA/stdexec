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

namespace stdexec {
  ////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.spawn] paragraph 9
  // [exec.spawn.future] paragraph 15
  //
  // spawn and spawn_future both have to choose an allocator and an injected environment, and they
  // do it in the same way; this namespace provides a couple of functions for making those choices
  namespace __spawn_common {
    struct __choose_alloc_fn {
      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.1) -- if the expression get_allocator(env) is well-formed, then alloc is
      //               the result of get_allocator(env)
      template <class _Env, class _SenderEnv>
        requires __callable<get_allocator_t, _Env>
      auto operator()(const _Env& __env, const _SenderEnv&) const {
        return get_allocator(__env);
      }

      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.2) -- otherwise, if the expression get_allocator(get_env(new_sender)) is
      //               well-formed, then alloc is the result of get_allocator(get_env(new_sender))
      template <class _Env, class _SenderEnv>
        requires(!__callable<get_allocator_t, const _Env&>)
             && __callable<get_allocator_t, const _SenderEnv&>
      auto operator()(const _Env&, const _SenderEnv& __env) const {
        return get_allocator(__env);
      }

      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.3) -- otherwise, alloc is allocator<void>()
      template <class _Env, class _SenderEnv>
        requires(!__callable<get_allocator_t, const _Env&>)
             && (!__callable<get_allocator_t, const _SenderEnv&>)
      std::allocator<void> operator()(const _Env&, const _SenderEnv&) const {
        return std::allocator<void>();
      }
    };

    inline constexpr __choose_alloc_fn __choose_alloc{};

    struct __choose_senv_fn {
      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.1) -- if the expression get_allocator(env) is well-formed, then ...
      //               senv is the expression env;
      template <class _Env, class _SenderEnv>
        requires __callable<get_allocator_t, const _Env&>
      const _Env& operator()(const _Env& __env, const _SenderEnv&) const {
        return __env;
      }

      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.2) -- otherwise, if the expression get_allocator(get_env(new_sender)) is
      //               well-formed, then ... senv is the expression
      //               JOIN-ENV(prop(get_allocator, alloc), env);
      template <class _Env, class _SenderEnv>
        requires(!__callable<get_allocator_t, const _Env&>)
             && __callable<get_allocator_t, const _SenderEnv&>
      auto operator()(const _Env& __env, const _SenderEnv& __sndrEnv) const {
        return __env::__join(prop(get_allocator, get_allocator(__sndrEnv)), __env);
      }

      // [exec.spawn] paragraph 9
      // [exec.spawn.future] paragraph 15
      //   (9/15.3) -- otherwise, ... senv is the expression env.
      template <class _Env, class _SenderEnv>
        requires(!__callable<get_allocator_t, const _Env&>)
             && (!__callable<get_allocator_t, const _SenderEnv&>)
      const _Env& operator()(const _Env& __env, const _SenderEnv&) const {
        return __env;
      }
    };

    inline constexpr __choose_senv_fn __choose_senv{};
  } // namespace __spawn_common
} // namespace stdexec
