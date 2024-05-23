/*
 * Copyright (c) 2024 Kirk Shoop
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

#include "../../stdexec/__detail/__execution_fwd.hpp"
#include "../../stdexec/__detail/__receivers.hpp"

#include "../../stdexec/concepts.hpp"

namespace exec {

// disable spurious warning in clang 
// https://github.com/llvm/llvm-project/issues/61566
STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wundefined-internal")

// fake receiver used to calculate whether inner connect is nothrow
template<class _Env>
struct __decl_receiver {
  using __t = __decl_receiver;
  using __id = __decl_receiver;

  using receiver_concept = stdexec::receiver_t;

  template <stdexec::same_as<stdexec::set_value_t> _Tag, class... _An>
  friend void tag_invoke(_Tag, __t&& __rcvr, _An&&... __an) noexcept;

  template <stdexec::same_as<stdexec::set_error_t> _Tag, class _Error>
  friend void tag_invoke(_Tag, __t&& __rcvr, _Error&& __err) noexcept;

  template <stdexec::same_as<stdexec::set_stopped_t> _Tag>
  friend void tag_invoke(_Tag, __t&& __rcvr) noexcept;

  template <stdexec::same_as<stdexec::get_env_t> _Tag>
  friend _Env tag_invoke(_Tag, const __t& __rcvr) noexcept;
};

STDEXEC_PRAGMA_POP()

} // namespace exec
