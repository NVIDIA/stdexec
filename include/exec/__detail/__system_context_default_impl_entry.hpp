/*
 * Copyright (c) 2024 Lucian Radu Teodorescu
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

// Assumes __EXEC__SYSTEM_CONTEXT__INLINE is defined

#include "__system_context_default_impl.hpp"
#include "__weak_attribute.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wattributes") // warning: inline function '[...]' declared weak

/// Gets the default system context implementation.
extern "C" __EXEC__SYSTEM_CONTEXT__INLINE __EXEC_WEAK_ATTRIBUTE __exec_system_context_interface*
  __get_exec_system_context_impl() {
  return exec::__system_context_default_impl::__instance_holder::__singleton()
    .__get_current_instance();
}

/// Sets the default system context implementation.
extern "C" __EXEC__SYSTEM_CONTEXT__INLINE __EXEC_WEAK_ATTRIBUTE void
  __set_exec_system_context_impl(__exec_system_context_interface* __instance) {
  return exec::__system_context_default_impl::__instance_holder::__singleton()
    .__set_current_instance(__instance);
}

STDEXEC_PRAGMA_POP()
