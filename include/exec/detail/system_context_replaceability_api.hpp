/*
 * Copyright (c) 2025 Lucian Radu Teodorescu, Lewis Baker
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

#ifndef STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H
#define STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H

#include "../../stdexec/__detail/__execution_fwd.hpp"
#include "../../stdexec/__detail/__parallel_scheduler_replacement_api.hpp"

#include <memory>

namespace experimental::execution
{
  namespace [[deprecated("Use the " STDEXEC_PP_STRINGIZE(STDEXEC)  //
                         "::parallel_scheduler_replacement namespace "
                         "instead.")]] system_context_replaceability
  {
    using STDEXEC::parallel_scheduler_replacement::__parallel_scheduler_backend_factory_t;

    /// Interface for the parallel scheduler backend.
    using parallel_scheduler_backend                                                             //
      [[deprecated("Use " STDEXEC_PP_STRINGIZE(STDEXEC)                                          //
                   "::parallel_scheduler_replacement::parallel_scheduler_backend instead.")]] =  //
      STDEXEC::parallel_scheduler_replacement::parallel_scheduler_backend;

    /// Get the backend for the parallel scheduler.
    /// Users might replace this function.
    [[deprecated("Use " STDEXEC_PP_STRINGIZE(STDEXEC) "::parallel_scheduler_replacement::query_"
                                                      "parallel_scheduler_backend "
                                                      "instead.")]]
    inline auto query_parallel_scheduler_backend()
      -> std::shared_ptr<STDEXEC::parallel_scheduler_replacement::parallel_scheduler_backend>
    {
      return STDEXEC::parallel_scheduler_replacement::query_parallel_scheduler_backend();
    }

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wdeprecated-declarations")
    STDEXEC_PRAGMA_IGNORE_MSVC(4996)  // warning C4996: 'function': was declared deprecated
    STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity)
    STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity_with_custom_message)
    /// Set a factory for the parallel scheduler backend.
    /// Can be used to replace the parallel scheduler at runtime.
    /// Out of spec.
    [[deprecated("Use " STDEXEC_PP_STRINGIZE(STDEXEC) "::parallel_scheduler_replacement::set_"
                                                      "parallel_scheduler_backend "
                                                      "instead.")]]
    inline auto set_parallel_scheduler_backend(__parallel_scheduler_backend_factory_t __new_factory)
      -> __parallel_scheduler_backend_factory_t
    {
      return STDEXEC::parallel_scheduler_replacement::set_parallel_scheduler_backend(__new_factory);
    }
    STDEXEC_PRAGMA_POP()

    /// Interface for completing a sender operation. Backend will call frontend though this interface
    /// for completing the `schedule` and `schedule_bulk` operations.
    using receiver [[deprecated("Use " STDEXEC_PP_STRINGIZE(STDEXEC)                              //
                                "::parallel_scheduler_replacement::receiver_proxy instead.")]] =  //
      STDEXEC::parallel_scheduler_replacement::receiver_proxy;

    /// Receiver for bulk scheduling operations.
    using bulk_item_receiver [[deprecated("Use " STDEXEC_PP_STRINGIZE(STDEXEC)  //
                                          "::parallel_scheduler_replacement::bulk_item_receiver_"
                                          "proxy instead.")]] =  //
      STDEXEC::parallel_scheduler_replacement::bulk_item_receiver_proxy;
  }  // namespace system_context_replaceability
}  // namespace experimental::execution

namespace exec = experimental::execution;

#endif
