/*
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

#include "__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__detail/__as_awaitable.hpp"                    // IWYU pragma: export
#include "__detail/__basic_sender.hpp"                    // IWYU pragma: export
#include "__detail/__bulk.hpp"                            // IWYU pragma: export
#include "__detail/__completion_signatures.hpp"           // IWYU pragma: export
#include "__detail/__connect_awaitable.hpp"               // IWYU pragma: export
#include "__detail/__continues_on.hpp"                    // IWYU pragma: export
#include "__detail/__cpo.hpp"                             // IWYU pragma: export
#include "__detail/__debug.hpp"                           // IWYU pragma: export
#include "__detail/__domain.hpp"                          // IWYU pragma: export
#include "__detail/__ensure_started.hpp"                  // IWYU pragma: export
#include "__detail/__env.hpp"                             // IWYU pragma: export
#include "__detail/__execute.hpp"                         // IWYU pragma: export
#include "__detail/__execution_legacy.hpp"                // IWYU pragma: export
#include "__detail/__inline_scheduler.hpp"                // IWYU pragma: export
#include "__detail/__into_variant.hpp"                    // IWYU pragma: export
#include "__detail/__intrusive_ptr.hpp"                   // IWYU pragma: export
#include "__detail/__intrusive_slist.hpp"                 // IWYU pragma: export
#include "__detail/__just.hpp"                            // IWYU pragma: export
#include "__detail/__let.hpp"                             // IWYU pragma: export
#include "__detail/__meta.hpp"                            // IWYU pragma: export
#include "__detail/__on.hpp"                              // IWYU pragma: export
#include "__detail/__operation_states.hpp"                // IWYU pragma: export
#include "__detail/__read_env.hpp"                        // IWYU pragma: export
#include "__detail/__receivers.hpp"                       // IWYU pragma: export
#include "__detail/__receiver_adaptor.hpp"                // IWYU pragma: export
#include "__detail/__receiver_ref.hpp"                    // IWYU pragma: export
#include "__detail/__run_loop.hpp"                        // IWYU pragma: export
#include "__detail/__schedule_from.hpp"                   // IWYU pragma: export
#include "__detail/__schedulers.hpp"                      // IWYU pragma: export
#include "__detail/__senders.hpp"                         // IWYU pragma: export
#include "__detail/__sender_adaptor_closure.hpp"          // IWYU pragma: export
#include "__detail/__split.hpp"                           // IWYU pragma: export
#include "__detail/__start_detached.hpp"                  // IWYU pragma: export
#include "__detail/__starts_on.hpp"                       // IWYU pragma: export
#include "__detail/__stopped_as_error.hpp"                // IWYU pragma: export
#include "__detail/__stopped_as_optional.hpp"             // IWYU pragma: export
#include "__detail/__submit.hpp"                          // IWYU pragma: export
#include "__detail/__sync_wait.hpp"                       // IWYU pragma: export
#include "__detail/__then.hpp"                            // IWYU pragma: export
#include "__detail/__transfer_just.hpp"                   // IWYU pragma: export
#include "__detail/__transform_sender.hpp"                // IWYU pragma: export
#include "__detail/__transform_completion_signatures.hpp" // IWYU pragma: export
#include "__detail/__type_traits.hpp"                     // IWYU pragma: export
#include "__detail/__upon_error.hpp"                      // IWYU pragma: export
#include "__detail/__upon_stopped.hpp"                    // IWYU pragma: export
#include "__detail/__unstoppable.hpp"                     // IWYU pragma: export
#include "__detail/__utility.hpp"                         // IWYU pragma: export
#include "__detail/__when_all.hpp"                        // IWYU pragma: export
#include "__detail/__with_awaitable_senders.hpp"          // IWYU pragma: export
#include "__detail/__write_env.hpp"                       // IWYU pragma: export

#include "functional.hpp" // IWYU pragma: export
#include "concepts.hpp"   // IWYU pragma: export
#include "coroutine.hpp"  // IWYU pragma: export
#include "stop_token.hpp" // IWYU pragma: export

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Ignore, stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(_Ignore&&, _Sender&&) noexcept;
