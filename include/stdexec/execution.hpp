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
// IWYU pragma: begin_exports
#include "__detail/__as_awaitable.hpp"
#include "__detail/__associate.hpp"
#include "__detail/__basic_sender.hpp"
#include "__detail/__bulk.hpp"
#include "__detail/__completion_signatures.hpp"
#include "__detail/__connect_awaitable.hpp"
#include "__detail/__continues_on.hpp"
#include "__detail/__counting_scopes.hpp"
#include "__detail/__cpo.hpp"
#include "__detail/__debug.hpp"
#include "__detail/__domain.hpp"
#include "__detail/__ensure_started.hpp"
#include "__detail/__env.hpp"
#include "__detail/__execute.hpp"
#include "__detail/__execution_legacy.hpp"
#include "__detail/__inline_scheduler.hpp"
#include "__detail/__into_variant.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__intrusive_slist.hpp"
#include "__detail/__just.hpp"
#include "__detail/__let.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__on.hpp"
#include "__detail/__operation_states.hpp"
#include "__detail/__read_env.hpp"
#include "__detail/__receiver_adaptor.hpp"
#include "__detail/__receivers.hpp"
#include "__detail/__run_loop.hpp"
#include "__detail/__schedule_from.hpp"
#include "__detail/__schedulers.hpp"
#include "__detail/__scope_concepts.hpp"
#include "__detail/__sender_adaptor_closure.hpp"
#include "__detail/__senders.hpp"
#include "__detail/__spawn.hpp"
#include "__detail/__spawn_future.hpp"
#include "__detail/__split.hpp"
#include "__detail/__start_detached.hpp"
#include "__detail/__starts_on.hpp"
#include "__detail/__stopped_as_error.hpp"
#include "__detail/__stopped_as_optional.hpp"
#include "__detail/__submit.hpp"
#include "__detail/__sync_wait.hpp"
#include "__detail/__task.hpp"
#include "__detail/__task_scheduler.hpp"
#include "__detail/__then.hpp"
#include "__detail/__transfer_just.hpp"
#include "__detail/__transform_completion_signatures.hpp"
#include "__detail/__transform_sender.hpp"
#include "__detail/__type_traits.hpp"
#include "__detail/__unstoppable.hpp"
#include "__detail/__upon_error.hpp"
#include "__detail/__upon_stopped.hpp"
#include "__detail/__utility.hpp"
#include "__detail/__when_all.hpp"
#include "__detail/__with_awaitable_senders.hpp"
#include "__detail/__write_env.hpp"

#include "concepts.hpp"
#include "coroutine.hpp"
#include "functional.hpp"
#include "stop_token.hpp"
// IWYU pragma: end_exports

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <STDEXEC::sender _Ignore, STDEXEC::sender _Sender>
  requires STDEXEC::__ok<STDEXEC::__bad_pipe_sink_t<_Sender>>
auto operator|(_Ignore&&, _Sender&&) noexcept;
