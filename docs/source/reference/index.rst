.. SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

.. _Reference:

Reference
=========

.. toctree::
  :maxdepth: 2

.. _ref-section-concepts:

Concepts
--------

The stdexec API is structured around a small set of foundational concepts.
Most sender adaptors and consumers express their requirements in terms of
these concepts, so understanding them — and which one to reach for in
which situation — pays off across the rest of the reference.

The concepts fall into three layers:

- **Sender side:** :ref:`ref-concept-sender`, :ref:`ref-concept-sender_in`,
  :ref:`ref-concept-sender_to`. A *sender* is the basic unit of
  composition — a value that *describes* (but does not yet execute) an
  async computation.
- **Receiver / operation-state side:**
  :ref:`ref-concept-receiver`, :ref:`ref-concept-receiver_of`,
  :ref:`ref-concept-operation_state`. These describe the *consumer*
  half of a sender/receiver pair — the destination of completion signals
  and the running operation that delivers them.
- **Context side:** :ref:`ref-concept-scheduler`,
  :ref:`ref-concept-scope_token`, :ref:`ref-concept-scope_association`.
  These describe how work is dispatched onto execution resources and how
  its lifetime is tracked.

Sender concepts
~~~~~~~~~~~~~~~

.. _ref-concept-sender:

``sender``
^^^^^^^^^^

.. doxygenconcept:: stdexec::sender

.. doxygenstruct:: stdexec::sender_tag

.. doxygenvariable:: stdexec::enable_sender

.. _ref-concept-sender_in:

``sender_in``
^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::sender_in

.. _ref-concept-sender_to:

``sender_to``
^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::sender_to

Receiver and operation-state concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _ref-concept-receiver:

``receiver``
^^^^^^^^^^^^

.. doxygenconcept:: stdexec::receiver

.. doxygenstruct:: stdexec::receiver_tag

.. _ref-concept-receiver_of:

``receiver_of``
^^^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::receiver_of

.. _ref-concept-operation_state:

``operation_state``
^^^^^^^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::operation_state

.. doxygenstruct:: stdexec::operation_state_tag

Context concepts
~~~~~~~~~~~~~~~~

.. _ref-concept-scheduler:

``scheduler``
^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::scheduler

.. _ref-concept-scope_token:

``scope_token``
^^^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::scope_token

.. _ref-concept-scope_association:

``scope_association``
^^^^^^^^^^^^^^^^^^^^^

.. doxygenconcept:: stdexec::scope_association

.. _ref-section-cpos:

Core Customization Points
-------------------------

The customization points listed here are the *defining operations* of the
sender model. They are what every sender, receiver, and operation state
type must support (each in its own way) to participate in the protocol.
Most user code never calls these directly — sender adaptors and consumers
do — but anyone *writing* a new sender, receiver, or scheduler will
implement one or more of these.

The CPOs fall into three layers:

- **Sender-side:** :ref:`connect <ref-cpo-connect>`,
  :ref:`get_completion_signatures <ref-cpo-get_completion_signatures>`,
  :ref:`get_env <ref-cpo-get_env>`. These describe how a sender exposes
  its computation and attributes to the framework.
- **Operation-state-side:** :ref:`start <ref-cpo-start>`. The trigger
  that turns a connected sender into a running operation.
- **Receiver-side:** :ref:`set_value <ref-cpo-set_value>`,
  :ref:`set_error <ref-cpo-set_error>`,
  :ref:`set_stopped <ref-cpo-set_stopped>`,
  :ref:`get_env <ref-cpo-get_env>`. These describe how an operation
  state delivers a completion to its receiver and queries the receiver's
  environment.

See the :ref:`Developer's Guide <CoreConceptsForDevelopers>` for a
narrative walkthrough of how these fit together when writing a new
sender adaptor.

Sender-side
~~~~~~~~~~~

.. _ref-cpo-connect:

``connect``
^^^^^^^^^^^

.. doxygenstruct:: stdexec::connect_t
   :members:

.. doxygenvariable:: stdexec::connect

.. _ref-cpo-get_completion_signatures:

``get_completion_signatures``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike the other entries in this section, ``get_completion_signatures``
is a *function template* (not a CPO instance), so it has no underlying
struct. The two forms — environment-free and environment-dependent —
share documentation:

.. doxygenfile:: __get_completion_signatures.hpp
   :sections: briefdescription detaileddescription

Operation-state-side
~~~~~~~~~~~~~~~~~~~~

.. _ref-cpo-start:

``start``
^^^^^^^^^

.. doxygenstruct:: stdexec::start_t
   :members:

.. doxygenvariable:: stdexec::start

Receiver-side
~~~~~~~~~~~~~

.. _ref-cpo-set_value:

``set_value``
^^^^^^^^^^^^^

.. doxygenstruct:: stdexec::set_value_t
   :members:

.. doxygenvariable:: stdexec::set_value

.. _ref-cpo-set_error:

``set_error``
^^^^^^^^^^^^^

.. doxygenstruct:: stdexec::set_error_t
   :members:

.. doxygenvariable:: stdexec::set_error

.. _ref-cpo-set_stopped:

``set_stopped``
^^^^^^^^^^^^^^^

.. doxygenstruct:: stdexec::set_stopped_t
   :members:

.. doxygenvariable:: stdexec::set_stopped

.. _ref-cpo-get_env:

``get_env``
^^^^^^^^^^^

.. doxygenstruct:: stdexec::get_env_t
   :members:

.. doxygenvariable:: stdexec::get_env

Sender Factories
----------------

A *sender factory* is an algorithm that produces a sender from non-sender
inputs (values, an error, a scheduler, an environment query). Factories
sit at the *head* of a sender pipeline.

.. _ref-just:

``just`` — produce a sender from values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that synchronously completes with the given values on
the *value* channel. The canonical way to inject literal values into a
sender pipeline. See :ref:`UserGuide_just` for an approachable
introduction with worked examples.

.. doxygenstruct:: stdexec::just_t
   :members:

.. doxygenvariable:: stdexec::just

.. _ref-just_error:

``just_error`` — produce a sender from an error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that synchronously completes with the given error on
the *error* channel. Mostly useful for testing error-handling adaptors.

.. doxygenstruct:: stdexec::just_error_t
   :members:

.. doxygenvariable:: stdexec::just_error

.. _ref-just_stopped:

``just_stopped`` — produce a stopped sender
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that synchronously completes on the *stopped* channel.
Mostly useful for testing cancellation-handling adaptors.

.. doxygenstruct:: stdexec::just_stopped_t
   :members:

.. doxygenvariable:: stdexec::just_stopped

.. _ref-read_env:

``read_env`` — produce a sender from an environment query
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender whose value completion is the result of querying the
connected receiver's environment with a given query CPO. It is the
primitive behind the standard ``get_stop_token()``, ``get_allocator()``,
``get_scheduler()`` helpers. See :ref:`UserGuide_read_env` for an
approachable introduction.

.. doxygenvariable:: stdexec::read_env

.. _ref-schedule:

``schedule`` — produce a sender from a scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that, when connected and started, value-completes
from the context of the given scheduler. It is the bridge between the
scheduler and sender abstractions, and the way to begin a pipeline that
must run on a specific execution context. See :ref:`UserGuide_schedule`
for an approachable introduction.

.. doxygenstruct:: stdexec::schedule_t
   :members:

.. doxygenvariable:: stdexec::schedule

.. _ref-just_from:

``just_from`` (experimental) — like ``just`` but value-producing via a function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. doxygenvariable:: experimental::execution::just_from

.. doxygenvariable:: experimental::execution::just_error_from

.. doxygenvariable:: experimental::execution::just_stopped_from

Sender Adaptors
---------------

TODO: More sender adaptor algorithms

.. _ref-then:

``then`` — apply a function to the value channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transforms a predecessor sender's value completion by invoking a callable
with the values it produces. See :ref:`UserGuide_then` for an approachable
introduction with worked examples; the complete reference (including
completion-signature transformation rules, exception behavior, and the
``operator()`` overloads) follows.

.. doxygenstruct:: stdexec::then_t
   :members:

.. doxygenvariable:: stdexec::then

.. _ref-upon_error:

``upon_error`` — handle the error channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Handles a predecessor sender's error completion by invoking a callable on
the error datum and delivering the result as a *value* completion — the
canonical way to recover from an error and continue the pipeline. See
:ref:`UserGuide_upon_error` for an approachable introduction with worked
examples.

.. doxygenstruct:: stdexec::upon_error_t
   :members:

.. doxygenvariable:: stdexec::upon_error

.. _ref-upon_stopped:

``upon_stopped`` — handle the stopped channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Handles a predecessor sender's stopped completion by invoking a nullary
callable and delivering its return value as a *value* completion — the
canonical way to recover from cancellation. See :ref:`UserGuide_upon_stopped`
for an approachable introduction with worked examples.

.. doxygenstruct:: stdexec::upon_stopped_t
   :members:

.. doxygenvariable:: stdexec::upon_stopped

.. _ref-let_value:

``let_value`` — chain a sender-returning function on the value channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chains a *sender-returning* function onto a predecessor's value completion.
The returned sender is connected and started, and its completions become
the completions of the overall pipeline. This is the way to launch another
asynchronous operation based on a predecessor's values. See
:ref:`UserGuide_let_value` for an approachable introduction with worked
examples.

.. doxygenstruct:: stdexec::let_value_t

.. doxygenvariable:: stdexec::let_value

.. _ref-let_error:

``let_error`` — chain a sender-returning function on the error channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chains a *sender-returning* function onto a predecessor's error completion
— the way to launch another asynchronous operation to recover from an
error.

.. doxygenstruct:: stdexec::let_error_t

.. doxygenvariable:: stdexec::let_error

.. _ref-let_stopped:

``let_stopped`` — chain a sender-returning function on the stopped channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chains a *sender-returning* nullary function onto a predecessor's stopped
completion — the way to launch another asynchronous operation to recover
from cancellation.

.. doxygenstruct:: stdexec::let_stopped_t

.. doxygenvariable:: stdexec::let_stopped

Scheduling adaptors
~~~~~~~~~~~~~~~~~~~

These adaptors move work between execution contexts. ``starts_on`` begins a
sender on a new scheduler; ``continues_on`` transfers execution to a new
scheduler after a sender completes; ``on`` runs work on a different
scheduler and then returns to where it started. See
:ref:`UserGuide_scheduling_adaptors` for a side-by-side comparison.

.. _ref-starts_on:

``starts_on`` — run a sender on a scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that runs an input sender starting on a given
scheduler's execution resource. The completion is delivered on that
same resource (no round-trip back).

.. doxygenstruct:: stdexec::starts_on_t
   :members:

.. doxygenvariable:: stdexec::starts_on

.. _ref-continues_on:

``continues_on`` — transfer to a scheduler after completion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Produces a sender that runs the input sender to completion, then transfers
execution to a given scheduler's resource before delivering the
completion downstream. Anything chained after ``continues_on`` runs on
the new scheduler.

.. doxygenstruct:: stdexec::continues_on_t
   :members:

.. doxygenvariable:: stdexec::continues_on

.. _ref-on:

``on`` — run on a scheduler and return to the original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The "go there, do work, come back" adaptor. Two forms:
``on(sched, sndr)`` runs ``sndr`` on ``sched`` and returns to the start
scheduler; ``on(sndr, sched, closure)`` (and its pipe form
``sndr | on(sched, closure)``) hops to ``sched`` for an inserted
closure then hops back. See :ref:`UserGuide_on` for guidance on when to
reach for which form.

.. doxygenstruct:: stdexec::on_t
   :members:

.. doxygenvariable:: stdexec::on

Composition adaptors
~~~~~~~~~~~~~~~~~~~~

These adaptors combine multiple senders into one. They are the building
blocks for parallel and fan-out patterns.

.. _ref-when_all:

``when_all`` — run senders concurrently and concatenate values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Takes one or more senders and produces a sender that, when started, runs
all of them concurrently and completes when every input has completed.
The resulting value-completion is the *concatenation* of every input's
value datums. If any input fails or is stopped, the others are
cancelled and the result is the first error/stopped completion observed.
Each input must have exactly one value-completion shape; for senders
that can succeed in more than one way, see :ref:`ref-when_all_with_variant`.

.. doxygenstruct:: stdexec::when_all_t
   :members:

.. doxygenvariable:: stdexec::when_all

.. _ref-when_all_with_variant:

``when_all_with_variant`` — like ``when_all`` for multi-completion senders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like ``when_all``, but lifts the "exactly one value-completion per
input" restriction by wrapping each input in
:cpp:member:`stdexec::into_variant`. Each output value-completion
position is a ``std::variant<std::tuple<...>, ...>`` of that input's
possible shapes.

.. doxygenstruct:: stdexec::when_all_with_variant_t
   :members:

.. doxygenvariable:: stdexec::when_all_with_variant

.. _ref-into_variant:

``into_variant`` — collapse multi-completion senders into one
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reshapes a sender that may value-complete in more than one way into a
sender that always value-completes with a single
``std::variant``-of-tuples datum. It is the building block behind
:ref:`when_all_with_variant <ref-when_all_with_variant>` and
:ref:`sync_wait_with_variant <ref-sync_wait_with_variant>`,
and is occasionally useful on its own when a downstream algorithm
requires the single-value-completion form.

.. doxygenstruct:: stdexec::into_variant_t
   :members:

.. doxygenvariable:: stdexec::into_variant

``transfer_when_all`` *(deprecated)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Deprecated
   :class: warning

   This adaptor is not part of the C++26 working draft and is retained
   only for backwards compatibility. Write
   ``when_all(sndrs...) | continues_on(sch)`` instead — the behavior is
   identical.

.. doxygenstruct:: stdexec::transfer_when_all_t
   :members:

.. doxygenvariable:: stdexec::transfer_when_all

``transfer_when_all_with_variant`` *(deprecated)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Deprecated
   :class: warning

   This adaptor is not part of the C++26 working draft and is retained
   only for backwards compatibility. Write
   ``when_all_with_variant(sndrs...) | continues_on(sch)`` instead.

.. doxygenstruct:: stdexec::transfer_when_all_with_variant_t
   :members:

.. doxygenvariable:: stdexec::transfer_when_all_with_variant

Parallel-loop adaptors
~~~~~~~~~~~~~~~~~~~~~~

The ``bulk`` family invokes a callable over an integer index space,
under a given execution policy. They are the parallel-loop primitives
of the sender model — the entry point for GPU/parallel-scheduler
customizations to take over.

.. _ref-bulk:

``bulk`` — apply a function to each index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Invokes ``f(i, vs...)`` for every ``i`` in ``[0, shape)`` under the
given execution policy. Lowers to ``bulk_chunked`` internally so that
domain customizations of ``bulk_chunked`` apply transparently. See
:ref:`UserGuide_bulk` for a worked example and policy discussion.

.. doxygenstruct:: stdexec::bulk_t

.. doxygenvariable:: stdexec::bulk

.. _ref-bulk_chunked:

``bulk_chunked`` — apply a function per chunk of indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Invokes ``f(begin, end, vs...)`` for chunks of the index space.
The implementation may split into any number of chunks (one,
shape, anything between).

.. doxygenstruct:: stdexec::bulk_chunked_t

.. doxygenvariable:: stdexec::bulk_chunked

.. _ref-bulk_unchunked:

``bulk_unchunked`` — apply a function per index, no chunking allowed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like ``bulk`` but forbids the implementation from combining multiple
indices into a single call. Use when per-iteration state (thread-local
accumulators, per-index hardware resources) prevents batching.

.. doxygenstruct:: stdexec::bulk_unchunked_t

.. doxygenvariable:: stdexec::bulk_unchunked

Stopped-channel translator adaptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These adaptors don't change the *behavior* of a pipeline; they
translate one completion channel into another, exposing a friendlier
shape to downstream code.

.. _ref-stopped_as_error:

``stopped_as_error`` — translate stopped into an error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts a ``set_stopped`` completion into a ``set_error`` completion
carrying a caller-supplied error datum. The resulting sender no longer
has a stopped channel.

.. doxygenstruct:: stdexec::stopped_as_error_t
   :members:

.. doxygenvariable:: stdexec::stopped_as_error

.. _ref-stopped_as_optional:

``stopped_as_optional`` — translate stopped into a value-channel ``nullopt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts a ``set_stopped`` completion into a value-channel
``std::optional<T>{std::nullopt}``, wrapping the predecessor's value
in ``std::optional<T>``. Requires the predecessor to have exactly one
value completion with one argument.

.. doxygenstruct:: stdexec::stopped_as_optional_t
   :members:

.. doxygenvariable:: stdexec::stopped_as_optional

Environment adaptors
~~~~~~~~~~~~~~~~~~~~

.. _ref-write_env:

``write_env`` — inject values into the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Augments the environment seen by a predecessor sender with
additional queries. The inverse of :ref:`read_env <ref-read_env>`:
``read_env`` exposes environment values into the value channel,
``write_env`` injects environment values into a child sender's
environment.

.. doxygenvariable:: stdexec::write_env


Sender Consumers
----------------

A *sender consumer* takes a sender, connects it to a receiver, and starts
the resulting operation. Consumers sit at the *tail* of a sender pipeline
— they are the point at which asynchronous work actually runs. They fall
into two broad families:

- **Synchronous waiters** (:ref:`sync_wait <ref-sync_wait>`,
  :ref:`sync_wait_with_variant <ref-sync_wait_with_variant>`) block the
  calling thread until the pipeline completes and return the result.
- **Eager launchers** (:ref:`start_detached <ref-start_detached>`,
  :ref:`spawn <ref-spawn>`, :ref:`spawn_future <ref-spawn_future>`)
  start the pipeline immediately and either discard the result
  (:ref:`spawn <ref-spawn>` and :ref:`start_detached <ref-start_detached>`)
  or expose it as a sender that observes the running operation
  (:ref:`spawn_future <ref-spawn_future>`).

See :ref:`UserGuide_sender_consumers` for a side-by-side comparison and
guidance on which consumer to reach for.

.. _ref-sync_wait:

``sync_wait`` — block until the sender completes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronously waits for a single-value-completion sender to complete on
the calling thread. Returns an engaged ``std::optional<std::tuple<...>>``
on success, an empty optional on stopped, and throws on error.

.. doxygenstruct:: stdexec::sync_wait_t
   :members:

.. doxygenvariable:: stdexec::sync_wait

.. _ref-sync_wait_with_variant:

``sync_wait_with_variant`` — block until a multi-completion sender completes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like ``sync_wait`` but for senders that may complete with more than one
value-completion shape. Returns an engaged
``std::optional<std::variant<std::tuple<...>...>>`` on success.

.. doxygenstruct:: stdexec::sync_wait_with_variant_t
   :members:

.. doxygenvariable:: stdexec::sync_wait_with_variant

.. _ref-start_detached:

``start_detached`` *(extension)* — fire and forget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eagerly starts a sender and discards its result. The operation state is
heap-allocated and cleans itself up on completion. **stdexec extension**
— not part of the C++26 working draft. For the standardized
scope-tracked equivalent, see :ref:`spawn <ref-spawn>`.

.. doxygenstruct:: experimental::execution::start_detached_t
   :members:

.. doxygenvariable:: experimental::execution::start_detached

.. _ref-spawn:

``spawn`` — fire and forget into an async scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eagerly starts a sender and ties its lifetime to a given async scope.
The argument sender must not be able to complete with ``set_error``.
The standardized way to launch fire-and-forget work whose lifetime
should be tracked.

.. doxygenstruct:: stdexec::spawn_t
   :members:

.. doxygenvariable:: stdexec::spawn

.. _ref-spawn_future:

``spawn_future`` — fire and observe via a sender
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like ``spawn`` but additionally returns a sender that completes when the
spawned operation completes. The returned sender is a *one-shot
observer* of work that is already running, not a re-runnable handle.

.. doxygenstruct:: stdexec::spawn_future_t
   :members:

.. doxygenvariable:: stdexec::spawn_future

Utilities
---------

TODO: Add utilities section
