.. =============================================================================
..  Copyright 2025 NVIDIA Corporation
.. 
..  Licensed under the Apache License, Version 2.0 (the "License");
..  you may not use this file except in compliance with the License.
..  You may obtain a copy of the License at
.. 
..      http://www.apache.org/licenses/LICENSE-2.0
.. 
..  Unless required by applicable law or agreed to in writing, software
..  distributed under the License is distributed on an "AS IS" BASIS,
..  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
..  See the License for the specific language governing permissions and
..  limitations under the License.
.. =============================================================================

User's Guide
============

This section will eventually contain a user's guide for ``stdexec`` that describes what the library
is, the high-level concepts on which it is based, and how to use it.

TODO

.. _CoreConceptsForUsers:

🧱 Core Concepts for Users
--------------------------

From the perspective of the user, the core concepts of the Sender model are the
**sender** abstraction, the **scheduler** abstraction, and **sender algorithms**.

1. Scheduler
^^^^^^^^^^^^

A **scheduler** is an object that provides a way to schedule work. They are lightweight
handles to what is often a heavy-weight and immovable **execution context**. Execution
contexts are where work actually happens, and they can be anything that can execute code.
Examples of execution contexts:

- Thread pools.
- Event loops.
- GPUs.
- An I/O subsystem.
- Any other execution model.

.. code-block:: cpp

    auto sched = stdexec::get_parallel_scheduler(); // Obtain the default system scheduler
    auto sndr  = stdexec::schedule(sched);          // Create a sender from the scheduler

The sender you get back from ``stdexec::schedule`` is a factory for operations that,
when started, will immediately call ``set_value()`` on the receiver the operation was
constructed with. And crucially, it does so from the context of the scheduler. Work
is executed on a context by chaining continuations to one of these senders, and passing
it to one of the algorithms that starts work, like ``stdexec::sync_wait``.

2. Sender
^^^^^^^^^

A **sender** is an object that describes an asynchronous computation that may happen
later. It can do nothing on its own, but when connected to a receiver, it returns an
**operation state** that can start the work described by the sender.

``stdexec`` provides a set of generic **sender algorithms** that can be used to chain
operations together. There are also algorithms, like ``stdexec::sync_wait``, that can
be used to launch the sender. The sender algorithms take care of connecting the sender
to a receiver and managing the lifetime the operation state.

- Produces values (or errors) asynchronously.

- Can be requested to stop early.

- Supports composition.

- Is lazy (does nothing until connected and started).

.. code-block:: cpp

    auto sndr = stdexec::just(42);                    // Sender that yields 42 immediately
    auto [result] = stdexec::sync_wait(sndr).value(); // Start the work & wait for the result
    assert(result.value() == 42);

🧮 Composition via Algorithms
-----------------------------

One benefit of the lazy evaluation of senders is that it makes it possible to create
reusable algorithms that can be composed together. ``stdexec`` provides a rich set of
algorithms that can be used to build complex asynchronous workflows.

A **sender factory algorithm** creates a sender that can be used to start an operation.
Below are some of the key sender factory algorithms provided by ``stdexec``:

.. list-table:: Sender Factory Algorithms

  * - **CPO**
    - **Description**
  * - :cpp:member:`stdexec::schedule`
    - Obtains a sender from a scheduler.
  * - :cpp:member:`stdexec::just`
    - Creates a sender that will immediately complete with a set of values.
  * - :cpp:member:`stdexec::read_env`
    - Reads a value from the receiver's environment and completes with it.

A **sender adaptor algorithm** takes an existing sender (or several senders) and
transforms it into a new sender with additional behavior. Below are some key sender
adaptor algorithms. Check the :ref:`Reference` section for additional algorithms.

.. list-table:: Sender Adaptor Algorithms
  :class: tight-table

  * - **CPO**
    - **Description**
  * - :cpp:member:`stdexec::then`
    - Applies a function to the value from a sender.
  * - :cpp:member:`stdexec::starts_on`
    - Executes an async operation on the specified scheduler.
  * - :cpp:member:`stdexec::continues_on`
    - Executes an async operation on the current scheduler and then transfers
      execution to the specified scheduler.
  * - :cpp:member:`stdexec::on`
    - Executes an async operation on a different scheduler and then transitions
      back to the original scheduler.
  * - :cpp:member:`stdexec::when_all`
    - Combines multiple senders, making it possible to execute them in parallel.
  * - :cpp:member:`stdexec::let_value`
    - Executes an async operation dynamically based on the results of a specified
      sender.
  * - :cpp:member:`stdexec::write_env`
    - Writes a value to the receiver's environment, allowing it to be used by
      child operations.

A **sender consumer algorithm** takes a sender connects it to a receiver and starts the
resulting operation. Here are some key sender consumer algorithms:

.. list-table:: Sender Consumer Algorithms

  * - **CPO**
    - **Description**
  * - :cpp:member:`stdexec::sync_wait`
    - Blocks the calling thread until the sender completes and returns the result.
  * - :cpp:member:`exec::start_detached`
    - Starts the operation without waiting for it to complete.

Here is an example of using sender algorithms to create a simple async pipeline:

.. code-block:: cpp

                    // Create a sender that produces a value:
    auto pipeline = stdexec::just(42)
                    // Transform the value using `then` on the specified scheduler:
                  | stdexec::on(some_scheduler, stdexec::then([](int i) { return i * 2; }))
                    // Further transform the value using `then` on the starting scheduler:
                  | stdexec::then([](int i) { return i + 1; });
                    // Finally, wait for the result:
    auto [result] = stdexec::sync_wait(std::move(pipeline)).value();


📖 Algorithms in Depth
----------------------

This section gives a more approachable, example-driven introduction to the
individual sender algorithms. For exhaustive technical reference — including
template parameters, completion-signature transformation rules, and exception
behavior — see the :ref:`Reference` section.

Sender factories
~~~~~~~~~~~~~~~~

Factories produce a sender from non-sender inputs. They sit at the *head*
of a pipeline.

.. _UserGuide_just:

``just`` — inject literal values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::just` is the simplest sender factory. You give it
zero or more values; you get back a sender that, when started, immediately
delivers those values to its receiver as a value completion. No context
transition, no asynchrony — just a synchronous handoff of values into the
sender world.

.. code-block:: cpp

    auto sndr = stdexec::just(21)
              | stdexec::then([](int x) { return x * 2; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

``just`` can take any number of values, including zero:

.. code-block:: cpp

    auto s0 = stdexec::just();              // value-completes with no datums
    auto s2 = stdexec::just(1, "hello");    // value-completes with int, string

Use ``just`` whenever you need to start a pipeline with a fixed value, or
to feed test data into an algorithm during unit tests.

.. _UserGuide_just_error:

``just_error`` — inject a literal error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::just_error` is to the error channel what ``just``
is to the value channel: it produces a sender that immediately completes
with the given error.

.. code-block:: cpp

    auto sndr = stdexec::just_error(std::error_code{ENOENT, std::system_category()})
              | stdexec::upon_error([](std::error_code) { return -1; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == -1

This is mostly useful in tests, where you want to drive an error-handling
adaptor (:cpp:member:`stdexec::upon_error`, :cpp:member:`stdexec::let_error`)
without having to construct a sender that actually fails.

.. _UserGuide_just_stopped:

``just_stopped`` — inject a literal cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::just_stopped` produces a sender that immediately
completes via ``set_stopped`` (no datums — the stopped channel carries
none).

.. code-block:: cpp

    auto sndr = stdexec::just_stopped()
              | stdexec::upon_stopped([] { return 42; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

Useful, like ``just_error``, primarily for testing cancellation-handling
adaptors.

.. _UserGuide_read_env:

``read_env`` — read a value from the receiver's environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::read_env` lets a pipeline inspect properties of its
*receiver* — the receiver's stop token, allocator, preferred scheduler,
and anything else the environment exposes. You give it a *query CPO*
(``stdexec::get_stop_token``, ``stdexec::get_scheduler``, …) and get
back a sender that, when started, evaluates that query against the
connected receiver's environment and delivers the result as a value.

.. code-block:: cpp

    auto sndr =
      stdexec::read_env(stdexec::get_stop_token)
      | stdexec::then([](auto tok) { return tok.stop_requested(); });

The standard helpers ``stdexec::get_stop_token()``,
``stdexec::get_scheduler()``, ``stdexec::get_allocator()``, and
``stdexec::get_delegation_scheduler()`` are all defined as one-line
calls to ``read_env`` with the corresponding query.

**When *not* to use** ``read_env`` **:**
If you only want to *use* the stop token / allocator / scheduler in your
own algorithm, you usually want the helper (e.g. ``get_stop_token()``)
rather than wiring ``read_env`` directly — the helper is the same
thing with a shorter name.

.. _UserGuide_schedule:

``schedule`` — start a pipeline on a scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::schedule` is how you say "begin work on *this*
execution context". You give it a *scheduler* — a lightweight handle to
a thread pool, GPU stream, event loop, etc. — and get back a sender that,
when started, value-completes (with no datums) *from the context of that
scheduler*. Anything you chain after it runs on that context.

.. code-block:: cpp

    auto sched = stdexec::get_parallel_scheduler();

    auto sndr =
      stdexec::schedule(sched)               // hop onto sched
      | stdexec::then([] { return 42; });    // ... and compute on it

    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42, computed on the parallel scheduler

The schedule-sender carries no value datum — the *point* of ``schedule``
is the context transition, not the value. Use :cpp:member:`stdexec::then`
or :cpp:member:`stdexec::let_value` to produce the actual work.

**Use** ``schedule`` **or** :cpp:member:`stdexec::starts_on` **or**
:cpp:member:`stdexec::continues_on` **?**

- ``schedule(sched)`` is the *primitive*: it gives you a fresh sender on
  ``sched``. Use it when you're starting a new pipeline.
- ``starts_on(sched, sndr)`` runs ``sndr`` starting on ``sched``. It is
  shorthand for ``schedule(sched) | let_value([&] { return sndr; })`` (or
  equivalent).
- ``continues_on(sndr, sched)`` runs ``sndr`` to completion, then
  transfers execution to ``sched`` for whatever follows. Use this to
  *change contexts* mid-pipeline.

Sender adaptors
~~~~~~~~~~~~~~~

Adaptors take an existing sender and produce a new sender with additional
behavior. They sit in the *middle* of a pipeline.

.. _UserGuide_then:

``then`` — transform a value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::then` is the asynchronous counterpart to "apply a
function to a result". You give it a predecessor sender and a callable; you
get back a new sender that, when its predecessor completes with values,
invokes the callable on those values and forwards the *return value*
downstream. If the predecessor fails or is cancelled, the callable is never
invoked and the failure or cancellation flows through unchanged.

The simplest possible example:

.. code-block:: cpp

    auto sndr = stdexec::just(21)
              | stdexec::then([](int x) { return x * 2; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

``then`` can be called in two equivalent ways:

.. code-block:: cpp

    // Direct call form:
    auto s1 = stdexec::then(sndr, f);

    // Pipe (sender-adaptor-closure) form — usually preferred in chains:
    auto s2 = sndr | stdexec::then(f);

Chaining several transformations is the bread-and-butter use of ``then``:

.. code-block:: cpp

    auto pipeline = stdexec::just(std::string{"hello"})
                  | stdexec::then([](std::string s) { return s + ", world"; })
                  | stdexec::then([](std::string s) { return s.size(); });
    auto [n] = stdexec::sync_wait(std::move(pipeline)).value();
    // n == 12

A function that returns ``void`` is allowed; the resulting sender completes
with a *value completion with no datums*. This is useful when you want to
perform a side effect mid-pipeline but have nothing to forward downstream:

.. code-block:: cpp

    auto pipeline = stdexec::just(42)
                  | stdexec::then([](int x) { std::println("got {}", x); })
                  | stdexec::then([]      { return "done"; });

**What happens on error?**
If the predecessor sender completes with an error, ``then`` forwards the
error and does not invoke your callable. If your callable itself *throws*,
the exception is caught and delivered through the error channel as a
``std::exception_ptr``. To handle the error in-pipeline, follow up with
:cpp:member:`stdexec::upon_error` or :cpp:member:`stdexec::let_error`.

**What happens on cancellation?**
If the predecessor sender completes via ``set_stopped``, ``then`` forwards
the stopped completion and does not invoke your callable. ``then`` itself
never consults the receiver's stop token.

**When *not* to use** ``then`` **:**
If the function you want to apply *itself* returns a sender (i.e. it starts
another asynchronous operation), reach for :cpp:member:`stdexec::let_value`
instead. ``then`` would forward the returned sender as a *value* — almost
certainly not what you want.

.. code-block:: cpp

    // Wrong: the resulting value type is a sender, not the eventual int.
    auto bad  = stdexec::just(7) | stdexec::then(fetch_async);
    // Right: let_value chains the returned sender into the pipeline.
    auto good = stdexec::just(7) | stdexec::let_value(fetch_async);

.. _UserGuide_upon_error:

``upon_error`` — recover from an error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::upon_error` is to the *error channel* what
:cpp:member:`stdexec::then` is to the *value channel*. You give it a
predecessor sender and a callable; if the predecessor completes with an
error, the callable is invoked with the error datum and its return value is
delivered downstream as a regular *value* completion. If the predecessor
succeeds (or is cancelled), ``upon_error`` is a no-op — your callable is
never invoked and the completion is forwarded unchanged.

.. code-block:: cpp

    auto sndr = stdexec::just_error(std::error_code{ENOENT, std::system_category()})
              | stdexec::upon_error([](std::error_code) { return -1; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == -1

The error channel of the input sender is *consumed* — the resulting sender
will never complete via ``set_error`` (unless the callable itself throws,
in which case the exception is rethrown via ``set_error(exception_ptr)``).

**What happens on success or cancellation?**
The corresponding completion is forwarded unchanged; the callable is never
invoked.

**When *not* to use** ``upon_error`` **:**
If your recovery step *itself* needs to perform another async operation
(e.g. retry against a different server), reach for
:cpp:member:`stdexec::let_error` instead — it expects a callable that
returns a sender.

.. _UserGuide_upon_stopped:

``upon_stopped`` — recover from cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::upon_stopped` handles the *stopped* completion. You
give it a predecessor sender and a *nullary* callable; if the predecessor
is cancelled, the callable is invoked with no arguments and its return
value is delivered downstream as a value completion. Successful values and
errors are forwarded unchanged.

.. code-block:: cpp

    auto sndr = stdexec::just_stopped()
              | stdexec::upon_stopped([] { return 42; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

Like :cpp:member:`stdexec::upon_error`, ``upon_stopped`` *consumes* its
channel — the resulting sender will not complete via ``set_stopped``.

**When *not* to use** ``upon_stopped`` **:**
If your fallback step is itself asynchronous, reach for
:cpp:member:`stdexec::let_stopped` — it expects a callable that returns a
sender.

.. _UserGuide_let_value:

``let_value`` — chain another async operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::let_value` is the way to launch *another* async
operation based on the values from a predecessor. Where ``then`` takes a
function returning a *value*, ``let_value`` takes a function returning a
*sender* — the returned sender is then run as part of the pipeline.

.. code-block:: cpp

    auto fetch_async = [](int id) {
      return stdexec::just(id * 10);  // pretend this is a non-trivial async op
    };

    auto sndr = stdexec::just(7)
              | stdexec::let_value(fetch_async);
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 70

The completion signatures of the overall pipeline are the *union* of the
signatures of every sender the callable can return. So a callable that
sometimes returns a sender completing with ``int`` and sometimes with
``string`` gives a pipeline that may complete with either.

**Use** ``then`` **or** ``let_value`` **?**
Use ``then`` when the function returns a *value*. Use ``let_value`` when
the function returns a *sender*. Passing a sender-returning function to
``then`` is almost always a bug — the resulting pipeline forwards the
sender as a value rather than running it.

**Use** ``let_value`` **or coroutines?**
Either works; ``let_value`` is the explicit, sender-graph form, while
``co_await`` inside a ``stdexec::task`` reads more sequentially. Mix them
freely.

.. _UserGuide_let_error:

``let_error`` — retry asynchronously after an error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::let_error` is to :cpp:member:`stdexec::upon_error`
what :cpp:member:`stdexec::let_value` is to :cpp:member:`stdexec::then`: it
takes a callable that returns a *sender* instead of a value, so the
recovery step can itself be asynchronous (a retry, a fallback fetch, etc.).

.. code-block:: cpp

    auto retry_async = [](std::error_code) { return stdexec::just(7); };

    auto sndr = stdexec::just_error(std::error_code{ENOENT, std::system_category()})
              | stdexec::let_error(retry_async);
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 7

.. _UserGuide_let_stopped:

``let_stopped`` — fall back asynchronously after cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::let_stopped` is to :cpp:member:`stdexec::upon_stopped`
what :cpp:member:`stdexec::let_value` is to :cpp:member:`stdexec::then`: it
takes a *nullary* callable that returns a *sender*, so the cancellation
fallback can itself be asynchronous.

.. code-block:: cpp

    auto fallback_async = [] { return stdexec::just(42); };

    auto sndr = stdexec::just_stopped()
              | stdexec::let_stopped(fallback_async);
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

Scheduling adaptors
~~~~~~~~~~~~~~~~~~~

Once you have a scheduler, you need to *move work onto it* — either to
begin a pipeline on a particular execution context, transfer between
contexts mid-pipeline, or take a brief detour. stdexec offers three
adaptors for this: ``starts_on``, ``continues_on``, and ``on``. They
look superficially similar; the table below disambiguates them.

.. _UserGuide_scheduling_adaptors:

``starts_on`` vs. ``continues_on`` vs. ``on``: which one?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
  :header-rows: 1
  :widths: 24 28 28 30

  * - Adaptor
    - Form
    - Where the work runs
    - Where the completion is delivered
  * - ``schedule(sched)``
    - factory
    - on ``sched``
    - on ``sched``
  * - ``starts_on(sched, sndr)``
    - factory-ish
    - on ``sched``
    - on ``sched``
  * - ``continues_on(sndr, sched)``
    - adaptor (pipeable)
    - on ``sndr``'s scheduler
    - on ``sched``
  * - ``on(sched, sndr)``
    - adaptor
    - on ``sched``
    - on the *start* scheduler (round-trip)
  * - ``on(sndr, sched, closure)``
    - adaptor (pipeable)
    - ``sndr`` in place; ``closure`` on ``sched``
    - on ``sndr``'s original completion scheduler (round-trip)

In short:

- Reach for **``starts_on``** when you want the whole pipeline (from
  some point onward) to run on a specific scheduler and stay there.
- Reach for **``continues_on``** when you want to switch contexts at a
  specific point: "produce on the I/O thread, but compute the result on
  the worker pool."
- Reach for **``on``** when you want a *side trip* to another
  scheduler — do some work there, then come back to where you started.

.. _UserGuide_starts_on:

``starts_on`` — begin work on a scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::starts_on` takes a scheduler and a sender, and
produces a sender that runs the given sender starting on the scheduler's
context. The completion is delivered on that *same* context — there is
no transfer back to the caller's scheduler. Unlike most adaptors,
``starts_on`` has no pipe form; the scheduler always comes first.

.. code-block:: cpp

    auto sched = stdexec::get_parallel_scheduler();

    auto sndr =
      stdexec::starts_on(sched,
        stdexec::just(21)
        | stdexec::then([](int x) { return x * 2; }));
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42, computed on `sched`

Equivalently — and this is the spec's defining identity:

.. code-block:: cpp

    // starts_on(sch, sndr) is semantically equivalent to:
    stdexec::schedule(sch) | stdexec::let_value([sndr]() mutable {
      return std::move(sndr);
    });

stdexec's implementation differs internally (for GPU efficiency), but
the observable semantics match.

.. _UserGuide_continues_on:

``continues_on`` — transfer contexts mid-pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::continues_on` takes a sender and a scheduler, runs
the sender to completion, then transfers execution to the scheduler
before forwarding the completion downstream. It's the canonical way to
*hand off* between execution contexts.

.. code-block:: cpp

    auto io_sched  = stdexec::get_parallel_scheduler();  // pretend: I/O
    auto cpu_sched = stdexec::get_parallel_scheduler();  // pretend: compute

    auto sndr =
      stdexec::starts_on(io_sched, stdexec::just(42))  // produce on io_sched
      | stdexec::continues_on(cpu_sched)               // hop to cpu_sched
      | stdexec::then([](int x) { return x * 2; });    // then() runs on cpu_sched
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 84

``continues_on`` does *not* alter the values, errors, or stopped status
of its predecessor — it only changes the execution context they're
delivered on. If you want to also transform the value, chain a
:cpp:member:`stdexec::then` after.

**When *not* to use** ``continues_on`` **:**
If you only want to *temporarily* run on a different scheduler and then
come back, use :cpp:member:`stdexec::on` (Form 2) — it round-trips,
``continues_on`` doesn't.

.. _UserGuide_on:

``on`` — take a side trip to another scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::on` is the round-trip scheduling adaptor: it runs
work on another scheduler and then *transfers execution back* to the
scheduler that started the operation. There are two forms.

**Form 1 — ``on(sched, sndr)``** runs the entirety of ``sndr`` on
``sched`` and then returns to the start scheduler:

.. code-block:: cpp

    auto sched = stdexec::get_parallel_scheduler();
    auto sndr  = stdexec::on(sched, stdexec::just(21)
                                  | stdexec::then([](int x){ return x*2; }));

This differs from ``starts_on(sched, ...)`` in *exactly one* way: after
``sndr`` completes, ``on`` transfers back to wherever the operation
originated; ``starts_on`` stays put. Use ``on`` when downstream code
needs to run on the caller's scheduler again.

**Form 2 — ``on(sndr, sched, closure)``** (and the pipe form
``sndr | on(sched, closure)``) is the "side trip" pattern. The
predecessor runs on its own scheduler; we hop to ``sched`` for the
closure; we hop back when the closure completes:

.. code-block:: cpp

    auto gpu = stdexec::get_parallel_scheduler();  // pretend: GPU

    auto sndr =
      stdexec::just(21)
      | stdexec::on(gpu, stdexec::then([](int x) { return x * 2; }));
    //  ^^^^^^^^^^^^^^^   the then() inside runs on `gpu`, but
    //                    sync_wait() below sees the result on its
    //                    own context — we round-tripped.
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42

Use Form 2 when a small, well-defined chunk of your pipeline needs a
different scheduler (a compute-bound transform, a GPU kernel, a blocking
syscall hidden in a thread pool) and the rest should stay where it is.

**Picking a form.**
If you're starting a fresh pipeline and want to *stay* on a scheduler,
use :cpp:member:`stdexec::starts_on`. If you want to hand off
permanently to a new scheduler, use :cpp:member:`stdexec::continues_on`.
If you want a side trip and then back, use ``on``.

Composition adaptors
~~~~~~~~~~~~~~~~~~~~

So far every adaptor we've seen takes one sender and produces another.
Composition adaptors take *many* senders and combine them into one —
they're how you express parallel and fan-out patterns.

.. _UserGuide_when_all:

``when_all`` — run senders concurrently and gather their values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::when_all` is the parallel-composition primitive
of the sender model. You give it one or more senders; you get back a
single sender that, when started, runs all of them concurrently. When
every input has completed, ``when_all``'s sender value-completes with a
tuple that is the *concatenation* of all the inputs' value datums.

.. code-block:: cpp

    auto sndr = stdexec::when_all(
      stdexec::just(1),
      stdexec::just(2.5),
      stdexec::just(std::string{"x"}));
    auto [i, d, s] = stdexec::sync_wait(std::move(sndr)).value();
    // i == 1, d == 2.5, s == "x"

Two key things to internalize:

1. **Lazy, not eager.** Like every other adaptor, ``when_all`` does
   *nothing* until its result sender is connected and started. The
   inputs aren't running yet just because you named them in a call to
   ``when_all`` — they're stored, and they all start the moment the
   outer pipeline starts.

2. **"Concurrently" means "not sequenced".** The inputs are started in
   a fold over the pack — they're not awaited in order. Whether they
   actually execute in parallel depends on the schedulers they're
   attached to. To get true parallelism, chain each branch through its
   own :cpp:member:`stdexec::starts_on`:

   .. code-block:: cpp

       auto cpu = stdexec::get_parallel_scheduler();

       auto sndr = stdexec::when_all(
         stdexec::starts_on(cpu, sndr_a),
         stdexec::starts_on(cpu, sndr_b),
         stdexec::starts_on(cpu, sndr_c));

   Without that, all the branches just run synchronously inside the
   caller's :cpp:member:`stdexec::start` (still useful for type-level
   composition, but not actually parallel).

**Fail-fast semantics.**
If any one input fails (``set_error``) or is stopped (``set_stopped``),
``when_all`` *requests stop* on all the others via an internal stop
source and completes with that error/stopped completion. The first one
observed wins; subsequent failures are discarded. This makes
``when_all`` naturally short-circuiting on errors — siblings get a
chance to wind down promptly instead of running to completion.

**Single value-completion per input.**
``when_all`` requires that each input sender have *exactly one*
``set_value_t`` completion shape — otherwise the output's value type
would explode into all possible concatenations. If you have inputs with
multiple shapes, use :cpp:member:`stdexec::when_all_with_variant`,
which wraps each in a ``std::variant`` first.

**``when_all`` vs.** :cpp:member:`stdexec::spawn_future` **:**
Both can express "run N things concurrently and collect their results,"
but they differ on *when* the work starts. ``when_all`` is lazy: the
work starts when the composed sender is started. ``spawn_future`` is
eager: the work starts the moment you call it, and you observe results
later through the returned sender. Use ``when_all`` for composition
inside a pipeline; use ``spawn_future`` when you want to overlap async
work with synchronous code or to start work before you know how many
results you'll need to collect.

.. _UserGuide_when_all_with_variant:

``when_all_with_variant`` — like ``when_all`` for multi-shape inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::when_all_with_variant` is the multi-completion
sibling of ``when_all``. It wraps each input in
:cpp:member:`stdexec::into_variant`, so an input that may
value-complete with either ``int`` or ``std::string`` is collapsed into
a single ``std::variant<std::tuple<int>, std::tuple<std::string>>``
before being passed to the ordinary ``when_all`` machinery.

.. code-block:: cpp

    // sndr_a value-completes with either set_value_t(int) or set_value_t(std::string);
    // sndr_b value-completes with set_value_t(float).
    auto sndr = stdexec::when_all_with_variant(sndr_a, sndr_b);
    auto [va, vb] = stdexec::sync_wait(std::move(sndr)).value();
    //   va: std::variant<std::tuple<int>, std::tuple<std::string>>
    //   vb: std::variant<std::tuple<float>>

If every input has a single value-completion shape, prefer plain
``when_all`` — it produces friendlier ``std::tuple`` values directly.

.. _UserGuide_into_variant:

``into_variant`` — collapse multi-completion senders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::into_variant` reshapes a sender that can
value-complete in more than one way into a sender that always
value-completes with a single ``std::variant``-of-tuples datum. It is
the primitive behind :cpp:member:`stdexec::when_all_with_variant` and
:cpp:member:`stdexec::sync_wait_with_variant`, but you can use it
directly whenever a downstream algorithm wants the single-completion
form.

.. code-block:: cpp

    // sndr value-completes with either set_value_t(int) or set_value_t(std::string).
    auto single = stdexec::into_variant(sndr);
    // single value-completes with:
    //   set_value_t(std::variant<std::tuple<int>, std::tuple<std::string>>)
    auto [v] = stdexec::sync_wait(std::move(single)).value();
    std::visit([](auto&& tup) { /* ... */ }, v);

The pipe form ``sndr | into_variant()`` is equivalent. Note the *empty*
parentheses — there are no other arguments to pass; the closure exists
purely for the pipe syntax.

Parallel-loop adaptors
~~~~~~~~~~~~~~~~~~~~~~

.. _UserGuide_bulk:

``bulk`` — apply a function over an index space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::bulk` is the parallel-loop primitive of the
sender model. You give it a sender, an *execution policy*, an integral
*shape*, and a callable; you get back a sender that, when started,
invokes the callable once for each integer in ``[0, shape)``.

.. code-block:: cpp

    std::vector<int> buf(1024, 0);

    auto pipeline = stdexec::just()
                  | stdexec::bulk(stdexec::par, buf.size(),
                                  [&](std::size_t i) { buf[i] = compute(i); });
    stdexec::sync_wait(std::move(pipeline)).value();

The callable receives the index as its first argument; any values from
the predecessor's value completion are passed as additional arguments
(and shared across all iterations — *the same values, not a per-index
view*).

The execution policy works like the policies in ``<execution>``:

- ``stdexec::seq`` — sequential, no parallelism.
- ``stdexec::par`` — parallelism permitted.
- ``stdexec::par_unseq`` — parallelism *and* vectorization permitted.

Whether iterations actually run in parallel depends on the scheduler.
On an :ref:`inline scheduler <building-a-custom-scheduler-simple-inline-scheduler>`
(the implicit one used by ``just`` plus ``sync_wait``) every iteration
runs synchronously on the calling thread regardless of the policy. On
a thread-pool or GPU scheduler — typically used in conjunction with
:cpp:member:`stdexec::starts_on` — the policy is honored.

**Two variants.**

:cpp:member:`stdexec::bulk_chunked` invokes the callable with *ranges*
``(begin, end, vs...)`` instead of single indices. Use it when the
per-iteration body benefits from per-chunk amortization
(thread-local accumulators, vectorization setup, batched allocations).
``bulk`` is internally implemented in terms of ``bulk_chunked`` and
delegates the chunk-size decisions to the runtime.

:cpp:member:`stdexec::bulk_unchunked` is like ``bulk`` but explicitly
*forbids* chunking — each index is guaranteed its own invocation. Use
when per-iteration state cannot be batched.

**``bulk`` as the GPU hook.**
A custom scheduler can take over ``bulk`` via the
:ref:`domain customization <customizing-stdexec-s-algorithms-via-domains>`
mechanism, lowering it to a parallel-kernel launch on its own
execution context. nvexec does exactly this for CUDA.

Stopped-channel translators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These adaptors *re-route* a stopped completion onto a different
channel. They don't change the underlying behavior; they translate.

.. _UserGuide_stopped_as_error:

``stopped_as_error`` — turn cancellation into an error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::stopped_as_error` converts a ``set_stopped``
completion into a ``set_error`` completion carrying a caller-supplied
error. Use it when downstream code can't (or shouldn't) distinguish
cancellation from other failure modes.

.. code-block:: cpp

    auto sndr = stdexec::just_stopped()
              | stdexec::stopped_as_error(std::runtime_error{"cancelled"});

    try {
      stdexec::sync_wait(std::move(sndr));
    } catch (std::runtime_error const& e) {
      assert(std::string{e.what()} == "cancelled");
    }

It's a thin wrapper over :cpp:member:`stdexec::let_stopped` +
:cpp:member:`stdexec::just_error` — reach for it whenever you would
have written that pattern by hand.

.. _UserGuide_stopped_as_optional:

``stopped_as_optional`` — turn cancellation into ``std::nullopt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::stopped_as_optional` is the *value-channel*
analogue. It converts a ``set_stopped`` completion into a
*value* completion of ``std::optional<T>{std::nullopt}``, wrapping the
predecessor's value (when it completes successfully) in a
``std::optional<T>``.

.. code-block:: cpp

    auto sndr = stdexec::just(42) | stdexec::stopped_as_optional();
    auto [opt] = stdexec::sync_wait(std::move(sndr)).value();
    // opt == std::optional<int>{42}

The predecessor must have *exactly one* value-completion signature with
exactly one argument — otherwise the resulting sender wouldn't have a
unique ``std::optional<T>`` shape to use. The static assertion will
say so.

**Use** ``stopped_as_optional`` **or** ``sync_wait`` **?**

:cpp:member:`stdexec::sync_wait` already returns
``std::optional<std::tuple<...>>`` and gives you ``nullopt`` on stopped.
Use ``stopped_as_optional`` when you want that optional-shape *inside
the pipeline* rather than at the consumer — for example, to feed it
into a :cpp:member:`stdexec::then` that branches on the optional.

Environment adaptors
~~~~~~~~~~~~~~~~~~~~

.. _UserGuide_write_env:

``write_env`` — inject values into a sub-pipeline's environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::write_env` is the inverse of
:cpp:member:`stdexec::read_env`. ``read_env`` *reads* a value from the
receiver's environment and exposes it on the value channel;
``write_env`` *injects* values into the environment a sub-pipeline
sees. The supplied environment is overlaid on the receiver's
environment, so child senders see the merged view.

.. code-block:: cpp

    auto inner = stdexec::read_env(stdexec::get_stop_token)
               | stdexec::then([](auto tok) { return tok.stop_requested(); });

    stdexec::stop_source src;
    auto pipeline =
      inner
      | stdexec::write_env(stdexec::prop{stdexec::get_stop_token, src.get_token()});

    auto [requested] = stdexec::sync_wait(std::move(pipeline)).value();
    // The inner pipeline sees `src`'s token, not the outer pipeline's.

Common uses:

- Injecting a different stop token so a sub-pipeline can be cancelled
  independently of the surrounding work.
- Supplying an allocator to a sub-pipeline that allocates internally.
- Pinning a domain when a sender doesn't have a scheduler in its
  chain.

The supplied environment shadows the receiver's environment for any
query it can answer; queries it can't answer fall through.

Sender consumers
~~~~~~~~~~~~~~~~

Consumers are how a pipeline actually *runs*. They take a sender,
connect it to a built-in receiver, and start the resulting operation.
Until a consumer is called, a sender does nothing — it is just a
description of work.

.. _UserGuide_sender_consumers:

Picking a consumer
^^^^^^^^^^^^^^^^^^

Five consumers cover the common cases. The first question to ask is:
*does my caller need to wait for the result?*

.. list-table::
  :header-rows: 1
  :widths: 22 26 30 22

  * - Consumer
    - Returns
    - Use when
    - Eager or lazy?
  * - :cpp:member:`stdexec::sync_wait`
    - ``std::optional<std::tuple<...>>``
    - Top-level synchronous wait; single value-completion shape.
    - lazy
  * - :cpp:member:`stdexec::sync_wait_with_variant`
    - ``std::optional<std::variant<std::tuple<...>...>>``
    - Same, but the sender has multiple value-completion shapes.
    - lazy
  * - :cpp:member:`exec::start_detached`
    - ``void``
    - Top-level fire-and-forget; no owning scope. **stdexec extension.**
    - eager
  * - :cpp:member:`stdexec::spawn`
    - ``void``
    - Fire-and-forget into an async scope that will be joined later.
    - eager
  * - :cpp:member:`stdexec::spawn_future`
    - sender
    - Spawn into a scope *and* observe the result without blocking.
    - eager

The other axis is *who owns the lifetime of the operation state*:

- For ``sync_wait`` / ``sync_wait_with_variant`` the *caller's stack
  frame* owns it (the operation runs synchronously to completion).
- For ``start_detached`` the operation owns itself — it heap-allocates
  and deallocates on completion.
- For ``spawn`` and ``spawn_future`` the *scope* owns it: the operation
  is associated with a scope token that must outlive the work and is
  eventually joined.

.. _UserGuide_sync_wait:

``sync_wait`` — block until a result is ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::sync_wait` is the bridge from senders back into
synchronous code. It connects the sender, drives an internal
``run_loop`` on the calling thread until completion, and returns the
result wrapped in a ``std::optional<std::tuple<...>>``.

.. code-block:: cpp

    auto [v] = stdexec::sync_wait(stdexec::just(42)).value();
    // v == 42

The return shape is uniform: an engaged optional on ``set_value``, a
disengaged optional on ``set_stopped``, and a thrown exception on
``set_error`` (rethrown directly for ``std::exception_ptr``, wrapped in
``std::system_error`` for ``std::error_code``, thrown as-is otherwise).

.. code-block:: cpp

    if (auto result = stdexec::sync_wait(std::move(sndr))) {
      auto [v] = *result;     // succeeded
    } else {
      // cancelled (set_stopped)
    }

**Single value-completion only.**
``sync_wait`` requires a sender with exactly one ``set_value_t``
completion signature. If the sender can succeed in more than one way,
the static assertion will steer you to :cpp:member:`stdexec::sync_wait_with_variant`.

**Don't use** ``sync_wait`` **on an executor thread.** It blocks. It is
for top-level code (``main``, tests, leaf utilities), not for the
middle of a pipeline. If you need to "wait" mid-pipeline, you almost
certainly want :cpp:member:`stdexec::let_value` or a coroutine
``co_await`` instead.

.. _UserGuide_sync_wait_with_variant:

``sync_wait_with_variant`` — block until a multi-shape result is ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::sync_wait_with_variant` is for the case where a
sender can succeed in more than one way. The result is wrapped in a
``std::optional<std::variant<std::tuple<...>...>>``:

.. code-block:: cpp

    // sndr can complete with either set_value_t(int) or set_value_t(std::string)
    if (auto opt = stdexec::sync_wait_with_variant(std::move(sndr))) {
      std::visit([](auto&& tup) {
        // tup is std::tuple<int> or std::tuple<std::string>
      }, *opt);
    }

If your sender has only one value-completion shape, use
:cpp:member:`stdexec::sync_wait` — it returns a friendlier
``std::tuple`` directly.

.. _UserGuide_start_detached:

``start_detached`` *(extension)* — fire and forget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`exec::start_detached` eagerly starts a sender and discards
its result. The operation state is heap-allocated and self-destructs on
completion. The sender **must not** complete with ``set_error`` — there
is no caller to deliver the error to. The static assertion enforces
this; if your sender can fail, handle the error inline with
:cpp:member:`stdexec::upon_error` / :cpp:member:`stdexec::let_error`
first.

.. code-block:: cpp

    exec::start_detached(
      stdexec::just(42)
      | stdexec::then([](int x) { std::println("background: {}", x); }));

This is an **stdexec extension** — it isn't in the C++26 working draft.
The standardized scope-tracked equivalent is
:cpp:member:`stdexec::spawn`; reach for that when you have an async
scope that should own the work's lifetime.

.. _UserGuide_spawn:

``spawn`` — fire and forget into a scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::spawn` is the standardized way to launch
fire-and-forget work whose lifetime is owned by an *async scope*. You
pass a sender and a *scope token* (a handle you get from an async
scope); ``spawn`` allocates and starts the operation, and the scope
tracks it for later joining.

.. code-block:: cpp

    exec::async_scope scope;

    stdexec::spawn(
      stdexec::just(42)
      | stdexec::then([](int x) { std::println("background: {}", x); }),
      scope.get_token());

    // ... later, before scope is destroyed ...
    stdexec::sync_wait(scope.join());

As with ``start_detached``, the sender must not be able to complete
with ``set_error`` — ``spawn`` cannot deliver an error to a
non-existent caller.

**``spawn`` vs.** :cpp:member:`exec::start_detached` **:**
Prefer ``spawn`` whenever there's a natural owning scope (a request, a
session, a worker, the program as a whole). Reserve ``start_detached``
for one-shot top-level work where adding a scope would be ceremony.

.. _UserGuide_spawn_future:

``spawn_future`` — fire and observe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::spawn_future` is ``spawn`` plus an observation
channel. It eagerly starts the sender into the scope *and* returns a
sender that, when later connected and started, delivers the spawned
operation's result.

.. code-block:: cpp

    exec::async_scope scope;

    auto future =
      stdexec::spawn_future(stdexec::just(42)
                          | stdexec::then([](int x){ return x * 2; }),
                          scope.get_token());

    // The work is already running. Do something else here ...

    auto [v] = stdexec::sync_wait(std::move(future)).value();
    // v == 84

    stdexec::sync_wait(scope.join());

The key thing to internalize is that the work is **eager**: it starts
at the moment ``spawn_future`` is called, not when you connect the
returned sender. The returned sender is a *one-shot observer* of work
that is already running. This is what makes ``spawn_future`` good for
fan-out: spawn N pieces of concurrent work, collect their results
individually.

**``spawn_future`` vs.** :cpp:member:`stdexec::when_all` **:**
``when_all`` is *lazy* — it composes senders without starting them,
and the resulting sender only runs the children when *it* is started.
``spawn_future`` is the right choice when work needs to start
*immediately* (perhaps to overlap with synchronous code) and you'll
collect results later.

🔄 Coroutine Integration
------------------------

Senders can be ``co_await``-ed inside a coroutine whose promise type
participates in stdexec's awaitable-sender protocol (e.g.
``stdexec::task``). Any sender with exactly one successful completion
shape is awaitable in such a coroutine.

.. code-block:: cpp

    auto my_task() -> stdexec::task<int> {
      int x = co_await some_sender();
      co_return x + 1;
    }

If a sender that is being ``co_await``-ed completes with an error, the coroutine will
throw an exception. If it completes with a stop, the coroutine will be canceled. That is,
the coroutine will never be resumed; rather, it and its calling coroutines will be
destroyed.

In addition, all awaitable types can be used as senders, allowing them to be composed with
sender algorithms.

This allows ergonomic, coroutine-based async programming with sender semantics under the
hood.
