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

.. highlight:: cpp
   :linenothreshold: 5

Welcome to ``stdexec``
======================

``stdexec`` is the reference implementation for `C++26's asynchronous execution framework
<https://eel.is/c++draft/exec>`_,  ``std::execution``. It provides a modern,
composable, and efficient abstraction for asynchronous programming in C++.

Sender/Receiver Abstraction in C++26
====================================

The **sender/receiver abstraction** is a foundational model for asynchronous programming in
modern C++, proposed for standardization in **C++26** (originally targeted for C++23). It aims
to unify and modernize asynchronous workflows across the C++ ecosystem.

🔧 Motivation
--------------

C++'s legacy async mechanisms — ``std::async``, futures, coroutines — have several limitations:

- Inflexible and hard to compose.

- Inefficient or heap-heavy.

- Difficult to customize or extend.

- Incompatible across libraries (e.g., Boost, Folly).

The sender/receiver abstraction offers:

- Composable async operations.

- Customizable schedulers and execution strategies.

- Clean cancellation handling.

- Coroutine integration.

- Zero-cost abstractions.

🧱 Core Concepts
----------------

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

    auto sched = exec::get_parallel_scheduler();   // Obtain the default system scheduler
    auto sndr  = stdexec::schedule(sched);         // Create a sender from the scheduler

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

3. Receiver
^^^^^^^^^^^

A **receiver** is an object that consumes the result of a sender. It defines three
member functions that handle completion:

- ``.set_value(args...)``: Called on success.

- ``.set_error(err)``: Called on error.

- ``.set_stopped()``: Called if the operation is canceled.

.. code-block:: cpp

    struct MyReceiver {
      using receiver_concept = stdexec::receiver_t;

      void set_value(int v) noexcept                { /* success      */ }
      void set_error(std::exception_ptr e) noexcept { /* error        */ }
      void set_stopped() noexcept                   { /* cancellation */ }
    };

As a user, you typically won't deal with receivers directly.
They are an implementation detail of sender algorithms.

4. Operation State
^^^^^^^^^^^^^^^^^^

Connecting a sender to a receiver yields an **operation state**, which:

- Represents the in-progress computation.

- Is started explicitly via ``.start()``.

.. code-block:: cpp

    auto op = stdexec::connect(sndr, MyReceiver{}); // Connect sender to receiver
    stdexec::start(op);                             // Start the operation

Operation states are immovable, and once started, they must be kept alive until the
operation completes. As a user though, you typically will not manage them directly.
They are handled by the sender algorithms.

5. Environments
^^^^^^^^^^^^^^^^^^

Environments are a key concept in the sender/receiver model. An **environment** is an
unordered collection of key/value pairs, queryable at runtime via tag types. Every
receiver has a (possibly empty) environment that can be obtained by passing the receiver
to ``stdexec::get_env()``.

Environments provide a way to pass contextual information like stop tokens, allocators, or
schedulers to asynchronous operations. That information is then used by the operation to
customize its behavior.

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
  * - ``stdexec::schedule``
    - Obtains a sender from a scheduler.
  * - ``stdexec::just``
    - Creates a sender that will immediately complete with a set of values.
  * - ``stdexec::read_env``
    - Reads a value from the receiver's environment and completes with it.

A **sender adaptor algorithm** takes an existing sender (or several senders) and
transforms it into a new sender with additional behavior. Below are some key sender
adaptor algorithms. Check the :ref:`Reference` section for additional algorithms.

.. list-table:: Sender Adaptor Algorithms
  :class: tight-table

  * - **CPO**
    - **Description**
  * - ``stdexec::then``
    - Applies a function to the value from a sender.
  * - ``stdexec::starts_on``
    - Executes an async operation on the specified scheduler.
  * - ``stdexec::continues_on``
    - Executes an async operation on the current scheduler and then transfers
      execution to the specified scheduler.
  * - ``stdexec::on``
    - Executes an async operation on a different scheduler and then transitions
      back to the original scheduler.
  * - ``stdexec::when_all``
    - Combines multiple senders, making it possible to execute them in parallel.
  * - ``stdexec::let_value``
    - Executes an async operation dynamically based on the results of a specified
      sender.
  * - ``stdexec::write_env``
    - Writes a value to the receiver's environment, allowing it to be used by
      child operations.

A **sender consumer algorithm** takes a sender connects it to a receiver and starts the
resulting operation. Here are some key sender consumer algorithms:

.. list-table:: Sender Consumer Algorithms

  * - **CPO**
    - **Description**
  * - ``stdexec::sync_wait``
    - Blocks the calling thread until the sender completes and returns the result.
  * - ``stdexec::start_detached``
    - Starts the operation without waiting for it to complete.

Sender algorithms are defined in terms of **core customization points**. Below are the
core customization points that define how senders and receivers interact:

.. list-table:: Core customization points

  * - **CPO**
    - **Description**
  * - ``stdexec::connect``
    - Connects a sender to a receiver resulting in an operation state.
  * - ``stdexec::start``
    - Starts the operation.
  * - ``stdexec::set_value``
    - Called by the operation state to deliver a value to the receiver.
  * - ``stdexec::set_error``
    - Called by the operation state to deliver an error to the receiver.
  * - ``stdexec::set_stopped``
    - Called by the operation state to indicate that the operation was stopped.
  * - ``stdexec::get_env``
    - Retrieves the environment from a receiver.

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


🔄 Coroutine Integration
------------------------

Senders can be ``co_await``-ed in coroutines if they model ``awaitable_sender``. Any sender
that can complete successfully in exactly one way is an awaitable sender.

.. code-block:: cpp

    auto my_task() -> exec::task<int> {
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

🚦 Standardization Status (as of 2025)
--------------------------------------

- The core sender/receiver model has been accepted into the C++ standard for C++26.

- Additional facilities have also been accepted such as

  * a system scheduler,
  * a sender-aware coroutine task type,
  * an async scope for spawning tasks dynamically.

- Interop with networking is being explored for C++29.

- Widely prototyped and tested in libraries and production settings.

🚀 Benefits
-----------

- ✅ Zero-cost abstractions: No heap allocations or runtime overhead.

- ✅ Composable: Express async pipelines clearly.

- ✅ Customizable: Plug in your own schedulers, tokens, adapters.

- ✅ Coroutine-friendly: Clean ``co_await`` support.

- ✅ Unified async model: Works for I/O, compute, UI, etc.

🔚 Summary
-----------

The sender/receiver abstraction:

- Brings modern, composable async programming to C++.

- Serves as a foundation for future concurrency features.

- Enables high-performance, coroutine-friendly workflows.

- Is set to become the standard async model in C++26.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  user/index

  developer/index

  reference/index

Indices and tables
------------------
* :ref:`genindex`
