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
  * - :cpp:member:`stdexec::start_detached`
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
