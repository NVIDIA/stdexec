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

The Sender Abstraction in C++26
===============================

The **Sender abstraction** is a foundational model for asynchronous programming in modern
C++, proposed for standardization in **C++26**. It aims to unify and modernize
asynchronous workflows across the C++ ecosystem.

🔧 Motivation
--------------

C++'s legacy async mechanisms — ``std::async``, futures, coroutines, callbacks, threads,
mutexes, etc. — have several limitations:

- Inflexible and hard to compose safely.

- No way to specify *where* work should execute.

- Runtime overhead, dynamic allocations, and shared ownership.

- Difficult to customize or extend.

- Incompatible across ecosystems (e.g., Boost, Folly, Abseil, etc.).


The Sender abstraction introduces a compositional model of async computations that
separates concerns cleanly, enabling:

- A unified async model: Works for compute, I/O, networking, UI, etc.

- Generic algorithms that capture common async patterns.

- Combinators for building async workflows.

- Structured cancellation and error handling.

- Coroutine integration: ``co_await`` senders directly within coroutines and pass
  awaitables to sender algorithms.

- Zero-overhead composition (compile-time plumbing with no runtime allocations or
  reference counting).

- Full support for customization: Plug in your own schedulers, senders, adaptors,
  allocators, stop tokens, etc.

Senders end `Callback Hell <https://chatgpt.com/s/t_687fe7703d708191b94247513ad28246>`_.


🚦 Standardization Status (as of 2025)
--------------------------------------

- The core Sender model has been accepted into the C++ standard for C++26.

- Additional facilities have also been accepted such as

  * a system scheduler,

  * a sender-aware coroutine task type,

  * an async scope for spawning tasks dynamically.

- Interop with networking is being explored for C++29.

- Widely prototyped and tested in libraries and production settings.

🔚 Summary
-----------

The Sender abstraction:

- Brings modern, composable async programming to C++.

- Serves as a foundation for future concurrency features.

- Enables high-performance, coroutine-friendly workflows.

- Is set to become the standard async model in C++26.

.. toctree::
  :maxdepth: 2
  :caption: 📚 Additional Reading

  user/index

  developer/index

  reference/index

Indices and tables
------------------
* :ref:`genindex`
