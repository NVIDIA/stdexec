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

Developer's Guide
=================

This section will eventually contain a guide for those interested in developing their own
asynchronous, sender-based algorithms and execution contexts.

.. _CoreConceptsForDevelopers:

Core Concepts for Developers
----------------------------

People wishing to extend the Sender model by writing their own sender algorithms or
schedulers, or by adapting another asynchronous model to the Sender model, should be
familiar with the core concepts of the Sender model. These concepts define how senders,
receivers, and schedulers interact.

.. graphviz::
    :caption: The Sender Model of Asynchrony
    :align: center

    digraph SenderModel {
        rankdir=TB;
        splines=false;
        fontsize=12;
        label=<<b>The Sender Model of Asynchrony</b><br/><br/>>;
        fontsize=18;
        labelloc=t;

        node [fontname="Courier New", fontsize=11, shape=box, style=filled];

        scheduler   [label=<<b>scheduler</b><br/><br/>schedule(<i>scheduler</i>) → <i>sender</i>>, fillcolor="#2c6a91", fontcolor=white];
        sender      [label=<<b>sender</b><br/><br/>connect(<i>sender</i>, <i>receiver</i>) → <i>operation-state</i>>, fillcolor="#d26937", fontcolor=white];
        receiver    [label=<<table border="0" cellborder="0"> <tr> <td>         <b>receiver</b></td> </tr> <tr> <td>set_value(<i>receiver</i>, vals...)</td> <td>→</td> <td>void</td> </tr> <tr> <td>set_error(<i>receiver</i>, err)</td> <td>→</td> <td>void</td> </tr> <tr> <td>set_stopped(<i>receiver</i>)</td> <td>→</td> <td>void</td> </tr> </table>>, fillcolor="#379dd2", fontcolor=white];
        environment [label=<<b>environment</b><br/>(key/value store)<br/><br/>key(<i>environment</i>) → value>, fillcolor="#2c682c", fontcolor=white];
        opstate     [label=<<b>operation state</b><br/><br/>start(<i>operation-state</i>) → void>, fillcolor="#933b94", fontcolor=white];

        // invisible dummy nodes
        connect  [shape=point, width=0.01, height=0.01, label="", style=invis];
        complete [shape=point, width=0.01, height=0.01, label="", style=invis];

        // Edges
        scheduler -> sender [penwidth=2, label="  schedule", labelfontname="Courier New"];
        environment -> receiver [penwidth=2, arrowhead=diamond];
        connect -> opstate [color=black, penwidth=2, labeldistance=2, labelangle=0, taillabel=< <table border="0" cellborder="0"> <tr> <td bgcolor="white">connect</td> </tr> </table> > ];
        sender -> receiver [color=black, penwidth=2, dir="both"];
        opstate -> complete [dir=none, style=dashed, label=" complete", fontsize=10, weight=0, labelangle=90, labeldistance=2];
        complete -> receiver [style=dashed, fontsize=10, weight=0];
        sender -> connect -> receiver [style=invis]

        // Layout tweaking
        { rank = same; scheduler; environment; }
        { rank = same; sender; connect; receiver; }
        { rank = same; opstate; complete; }
    }

In addition to the :ref:`CoreConceptsForUsers`, developers should also be familiar with the
following concepts:

1. Receiver
^^^^^^^^^^^

A **receiver** is an object that consumes the result of a sender. It defines three
member functions that handle completion:

- ``.set_value(args...)``: Called on success.

- ``.set_error(err)``: Called on error.

- ``.set_stopped()``: Called if the operation is canceled.

.. code-block:: cpp

    struct MyReceiver {
      using receiver_concept = stdexec::receiver_tag;

      void set_value(int v) noexcept                { /* success      */ }
      void set_error(std::exception_ptr e) noexcept { /* error        */ }
      void set_stopped() noexcept                   { /* cancellation */ }
    };

Receivers are an implementation detail of sender algorithms.

2. Operation State
^^^^^^^^^^^^^^^^^^

Connecting a sender to a receiver yields an **operation state**, which:

- Represents the in-progress computation.

- Is started explicitly via ``.start()``.

.. code-block:: cpp

    auto op = stdexec::connect(sndr, MyReceiver{}); // Connect sender to receiver
    stdexec::start(op);                             // Start the operation

Operation states are immovable, and once started, they must be kept alive until the
operation completes. Like receivers, operation states are typically an implementation
detail of sender algorithms.

3. Environments
^^^^^^^^^^^^^^^^^^

Environments are a key concept in the Sender model. An **environment** is an unordered
collection of key/value pairs, queryable at runtime via tag types. Every receiver has a
(possibly empty) environment that can be obtained by passing the receiver to
``stdexec::get_env``.

Environments provide a way to pass contextual information like stop tokens, allocators, or
schedulers to asynchronous operations. That information is then used by the operation to
customize its behavior.


Core Customization Points
-------------------------

Sender algorithms are defined in terms of a small set of **core
customization points** (CPOs) — the operations that every sender,
receiver, and operation state type must support in some form. Most user
code never touches these directly; sender adaptors and consumers do.
Anyone *writing* a new sender, receiver, or scheduler will implement one
or more of these.

The picker table below gives a one-liner per CPO; the sections that
follow expand each one with the customization patterns sender / receiver
/ operation-state authors actually use. Each CPO has a full-detail
reference entry under :ref:`Core Customization Points
<ref-section-cpos>` in the Reference section.

.. list-table:: Core customization points
  :header-rows: 1
  :widths: 32 28 40

  * - CPO
    - Lives on
    - Purpose
  * - :ref:`connect <ref-cpo-connect>`
    - sender
    - Connect a sender to a receiver, producing an operation state.
  * - :ref:`get_completion_signatures <ref-cpo-get_completion_signatures>`
    - sender (compile-time)
    - Compute what signals a sender can deliver.
  * - :ref:`start <ref-cpo-start>`
    - operation state
    - Begin execution of a connected operation.
  * - :ref:`set_value <ref-cpo-set_value>`
    - receiver
    - Deliver a successful value completion.
  * - :ref:`set_error <ref-cpo-set_error>`
    - receiver
    - Deliver a typed error completion.
  * - :ref:`set_stopped <ref-cpo-set_stopped>`
    - receiver
    - Deliver a stopped (cancellation) completion.
  * - :ref:`get_env <ref-cpo-get_env>`
    - sender *and* receiver
    - Obtain the environment (queries: stop token, allocator, …).

``connect`` — connect a sender to a receiver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sender author exposes :cpp:member:`stdexec::connect` via a
`.connect()` member that returns an
:ref:`operation_state <ref-concept-operation_state>`:

.. code-block:: cpp

    struct my_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(int)>;

      int value_;

      template <stdexec::receiver_of<completion_signatures> R>
      auto connect(R rcvr) && -> my_opstate<R> {
        return my_opstate<R>{std::move(rcvr), value_};
      }
    };

Notes:

- The returned operation state should be returned by value (it is
  immovable *after* construction, but stdexec relies on prvalue
  copy-elision to actually place it).
- The member is non-`const` and accepts the sender by value or
  rvalue reference — sender adaptors typically *move* their inputs
  into the operation state.
- Before dispatch, the framework runs ``transform_sender`` on the
  sender (passing in the receiver's environment), so domain-based
  customization happens *between* the user's call and the
  ``.connect()`` member. See :ref:`Customizing stdexec's algorithms
  <CoreConceptsForDevelopers>` below.

``get_completion_signatures`` — declare what a sender produces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Either expose a non-template type alias ``completion_signatures``
(for environment-independent senders — the common case):

.. code-block:: cpp

    struct my_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(int),
        stdexec::set_error_t(std::exception_ptr)>;
      // ... connect()
    };

…or, for senders whose signatures depend on the receiver's
environment, provide a static ``consteval`` member template:

.. code-block:: cpp

    struct env_dependent_sender {
      using sender_concept = stdexec::sender_tag;

      template <class Self, class... Env>
      static consteval auto get_completion_signatures() noexcept {
        // ... compute signatures from Env...
      }
      // ... connect()
    };

``start`` — begin execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An operation-state author provides a ``noexcept``, ``void``-returning
``start()`` member:

.. code-block:: cpp

    template <stdexec::receiver R>
    struct my_opstate {
      using operation_state_concept = stdexec::operation_state_tag;

      // Immovable after construction:
      my_opstate(my_opstate&&) = delete;

      R    rcvr_;
      int  value_;

      void start() noexcept {
        stdexec::set_value(std::move(rcvr_), value_);
      }
    };

The ``noexcept`` and ``void`` return are enforced by the dispatch site
with static asserts — the operation state must commit to never
throwing out of ``start``, and there is nothing to return.

``set_value`` / ``set_error`` / ``set_stopped`` — completion signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A receiver author opts into the three completion channels by exposing
matching ``noexcept`` members. Receivers typically implement *one* of
:cpp:member:`stdexec::set_value` and :cpp:member:`stdexec::set_error`,
and almost always :cpp:member:`stdexec::set_stopped`:

.. code-block:: cpp

    struct my_receiver {
      using receiver_concept = stdexec::receiver_tag;

      void set_value(int v) noexcept                { /* success      */ }
      void set_error(std::exception_ptr e) noexcept { /* error        */ }
      void set_stopped() noexcept                   { /* cancellation */ }
    };

Receivers may have multiple ``set_error`` overloads (one per error
type they understand), and the ``set_value`` arity must match the
sender's value-completion signatures.

The receiver promises that **exactly one** of the three completion
signals will be called on it, exactly once, after the operation has
been started. The receiver may not be called after destruction; the
operation state is responsible for ensuring this.

``get_env`` — expose the receiver's environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most receivers expose an environment so child operations can query
the stop token, allocator, preferred scheduler, etc. The
:cpp:member:`stdexec::get_env` CPO retrieves it:

.. code-block:: cpp

    struct my_receiver {
      using receiver_concept = stdexec::receiver_tag;

      stop_token_t stop_token_;

      auto get_env() const noexcept {
        return stdexec::env{
          stdexec::prop{stdexec::get_stop_token, stop_token_}};
      }
      // ... set_value, set_error, set_stopped
    };

The same CPO is used to query a *sender*'s attributes (e.g. its
completion scheduler) — the only difference is which member the
sender or receiver implements. A sender that has no attributes to
expose may simply omit ``get_env``; the CPO will default to an empty
environment via its ``__ignore`` overload.

.. _building-a-custom-algorithm-simple-then:

Building a Custom Algorithm: ``simple_then``
--------------------------------------------

This section is a worked example. We'll build a hand-rolled version of
:cpp:member:`stdexec::then` from scratch, using only the
:ref:`concepts <ref-section-concepts>` and :ref:`core customization
points <ref-section-cpos>` documented above. By the end you'll have a
complete, compilable, ~70-line sender adaptor that the rest of stdexec
treats as a first-class citizen.

We'll call it ``simple_then`` so it doesn't collide with the real one.
The semantics:

.. code-block:: cpp

    auto pipeline = simple_then(
      stdexec::just(21),
      [](int x) { return x * 2; });
    auto [v] = stdexec::sync_wait(std::move(pipeline)).value();
    // v == 42

To keep the focus on the *structure* of an adaptor rather than the
type-system gymnastics, we will hardcode the completion signatures.
That's a real limitation; we'll discuss how to lift it at the end.

The shape of an adaptor
^^^^^^^^^^^^^^^^^^^^^^^

Every sender adaptor is structurally three pieces, even though we'll
only have to write *two* of them. Recall the protocol:

.. code-block:: text

                     +-------- predecessor -------- our adaptor ----+
                     |                                              |
    1. caller -----> | sender ---connect---> opstate ---start---+   |
                     |                                          |   |
    2.               |                                          v   |
                     |                                set_value/error/stopped
                     |                                          |   |
                     +------------------------------------------+---+
                                                                |
                                                                v
                                                          receiver

A predecessor sender is the input. When connected to a receiver and
started, it eventually completes by calling
:cpp:member:`stdexec::set_value` (or ``set_error``, or ``set_stopped``)
on its receiver. Our adaptor's job is to *intercept* the value
completion and apply our callable to it, forwarding the result.

The trick is to *wrap the receiver*, not the sender. Our adaptor needs
three components:

1. A **sender type** holding the predecessor and the callable.
2. A **wrapping receiver** that intercepts ``set_value`` (transforming
   it) and forwards everything else verbatim to the real receiver.
3. An **operation state**. We'll get this for free: when we ``connect``
   the predecessor to our wrapping receiver, the predecessor returns
   its own operation state, and we can return that directly. Many
   adaptors do this; the only ones that build their own operation state
   are ones that need extra storage (cancellation callbacks, child
   variants, etc.).

Step 1: the wrapping receiver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The receiver is where the actual work happens. Its
:cpp:member:`stdexec::set_value` member runs the callable and forwards
the result; its ``set_error``, ``set_stopped``, and ``get_env`` members
just pass through to the inner receiver.

.. code-block:: cpp

    template <stdexec::receiver R, class Fn>
    struct simple_then_receiver {
      using receiver_concept = stdexec::receiver_tag;

      R  rcvr_;
      Fn fn_;

      template <class... Vs>
      void set_value(Vs&&... vs) noexcept {
        try {
          stdexec::set_value(
            std::move(rcvr_),
            std::invoke(std::move(fn_), static_cast<Vs&&>(vs)...));
        } catch (...) {
          stdexec::set_error(std::move(rcvr_), std::current_exception());
        }
      }

      template <class E>
      void set_error(E&& e) noexcept {
        stdexec::set_error(std::move(rcvr_), static_cast<E&&>(e));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(std::move(rcvr_));
      }

      auto get_env() const noexcept {
        return stdexec::get_env(rcvr_);
      }
    };

Things to notice:

- ``receiver_concept = receiver_tag`` opts the type into the
  :ref:`stdexec::receiver concept <ref-concept-receiver>`. Without this
  alias, our type wouldn't satisfy the concept and the framework would
  refuse to connect it.
- Every completion-signal member is ``noexcept`` and returns ``void``.
  The :cpp:member:`stdexec::set_value` dispatch site enforces this with
  static asserts — drop the ``noexcept`` and you get a compile error.
- If ``fn_`` throws, we catch and re-deliver via
  :cpp:member:`stdexec::set_error`. This is the standard convention; it
  is *why* our completion signatures include
  ``set_error_t(std::exception_ptr)``.
- ``get_env`` forwards the environment of the *inner* receiver. The
  predecessor sender needs to see the same stop token, allocator,
  scheduler etc. as the eventual consumer — our adaptor is invisible to
  environment queries. (If we wanted to *modify* the environment — say,
  to inject a different stop token — this is where we'd do it.)

Step 2: the sender
^^^^^^^^^^^^^^^^^^

The sender is just a value type that holds the predecessor and the
callable, plus three things that wire it into the framework:

.. code-block:: cpp

    template <stdexec::sender Sndr, class Fn>
    struct simple_then_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(int),
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_stopped_t()>;

      Sndr sndr_;
      Fn   fn_;

      template <stdexec::receiver R>
      auto connect(R rcvr) && {
        return stdexec::connect(
          std::move(sndr_),
          simple_then_receiver<R, Fn>{std::move(rcvr), std::move(fn_)});
      }
    };

The three wiring elements:

- ``sender_concept = sender_tag`` opts the type into
  :ref:`stdexec::sender <ref-concept-sender>`.
- ``completion_signatures`` (a type alias to a
  ``stdexec::completion_signatures`` specialization) declares what this
  sender can complete with. The framework consults this to type-check
  adaptors downstream. **We've hardcoded** ``set_value_t(int)`` here
  for simplicity — see *Going further* below for how to compute it
  properly from the predecessor's signatures.
- The ``connect`` member is what
  :cpp:member:`stdexec::connect` dispatches to. Our implementation
  wraps the user's receiver in a ``simple_then_receiver`` and connects
  *that* to the predecessor. The predecessor's operation state is what
  comes back — we don't need our own.

Note that ``connect`` takes ``this`` as an rvalue (``&&``). Senders are
typically *moved into* the operation state, not copied, so a sender
adaptor takes its inputs as rvalues. The framework arranges for this:
:cpp:member:`stdexec::connect` perfect-forwards both the sender and the
receiver, but the conventional sender object is short-lived (a temporary
in a pipeline).

Step 3: a helper factory
^^^^^^^^^^^^^^^^^^^^^^^^

Class-template type deduction would force callers to spell out the
template parameters of ``simple_then_sender``. A one-line factory
function fixes that:

.. code-block:: cpp

    template <class Sndr, class Fn>
    auto simple_then(Sndr&& sndr, Fn&& fn) {
      return simple_then_sender<std::decay_t<Sndr>, std::decay_t<Fn>>{
        static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn)};
    }

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

Here is the full, compilable example:

.. code-block:: cpp

    #include <stdexec/execution.hpp>
    #include <cassert>
    #include <exception>
    #include <utility>
    #include <type_traits>

    // ---------- The wrapping receiver -----------------------------------

    template <stdexec::receiver R, class Fn>
    struct simple_then_receiver {
      using receiver_concept = stdexec::receiver_tag;

      R  rcvr_;
      Fn fn_;

      template <class... Vs>
      void set_value(Vs&&... vs) noexcept {
        try {
          stdexec::set_value(
            std::move(rcvr_),
            std::invoke(std::move(fn_), static_cast<Vs&&>(vs)...));
        } catch (...) {
          stdexec::set_error(std::move(rcvr_), std::current_exception());
        }
      }

      template <class E>
      void set_error(E&& e) noexcept {
        stdexec::set_error(std::move(rcvr_), static_cast<E&&>(e));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(std::move(rcvr_));
      }

      auto get_env() const noexcept {
        return stdexec::get_env(rcvr_);
      }
    };

    // ---------- The sender ----------------------------------------------

    template <stdexec::sender Sndr, class Fn>
    struct simple_then_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(int),
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_stopped_t()>;

      Sndr sndr_;
      Fn   fn_;

      template <stdexec::receiver R>
      auto connect(R rcvr) && {
        return stdexec::connect(
          std::move(sndr_),
          simple_then_receiver<R, Fn>{std::move(rcvr), std::move(fn_)});
      }
    };

    // ---------- The factory ---------------------------------------------

    template <class Sndr, class Fn>
    auto simple_then(Sndr&& sndr, Fn&& fn) {
      return simple_then_sender<std::decay_t<Sndr>, std::decay_t<Fn>>{
        static_cast<Sndr&&>(sndr), static_cast<Fn&&>(fn)};
    }

    // ---------- Try it out ----------------------------------------------

    int main() {
      auto pipeline = simple_then(
        stdexec::just(21),
        [](int x) { return x * 2; });

      auto [v] = stdexec::sync_wait(std::move(pipeline)).value();
      assert(v == 42);
    }

That's a complete sender adaptor. The framework treats it just like a
built-in — you can chain it with other adaptors, pass it to
:cpp:member:`stdexec::sync_wait`, run it on a scheduler with
:cpp:member:`stdexec::starts_on`, ``co_await`` it from a coroutine, etc.

Going further
^^^^^^^^^^^^^

The version above leaves several things on the table — each is a
realistic next step.

**Computing the right completion signatures.**
We hardcoded ``set_value_t(int)``. For a real ``then`` we want to
compute the output's completion signatures from the input's: each
``set_value_t(Vs...)`` of the predecessor becomes a
``set_value_t(R)`` where ``R = decltype(std::invoke(fn, Vs...))``, and
the error/stopped signatures pass through unchanged. stdexec provides
``stdexec::transform_completion_signatures`` for exactly this. Replace
the hardcoded type alias with a ``static consteval``
:cpp:member:`stdexec::get_completion_signatures` member that calls into
that utility (the implementation of the built-in
:cpp:member:`stdexec::then` does this — see
``include/stdexec/__detail/__then.hpp``).

**Pipe syntax.**
``simple_then(sndr, fn)`` works but ``sndr | simple_then(fn)`` does
not, because ``simple_then`` with one argument doesn't return a
*sender-adaptor closure*. The fix is an overload that captures ``fn``
into a closure object, which stdexec supplies via
``stdexec::__closure`` (currently an internal helper) and which C++26
calls a ``pipeable_sender_adaptor_closure``.

**A real operation state.**
Our version inherits the predecessor's operation state. Adaptors that
need their own — to allocate a child variant, hook up a stop callback,
or store value datums across a hop — write their own operation-state
type with an :ref:`operation_state_concept <ref-concept-operation_state>`
alias and a ``noexcept`` ``start()`` member, and connect the
predecessor into a child member at construction time. The
:ref:`Customization-points walkthrough <ref-section-cpos>` sketches this
pattern.

**Forwarding the environment with modifications.**
If your adaptor wants to *change* something the predecessor sees — say,
inject a different stop token — override ``get_env`` to return a
modified environment (e.g. via ``stdexec::env`` and ``stdexec::prop``)
instead of just forwarding ``stdexec::get_env(rcvr_)``.

Each of these is a small extension of the same protocol — the
structure (sender, wrapping receiver, completion-signal forwarding) is
unchanged.

.. _building-a-custom-scheduler-simple-inline-scheduler:

Building a Custom Scheduler: ``simple_inline_scheduler``
--------------------------------------------------------

This section is a worked example, mirroring the structure of
:ref:`simple_then <building-a-custom-algorithm-simple-then>` for the
scheduler side of the protocol. We'll build a minimal scheduler whose
``schedule()`` produces a sender that completes *synchronously on the
calling thread* — an "inline scheduler".

stdexec ships a real inline scheduler in
``include/stdexec/__detail/__inline_scheduler.hpp``; we'll call our
walkthrough version ``simple_inline_scheduler`` so it doesn't collide.

Inline schedulers are useful in tests (synchronous, deterministic) and
as the "default trivial scheduler" when scheduler abstraction is wanted
but actual asynchrony is not. Writing one is also the simplest way to
exercise the *whole* scheduler protocol — there's no queue, no thread
management, no allocator interaction, just the three structural pieces.

What we're building
^^^^^^^^^^^^^^^^^^^

The user-visible API is the standard scheduler shape:

.. code-block:: cpp

    simple_inline_scheduler sched;

    auto s = stdexec::schedule(sched) | stdexec::then([] { return 42; });
    auto [v] = stdexec::sync_wait(std::move(s)).value();
    // v == 42

By the end you'll have a ~45-line scheduler that satisfies the
:ref:`stdexec::scheduler concept <ref-concept-scheduler>` and works
with :cpp:member:`stdexec::starts_on`,
:cpp:member:`stdexec::continues_on`, :cpp:member:`stdexec::on`, and
everything else that takes a scheduler.

The shape of a scheduler
^^^^^^^^^^^^^^^^^^^^^^^^

A scheduler is a small, value-typed *handle* to an execution context.
The :ref:`scheduler concept <ref-concept-scheduler>` requires only one
operation — :cpp:member:`stdexec::schedule` — plus plumbing
(equality-comparable, copy-constructible, nothrow-move-constructible).

So three pieces:

1. The **scheduler type** itself — a handle, equality-comparable, with
   a ``schedule()`` member.
2. The **schedule-sender** — the sender returned by
   ``schedule()``. It satisfies :ref:`stdexec::sender
   <ref-concept-sender>` and value-completes on the scheduler's
   execution resource.
3. The **operation state** — what
   :cpp:member:`stdexec::connect`-ing the schedule-sender to a
   receiver produces. Unlike the simple_then walkthrough, we have to
   write this from scratch — there's no predecessor whose op-state we
   can forward to.

Building it bottom-up — opstate, then sender, then scheduler — makes
each type a complete piece its dependent can refer to.

Step 1: the operation state
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The op-state holds the receiver and, when started, delivers an empty
value completion synchronously:

.. code-block:: cpp

    template <stdexec::receiver R>
    struct simple_inline_opstate {
      using operation_state_concept = stdexec::operation_state_tag;

      R rcvr_;

      explicit simple_inline_opstate(R rcvr) noexcept
        : rcvr_(std::move(rcvr)) {}

      // Operation states must remain at a stable address once started:
      simple_inline_opstate(simple_inline_opstate&&) = delete;

      void start() noexcept {
        stdexec::set_value(std::move(rcvr_));
      }
    };

A few things to notice:

- ``operation_state_concept = operation_state_tag`` opts the type into
  the :ref:`operation_state concept <ref-concept-operation_state>`.
- ``start()`` is ``noexcept`` and returns ``void`` — the
  :cpp:member:`stdexec::start` dispatch site enforces both with
  static asserts.
- ``start()`` calls :cpp:member:`stdexec::set_value` *directly* —
  there's nothing async about an inline scheduler. The receiver
  observes a completion that happens before ``start`` returns.
- The deleted move constructor is the standard way to assert
  immovability. Once an operation state is connected, the framework
  (and the receiver inside it) may hold pointers into its storage;
  letting it move would dangle those.
- We add an explicit constructor because deleting the move makes the
  type non-aggregate — brace-initialization no longer works without a
  matching constructor.

Step 2: the schedule-sender
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The schedule-sender is what ``schedule()`` returns. It's a regular
sender — same shape as in the :ref:`simple_then walkthrough
<building-a-custom-algorithm-simple-then>` — but instead of wrapping a
predecessor, it constructs an ``simple_inline_opstate`` directly:

.. code-block:: cpp

    struct simple_inline_schedule_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t()>;

      template <stdexec::receiver R>
      auto connect(R rcvr) const noexcept {
        return simple_inline_opstate<R>{std::move(rcvr)};
      }
    };

The interesting parts:

- The completion signatures declare *one* completion, ``set_value_t()``
  — an empty value completion, no datums. This is the canonical signal
  for "we are now on the scheduler's resource; carry on."
- ``connect`` is ``const`` here, not ``&&``. Schedule-senders are
  cheap, default-constructible, and stateless (for our scheduler);
  copying them is fine and lets ``stdexec::starts_on`` work without
  worrying about ownership.

A more sophisticated scheduler — say, a thread pool — would pass a
pointer/reference to its execution context through the schedule-sender
into the op-state, so that ``start()`` knows *where* to enqueue.

Step 3: the scheduler
^^^^^^^^^^^^^^^^^^^^^

The scheduler is just a handle with a ``schedule()`` member:

.. code-block:: cpp

    struct simple_inline_scheduler {
      auto schedule() const noexcept {
        return simple_inline_schedule_sender{};
      }

      bool operator==(simple_inline_scheduler const&) const noexcept = default;
    };

That's the whole thing. The :ref:`scheduler concept
<ref-concept-scheduler>` requires:

- ``schedule(s)`` is well-formed and returns a sender. ✓
- The type is equality-comparable. ✓ (``= default``)
- The type is copy-constructible. ✓ (implicit)
- The type is nothrow-move-constructible. ✓ (implicit)

Note that :cpp:member:`stdexec::schedule(sched)` is
expression-equivalent to ``sched.schedule()`` — that's how the CPO
dispatches. Notice also that we did *not* need to opt into any concept
explicitly with a tag-alias; the scheduler concept is structural (it
just checks for the ``.schedule()`` member and the value-semantics
plumbing).

**On equality.**

Two schedulers compare equal *iff they refer to the same execution
resource*. For our inline scheduler this is uninteresting: there's
only one "calling thread" in any meaningful sense, so any two
``simple_inline_scheduler`` instances are equivalent. ``= default``
gives us that.

For a thread-pool scheduler, equality would compare the underlying
pool pointer — adaptors like ``continues_on`` use this to elide
redundant scheduler hops (if the target is the same as the current,
the hop is a no-op).

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

Here is the complete compilable example:

.. code-block:: cpp

    #include <stdexec/execution.hpp>
    #include <cassert>
    #include <utility>

    // ---------- Operation state -----------------------------------------

    template <stdexec::receiver R>
    struct simple_inline_opstate {
      using operation_state_concept = stdexec::operation_state_tag;

      R rcvr_;

      explicit simple_inline_opstate(R rcvr) noexcept
        : rcvr_(std::move(rcvr)) {}

      simple_inline_opstate(simple_inline_opstate&&) = delete;

      void start() noexcept {
        stdexec::set_value(std::move(rcvr_));
      }
    };

    // ---------- Schedule-sender -----------------------------------------

    struct simple_inline_schedule_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t()>;

      template <stdexec::receiver R>
      auto connect(R rcvr) const noexcept {
        return simple_inline_opstate<R>{std::move(rcvr)};
      }
    };

    // ---------- Scheduler -----------------------------------------------

    struct simple_inline_scheduler {
      auto schedule() const noexcept {
        return simple_inline_schedule_sender{};
      }

      bool operator==(simple_inline_scheduler const&) const noexcept = default;
    };

    // ---------- Try it out ----------------------------------------------

    int main() {
      // Use schedule() directly:
      auto s = stdexec::schedule(simple_inline_scheduler{})
             | stdexec::then([] { return 42; });
      auto [v] = stdexec::sync_wait(std::move(s)).value();
      assert(v == 42);

      // Use starts_on to run an entire pipeline on the scheduler:
      auto t = stdexec::starts_on(
        simple_inline_scheduler{},
        stdexec::just(21) | stdexec::then([](int x) { return x * 2; }));
      auto [w] = stdexec::sync_wait(std::move(t)).value();
      assert(w == 42);

      // The scheduler concept is satisfied:
      static_assert(stdexec::scheduler<simple_inline_scheduler>);
    }

That's a complete scheduler. Anywhere an stdexec scheduler is accepted
— :cpp:member:`stdexec::starts_on`,
:cpp:member:`stdexec::continues_on`, :cpp:member:`stdexec::on`,
:cpp:member:`stdexec::schedule_from`, or even as the value of a
``get_scheduler`` query on a receiver's environment — this scheduler
just works.

Going further
^^^^^^^^^^^^^

The inline scheduler is the minimal possible scheduler. Realistic
schedulers add three things, each a relatively small extension:

**A queue and a context.**
A *deferred* scheduler — one whose ``schedule()`` doesn't complete
synchronously — holds a pointer or reference to an execution context
(thread pool, run loop, event loop). The schedule-sender carries
that context pointer through to the operation state, and ``start()``
enqueues the op-state for later execution rather than calling
``set_value`` immediately. The op-state typically derives from a
linked-list node so the queue can intrusively link it. stdexec's
``stdexec::run_loop`` (in
``include/stdexec/__detail/__run_loop.hpp``) is a good first reference
— it's a single-threaded run-loop scheduler in ~250 lines.

**Stop-token observation.**
A real scheduler reads the receiver's environment for a stop token
(via :cpp:member:`stdexec::get_env` followed by
``stdexec::get_stop_token``), registers a callback on it, and
completes with :cpp:member:`stdexec::set_stopped` if cancellation is
requested before the operation gets to run. Add a
``set_stopped_t()`` to the schedule-sender's completion signatures and
a stop-callback member to the operation state.

**Allocator handling.**
Schedulers that allocate per-operation state (most of them) should
consult ``stdexec::get_allocator`` on the receiver's environment so
allocations honor the caller's preferences. Falling back to
``std::allocator`` is fine for the default.

**Domain customization.**
A scheduler can additionally publish a *domain* tag in its
environment, which lets it intercept and rewrite sender expressions
specifically targeted at it — e.g. the GPU scheduler taking over a
``then`` chain so the lambdas run on-device. This is the topic of
the (still-to-be-written) *Customizing stdexec's algorithms* section
below.

Each of these extensions composes with the rest — you can add them
one at a time without redesigning the basic structure.

.. _customizing-stdexec-s-algorithms-via-domains:

Customizing stdexec's algorithms via domains
--------------------------------------------

The previous walkthroughs taught a scheduler how to *host* execution,
but they left the algorithms themselves untouched. Calling
:cpp:member:`stdexec::then` on a sender hosted by your scheduler still
runs the user's lambda through plain ``std::invoke`` — on the CPU, in
the receiver-completion thread.

For some execution contexts that is wrong. A GPU scheduler wants
:cpp:member:`stdexec::then` lambdas to execute *on the device*; a tracing
scheduler wants every algorithm wrapped with span-recording code; a
fault-injection scheduler wants every algorithm interceptible from a
test harness. None of these can be expressed by writing a different
scheduler — the *algorithms* themselves need to know they're operating
in a special context.

stdexec exposes this through *domains*. A scheduler publishes a domain
through its environment; the framework consults that domain at
``connect`` time, giving it a chance to rewrite any sender expression
in the pipeline before the algorithms see it. This is the same hook
nvexec uses to make standard algorithms compile to CUDA kernels.

What a domain is
^^^^^^^^^^^^^^^^

A domain is just a *tag type* — usually empty — with a particular
shape of customization member:

.. code-block:: cpp

    struct my_domain {
      template <class OpTag, class Sndr, class Env>
      static auto transform_sender(OpTag, Sndr&& sndr, Env const&)
        /* -> some-new-sender-or-the-same-sender */;
    };

The framework calls ``my_domain{}.transform_sender(tag, sndr, env)`` and
uses the *returned* sender — whatever that is — in place of the
original. Returning the input unchanged is a no-op customization;
returning a structurally different sender is how a domain *rewrites*
the pipeline.

The ``OpTag`` argument is the *kind* of customization being requested:

- ``stdexec::set_value_t`` — the sender will be connected on its
  *completion* domain (the domain advertised by the predecessor).
  This is what GPU schedulers hook to take over algorithms that
  produced data on the GPU.
- ``stdexec::start_t`` — the sender will be started in the *current*
  domain (the domain in the receiver's environment). This is the path
  used to react to the consumer side: "I am about to start on this
  domain, transform me first."

A given domain may handle both. Most real-world domains primarily care
about ``set_value_t``.

How the framework consults a domain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hook lives inside :cpp:member:`stdexec::connect`. When you write

.. code-block:: cpp

    auto op = stdexec::connect(sndr, rcvr);

the framework expands roughly to

.. code-block:: cpp

    // pseudo-code
    auto env       = stdexec::get_env(rcvr);
    auto completing_domain = /* read get_completion_domain<set_value_t>
                                from sndr's attributes */;
    auto starting_domain   = /* read get_domain from env (or default) */;

    auto sndr1 = completing_domain.transform_sender(set_value_t{}, sndr,  env);
    auto sndr2 = starting_domain  .transform_sender(start_t{},     sndr1, env);

    auto op = sndr2.connect(rcvr);   // (or static_connect, awaitable, ...)

Each ``transform_sender`` call delegates by default to a tag-type hook
on the sender (more below). If the sender's tag type does not provide
``transform_sender``, the default behavior is to return the sender
unchanged.

This is a two-phase model:

1. **Completing-domain transformation** — driven by the sender's
   advertised completion domain. Used by algorithms like
   :cpp:member:`stdexec::on`, whose ``transform_sender`` rewrites
   ``on(sch, sndr)`` into ``continues_on(starts_on(sch, sndr),
   orig_sch)``.
2. **Starting-domain transformation** — driven by the receiver's
   environment. Used to allow the *eventual consumer* to inject
   transformations.

Both happen at the same point (inside ``connect``) — there is no
"early" customization before ``connect`` any more. (Older versions of
the design had an "early" form, performed at sender construction
time and without an environment; it has since been removed.)

Two ways to participate
^^^^^^^^^^^^^^^^^^^^^^^

There are two customization paths, depending on whether you're writing
an algorithm or a scheduler.

**Tag-type customization** — for sender adaptor authors. Define a
``static transform_sender`` member on the sender's *tag type* (the
``foo_t`` of your ``foo``-adaptor) — the default domain's
``transform_sender`` finds it and forwards to it. This is what
``stdexec::on_t::transform_sender`` does to expand
``on(sch, sndr)`` into a combination of ``starts_on`` and
``continues_on``:

.. code-block:: cpp

    // Lightly paraphrased from include/stdexec/__detail/__on.hpp.
    struct on_t {
      // ... operator() overloads ...

      template <class Sender, class Env>
      static auto transform_sender(set_value_t, Sender&& sndr, Env&& env)
      {
        auto& [tag, data, child] = sndr;          // destructure the s-expression
        return /* continues_on(starts_on(data, child), orig_sch) */;
      }
    };

Tag-type customizations are how an adaptor implements its semantics
without writing the operation state by hand. They run regardless of
which domain is active — the rewrite is universal.

**Domain-level customization** — for scheduler authors. Define a
custom domain type with a ``transform_sender`` member that intercepts
algorithms in *its* execution context. Publish the domain through the
scheduler's environment. The framework will route every sender flowing
through a pipeline anchored on your scheduler through your domain's
``transform_sender`` first.

This is the GPU-scheduler story: nvexec's domain has a
``transform_sender`` that recognizes ``then_t``, ``bulk_t``,
``when_all_t`` etc. coming from its own schedule-sender and rewrites
them into CUDA-kernel-launching senders.

Worked example: a scheduler with a noticing domain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pedagogy, we'll skip the rewriting payload (it requires reaching
into a specific sender's structure, which varies per algorithm) and
focus on the *plumbing*. Our domain will simply *count* how many times
it is consulted at ``connect`` time — proof that the hook is live.

The structure is identical to the
:ref:`simple_inline_scheduler walkthrough
<building-a-custom-scheduler-simple-inline-scheduler>`, with three
additions:

1. A ``my_domain`` type with a ``transform_sender`` member.
2. The schedule-sender's attributes (``attrs_t``) advertise both
   ``get_completion_scheduler<set_value_t>`` (returning the scheduler
   itself) and ``get_completion_domain<set_value_t>`` (returning
   ``my_domain``).
3. The scheduler also answers ``get_completion_scheduler`` and
   ``get_completion_domain`` — the framework cross-checks that the
   scheduler's domain matches the schedule-sender's advertised domain
   and refuses to compile if they disagree.

.. code-block:: cpp

    #include <stdexec/execution.hpp>
    #include <atomic>
    #include <cassert>
    #include <utility>

    // Module-level state so we can observe whether our domain was consulted.
    static std::atomic<int> g_transform_count{0};

    // ---------- The domain ----------------------------------------------

    struct my_domain {
      // The framework calls this at connect time. We don't rewrite the
      // sender — we just record that we were consulted, then forward.
      template <class OpTag, class Sndr, class Env>
      static auto transform_sender(OpTag, Sndr&& sndr, Env const&) {
        g_transform_count.fetch_add(1, std::memory_order_relaxed);
        return static_cast<Sndr&&>(sndr);
      }
    };

    // ---------- A scheduler that publishes my_domain --------------------

    template <stdexec::receiver R>
    struct my_sched_opstate {
      using operation_state_concept = stdexec::operation_state_tag;
      R rcvr_;

      explicit my_sched_opstate(R rcvr) noexcept : rcvr_(std::move(rcvr)) {}
      my_sched_opstate(my_sched_opstate&&) = delete;

      void start() noexcept { stdexec::set_value(std::move(rcvr_)); }
    };

    struct my_scheduler;  // forward

    struct my_schedule_sender {
      using sender_concept = stdexec::sender_tag;
      using completion_signatures =
        stdexec::completion_signatures<stdexec::set_value_t()>;

      struct attrs_t {
        auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>)
          const noexcept -> my_scheduler;
        auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>)
          const noexcept { return my_domain{}; }
      };

      auto get_env() const noexcept { return attrs_t{}; }

      template <stdexec::receiver R>
      auto connect(R rcvr) const noexcept {
        return my_sched_opstate<R>{std::move(rcvr)};
      }
    };

    struct my_scheduler {
      auto schedule() const noexcept { return my_schedule_sender{}; }

      // A scheduler is its own completion scheduler, and claims my_domain
      // as the domain on which it completes.
      auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>)
        const noexcept { return *this; }
      auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>)
        const noexcept { return my_domain{}; }

      bool operator==(my_scheduler const&) const noexcept = default;
    };

    inline auto my_schedule_sender::attrs_t::query(
      stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept
      -> my_scheduler {
      return {};
    }

    // ---------- Try it out ----------------------------------------------

    int main() {
      static_assert(stdexec::scheduler<my_scheduler>);

      auto count_before = g_transform_count.load();

      auto pipeline = stdexec::schedule(my_scheduler{})
                    | stdexec::then([] { return 42; });
      auto [v] = stdexec::sync_wait(std::move(pipeline)).value();
      assert(v == 42);

      // The domain was consulted at least once at connect time:
      assert(g_transform_count.load() > count_before);
    }

Running this and observing ``g_transform_count > 0`` after
:cpp:member:`stdexec::sync_wait` returns is the empirical proof that
the domain hook is wired up. From there, replacing the
``return static_cast<Sndr&&>(sndr);`` line with an actual rewrite —
inspecting the sender's tag type and rebuilding it as something
different — is what turns this scaffolding into a real customization.

When and what to rewrite
^^^^^^^^^^^^^^^^^^^^^^^^

The body of a real ``transform_sender`` typically inspects
``stdexec::tag_of_t<Sndr>`` (or equivalently the result of pattern-matching
on the sender's tag) and rewrites only the senders whose tags are
"interesting" to this domain. Everything else is forwarded unchanged.

The canonical pattern is a chain of ``if constexpr`` branches:

.. code-block:: cpp

    template <class OpTag, class Sndr, class Env>
    static auto transform_sender(OpTag op, Sndr&& sndr, Env const& env) {
      using tag = stdexec::tag_of_t<Sndr>;
      if constexpr (std::same_as<tag, stdexec::then_t>) {
        return /* my-domain's version of then(sndr, fn) */;
      } else if constexpr (std::same_as<tag, stdexec::bulk_t>) {
        return /* my-domain's version of bulk(...)     */;
      } else {
        return static_cast<Sndr&&>(sndr);                 // pass through
      }
    }

Each rewrite typically destructures the original sender via the s-expression
machinery, extracts its data and child(ren), and rebuilds an equivalent
sender that runs on the domain's resource. nvexec's ``stream_domain`` does
this for every CUDA-kernel-compatible algorithm.

Pitfalls
^^^^^^^^

**Consistency between scheduler and schedule-sender attrs.**
If both the scheduler and the schedule-sender's attrs advertise a
domain, the framework cross-checks them at compile time and emits the
static assertion ``"the sender claims to complete on a domain that is
not the domain of its completion scheduler"`` if they disagree. Keep
them in lock-step.

**Don't capture the environment.**
``transform_sender`` is given the environment by const-reference;
rewriting it into a different sender that *captures* the environment
would dangle. Use the environment only for compile-time decisions
(constraint checks, alternative-selection).

**``transform_sender`` runs at connect time, once per call.**
It is not invoked at sender construction time, and it doesn't see the
op-state. If your rewrite needs runtime state, smuggle it through the
*sender itself* (as a data member of the type your rewrite produces).

Topics still to be written
^^^^^^^^^^^^^^^^^^^^^^^^^^

* A worked example of an actual rewriting domain (e.g. a tracing
  ``then`` that wraps the user's callable).
