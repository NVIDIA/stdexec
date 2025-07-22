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
      using receiver_concept = stdexec::receiver_t;

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

Sender algorithms are defined in terms of **core customization points**. Below are the
core customization points that define how senders and receivers interact:

.. list-table:: Core customization points

  * - **CPO**
    - **Description**
  * - :cpp:member:`stdexec::connect`
    - Connects a sender to a receiver resulting in an operation state.
  * - :cpp:member:`stdexec::start`
    - Starts the operation.
  * - :cpp:member:`stdexec::set_value`
    - Called by the operation state to deliver a value to the receiver.
  * - :cpp:member:`stdexec::set_error`
    - Called by the operation state to deliver an error to the receiver.
  * - :cpp:member:`stdexec::set_stopped`
    - Called by the operation state to indicate that the operation was stopped.
  * - :cpp:member:`stdexec::get_env`
    - Retrieves the environment from a receiver.




* Receivers
* Custom Algorithms
* Custom Schedulers
* Customizing ``stdexec``'s algorithms
    * Domains
    * Early algorithm customization
    * Late algorithm customization
