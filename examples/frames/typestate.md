# Operation states as sequences of phases

`just.hpp`, `then.hpp`, `let_value.hpp`, `sequence.hpp`, `repeat_n.hpp` and
`when_all.hpp` reimplement `just`, `then`, `let_value`, a binary `sequence`,
`repeat_n` and `when_all` with operation states that hold one *phase* at a
time, written against the
machinery in `typestate.hpp`. The aim is a standard way of expressing an
operation's states and transitions, so that lifetimes are correct by
construction, at no cost relative to the established approach. The machinery
is where the complexity lives, in about 1600 lines; each algorithm takes 125
to 180, about a fifth of them completion signatures, except `when_all`,
whose 395 include about 120 working out its completion signatures and the
types it stores.

- **`phased<Rcvr, Parts, Carried, Phases...>`** holds the operation's
  receiver, a *carried* region that lives as long as the operation, and
  storage sized for the largest phase. A phase may also *receive* values
  from the phase before it, built from that phase's results while it's still
  alive, since they may refer into it. Where those live is the layout
  policy's choice, made inside `phased`. Algorithms never construct or
  destroy phases themselves: from a phase's completion they call
  `transition<From, To>(prepare, ...)`, `finish<Phase>(tag, ...)` or
  `finish_with<Phase>(fn, ...)`, and `phased` does the destroying,
  constructing, starting and exception handling. Their bodies contain no
  placement new, explicit destruction, liveness flags or `try`.
- **`phase_receiver<Op, Rcvr, Tag>`** is the receiver a phase's child
  completes through. Its type names the phase, so each completion handler
  knows statically what the storage holds. `let_value` has one possible
  second phase per value signature of its child, each with its own receiver
  type. `phased` also keeps one byte naming the live phase, for its
  destructor and exception paths.
- **A phase's children may see an environment of their own**, built from
  what the phase holds. `phase_receiver`'s last parameter names its type, and
  the operation's `env(Tag)` builds it, from `live<Phase>()`; otherwise a
  child sees the operation's receiver's environment. The type is named up
  front because the operation is incomplete while its children's types are
  worked out. Keyed by the tag, it can differ per child, as a fork of
  automatic storage would need. `env_with<Query, Value, Env>` replaces one
  query and carries over the rest, policies included.
- **Children that run at once and stop together** share a `join_scope`: a
  stop source whose token their environments carry, a callback that forwards
  the receiver's stop requests to it from `open()` until the last child
  arrives, and a count of the children still running. `when_all` holds one
  in its single phase. A child that fails or stops arrives with
  `stop_and_arrive()`, which requests stop before counting the child out, so
  no child can end the scope during its own `request_stop`. The last to
  arrive (`arrive()` returns `true`) drops the forwarding callback, waiting
  for one running on another thread, before the operation finishes.
- **A transition** runs inside the live phase's completion. `prepare` sees
  the dying phase and the completion's arguments while they're alive; it may
  store into the carried region, and returns what the next phase's
  constructor needs, or `hand_over(vs...)` to hand it values. Then the live
  phase is destroyed and the next one is constructed in the same storage and
  started.

Four policies, chosen together, as `policies<Ending, Layout, Entry, Data>`,
for a whole tree of operations. The algorithms take none of them as
parameters: `phased` asks its receiver's environment for them with the
forwarding query `get_policies`, so every operation in a tree sees the
policies asked for at its top, including operations connected at run time
(the sender `let_value`'s function returns), and stdexec's algorithms pass
the query through to their children. An environment that doesn't answer it
gets `default_policies`, the first of each below. `with_policies<P>(sndr)`
asks for `P` with `write_env`; connecting to a receiver whose environment
answers the query does the same without an adaptor.

The four:

- **Ending.** `eager_teardown` destroys the last phase before completing,
  so a completed operation holds nothing live; results may refer into that
  phase, so they're moved onto the stack first. `forward_results` completes
  from inside the last phase, as stdexec's algorithms do, and leaves the
  phase to the destructor.
- **Layout** of values a phase receives (`let_value`'s values). The
  algorithm only says what it hands over; `phased` places it.
  `carried_layout`, the default, keeps the values beside the phases for as
  long as the operation, as stdexec does, so they're constructed in place.
  `overlapped_layout` keeps them in the phases' bytes, just before the phase
  that receives them, so they reuse the earlier phase's bytes, at the price
  of a move through the stack: data whose destination overlaps its source
  has to be somewhere else while the source is destroyed. `scoped_layout`
  keeps them beside the phases, constructed in place, but only for as long
  as the phase that receives them, and with no `std::variant` index, since
  the live phase already says which values are there.
- **Entry**: when the first phase is entered. `connect_entry`, the default,
  enters it in `connect`. `start_entry` enters it in `start`, so a child
  whose `connect` throws makes `start` complete with `set_error` instead of
  making `connect` throw.
- **Data**: where the sender's data lives once it's connected.
  `stored_sender`, the default, has `connect` move the sender into a small
  `root` operation, and every typestate sender then *binds* rather than
  connects (`bind(rcvr) &`): its operation refers to the stored sender,
  using its function in place and binding its children to its own child
  senders, so the stored sender is the carried state of the whole tree and
  nothing below the root copies it. A sender that can't bind (any stdexec
  sender) is moved into `connect` when its phase begins. `split_sender`
  stores no sender: `connect` moves each part the operation uses into the
  operation, except the child sender that a first phase entered by `connect`
  connects, which goes straight into that child's `connect`, as stdexec's
  algorithms do. Each algorithm lists the parts it uses (`parts<&S::sndr_,
  &S::fn_>`) and each phase the part its child is made from (`connects`); it
  reads a part with `part<M>()` and makes a child with
  `connect_part<M>(rcvr)`, which binds under one policy and connects under
  the other, so the algorithm doesn't depend on which. Senders that
  `let_value`'s function returns are created at run time, so their phase
  owns them either way.

Under `stored_sender`, binding comes in two forms, like `connect` for rvalues and const lvalues,
minus the move or copy. Through a non-const reference, `bind(rcvr) &`
*consumes*: the operation may move from the stored sender (as `just` does
when it hands over its values), which is then spent. Through a const
reference, `bind(rcvr) const &` doesn't: the operation only reads the
stored sender, so it's unchanged afterwards and can be bound again. `then`
calls its function as `const`, `just` copies its values as it sends them,
and `let_value` and `sequence` bind their children the same way, so the
whole subtree is read-only. A sender without a member `bind` is connected
as a const lvalue, which copies. One semantic difference from connecting a
copy: a function is shared by every bind rather than copied, so it must be
callable as `const`.

`repeat_n(sndr, n)` uses the non-consuming form: one phase, re-entered for
each iteration, binding the stored child every time. Over 100 iterations
(eager+split is `eager_teardown` with `split_sender`):

| Moves (copies) | stdexec | eager | forward | eager+start | eager+split |
|---|---|---|---|---|---|
| `repeat_n(then(just(), fn capturing c))` | 904 (100) (Clang: 804) | 3 (0) | 3 (0) | 3 (0) | 303 (100) |
| `repeat_n(then(just(c), drop))` | 1104 (100) (Clang: 1004) | 103 (100) | 3 (100) | 103 (100) | 703 (100) |

`exec::repeat_n` reconnects a copy of its child every iteration, captures
included; bound to a stored sender, `repeat_n` never copies what its child
only reads. Under `split_sender` it keeps its child and connects a copy each
iteration too. Values the child sends are copied each time either way, since
the receiver gets an rvalue; under `eager_teardown` each copy is moved once
more onto the stack. Unlike `exec::repeat_n`, it doesn't trampoline, so a
child that completes inline makes each iteration a nested call.

Each exception path knows statically which phase, if any, is live, so the
byte recording the live phase serves only the destructor. With
`eager_teardown` and `start_entry` together, no phase is ever live when the
operation state is destroyed, so `phased` has no such byte at all and its
destructor destroys only the receiver and the carried region. Under
`stored_sender`, the root's destructor destroys the stored sender as well,
which is every level's carried data at once.

| Section of `typestate_demo.cpp` | Shows |
|---|---|
| 1 | The four algorithms composed under `sync_wait`. |
| 2 | `let_value` choosing between an `int` phase and a `string` phase by receiver type alone; errors and stopped passing through. |
| 3 | Sizes and moves against stdexec's own algorithms. |
| 4 | Exceptions: from `then`'s function; from every move of a value in turn, under four policies and again under each other layout; and where a throwing child `connect` surfaces under each entry. |
| 5 | Bound operations mixed with stdexec senders. |
| 6 | `repeat_n`: iteration counts, a failing iteration, and copies against `exec::repeat_n`. |
| 7 | With `-fsanitize=address -DFRAMES_SHOW_USE_AFTER_FINISH`: a child that touches its own state after completing, reported by AddressSanitizer. |
| 8 | When `let_value`'s values are destroyed, under each layout and both endings. |
| 9 | `when_all` giving its children its own stop token: a child's error, and the receiver's stop request, reaching the other children. |
| 10 | `then`'s function called as its child sent the values, under every ending. |

Costs (x86-64; GCC 13 and Clang 18, with libstdc++ or libc++, agree except
where noted; "eager+start" is `eager_teardown` with `start_entry`, and
"eager+split" is `eager_teardown` with `split_sender`). Each policy heads
two columns: the operation state's size in bytes, and how often a value
moves between `connect` and the receiver, with copies in parentheses. Both
connect the pipeline to a receiver whose environment asks for the column's
policies; for moves, it's then started, and completes inline. The two are
measured on slightly different pipelines. For bytes, `v` is a 256-byte `big`
(an `int` in the `then` row); for moves, it's a counted value `c`, which
`then`'s `fn` takes by value. stdexec's `sequence` is `exec::sequence`,
which requires every sender but the last to send no values, so `s` is
`then(just(big), drop)` for bytes, making the first phase big too, and
`just()` for moves.

```text
                              stdexec          eager          forward        eager+start      eager+split
                           -------------   -------------   -------------   -------------   -------------
                           bytes   moves   bytes   moves   bytes   moves   bytes   moves   bytes   moves
let_value(just(v), fn)       552   6 (0)     568   5 (0)     568   4 (0)     568   5 (0)     536   7 (0)
  ... overlapped_layout                      568   6 (0)     568   5 (0)     560   6 (0)     280   8 (0)
  ... scoped_layout                          568   5 (0)     568   4 (0)     560   5 (0)     536   7 (0)
sequence(s, just(v))*        560   9 (0)     568   5 (0)     568   3 (0)     552   5 (0)     552   8 (0)
then(just(v), fn)             24   7 (0)      40   7 (0)      40   5 (0)      32   7 (0)      24   9 (0)
when_all of two just(v)     1072  14 (0)    1144  12 (0)    1144   8 (0)    1136  12 (0)    1136  16 (0)

* stdexec moves 8 (0) with Clang
```

Under `stored_sender`, binding makes values move once, into the root, and
then be used in place, so the moves come out at or below stdexec's in every
row. The cost is in bytes: the stored sender keeps every level's data for
the whole operation, so nothing a later phase could reuse is moved out of
it. That's why `overlapped_layout` saves nothing there: `just`'s value stays
in the stored sender rather than in the first phase. A bound operation
doesn't store a pointer to its sender either: it asks its receiver, whose
parent knows which of its child senders each phase binds (`bound_sender()`),
and the root's receiver points at the stored sender. So the tree holds one
pointer to a sender in all, at the root. That pointer, and under
`connect_entry` the byte recording the live phase (an operation destroyed
before it starts still holds its first phase), are what separate
`then(just(1), fn)` from stdexec's 24 bytes. The price is a walk up the
receivers on each access to a sender's data. Receivers that are rebuilt from
the child's address instead of stored (P3425, worded for C++26 by P3986 as
`inlinable_receiver`) would turn that walk into constant offsets, and remove
the receivers' own pointers too.

`when_all` keeps its children's values in its phase until the last child
completes, as stdexec's does, and moves them less often except under
`split_sender`. Its extra bytes are mostly an error slot, since like every
algorithm here it declares `set_error_t(std::exception_ptr)` where stdexec's
keeps no room for errors its children can't send (16 bytes), and the root's
pointer and the bytes recording `when_all`'s and each `just`'s live phase,
with their padding (32).

`split_sender` trades the other way. Each operation moves in the data it
uses, so a value moves once per level rather than once in all, but the data
lives where its phase does: `overlapped_layout` saves bytes again, and
`then(just(1), fn)` matches stdexec's 24.

`eager_teardown` costs one move of the results per algorithm that tears
down. `overlapped_layout` trades a move for bytes where the received values
can reuse the dying phase's bytes. `scoped_layout` costs what
`carried_layout` does, less the variant's index where alignment doesn't
absorb it; what it changes is how long the values live. Under
`eager_teardown` they're destroyed before the operation completes, as they
are with `overlapped_layout`, rather than with the operation state (section
8).

Under AddressSanitizer, `phased` poisons the storage no phase occupies.
Touching a phase after it has ended then gets reported rather than silently
reading or corrupting memory: section 7's child increments a counter after
completing, inside a `then` with `eager_teardown`, and ASan reports a
use-after-poison at that line. Two limits: bytes the next phase reuses are
live again, so a transition only catches late accesses beyond the next
phase's size; and under `forward_results` the last phase is still alive when
a child returns from completing, so nothing is caught there. Received values
are poisoned the same way once they end, so under `overlapped_layout` or
`scoped_layout` with eager teardown, a child that reads `let_value`'s values
after completing is reported too; with `carried_layout` they're still alive
then. That makes eager teardown worth having at least in checked builds. The
whole demo runs clean with poisoning on, including the stdexec senders used
as children.

What else it taught:

- **`let_value`'s function stays put**, not passed between phases: the
  sender it returns may refer to its captures.
- **Every layer of `connect` moves the sender's data again** in stdexec at
  this commit, apparently because `connect` passes the sender through
  `transform_sender`, which can return it by value. Wrapping a typestate
  sender in `with_policies` (or any other adaptor) costs two more moves of
  its values, which is why the demo counts moves by connecting to a receiver
  that asks for the policies instead. Under `stored_sender`, a tree of
  typestate senders pays this once, at the root, since the operations below
  bind their children; under `split_sender`, each level connects its
  children through `connect`.
- **Stop requests end operations from inside their own callbacks.** A
  child that completes from its stop callback can make its parent change
  phase (`let_value`'s second child then registers its callback with a
  source that's in the middle of `request_stop`) or finish, and under
  `eager_teardown` end the phase holding the child, its callback and, in
  `when_all`, the stop source whose `request_stop` is running. That's
  allowed: a stop callback may be destroyed from its own invocation, and
  since #1889 stdexec's `inplace_stop_source` may be destroyed by one of its
  own callbacks. What it asks of children is the usual rule, applied
  strictly: since eager teardown ends a child's phase before the child's
  `start` or stop callback returns, a child that writes to itself after
  completing is caught every time, not only when its parent happens to
  destroy the operation. A child that registers its stop callback with
  `std::optional::emplace` and completes from the callback is one such: when
  stop has already been requested, the callback runs inside `emplace`, which
  sets its flag afterwards.
- **Completion signatures are still much of the work.** About 160 of
  `typestate.hpp`'s lines, and 20 to 30 of each algorithm's, compute them,
  and phases don't help with that; every algorithm here except `just`
  conservatively declares `set_error_t(std::exception_ptr)`.

## Open questions

- **Simplifications.** The typestate senders connect as rvalues only, and
  `let_value` doesn't add a completion scheduler to its second phase's
  environment.
