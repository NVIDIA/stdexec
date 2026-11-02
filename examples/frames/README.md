# Automatic storage for asynchronous operations

An exploration, not a proposal. In synchronous code, a function's frame lives
in automatic storage: the caller pushes it, and the space is reused by the
next call once the callee returns. This directory asks whether senders can
get the same thing: LIFO storage, threaded through the environment, that an
operation pushes its child's operation state onto when it can't nest it by
value (type erasure being the motivating case).

It builds with the other examples as `example.frames`, or on its own:

```sh
c++ -std=c++20 -isystem ../../include demo.cpp -o demo && ./demo
```

## The model

Three layers, from the top:

- **`automatic_storage`** is a copyable cursor. `push<T>()` returns room for
  a `T` and the storage *after* it. Like a saved stack pointer, a copy may be
  pushed from again once everything allocated from later copies is dead. So
  an operation that pushes keeps its own copy and pushes from it again after
  its child's operation state has been destroyed. There is no `pop`.
- **`slab_chain`** belongs to one *strand*: a chain of logically sequential
  operations. It isn't thread-safe and is used LIFO. It keeps every slab it
  has handed out until it's destroyed, so crossing a slab boundary
  repeatedly doesn't allocate each time.
- **A slab source**, here any `std::pmr::memory_resource`, is shared and
  thread-safe. Allocation is coarse (one call per slab), so erasing it is
  cheap, and it is the one place to customize where memory comes from.

An operation state that has room to spare can offer it to its children:
`automatic_storage(buffer, outer)` pushes into `buffer` first and falls back
to `outer`.

## The pieces in `automatic_storage.hpp`

| Piece | Role |
|---|---|
| `with_automatic_storage(sndr, source, slab_size)` | Root of a strand: owns a chain and offers its storage to `sndr`. |
| `indirect(sndr)` | Stand-in for type erasure: pushes its child's operation state; falls back to the heap with no storage. |
| `strand(sndr)` | A fork: runs `sndr` on a new chain from the same slab source. |
| `reserve_async_stack<N>(sndr)` | Offers `N` bytes of its own operation state before the storage it was given. |
| `get_automatic_storage` | The environment query (a forwarding query). |

## What the demo shows

1. **Sequential phases reuse storage.** Three `let_value` phases each push a
   frame; all three land at the same address. stdexec's `let_value` already
   destroys the first child's operation state inside that child's completion
   and constructs the second in the same storage, so the second phase's push
   finds the space free. No change to `let_value` was needed.
2. **Nesting stacks frames, spills, and reuses spare slabs.** Three nested
   frames: two fit in the first slab, the third spills into a second. A
   second phase repeats the nesting at the same addresses without allocating
   another slab.
3. **Concurrency needs a fork.** `when_all` forwards its environment to its
   children, so two children pushing from the same storage *overlap*. Build
   with `-DFRAMES_SHOW_OVERLAP` to see both frames land at one address and the
   result come out as `2 2` instead of `1 2` (undefined behaviour). Wrapping
   each child in `strand` gives each its own chain.
4. **Inline room first.** `reserve_async_stack<128>` takes the first two
   frames in its own buffer; the third falls back to the strand's slab.
5. **No storage, no change.** Without a root, `indirect` uses the heap.

## Part two: operation states as sequences of phases

`typestate.hpp` reimplements `just`, `then`, `let_value` and a binary
`sequence` with operation states that hold one *phase* at a time. The aim
is a standard way of expressing an operation's states and transitions, so
that lifetimes are correct by construction, at no cost relative to the
established approach.

- **`phased<Ending, Rcvr, Carried, Phases...>`** holds the operation's
  receiver, a *carried* region that lives as long as the operation, and
  storage sized for the largest phase. Algorithms never construct or destroy
  phases themselves: from a phase's completion they call
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
- **A transition** runs inside the live phase's completion. `prepare` sees
  the dying phase and the completion's arguments while they're alive; it
  may store into the carried region, and returns what the next phase's
  constructor needs. Then the live phase is destroyed and the next one is
  constructed in the same storage and started.

Two policies, chosen per algorithm instance:

- **Ending.** `eager_teardown` destroys the last phase before completing,
  so a completed operation holds nothing live; results may refer into that
  phase, so they're moved onto the stack first. `forward_results` completes
  from inside the last phase, as stdexec's algorithms do, and leaves the
  phase to the destructor.
- **Layout** of state that crosses a transition (`let_value`'s values,
  `sequence`'s second sender). `carried_layout`, the default, keeps it in
  the carried region, as stdexec does, so it's constructed in place.
  `overlapped_layout` keeps it in the phases, so it reuses the earlier
  phase's bytes, at the price of a move through the stack: data whose
  destination overlaps its source has to be somewhere else while the source
  is destroyed.

| Section | Shows |
|---|---|
| 6 | The four algorithms composed under `sync_wait`. |
| 7 | `let_value` choosing between an `int` phase and a `string` phase by receiver type alone; errors and stopped passing through. |
| 8 | Both of `let_value`'s phases pushing frames at the same automatic-storage address. |
| 9 | Sizes and moves against stdexec's own algorithms. |
| 10 | Exceptions: from `then`'s function, and from every move of a value in turn, under both endings. |

Costs (x86-64; GCC 13 and Clang 18 agree; `big` is 256 bytes; stdexec has
no `sequence`, so its column is a `let_value` returning a captured sender):

| Operation-state bytes | stdexec | eager | forward |
|---|---|---|---|
| `let_value(just(big), fn)` | 552 | 536 | 536 |
| ... `overlapped_layout` | | 280 | 280 |
| `sequence(just(big), just(big))` | 816 | 544 | 544 |
| ... `overlapped_layout` | | 536 | 536 |
| `then(just(1), fn)` | 24 | 24 | 24 |

| Moves of a value, connect to result | stdexec | eager | forward |
|---|---|---|---|
| `let_value(just(c), fn)` | 6 | 7 | 6 |
| ... `overlapped_layout` | | 9 | 8 |
| `then(just(c), fn)`, `fn` by value | 8 | 10 | 8 |
| `sequence(just(), just(c))` | 10 | 10 | 8 |
| ... `overlapped_layout` | | 11 | 9 |

So with `forward_results` and the default layout, the typestate versions
cost no more than stdexec's in bytes or moves. `eager_teardown` costs one
move of the results per algorithm that tears down; `overlapped_layout`
trades a move for bytes.

What else it taught:

- **`let_value`'s function is carried**, not passed between phases: the
  sender it returns may refer to its captures.
- **Completion signatures are still most of the work.** About 175 of the
  header's ~990 lines compute them, and phases don't help with that; every
  algorithm here except `just` conservatively declares
  `set_error_t(std::exception_ptr)`.

## Open questions

- **Who may forward the query.** Making the query non-forwarding wouldn't
  make stock `when_all` safe: in stdexec today, `when_all`'s children (and
  `then`'s, and `let_value`'s first child) see non-forwarding queries too,
  while `let_value`'s second child, `starts_on`'s and `write_env`'s don't. A
  real design needs every algorithm that introduces concurrency to fork, and
  some rule that stops an algorithm that doesn't know about forking from
  sharing a strand's storage between concurrent children.
- **Lifetimes.** All of this relies on an operation never touching its own
  state after signalling completion, since the parent may destroy it and
  reuse the space from inside the completion. stdexec's `let_value` already
  relies on the same rule.
- **Frame overhead.** Each receiver carries a copy of the storage cursor
  (five pointers here); that is most of each 56–64-byte frame in the demo.
- **Simplifications.** The typestate senders connect as rvalues only, and
  `let_value` doesn't add a completion scheduler to its second phase's
  environment. Fixed-size slabs (an oversized frame fails instead of
  getting an oversized slab), allocation failure always throws `bad_alloc`
  (or terminates without exceptions), no `fork(n)`, no `spawn`, no
  coroutines.
