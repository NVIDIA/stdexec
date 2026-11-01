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
`sequence` with operation states that hold one *phase* at a time:

- **`phased<Rcvr, Phases...>`** holds the operation's receiver and storage
  sized for the largest phase. It doesn't record which phase is live.
- **`phase_receiver<Op, Rcvr, Tag>`** is the receiver a phase's child
  completes through. Its type names the phase, so each completion handler
  knows statically what the storage holds. `let_value` has one possible
  second phase per value signature of its child, each with its own receiver
  type.
- **A transition** runs inside the live phase's completion: move what the
  next phase needs onto the stack, destroy the live phase, then construct the
  next phase in the same storage and start it.
- **Eager teardown.** The last phase is destroyed the same way before the
  operation's own receiver is completed, with the results moved onto the
  stack first. So the destructor needs one bit: was the operation started?

| Section | Shows |
|---|---|
| 6 | The four algorithms composed under `sync_wait`. |
| 7 | `let_value` choosing between an `int` phase and a `string` phase by receiver type alone; errors and stopped passing through. |
| 8 | Both of `let_value`'s phases pushing frames at the same automatic-storage address. |
| 9 | Operation-state sizes against stdexec's own algorithms. |
| 10 | An exception thrown by `then`'s function ending the phase and completing with `set_error`. |

Sizes (x86-64, GCC 13 and Clang 18 agree):

| Pipeline | stdexec | typestate |
|---|---|---|
| `let_value(just(big), fn)`, `big` 256 bytes | 552 | 288 |
| `sequence(just(big), just(big))` (stdexec: `let_value` making the second) | 560 | 536 |
| `then(just(1), fn)` | 24 | 32 |

`let_value` gains because its arguments live in the second phase, which
reuses the first phase's bytes; stdexec keeps them alongside the shared
storage. `sequence` gains nothing because the second sender waits in the
first phase. `then` loses 8 bytes to the started bit and its padding.

What it costs, and what it taught:

- **Copies.** Everything crossing a phase boundary, or leaving the last
  phase, is moved through the stack: `then`'s function and arguments,
  `let_value`'s function (twice) and arguments, every final result.
  stdexec's `then` passes its child's arguments straight to the function.
- **`let_value`'s function must move into the second phase**, not die on
  the stack: the sender it returns may refer to its captures.
- **The transition is boilerplate.** `then`, `let_value` and `sequence`
  each repeat the same "move out, leave, enter, start, or fail" sequence with
  its own exception handling. It wants to be one helper on `phased`.
- **Completion signatures are still most of the work.** About 130 of the
  header's ~780 lines compute them, and phases don't help with that; every
  algorithm here conservatively declares `set_error_t(std::exception_ptr)`.

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
