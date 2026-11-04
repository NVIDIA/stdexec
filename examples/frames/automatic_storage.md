# Automatic storage for asynchronous operations

An exploration, not a proposal. In synchronous code, a function's frame lives
in automatic storage: the caller pushes it, and the space is reused by the
next call once the callee returns. This part asks whether senders can get
the same thing: LIFO storage, threaded through the environment, that an
operation pushes its child's operation state onto when it can't nest it by
value (type erasure being the motivating case).

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

## What `automatic_storage_demo.cpp` shows

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
   children, so two children pushing from the same storage would *overlap*:
   both frames would land at one address, the second constructed over the
   first. Wrapping each child in `strand` gives each its own chain.
4. **Inline room first.** `reserve_async_stack<128>` takes the first two
   frames in its own buffer; the third falls back to the strand's slab.
5. **No storage, no change.** Without a root, `indirect` uses the heap.
6. **Phases reuse storage too.** The phase-based `let_value` of
   [part two](typestate.md) holds one phase at a time, and both phases push
   their children's frames at the same address.
7. **So do bound phases**, entered at `start` and bound to a sender stored at
   the root.

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
- **Simplifications.** Fixed-size slabs (an oversized frame fails instead of
  getting an oversized slab), allocation failure always throws `bad_alloc`
  (or terminates without exceptions), no `fork(n)`, no `spawn`, no
  coroutines.
