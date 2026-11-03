# Automatic storage and phase-based operation states

An exploration, not a proposal, in two parts:

- [Automatic storage for asynchronous operations](automatic_storage.md)
  (`automatic_storage.hpp`, `automatic_storage_demo.cpp`) asks whether senders
  can push their children's operation states onto LIFO storage threaded
  through the environment, the way a function's frame lives in automatic
  storage.
- [Operation states as sequences of phases](typestate.md) (`typestate.hpp`,
  the algorithm headers, `typestate_demo.cpp`) reimplements a few algorithms
  with operation states that hold one phase at a time, so that lifetimes are
  correct by construction, under policies chosen at `connect`.

The second part grew out of the first. Reusing storage LIFO depends on every
operation ending its children's lifetimes in order and never touching its own
state after completing; phases make those lifetimes explicit, and under
`eager_teardown` a completed operation holds nothing live. The phase-based
algorithms don't depend on automatic storage; the storage demo's last
sections run them under it.

Both demos build with the other examples, as `example.frames.automatic_storage`
and `example.frames.typestate`, or on their own:

```sh
c++ -std=c++20 -isystem ../../include automatic_storage_demo.cpp -o automatic_storage_demo
c++ -std=c++20 -isystem ../../include typestate_demo.cpp -o typestate_demo
```
