# Maintainers' Guide

This is a place to put design rationale and code idioms that maintainers of
stdexec should follow.

* Data, including downstream receivers, are best stored in the operation
  state. Receivers themselves should, in general, store nothing but a
  pointer back to the operation state.

* Assume that schedulers and receivers contain nothing but a pointer and
  are cheap to copy. Take them by value.

* In a sender adaptor, a reasonable way to constrain
  `ThisSender<InnerSender>::connect(OuterReceiver)` is by
  requiring `sender_to<InnerSender, ThisReceiver<OuterReceiver>>`.

* Place concept checks on public interfaces. Only place concept checks
  on implementation details if needed for correctness; otherwise, leave
  them unconstrained. Use `static_assert` if you must, but don't bother
  rechecking things that were already checked at the public interface.

* Regarding the modularized build:
  * Enable the modularized build with `-DSTDEXEC_BUILD_MODULES=1` at
    configure time. CI tests it with Clang built from llvm-project trunk
    (see the `modules` row in `.github/workflows/ci.cpu.yml`); Clang 22.1.x
    and 23.1.0 both hit ICEs in the modular build.
  * Every header that is transitively included by `modules/stdexec.cppm` must
    check its build context:
    * if `STDEXEC_USE_MODULES()` is true then modules are enabled
    * if `STDEXEC_IN_MODULE_PURVIEW` is defined then not only are modules
      enabled, but also the header is being included in the module definition
      unit (i.e. inside `modules/stdexec.cppm` for headers in
      `include/stdexec`)
    * otherwise, this is a normal include into a non-modules TU
  * When `STDEXEC_USE_MODULES()` is true and `STDEXEC_IN_MODULE_PURVIEW` is
    defined, the header's job is to define its part of the module unit so it
    should behave mostly like a traditional include. One key difference from
    traditional inclusion is that standard headers should *not* be included;
    instead, use `import std;`.
  * When `STDEXEC_USE_MODULES()` is true and `STDEXEC_IN_MODULE_PURVIEW` is
    not defined, the header should do nothing but `import stdexec;`
  * Otherwise, headers should behave as normal.
  * Within the header, use `STDEXEC_MODULE_EXPORT` to expose symbols through
    the module interface; for symbols that are part of the metaprogramming
    library in `include/stdexec/__detail/__meta.hpp` that are used outside
    the module's purview, export them with `STDEXEC_MODULE_EXPORT_META`;
    this is a stopgap that will need to be changed eventually. Similarly,
    for logically-module-private symbols that are used in the tests or in
    the `include/exec` directory, export them with
    `STDEXEC_MODULE_EXPORT_AUTHORING`.

* In tests, the include order should be:
  ```c++
  /* Copyright... */

  // Catch2 must come before any potential import statements
  #include <catch2/catch_all.hpp>

  // Pull in all the macros in __config.hpp and either include all of
  // stdexec's headers, or import stdexec, as appropriate
  #include <stdexec/execution.hpp>

  // other non-std headers, like anything in <exec/...> or in the test
  // utility library
  #include <exec/function.hpp>
  #include <test_common/receivers.hpp>

  // if the test has direct dependencies on std, the declarations thereof
  // must come last, either as import std, or the usual includes. This has
  // to come last because current compilers support including std headers
  // *before* import std, but not *after*.
  #if STDEXEC_USE_MODULES()
  import std;
  #else
  #  include <algorithm>
  #endif
  ```

## Writing a sender algorithm with the facade

An algorithm's completions, its completion signatures, and its completion
attributes (scheduler, domain, behavior) must agree. Written separately, they
drift apart. The sender facade in `include/stdexec/__detail/__sender_facade.hpp`
derives the signatures and the attributes from the code that performs the
completions, so there is only one thing to get right.

These algorithms use the facade, and are good examples:

| Algorithm | File | Shows |
|---|---|---|
| `just`, `just_error`, `just_stopped` | `__just.hpp` | completing from `start` |
| `then`, `upon_error`, `upon_stopped` | `__upon.hpp` | transforming a child's completion |
| `read_env` | `__read_env.hpp` | completions that depend on the environment |
| `continues_on` | `__continues_on.hpp` | a nested operation, saved results |
| `when_all` | `__when_all.hpp` | several children, stop tokens, the last to arrive |
| `when_all_with_variant`, `__transfer_when_all` | `__when_all.hpp` | senders that are lowered to other senders |
| `let_value`, `let_error`, `let_stopped` | `__let.hpp` | nested senders created at run time |
| `__sequence` | `__sequence.hpp` | connecting children in the state |
| `starts_on` | `__starts_on.hpp` | a child started by a nested operation |
| `on` | `__on.hpp` | a sender lowered to others in its environment |
| `schedule_from` | `__schedule_from.hpp` | nothing but the defaults (and the child's environment) |
| `bulk`, `bulk_chunked`, `bulk_unchunked` | `__bulk.hpp` | one algorithm for several tags |
| `exec::repeat_n`, `exec::repeat_until`, `exec::repeat` | `exec/detail/repeat.hpp` | loops |
| `exec::split`, `exec::ensure_started` | `exec/detail/shared.hpp` | a shared operation; a sender that describes another |

### The shape of an algorithm

An algorithm is a class that derives from `__facade::__defaults` and overrides
the hooks it needs. Its senders are `__facade::__sender<Tag, Data, Child...>`s,
and the specialization of `__facade::__algorithm<Tag>` for the algorithm's tag
gives them their behavior. Here is all of `then`:

```c++
template <class _AlgoTag, class _SetTag>
struct __algo : __facade::__defaults
{
  // A completion of the child (the 0-th child, __child<0>)
  template <class _State, class _Tag, class... _Args>
  static constexpr auto
  __complete(__facade::__child<0>, _State& __state, _Tag __tag, _Args&&... __args) noexcept
  {
    if constexpr (__same_as<_Tag, _SetTag>)
      return __facade::__set_value_from<_AlgoTag>(__state.__rcvr_,
                                                  static_cast<_State&&>(__state).__data_,
                                                  static_cast<_Args&&>(__args)...);
    else
      return __facade::__send(__state.__rcvr_, __tag, static_cast<_Args&&>(__args)...);
  }
};

template <>
struct __facade::__algorithm<then_t> : __upon::__algo<then_t, set_value_t>
{};

// A sender of then:
auto sndr = __facade::__sender{then_t(), fn, child};
```

The defaults store the sender's data in the state (as `__data_`), connect the
children, start them all from `start`, and forward every completion.

A `__facade::__sender` is a product type of its tag, its data, and its
children, like the senders of the standard algorithms: `auto& [tag, data,
child] = sndr;` works, and so do `tag_of_t` and `transform_sender`. To give
its senders a name, an algorithm can derive a type from `__facade::__sender`,
as `then_t::__sender` does. The algorithm's hooks see such a sender as the
`__facade::__sender` it derives from.

### Handlers return effect tokens

The two handlers are where an algorithm does its work:

* `__start(state, children)` runs when the operation starts. The default
  starts every child with `children.__start_all()`.
* `__complete(key, state, tag, args...)` runs when the child or nested
  operation identified by `key` completes with `tag(args...)`. The default
  forwards the completion.

A handler returns an *effect token*, a type that records what the handler did
synchronously. The facade walks the tokens at compile time, starting from
`__start`, to find every completion and where it happens. So a handler must
perform each of its effects through an operation that returns a token, and
return all of those tokens:

* `__facade::__send(state.__rcvr_, tag, args...)` is the only way to
  complete. It sends the completion to the receiver, and returns a token for
  it. Call it (and the other functions here) with a qualified name, so that
  argument-dependent lookup does not search the receiver's namespaces.
* `__facade::__start_nested<Key>(op)` starts the nested operation `op`, which
  `Key` identifies, and returns `__starts<Key>`.
* When control flow takes paths that do different things, join them with
  `__choose(cond, then_fn, else_fn)`, `__when(cond, fn)`, `__visit(variant,
  fn)`, or `__set_value_from<AlgoTag>(rcvr, fn, args...)`. Their tokens are the
  union of their paths' tokens. A token converts to any token for more effects,
  and `__facade::__effects<>{}` is the token for doing nothing.

A token that is never returned is an effect that the facade never sees. The
signatures then omit a completion that the algorithm performs.

### Keys

Every operation that an algorithm runs has a key, which is a type:

* The `I`-th child is `__facade::__child<I>`.
* The algorithm names its other (nested) operations with key types of its
  own, such as `continues_on`'s `__sched_op`, and says what each one's sender
  is with `__nested_sender(key, sndr, env...)`. Return the nested sender itself
  if it can be made from the algorithm's sender, as
  `schedule(__facade::__get_data(sndr))`. Attribute queries
  can then ask it for its own attributes. If the sender is made only at run
  time, as for `let_value`, return `__mtype<Sender>()`. The facade
  then knows only its type, and none of its runtime attributes.
* A nested sender's attributes are those of the sender. To use others, define
  `__nested_attrs(key, sndr, env...)`. For example, a schedule operation's
  scheduler knows better where it completes than its schedule sender does, so
  `continues_on` and `starts_on` return
  `__facade::__schedule_attrs<Scheduler>`.

The hooks that run at compile time and during attribute queries are given
the algorithm's sender, as a `__facade::__sender const &`.
`__facade::__get_data(sndr)` is its data, and `__facade::__get_child<I>(sndr)`
its `I`-th child. (Both keep the sender's value category, so an operation state
can use them to move the parts out of an rvalue sender.)

### The operation state

`__get_state(sndr, rcvr)` makes the state. The state must have a member
`__rcvr_`, the receiver. The default makes a `__facade::__basic_state`, which
holds the receiver and the data. An
algorithm with nested operations makes its own state, and keeps their
operation states in it:

* Connect a nested operation to a `__facade::__receiver_t<Algo, Key, State,
  Sender, Receiver>`, which calls `Algo::__complete(Key(), ...)`. Construct it
  from a pointer to the state. In the walk, where `Receiver` is an archetype,
  this is a receiver that does nothing, so analyzing the algorithm does not
  instantiate the code that would complete it. (A CUDA stream operation
  completes a receiver that is not a stream receiver from a kernel. Were that
  receiver the facade's, the walk would make the algorithm's handlers device
  code.)
* If the state's type depends on the completions of a child or a nested
  operation, for example because it stores their results, compute those
  completions with `__facade::__completions_of_t<Algo, Key, Sender,
  Receiver>`. When there is no environment, the walk asks the children for
  their completions without one. This type does the same, so the state agrees
  with the walk.
* If the algorithm cannot work with the sender and receiver it is given, have
  `__get_state` return an `__mexception` type instead of a state. The walk
  reports that error. (`when_all` does this for children with more than one
  value completion.)
* The facade connects the children, outside the state, and `__start` can
  start them. Their operations live as long as the algorithm's. If the
  algorithm needs to connect a child itself, set `static constexpr bool
  __connects_children = false;`, and connect it in the state with
  `__receiver_t<..., __child<I>, ...>`. This is necessary to connect a child
  later (as `__sequence` does), to start it from a nested operation's
  completion (`starts_on`), or to destroy its operation before the algorithm
  completes (`let_value` puts the child's operation in a variant, where the
  operation of the sender that the function returns replaces it).

An algorithm can start an operation again from that operation's own completion
(or from the completion of another operation that it started), which is a
loop. `exec::repeat_n` connects its child anew for each iteration, and starts
it from a trampoline scheduler's schedule operation, which it starts from the
child's value completion. See below for what the walk does with loops.

A nested operation need not be one that the algorithm connects. The operations
of `exec::split` share one operation, which whichever of them starts first
starts. Each waits for it with the key `__shared_op`: when the shared operation
completes, it calls `__complete(__shared_op(), ...)` for each waiting operation
(through a function pointer that each one registers). `__nested_sender` gives
the walk a sender whose completions are the results as the waiting operations
receive them, and whose attributes are the child's. They say the shared
operation completes asynchronously, because it completes asynchronously to an
operation that waits for it, even when it completes where it started: whichever
operation started it may not be this one. An operation that starts after the
shared operation has completed sends the results from `__start`, so its
completions happen where it starts.

The child's attributes are computed when the shared state is made, because
connecting the child consumes it: its completion schedulers are stored, and its
completion domains are those its attributes give *directly*. (Asking
`get_completion_domain` would also give the domains it falls back on, such as
`default_domain` for a channel that the child says nothing about. The facade
falls back on the child's dispatch domain instead, which is the domain that
implements the child.)

Mind the order of the state's members. They are initialized in declaration
order, so an operation that inspects a child must come before the operation
that consumes the child.

### Environments

`__env_for(key, sndr, env)` gives the environment of the operation that `key`
identifies. The walk uses it for the operation's completions, and attribute
queries use it for its attributes. At run time, `__get_env(key, state)` gives
the same environment, from the state, and the two must have the *same type*.
A `static_assert` in the facade's receivers checks this. So:

* Forward `env` with its value category, as in `static_cast<Env&&>(env)`; do
  not decay it. The receiver's environment may be a reference, or not
  copyable.
* A `__get_env` that returns `__fwd_env(get_env(state.__rcvr_))` must
  return `decltype(auto)`, as `__env_for` returns `__fwd_env_t<Env>`. The
  receiver's environment may be a reference to a forwarding environment, which
  `__fwd_env` returns as it is.
* Build the environment from the same parts in both. For example,
  `continues_on` stores its secondary environment in the state, and
  `__get_env` joins a copy of it with the receiver's environment (with the
  receiver's start scheduler and domain hidden, unless the child stays put),
  just as `__env_for` joins a freshly made one.

The defaults give every operation the receiver's environment, unchanged.

### Type errors

The walk instantiates the handlers with every completion that their children
and nested operations can send. So a handler must be well-formed for all of
them, even ones that are errors:

* Guard type-dependent code with `if constexpr`, and in the bad branch return
  `__facade::__type_error(__mexception<...>())`. The walk reports the error the
  way stdexec reports any other: by throwing, when constexpr exceptions are
  available, and otherwise by returning it as the completion signatures.
* `__facade::__decay_copy_error<AlgoTag, Args...>()` is the error for results
  that the algorithm must store but cannot decay-copy.
* Computing a deduced return type instantiates the function's body, and the
  bodies of the nested receivers' member functions that it uses. So code that
  is reachable from a handler must not `static_assert` about the types it
  handles. Let the walk report the problem instead. (A sender whose walk
  fails cannot be connected, so the bad branch never runs.)
* The state's type must be well-formed even for inputs that the handlers
  reject, or a hard error hides the handler's diagnostic. For example, if a
  function returns something that cannot be connected, `let_value`'s state
  holds a placeholder in place of its operation, and the handler reports
  `_FUNCTION_DOES_NOT_RETURN_A_SENDER_`.
* A handler that connects a sender made at run time must check that it can.
  If the sender does not type-check in its environment, return the token that
  starts it anyway (without connecting it): the walk then asks the sender for
  its completions, and reports its error. (`let_value` does this for a
  function that returns `nvexec::get_stream()`, which type-checks only in a
  stream algorithm's environment.)

The receivers of nested operations have member functions that are not
`constexpr`. Clang instantiates `constexpr` functions eagerly, and when a
nested operation type-erases its receiver, that happens while the state is
still incomplete.

### Environment dependence

A sender is *dependent* if its completions depend on its environment. A
dependent child makes its parent dependent without anything further. If the
algorithm's own completions depend on the environment, as `read_env`'s and
`on`'s do, define

```c++
template <class Sender>
static constexpr bool __depends_on_env = true;
```

Without an environment, the walk then reports the sender as dependent. (It
checks the children first, so a child's type error is still reported when the
sender is made.)

### Attributes

The facade derives the completion attributes from where each completion
happens. Nothing needs to be written for them, but the rules are worth
knowing:

* A channel's *behavior* joins the behaviors of its completions. A completion
  that happens inside a nested operation's completion is sequenced after it:
  it is inline only if the nested operation completes inline, and the path
  that started the nested operation is inline too.
* A completion's *domain* is the domain where it happens. If a channel can
  complete in several domains, the channel's domain is their common type, if
  they have one (when one domain derives from the others), and otherwise
  `indeterminate_domain<Domains...>`.
* A completion *scheduler* is a runtime value. A channel reports one only if
  all its completions get their scheduler from the same place. Completions of
  different kinds of a nested facade sender get it from the same place if they
  get it from the same place in that sender. (When `fn` may throw,
  `then(schedule(sch), fn)` completes with a value or an error on `sch`, so
  `upon_error` over it completes with a value on `sch`, however it completes.)
  A sender that is not a facade sender can say that its attributes report the
  same scheduler for every kind of completion, with
  `static constexpr bool __one_completion_scheduler = true;` in the type of its
  attributes. A schedule sender whose operation completes on its scheduler,
  however it completes, can; `run_loop`'s and `exec::static_thread_pool`'s do.
* A completion that happens inside a nested operation completes where that
  operation was started, if the operation completes where it starts and where
  it was started is known. (So completions that start in the same place agree
  on their scheduler, even when they come from different operations.)
  Otherwise it completes where the operation's own scheduler says it does,
  with one exception. An operation that completes where it starts, in an
  environment that does not say where that is, can only name a scheduler
  relative to wherever it started: `inline_scheduler` names itself, which the
  one who asks takes to mean its own start scheduler. That is believed only of
  an operation that was started where the whole operation started. If
  nothing says where, the completion happens in the operation's dispatch
  domain (see below): the domain that implements the operation can complete it
  in ways that its sender's attributes do not mention. (nvexec adds
  `set_error_t(cudaError_t)` completions, and `split` a `set_stopped_t()`
  completion, which happen on the stream.)
* A loop's iterations can start, and so complete, in different places: an
  iteration starts where the one before it ended. But a place has few
  attributes (a behavior, a domain, and where its scheduler comes from), and
  the attributes of an operation's completions depend only on those of the
  place where it starts. So the walk follows a loop until an iteration starts
  an operation in a place where an earlier iteration on the same path started
  it, and then stops: the later iterations complete as earlier ones did. (In
  `repeat_n(schedule(pool) | then(f), n)`, the trampoline's first iteration
  can stop on the start scheduler, and its later ones on the pool, so the
  stopped channel has no completion scheduler. Stopping after the first
  iteration would report the start scheduler.) An algorithm that loops should
  also give the loop's operations an environment that is true of every
  iteration: `repeat_n` hides the receiver's start scheduler and domain from
  its child and its trampoline unless the child completes where it starts.
* Likewise, an operation that is started from a sender's completions sees the
  receiver's start scheduler only if it starts there: if the sender completes
  where it starts, however it completes (`__stays_put`). Otherwise the
  receiver's start scheduler and domain are hidden from it. `continues_on`
  starts its schedule operation from every completion of its child, so it
  then sees the child's completion scheduler if the child completes in only
  one way, and otherwise the domain of the child's completions. (Otherwise
  `continues_on(sndr, inline_scheduler{})` would claim to complete on the start
  scheduler, wherever `sndr` completes.)
* Attributes are available only if the walk succeeds. So a dependent sender
  has no completion attributes when it is queried without an environment.
* A channel on which nothing completes has no attributes, unless the algorithm
  names the provenance its completions would have, with `template <class Tag>
  using __empty_channel_source = ...;` (void for none). `continues_on` and
  `starts_on` name their schedule operation's value completion for
  `set_value_t`, so they report the scheduler's attributes even for a child
  that never completes with a value, as the standard's `SCHED-ATTRS` does.
* `get_completion_domain<>()` (with no tag) gives the domain in which to look
  for customizations of the algorithm. By default it is not answered, and
  `transform_sender` uses the domain of the value completions. To use another
  completion's domain, name it with `using __dispatch_source = __via<Key,
  Tag, __from_start>;` (`let_value` names the child's value completion). If
  that operation never completes that way, but its attributes say where it
  would (as `continues_on`'s do, following the standard's `SCHED-ATTRS`), the
  algorithm is dispatched there. (`exec::repeat_n` names its child's value
  completion, so `repeat_n(just_error(e) | continues_on(sch), n)` is
  customized in `sch`'s domain.)
  Unlike the other completion queries, it does not need the walk to succeed:
  a customization can be what makes the sender type-check (for example, by
  giving a child a different environment).
* Other forwarding queries are answered by the sender that `__fwd_sender(sndr)`
  returns: by default the child, if there is exactly one.

An algorithm whose senders' completion signatures or attributes cannot be
derived can give them itself, with the hooks
`__get_completion_signatures<Sender, Env...>()` and `__get_attrs(sndr)`. The
facade uses a hook if the algorithm has one that accepts the sender, so a hook
can be constrained to the senders it is for. Use these only as a last resort:
the reason for the facade is that it derives these.

### Hook reference

Every hook is optional. An algorithm overrides a hook by declaring one of the
same name in its class, which hides the one in `__facade::__defaults`. Hooks
marked "opt-in" have no default, and the facade looks for them. Most
algorithms need only `__complete`, and perhaps `__start`.

| Hook | Default | Override it when | Example |
|---|---|---|---|
| `__complete(key, state, tag, args...)` | forwards the completion to the receiver | the algorithm transforms, saves, or counts a completion of a child or nested operation, or starts something from it | `then`, `when_all`, `continues_on` |
| `__start(state, children)` | starts every child with `children.__start_all()` | the algorithm completes from `start`, starts only some of its children, or starts a nested operation | `just`, `let_value`, `repeat_n` |
| `__get_state(sndr, rcvr)` | a `__basic_state` that holds the receiver and the sender's data | the state holds more than the data (nested operations, saved results, a stop source), or the algorithm rejects its inputs by returning an `__mexception` | `when_all`, `continues_on`, `let_value` |
| `__get_state(sndr, rcvr, __subscribing_t)` | the same as the above | the state of a subscribed (sequence) sender differs from that of a connected one | |
| `__state_type<Sender, Receiver, Subscribing...>` (opt-in) | the type that `__get_state` returns | the state holds nested operations. The walk then gets the state's type without asking whether making the state can throw, which would instantiate the operations below it. It must name the type that `__get_state` returns, or its `__mexception` | `let_value`, `__sequence`, `exec::ignore_all_values` |
| `__connects_children` | `true` | the algorithm connects a child itself, in its state: to connect it later, to start it from a nested operation, or to destroy its operation early | `__sequence`, `starts_on`, `let_value` |
| `__connect(key, state, sndr, rcvr)` | `STDEXEC::connect(sndr, rcvr)` | a child or nested operation must be connected to a different kind of receiver, as when a domain adapts a child whose operation completes from a kernel | nvexec |
| `__receiver_base` | an empty struct | a domain's algorithm needs the facade's receivers to have a base class, so that its senders complete them in the way they expect | nvexec |
| `__nested_sender(key, sndr, env...)` | for `__child<I>`, the `I`-th child | the algorithm has nested operations of its own, to say what their senders are (the sender itself, or `__mtype<Sender>()` if it is made at run time) | `continues_on`, `let_value`, `exec::split` |
| `__nested_attrs(key, sndr, env...)` (opt-in) | the nested sender's attributes | a nested operation knows better where it completes than its sender's attributes say | `continues_on`, `starts_on` (`__schedule_attrs`) |
| `__env_for(key, sndr, env)` | the receiver's environment, unchanged | the algorithm gives a child or nested operation an environment of its own (it must forward `env` with its value category) | `read_env`, `continues_on`, `repeat_n`, `on` |
| `__get_env(key, state)` | the receiver's environment, at run time | the algorithm overrides `__env_for`. It must return the same *type* as `__env_for` | the same |
| `__fwd_sender(sndr)` | the child, if there is exactly one | the other forwarding queries, which are not completion queries, should be answered by a sender other than the only child, or when there is more than one child | |
| `__depends_on_env<Sender>` | `false` | the algorithm's own completions depend on the environment (dependence through a child needs nothing) | `read_env`, `on` |
| `__dispatch_source` | `void`, the domain of the value completions | `transform_sender` should look for customizations in the domain of another completion | `let_value`, `exec::repeat_n` |
| `__empty_channel_source<Tag>` | `void`, no attributes for a channel that never completes | the algorithm reports attributes for a channel on which nothing completes, as the standard's `SCHED-ATTRS` does | `continues_on`, `starts_on` |
| `__item_source<Sender>` | `void` | one child's items are the algorithm's items, which makes the sender a sequence sender when that child is one | |
| `__subscribe(key, state, sndr, rcvr)` | transforms the sender, then subscribes it | the item source must be subscribed in a different way | |
| `__get_completion_signatures<Sender, Env...>()` (opt-in) | the signatures that the walk derives | the signatures cannot be derived, and only as a last resort. Constrain the hook to the senders it is for | `on`, `when_all_with_variant`, `__sequence` |
| `__get_attrs(sndr)` (opt-in) | the attributes that the facade derives | the same, for the attributes | `on`, `__sequence`, `starts_on` |
| `__retags_facade_senders` (a domain, opt-in) | not set | a domain customizes facade algorithms by re-tagging their senders, so that the facade answers completion queries as the re-tagged sender would | nvexec's `stream_domain` |

Choose the least powerful hook that does the job. A hook that only changes
what an algorithm does with a completion (`__complete`, `__start`) leaves
everything else for the facade to derive. A hook that changes the algorithm's
operations (`__get_state`, `__connect`, `__connects_children`) must still
keep every effect in the tokens the handlers return. The last two hooks,
`__get_completion_signatures` and `__get_attrs`, replace what the facade
derives, and so are what lets an algorithm's completions drift away from its
signatures again.

### Senders that describe other senders

`exec::split(sndr)` makes a sender for its domain to customize, and the sender
that it returns by default is a different one (it holds the shared state). The
first sender is a facade sender too, whose algorithm (`__shared::__describe_algo`)
derives from the second's, so it has the same completions and attributes, and
customizations are found where its child completes. Its nested senders are
known only by their types, because its child is not connected, and it cannot be
connected. Its state, too, is only a type: the state of the sender it describes
would connect the child, and the walk would then instantiate the code that
completes the child, which can be device code.

nvexec's `stream_domain` customizes `split` and `ensure_started` by making
facade senders of its own (`_strm::split_t` and `_strm::ensure_started_t`), on
`exec::__shared::__shared_attrs`. In the stream model, an operation receives the
results on its own stream, after the stream waits for them, whether the shared
operation completes before the operation starts or after. So both cases go
through the shared operation's key, and its attributes are the child's.

### Senders that are lowered to other senders

Some senders exist to be customized, and are otherwise lowered to other senders
by `transform_sender`: `when_all_with_variant(sndrs...)` becomes
`when_all(into_variant(sndrs)...)`, `__transfer_when_all(sch, sndrs...)` becomes
`continues_on(when_all(sndrs...), sch)`,
`__transfer_when_all_with_variant(sch, sndrs...)` becomes
`__transfer_when_all(sch, into_variant(sndrs)...)`, and `on(sch, sndr)` becomes
`continues_on(starts_on(sch, sndr), orig_sch)`, where `orig_sch` is the
scheduler that it returns to. Such a sender is a facade sender with the
attributes of the sender it is lowered to:

* If nothing fixes the sender's parts, they are built when the sender is, as
  the parts of the sender it is lowered to, and the algorithm is that sender's.
  `__transfer_when_all(sch, sndrs...)` has the scheduler and the child of
  `continues_on(when_all(sndrs...), sch)`, and `continues_on`'s algorithm, so
  lowering it is re-tagging it.
* The standard fixes `when_all_with_variant`'s parts: its children are the
  senders it is given. So `when_all`'s algorithm does for a
  `when_all_with_variant` what `when_all(into_variant(sndrs)...)` does: it saves
  each child's values in a variant of tuples. It also has the channels of that
  sender: an `exception_ptr` error, which `into_variant` always declares, and a
  stop source whenever there are two children or more (since `into_variant`'s
  error means that a child can fail). Its completion signatures are the lowered
  sender's (with the `__get_completion_signatures` hook), so that they are
  exactly what connecting it produces.
* The sender that `on` is lowered to depends on the environment, which says
  where `on` returns to, so `on` cannot be built as that sender. It gives that
  sender's completion signatures and attributes with the
  `__get_completion_signatures` and `__get_attrs` hooks. For an attribute
  query, it makes that sender in the query's environment, around an
  `__on::__proxy` for what it keeps, so that nothing is copied. (The proxy is
  never connected, but it says what it would connect to: the walk computes the
  operation types of the senders that it analyzes, as when an algorithm
  connects a child in its state.) `on(sch, sndr)` keeps `starts_on(sch, sndr)`,
  made when the sender is, so that it is not made again for each query or
  when the sender is lowered. Its structured bindings are still the
  standard's, `[on_t, sch, sndr]`, by the tuple protocol (see `__on::__sndr`).
  `on(sndr, sch, clsur)` cannot keep `clsur(continues_on(sndr, sch))`:
  applying the closure consumes `sndr` and the closure, which its structured
  bindings must produce, and the scheduler that it returns to is where `sndr`
  completes.

### Things the facade cannot do

The facade cannot connect a sequence sender, so `__sequence` and `starts_on`
cannot run one as a child. When its last child is a sequence sender,
`__sequence` lowers itself to `__seq::__sndr` with `transform_sender`, and
`starts_on` lowers itself to a `__sequence`. For those senders, the algorithms
give the completion signatures of the sender they are lowered to, and
attributes of their own, with the `__get_completion_signatures` and
`__get_attrs` hooks (see above).

### Customizing a facade algorithm in a domain

A domain customizes an algorithm with `transform_sender`. To customize a facade
algorithm with another facade algorithm, *re-tag* the sender: transform
`__facade::__sender<Tag, Data, Child...>` into `__facade::__sender<Tag2, Data,
Child...>`, moving the data and the children, and specialize
`__facade::__algorithm<Tag2>`. (nvexec customizes all of the facade algorithms
it supports this way.)

A sender's completion signatures in an environment are those of the sender as
its domain customizes it there (`get_completion_signatures` transforms the
sender first), so a parent sees the completions of its children as their
domains customize them. A domain that customizes facade algorithms by
re-tagging says so with `static constexpr bool __retags_facade_senders =
true;` (nvexec's `stream_domain` does), and then the facade makes the
attributes agree: in an environment in which such a domain re-tags a sender,
the sender's completion queries are answered as the re-tagged sender's are,
computed with `Tag2`'s algorithm from the sender as it is, without
transforming it. So `sync_wait(sndr | exec::repeat_n(n))`, for a `sndr` that
completes on a CUDA stream, sees that `repeat_n` completes there, as nvexec's
`repeat_n` does, even though `exec::repeat_n` would complete inline in
`start` when `n` is 0. And a parent sees its children's attributes as
customized, even for a child that type-checks only once it is customized
(`starts_on(sch, nvexec::get_stream())`).

The exception is `get_completion_domain<>()`, which decides the
customization: it is the sender's own, and the facade always answers it when
it can (with the dispatch domain, or else the value completion domain), so
that asking it never asks how the sender is customized. The facade asks only
the domains that say they re-tag: one that decides how to customize a sender
by asking the sender for its own attributes (as `task_scheduler_domain` does
for `bulk`) would be asked again, and asking every domain would cost every
attribute query a transformation. So a domain that re-tags must decide from
the sender's children and data, not from the sender's attributes. Other
customizations leave the attributes those of the uncustomized sender.

Without an environment, nothing is customized, so a sender that a re-tagging
domain would customize could answer differently with an environment and
without one. And the query functions ask again without an environment when a
sender gives no answer with one, so they would mix the customized sender's
answers with the uncustomized sender's. (`stdexec::on` over a CUDA stream
scheduler hit this: with the environment, nvexec's `schedule_from` reports
that its errors may happen in two domains and on no one scheduler, because a
CUDA error from `start` completes inline; without it, the uncustomized
`schedule_from` reported the stream scheduler, and `get_completion_domain`'s
consistency check failed.) So such a sender (one whose customizing domain
without an environment re-tags, and which that domain changes in an empty
environment) gives no completion answers without an environment. A sender that
the domain has customized already, which it leaves as it is, still answers.

A re-tagged algorithm's hooks must work with the uncustomized sender type as
well as the customized one: use `__facade::__get_data`,
`__facade::__get_child`, and the traits, not the tag.

An algorithm whose receivers must be of a particular kind can name their base
with `using __receiver_base = ...;`. nvexec's facade algorithms derive from
`_strm::stream_algorithm`, which names `stream_receiver_base`, so that stream
operations complete them on the host, with the work enqueued on the stream.

An algorithm can also adapt how its children are connected, with the hook
`__connect(key, state, sndr, rcvr)` (by default, `STDEXEC::connect(sndr,
rcvr)`). The facade connects the children with it, and a state that connects
operations itself can call it too. `stream_algorithm` uses it for a child that
would complete its receiver from a kernel: one that completes on a stream
scheduler without being a stream sender or completing inline. (Such a sender
wraps a stream sender with a receiver of its own.) It connects that child
through `_strm::enqueue_opstate`, which receives the completion on the device,
stores it, and enqueues a task that completes the algorithm's receiver on the
host. Any other child completes the receiver on the host, and is connected
directly. (Enqueueing a child that completes on the host, as a CPU scheduler's
sender does, would crash: the enqueue receiver is device code.)

Where a completion happens matters to the attributes, so a stream algorithm
reports its CUDA errors (a stream that could not be created, a failed
allocation or kernel launch) with `_strm::_report_error(state, status)`. That
attributes the error to the nested key `_strm::stream_error`, a sender known
only by its type, whose errors complete in the stream domain. (Sent from the
handler that detects them, they would be attributed to wherever the algorithm
started, or to its child.) Likewise, a stream algorithm that stops other than
by passing on a completion reports it with `_strm::_report_stopped(state)`,
which attributes it to `_strm::stream_stopped`: `split` when a receiver asks an
operation to stop, and `ensure_started` when its shared operation was stopped
before the operation starts. `continues_on` instead sends its errors from its
schedule operation, like its other completions, so that every completion it
makes has the scheduler's attributes.

nvexec's algorithms with one child (`then`, `upon_error`, `upon_stopped`,
`bulk`, `schedule_from`, and nvexec's own `launch`, `reduce` and `nvtx`) derive
from `_strm::stream_adaptor<Algo>` (in `nvexec/stream/adaptor.cuh`). It gives
the state (`adaptor_state`: the stream, the algorithm's data, and temporary
storage), the child's environment (sharing the stream), the check of the stream
in `__start`, and the passing on of the completions that the algorithm does not
handle. `_strm::_invoke_on_stream(state, fn, args...)` calls a function in a
kernel and sends its result.

A completion scheduler is reported only if all of a channel's completions get
it from the same place, so algorithms whose values come from two places (as
`upon_stopped`'s come from its child's values and from its function, when the
child is stopped, unless the child gets the schedulers of both from the same
place) report none, and `when_all` reports none for its values. Where
an nvexec algorithm needs a stream scheduler for its child (to know its stream),
such a child must be piped through `write_attrs` to proclaim one, as after
`let_value`. (`schedule_from` needs none, because such a sender, like `when_all`,
completes a receiver that is not a stream receiver on the host. So nvexec's
`schedule_from` leaves such a sender alone.)

nvexec's `repeat_n` starts each iteration with a schedule operation on the
stream (which the old implementation's `starts_on(sch, child)` also did), and
ends the loop there when no iterations are left. So all of its value
completions come from that operation, and it reports the stream scheduler as
its value completion scheduler, which nvexec's `sync_wait` needs. (Its child's
value completions come from elsewhere, so completing from them would leave no
scheduler that all the value completions share.)
