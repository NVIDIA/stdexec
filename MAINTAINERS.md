# Maintainers' Guide

This is a place to put design rationale and code idioms that maintainers of
stdexec should follow.

## ADL isolation

Stdexec makes use of the [`tag_invoke`](http://wg21.link/p1895) mechanism to
support end-user customization of algorithms. For the sake of compile-time
performance, this mechanism requires care to ensure that overload sets stay
small.

### Hidden friend functions

Define `tag_invoke` overloads for class types _in-situ_ as hidden friend
functions. Keep in mind that when calling a customizable funtion that
all the associated entities (classes and namespaces) of all the function's
arguments are searched for `tag_invoke` overloads. That means that for
a function call like:

```c++
connect(sender, receiver);
```

... the hidden friend functions of the _receiver_ will be considered in
addition to those of the sender. This can sometimes lead to surprising and
confusing compiler errors.

### Class template parameters

For a class template instantiation such as `N::S<A,B,C>`, the associated
entities include the associated entities of `A`, `B`, and `C`. This is
pretty much never what you want, and in a combinator library like stdexec,
it causes the set of associated entities to grow algorithmically as types
are composed.

To avoid that problem, we take advantage of a curious property of nested
classes: they don't inherit the associated entities of the template
parameters of the enclosing template. To illustrate, a class type such
as `N::S<A,B,C>::T` does _not_ inherit the associated entities of `A`,
`B`, or `C`.

Stdexec provides some utilities that bundle up that technique, but it
requires certain rules to be followed to get the full benefit. Rather
than defining a sender adaptor as:

```c++
template <class Sender, class Arg>
  struct my_sender {
    Sender sndr_;
    Arg arg_;
    // ... rest of sender implementation
  };
```

we define it as follows:

```c++
template <class SenderId, class Arg>
  struct my_sender_id {
    using Sender = stdexec::__t<SenderId>;

    struct __t {
      using __id = my_sender_id;
      Sender sndr_;
      Arg arg_;
      // ... rest of sender implementation
    };
  };

template <class Sender, class Arg>
  using my_sender =
    stdexec::__t<my_sender_id<stdexec::__id<Sender>, Arg>>;
```

Note that we use `stdexec::__id` to "encode" a type into an identifier
before passing it as a template argument. And we use `stdexec::__t`
to get back the original type from the identifier.

> **Note**
>
> We only really need to encode the type of the template arguments that
> are likely to have `tag_invoke` overloads that we want to exclude when
> looking for overloads for _this_ type. Hence, the `Sender` type is
> encoded but the `Arg` type is left alone. 

Additionally, we move the implementation of the type into a (non-template)
nested class type called `__t`. This nested class type must have a nested
type alias `__id` that is an alias for the enclosing class template.

When these guidelines are followed, we can ensure that the minimum number
of class templates are instantiated, and the types of composite senders
remains short, uncluttered, and readable.

## Assorted tips and tricks

* Data, including downstream receivers, are best stored in the operation
  state. Receivers themselves should, in general, store nothing but a
  pointer back to the operation state.
* Assume that schedulers and receivers contain nothing but a pointer and
  are cheap to copy. Take them by value.
* All `tag_invoke` overloads _must_ be constrained.
* In a sender adaptor, a reasonable way to constrain
  `tag_invoke(connect_t, ThisSender<InnerSender>, OuterReceiver)` is by
  requiring `sender_to<InnerSender, ThisReceiver<OuterReceiver>>`.
* Place concept checks on public interfaces. Only place concept checks
  on implementation details if needed for correctness; otherwise, leave
  them unconstrained. Use `static_assert` if you must, but don't bother
  rechecking things that were already checked at the public interface.
