# Maintainers' Guide

This is a place to put design rationale and code idioms that maintainers of
stdexec should follow.

* Data, including downstream receivers, are best stored in the operation
  state. Receivers themselves should, in general, store nothing but a
  pointer back to the operation state.

* Assume that schedulers and receivers contain nothing but a pointer and
  are cheap to copy. Take them by value.

* Do not use `tag_invoke` anywhere. It's deprecated.

* In a sender adaptor, a reasonable way to constrain
  `ThisSender<InnerSender>::connect(OuterReceiver)` is by
  requiring `sender_to<InnerSender, ThisReceiver<OuterReceiver>>`.

* Place concept checks on public interfaces. Only place concept checks
  on implementation details if needed for correctness; otherwise, leave
  them unconstrained. Use `static_assert` if you must, but don't bother
  rechecking things that were already checked at the public interface.
