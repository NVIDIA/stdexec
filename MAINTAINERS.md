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

* Regarding the modularized build:
  * Enable the modularized build with `-DSTDEXEC_BUILD_MODULES=1` at
    configure time. It has so far only been tested with Clang 22.
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
