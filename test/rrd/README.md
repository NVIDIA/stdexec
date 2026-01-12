## Relacy tests

[Relacy (RRD)](https://www.1024cores.net/home/relacy-race-detector/rrd-introduction)
is a data race detector. It replaces the OS scheduler with a scheduler that
explores many different thread interleavings, and logs detected races or assertion
failures. Relacy can also simulate relaxed hardware by simulating old values of a
variable as allowed by the C++11 memory model.

Relacy requires a specialized build. In particular, it is a header only library that
replaces the standard library and pthread APIs at compile time. Since it replaces some
standard library includes, writing new tests may require working around certain
limitations in terms of what the replacement headers and accompanying runtime can
support. For example, Relacy's atomic replacements cannot support `++x`, so the
STDEXEC library could needs to use `x.fetch_add(1)` to be compatible with Relacy.

## Instructions

Run the following commands from within this directory (`./tests/rrd`).

```
git clone -b STDEXEC https://github.com/dvyukov/relacy
CXX=g++-11 make -j 4
./build/split
./build/async_scope
```

## Recommended use

The Relacy tests can be manually built and executed. New tests can be written for
new algorithms, or new use cases in the STDEXEC library.

At this time, integrating the tests into CI is not yet recommended. If we can figure
out a more stable build on all environments/compilers, we should revisit this.

## Supported platforms

The STDEXEC Relacy tests have been verified to build and run on
 * Linux based GCC+11 with libstdc++ (`x86_64`)
 * Mac with Apple Clang 15 with libc++ (`x86_64`)

G++12 and newer are known to have issues that could be addressed with patches
to Relacy.
