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

Configure and build stdexec following the build instructions in the top level
[README.md](../../README.md). There are a couple relacy specific build and ctest
targets, though they are part of the standard build and ctest and will be run
automatically if cmake is configured with `-DSTDEXEC_BUILD_RELACY_TESTS=1`.
`STDEXEC_BUILD_RELACY_TESTS` is set by default for GCC today.

Run the following on a Linux machine with GCC as the toolchain.

```
mkdir build && cd build
cmake ..
make relacy-tests -j 4
ctest -R relacy # Run all relacy tests
./test/rrd/sync_wait # Run a specific relacy test directly
```

## Recommended use

The Relacy tests can be manually built and executed. New tests can be written for
new algorithms, or new use cases in the STDEXEC library.

At this time, integrating the tests into CI is not yet recommended. If we can figure
out a more stable build on all environments/compilers, we should revisit this.

## Supported platforms

The STDEXEC Relacy tests have been verified to build and run on
 * Linux based GCC+12-14 with libstdc++ (`x86_64`)
 * Mac with Apple Clang 15 and 17 with libc++ (`x86_64`)

## Caveat

Relacy relies on a less than robust approach to implement its runtime: it replaces
std:: names with its own versions, for example, std::atomic and std::mutex, as well
as pthread_* APIs. As libstdc++/libc++ evolve, newer versions may not be compatible with
Relacy. In these cases, changes to Relacy are needed to correctly intercept and replace
std:: names.

When the compilers and standard libraries release new versions, we will need to test the
new versions can compile the stdexec Relacy tests before enabling the new compiler.
