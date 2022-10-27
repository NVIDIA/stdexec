# Senders - A Standard Model for Asynchronous Execution in C++

`stdexec` is an experimental reference implementation of the _Senders_ model of asynchronous programming proposed by [**P2300 - `std::execution`**](http://wg21.link/p2300) for adoption into the C++ Standard.

**Purpose of this Repository:** 
1. Provide a proof-of-concept implementation of the design proposed in [P2300](http://wg21.link/p2300).
2. Provide early access to developers looking to experiment with the Sender model.
3. Colloborate with those interested in participating or contributing to the design of P2300 (contributions welcome!).

## Disclaimer

`stdexec` is experimental in nature and subject to change without warning. 
The authors and NVIDIA do not guarantee that this code is fit for any purpose whatsoever.

[![CI](https://github.com/NVIDIA/stdexec/workflows/CI/badge.svg)](https://github.com/NVIDIA/stdexec/actions)

## Example

Below is a simple program that parallelizes some compute intensive work by executing it on a thread pool.

```c++
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <vector>
#include <print> // from C++23

extern int compute(int);

int main()
{
    // Declare a pool of 8 worker threads:
    exec::static_thread_pool pool(8);

    // Get a handle to the thread pool:
    auto sched = pool.get_scheduler();

    // Describe some work:
    auto fun = [](int i) { return compute(i); };
    auto work = stdexec::when_all(
        stdexec::on(sched, stdexec::just(0) | stdexec::then(fun)),
        stdexec::on(sched, stdexec::just(1) | stdexec::then(fun)),
        stdexec::on(sched, stdexec::just(2) | stdexec::then(fun))
    );

    // Launch the work and wait for the result:
    auto [i, j, k] = stdexec::sync_wait(std::move(work)).value();

    // Print the results:
    std::print("{}, {}, {}", i, j, k);
}
```

## Structure

This library is header-only, so all the source code can be found in the `include/` directory. The physical and logical structure of the code can be summarized by the following table:

| Kind | Path | Namespace |
|------|------|-----------|
| Things approved for the C++ standard | `<stdexec/...>` | `::stdexec` |
| Generic additions and extensions | `<exec/...>` | `::exec` |
| Vendor-specific extensions and customizations | <code>&lt;<i>(vendor)</i>exec/...&gt;</code> | <code>::<i>(vendor)</i>exec</code> |
| | |



## Building

The following tools are needed:

* [`CMake`](https://cmake.org/)
* GCC 11+ or clang 12+

Perform the following actions:

```bash
# Configure the project
cmake -S . -B build -G<gen>
# Build the project
cmake --build build
```

Here, `<gen>` can be `Ninja`, `"Unix Makefiles"`, `XCode`, `"Visual Studio 15 Win64"`, etc.

### Specifying the compiler

You can set the C++ compiler via `-D CMAKE_CXX_COMPILER`:

```bash
# Use GCC:
cmake -S . -B build/g++ -DCMAKE_CXX_COMPILER=$(which g++)
cmake --build build/g++

# Or clang:
cmake -S . -B build/clang++ -DCMAKE_CXX_COMPILER=$(which clang++)
cmake --build build/clang++
```

### Specifying the stdlib

If you want to use `libc++` with clang instead of `libstdc++`, you can specify the standard library as follows:

```bash
# Do the actual build
cmake -S . -B build/clang++ -G<gen> \
    -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
    -DCMAKE_CXX_COMPILER=$(which clang++)

cmake --build build/clang++
```
