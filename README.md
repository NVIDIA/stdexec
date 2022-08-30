# std::execution

`std::execution`, the proposed C++ framework for asynchronous and parallel programming.

You can see a rendered copy of the current draft [here](https://brycelelbach.github.io/wg21_p2300_std_execution/std_execution.html).

## Reference implementation

[![CI](https://github.com/brycelelbach/wg21_p2300_std_execution/workflows/CI/badge.svg)](https://github.com/brycelelbach/wg21_p2300_std_execution/actions)

### Building

The following tools are needed:

* [`CMake`](https://cmake.org/)
* GCC 11+ or clang 14+

Perform the following actions:

```bash
# Configure the project
cmake S . -B build -G<gen>
# Build the project
cmake --build build
```

Here, `<gen>` can be `Ninja`, `"Unix Makefiles"`, `XCode`, `"Visual Studio 15 Win64"`, etc.

#### Specifying the compiler

You can set the C++ compiler via `-D CMAKE_CXX_COMPILER`:

```bash
# Use GCC:
cmake -S . -B build/g++ -DCMAKE_CXX_COMPILER=$(which g++)
cmake --build build/g++

# Or clang:
cmake -S . -B build/clang++ -DCMAKE_CXX_COMPILER=$(which clang++)
cmake --build build/clang++
```

#### Specifying the stdlib

If you want to use `libc++` with clang instead of `libstdc++`, you can specify the standard library as follows:

```bash
# Do the actual build
cmake -S . -B build/clang++ -G<gen> \
    -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
    -DCMAKE_CXX_COMPILER=$(which clang++)

cmake --build build/clang++
```
