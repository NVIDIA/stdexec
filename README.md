# Senders - A Standard Model for Asynchronous Execution in C++

`stdexec` is an experimental reference implementation of the _Senders_ model of asynchronous programming proposed by [**P2300 - `std::execution`**](http://wg21.link/p2300) for adoption into the C++ Standard.

**Purpose of this Repository:** 
1. Provide a proof-of-concept implementation of the design proposed in [P2300](http://wg21.link/p2300).
2. Provide early access to developers looking to experiment with the Sender model.
3. Colloborate with those interested in participating or contributing to the design of P2300 (contributions welcome!).

## Disclaimer

`stdexec`is experimental in nature and subject to change without warning. 
The authors and NVIDIA do not guarantee that this code is fit for any purpose whatsoever.

[![CI](https://github.com/NVIDIA/stdexec/workflows/CI/badge.svg)](https://github.com/NVIDIA/stdexec/actions)

## Building

The following tools are needed:

* [`CMake`](https://cmake.org/)
* GCC 11+ or clang 12+

Perform the following actions:

```bash
# Configure the project
cmake S . -B build -G<gen>
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
