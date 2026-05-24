# stdexec — Senders for C++

**A reference implementation of `std::execution` ([\[exec\]](https://wg21.link/exec)), the C++26 model for asynchronous and parallel programming.**

[![CI (CPU)](https://github.com/NVIDIA/stdexec/actions/workflows/ci.cpu.yml/badge.svg)](https://github.com/NVIDIA/stdexec/actions/workflows/ci.cpu.yml)
[![CI (GPU)](https://github.com/NVIDIA/stdexec/actions/workflows/ci.gpu.yml/badge.svg)](https://github.com/NVIDIA/stdexec/actions/workflows/ci.gpu.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM--exception-blue.svg)](LICENSE.txt)
[![C++](https://img.shields.io/badge/C%2B%2B-20%2B-blue.svg)](https://en.cppreference.com/w/cpp/compiler_support)
[![Try on Godbolt](https://img.shields.io/badge/try-godbolt-orange.svg)](https://godbolt.org/z/zjjvWoWPW)
[![Documentation](https://img.shields.io/badge/docs-nvidia.github.io%2Fstdexec-blue.svg)](https://nvidia.github.io/stdexec)

`stdexec` lets you express asynchronous work as composable, lazy *sender* pipelines that can run on threads, thread pools, GPUs, or any custom execution context — with structured concurrency guarantees.

> [!WARNING]
> `stdexec` is experimental and tracks an evolving standard. APIs may change without notice. NVIDIA does not guarantee fitness for any particular purpose.

## Table of contents

- [Example](#example)
- [Features](#features)
- [Compiler support](#compiler-support)
- [Installation](#installation)
- [Quick start](#quick-start)
- [GPU support](#gpu-support)
- [Examples gallery](#examples-gallery)
- [Documentation](#documentation)
- [Building tests and examples](#building-tests-and-examples)
- [IDE support](#ide-support)
- [Resources](#resources)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Example

Run three pieces of work concurrently on the system thread pool. Try it live on [godbolt](https://godbolt.org/z/zjjvWoWPW).

```c++
#include <stdexec/execution.hpp>
#include <cstdio>

namespace ex = stdexec;

int main() {
    auto sched = ex::get_parallel_scheduler();
    auto fun   = [](int i) { return i * i; };

    // Build a lazy pipeline: three squares, computed in parallel.
    auto work = ex::when_all(ex::on(sched, ex::just(0) | ex::then(fun)),
                             ex::on(sched, ex::just(1) | ex::then(fun)),
                             ex::on(sched, ex::just(2) | ex::then(fun)));

    // Launch the work and wait for the result.
    auto [i, j, k] = ex::sync_wait(std::move(work)).value();
    std::printf("%d %d %d\n", i, j, k); // prints "0 1 4"
}
```

## Features

- **C++26 reference implementation** of `std::execution` (P2300).
- **Header-only**, no external dependencies.
- **Composable algorithms**: `then`, `let_value`, `when_all`, `bulk`, `split`, `transfer`, `upon_*`, ...
- **Structured concurrency primitives**: `async_scope`, `task`, `finally`, `when_any`, `repeat_n`, ...
- **Pluggable schedulers**: system parallel scheduler, static thread pool, Linux `io_uring` context, NVIDIA GPU contexts, your own.
- **GPU offload** via `nvexec` schedulers (`nvc++` compiler).
- **Coroutine interop**: senders are awaitable; awaitables are senders.
- **Generic extensions** (`<exec/...>`) for primitives not (yet) in the standard.

## Compiler support

| Compiler | Minimum version | Notes |
|---|---|---|
| GCC | 12 | |
| Clang | 16 | |
| MSVC | 14.43 | |
| Xcode (Apple Clang) | 16 | |
| nvc++ | 25.9 | required for [GPU support](#gpu-support) |

Requires `-std=c++20` or later.

> [!NOTE]
> `stdexec` does not yet support NVIDIA's `nvcc` compiler.

## Installation

Pick whichever fits your project.

### CPM (recommended)

[CPM](https://github.com/cpm-cmake/CPM.cmake) fetches and configures `stdexec` automatically from your `CMakeLists.txt`:

```cmake
CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main  # or a specific tag
)

target_link_libraries(my_target PRIVATE STDEXEC::stdexec)
```

### `add_subdirectory`

Clone alongside your project and add it as a subdirectory:

```bash
git clone https://github.com/NVIDIA/stdexec.git
```

```cmake
add_subdirectory(stdexec)
target_link_libraries(my_target PRIVATE STDEXEC::stdexec)
```

### Conan

A [`conanfile.py`](conanfile.py) is provided for use with the [Conan](https://conan.io) package manager.

### NVIDIA HPC SDK

Starting with [NVHPC SDK 22.11](https://developer.nvidia.com/nvidia-hpc-sdk-releases), `stdexec` is bundled with `nvc++`. Pass `--experimental-stdpar` to put `stdexec` headers on the include path. Add `-stdpar=gpu` for GPU features. See the [godbolt example](https://godbolt.org/z/qc1h3sqEv).

### Manual include path

`stdexec` is header-only, so adding `-I<stdexec root>/include` to your compile command is sufficient. Using the CMake target is recommended because it sets the required compile flags.

## Quick start

A minimal `CMakeLists.txt` using CPM:

```cmake
cmake_minimum_required(VERSION 3.25.0)
project(stdexec_example LANGUAGES CXX)

include(CPM.cmake)  # see https://github.com/cpm-cmake/CPM.cmake#adding-cpm

CPMAddPackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main
)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE STDEXEC::stdexec)
```

## GPU support

`stdexec` ships GPU schedulers in [`<nvexec/...>`](include/nvexec/) for use with `nvc++ -stdpar=gpu`:

| Scheduler | Header | Description |
|---|---|---|
| `nvexec::stream_scheduler` | [`<nvexec/stream_context.cuh>`](include/nvexec/stream_context.cuh) | Single-GPU scheduler (device 0). |
| `nvexec::multi_gpu_stream_scheduler` | [`<nvexec/multi_gpu_context.cuh>`](include/nvexec/multi_gpu_context.cuh) | Multi-GPU scheduler across all visible devices. |

Live example: <https://godbolt.org/z/h7rh5qGhj>

## Examples gallery

The [`examples/`](examples/) directory contains runnable programs demonstrating the library.

| Example | What it shows |
|---|---|
| [`hello_world.cpp`](examples/hello_world.cpp) | The "hello world" of senders. |
| [`hello_coro.cpp`](examples/hello_coro.cpp) | Awaiting a sender from a coroutine. |
| [`then.cpp`](examples/then.cpp) | Writing a `then` algorithm from scratch. |
| [`retry.cpp`](examples/retry.cpp) | Writing a `retry` algorithm from scratch. |
| [`scope.cpp`](examples/scope.cpp) | Structured concurrency with `async_scope`. |
| [`io_uring.cpp`](examples/io_uring.cpp) | Async I/O via the Linux `io_uring` context. |
| [`sudoku.cpp`](examples/sudoku.cpp) | A parallel sudoku solver. |
| [`server_theme/`](examples/server_theme/) | Server-style patterns (`let_value`, `split`, `bulk`, `transfer`). |
| [`nvexec/`](examples/nvexec/) | GPU schedulers, including the Maxwell solver. |

## Documentation

**📖 Full documentation: <https://nvidia.github.io/stdexec>**

- **User guide**: <https://nvidia.github.io/stdexec/user/> ([source](docs/source/user/))
- **Reference**: <https://nvidia.github.io/stdexec/reference/> ([source](docs/source/reference/))
- **Developer docs**: <https://nvidia.github.io/stdexec/developer/> ([source](docs/source/developer/))
- **Contributing to docs**: [`docs/CONTRIBUTING-docs.md`](docs/CONTRIBUTING-docs.md)
- **The proposal**: [`[exec]` — `std::execution`](https://wg21.link/exec)

The library is organized into three namespaces:

| Namespace | Headers | Contents |
|---|---|---|
| `::stdexec` | `<stdexec/...>` | Things in (or proposed for) the C++ standard. |
| `::exec` | `<exec/...>` | Generic additions and extensions. |
| `::nvexec` | `<nvexec/...>` | NVIDIA-specific schedulers and customizations. |

## Building tests and examples

The library itself is header-only — these steps are only needed if you want to build the test suite or the examples.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

To select a specific compiler:

```bash
cmake -S . -B build/clang -DCMAKE_CXX_COMPILER=$(which clang++)
cmake --build build/clang
```

To use `libc++` with Clang:

```bash
cmake -S . -B build/libcxx \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CXX_FLAGS=-stdlib=libc++
cmake --build build/libcxx
```

## IDE support

A [VSCode extension](https://marketplace.visualstudio.com/items?itemName=ericniebler.erics-build-output-colorizer) is available that colorizes compiler diagnostics from `stdexec`, making the long template error messages much easier to read. Source and configuration: <https://github.com/ericniebler/buildoutputcolorizer>.

## Resources

### Standards papers

- [P2300 — `std::execution`](https://wg21.link/p2300) — the proposal accepted into C++26.

### Talks

- [Working with Asynchrony Generically: A Tour of Executors](https://www.youtube.com/watch?v=xLboNIf7BTg) ([Part 2](https://www.youtube.com/watch?v=6a0zzUBUNW4)) — comprehensive introduction.
- [From Zero to Sender/Receiver in ~60 Minutes](https://www.youtube.com/watch?v=xiaqNvqRB2E) — live-coding a toy sender/receiver from scratch.
- [A Unifying Abstraction for Async in C++](https://www.youtube.com/watch?v=h-ExnuD6jms) — concepts behind P2300.
- [Structured Concurrency](https://www.youtube.com/watch?v=1Wy5sq3s2rg) — what structured concurrency means and why.
- [Structured Networking in C++](https://www.youtube.com/watch?v=XaNajUp-sGY) — what a P2300-style networking library could look like.

### Articles and blog posts

- [What are Senders Good For, Anyway?](https://ericniebler.com/2024/02/04/what-are-senders-good-for-anyway/) — wrapping a C-style async API in a sender.
- [A Universal Async Abstraction for C++](https://cor3ntin.github.io/posts/executors/) — an introduction to senders.
- [A Universal I/O Abstraction for C++](https://cor3ntin.github.io/posts/iouring/) — senders meet `io_uring`.
- [Executors: a Change of Perspective](https://accu.org/journals/overload/29/165/teodorescu/) — on the computational completeness of senders.
- [Structured Concurrency in C++](https://accu.org/journals/overload/30/168/teodorescu/) — how senders manifest structured concurrency.
- [HPCWire: New C++ Sender Library Enables Portable Asynchrony](https://www.hpcwire.com/2022/12/05/new-c-sender-library-enables-portable-asynchrony/).

### NVIDIA

- [NVIDIA HPC SDK documentation](https://docs.nvidia.com/hpc-sdk/index.html).

## Contributing

Contributions are welcome. Before opening a PR, please review:

- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- [`MAINTAINERS.md`](MAINTAINERS.md)
- [`docs/CONTRIBUTING-docs.md`](docs/CONTRIBUTING-docs.md) for documentation contributions.

Bug reports and feature requests belong in [GitHub Issues](https://github.com/NVIDIA/stdexec/issues); design discussion in [GitHub Discussions](https://github.com/NVIDIA/stdexec/discussions).

## Citation

If you reference `stdexec` in academic work, please cite the standards proposal:

```bibtex
@techreport{P2300,
  author = {Niebler, Eric and Shoop, Kirk and Baker, Lewis and Dominiak, Michał and
            Evtushenko, Georgy and Teodorescu, Lucian Radu and Howes, Lee and Garland,
            Michael and Lelbach, Bryce Adelstein}
  title  = {{P2300R10}: \texttt{std::execution}},
  institution = {ISO/IEC JTC1/SC22/WG21},
  year   = {2024},
  url    = {https://wg21.link/p2300}
}
```

## License

`stdexec` is licensed under the **Apache License 2.0 with LLVM Exceptions**. See [LICENSE.txt](LICENSE.txt) for the full text.
