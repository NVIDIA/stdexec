`std::execution`, the proposed C++ framework for asynchronous and parallel programming.

You can see a rendered copy of the current draft [here](https://brycelelbach.github.io/wg21_p2300_std_execution/std_execution.html).


## Reference implementation

[![CI](https://github.com/brycelelbach/wg21_p2300_std_execution/workflows/CI/badge.svg)](https://github.com/brycelelbach/wg21_p2300_std_execution/actions)

### Building

The following tools are needed:
* [`conan`](https://www.conan.io/)
* [`CMake`](https://cmake.org/)

Perform the following actions:
```
# Go to the build directory
mkdir -p build
pushd build

# Install dependencies
conan install .. --build=missing -s build_type=Release

# Do the actual build
cmake -G<gen> -D CMAKE_BUILD_TYPE=Release ..
cmake --build .

popd build
```

Here, `<gen>` can be `Ninja`, `make`, `XCode`, `"Visual Studio 15 Win64"`, etc.