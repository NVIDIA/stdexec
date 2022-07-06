`std::execution`, the proposed C++ framework for asynchronous and parallel programming.

You can see a rendered copy of the current draft [here](https://brycelelbach.github.io/wg21_p2300_std_execution/std_execution.html).


## Reference implementation

[![CI](https://github.com/brycelelbach/wg21_p2300_std_execution/workflows/CI/badge.svg)](https://github.com/brycelelbach/wg21_p2300_std_execution/actions)

### Building

The following tools are needed:
* [`conan`](https://www.conan.io/)
* [`CMake`](https://cmake.org/)

Perform the following actions:

```bash
# Go to the build directory
mkdir -p build
pushd build

# Install dependencies
conan install .. --build=missing -s build_type=Release -s compiler.cppstd=20

# Do the actual build
cmake -G<gen> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 ..
cmake --build .

popd build
```

Here, `<gen>` can be `Ninja`, `make`, `XCode`, `"Visual Studio 15 Win64"`, etc.

#### Specifying the compiler

If you want to build with a specific compiler, you first need to teach conan about the compiler. You can do that with a conan "profile". Do the following:

```bash
# Create a conan profile for your compiler. In this case, the profile named
# "clang-14" is created that refers to /usr/bin/clang++-14
CC=/usr/bin/clang-14 \
  CXX=/usr/bin/clang++-14 \
  conan profile new --detect clang-14
```

Then edit the resulting file `~/.conan/profiles/clang-14` and add the following to the `[env]` section:

```
CC=/usr/bin/clang-14
CXX=/usr/bin/clang++-14
```

Then when you are installing the conan dependencies (see above), you would add `--profile clang-14` to the command line; e.g.,

```bash
# Install dependencies
conan install .. --build=missing  --profile clang-14 \
    -s build_type=Release -s compiler.cppstd=20
```

Finally, when configuring CMake, you would specify the same compiler via the `CMAKE_CXX_COMPILER` define on the command line; e.g.,

```bash
# Do the actual build
cmake -G<gen> -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CXX_STANDARD=20 \
              -DCMAKE_CXX_COMPILER=/usr/bin/clang++-14 \
              ..
cmake --build .
```

#### Specifying the stdlib

If you want to use libc++ with clang instead of libstdc++, then you first need to tell conan. You have two options: add "`-s compiler.libcxx=libc++`" to the `conan install` command, or update the profile directly. Continuing the previous example, to update the clang-14 profile you would do:

```bash
# Set the compiler.libcxx property to libc++
conan profile update 'compiler.libcxx=libc++' clang-14
```

All future uses of the clang-14 profile will now automatically use libc++ instead of libstdc++.

Finally, when configuring CMake, you need to additionally specify the standard library as follows:

```bash
# Do the actual build
cmake -G<gen> -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CXX_STANDARD=20 \
              -DCMAKE_CXX_COMPILER=/usr/bin/clang++-14 \
              -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
              ..
cmake --build .
```
