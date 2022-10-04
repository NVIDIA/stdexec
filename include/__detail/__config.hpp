/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#define _P2300_CAT_(X, ...) X ## __VA_ARGS__
#define _P2300_CAT(X, ...) _P2300_CAT_(X, __VA_ARGS__)

#define _P2300_EXPAND(...) __VA_ARGS__
#define _P2300_EVAL(M, ...) M(__VA_ARGS__)

#define _P2300_NOT(X) _P2300_CAT(_P2300_NOT_, X)
#define _P2300_NOT_0 1
#define _P2300_NOT_1 0

#define _P2300_IIF_0(Y,...) __VA_ARGS__
#define _P2300_IIF_1(Y,...) Y
#define _P2300_IIF(X,Y,...) \
    _P2300_EVAL(_P2300_CAT(_P2300_IIF_, X), Y, __VA_ARGS__)

#define _P2300_COUNT(...) \
    _P2300_EXPAND(_P2300_COUNT_(__VA_ARGS__,10,9,8,7,6,5,4,3,2,1))
#define _P2300_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _N, ...) _N

#define _P2300_CHECK(...) _P2300_EXPAND(_P2300_CHECK_N(__VA_ARGS__, 0,))
#define _P2300_CHECK_N(x, n, ...) n
#define _P2300_PROBE(x) x, 1,

#if defined(__NVCOMPILER)
#define _P2300_NVHPC() 1
#elif defined(__clang__)
#define _P2300_CLANG() 1
#elif defined(__GNUC__)
#define _P2300_GCC() 1
#elif defined(_MSC_VER)
#define _P2300_MSVC() 1
#endif

#ifndef _P2300_NVHPC
#define _P2300_NVHPC() 0
#endif
#ifndef _P2300_CLANG
#define _P2300_CLANG() 0
#endif
#ifndef _P2300_GCC
#define _P2300_GCC() 0
#endif
#ifndef _P2300_MSVC
#define _P2300_MSVC() 0
#endif
