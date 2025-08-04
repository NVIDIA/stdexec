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

#define STDEXEC_STRINGIZE_(...)   #__VA_ARGS__
#define STDEXEC_STRINGIZE(...)    STDEXEC_STRINGIZE_(__VA_ARGS__)

#define STDEXEC_CAT_(_XP, ...)    _XP##__VA_ARGS__
#define STDEXEC_CAT(_XP, ...)     STDEXEC_CAT_(_XP, __VA_ARGS__)

#define STDEXEC_EXPAND(...)       __VA_ARGS__
#define STDEXEC_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)
#define STDEXEC_EAT(...)

#define STDEXEC_IIF(_XP, _YP, ...)                                                                 \
  STDEXEC_IIF_EVAL(STDEXEC_CAT(STDEXEC_IIF_, _XP), _YP, __VA_ARGS__)
#define STDEXEC_IIF_0(_YP, ...)       __VA_ARGS__
#define STDEXEC_IIF_1(_YP, ...)       _YP
#define STDEXEC_IIF_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)

#define STDEXEC_COMPL(_B)             STDEXEC_COMPL_CAT(STDEXEC_COMPL_, _B)
#define STDEXEC_COMPL_0               1 // NOLINT(modernize-macro-to-enum)
#define STDEXEC_COMPL_1               0 // NOLINT(modernize-macro-to-enum)
#define STDEXEC_COMPL_CAT(_XP, ...)   _XP##__VA_ARGS__

#define STDEXEC_COUNT(...)                                                                         \
  STDEXEC_EXPAND(STDEXEC_COUNT_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define STDEXEC_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _NP, ...) _NP

#define STDEXEC_CHECK(...)                                                STDEXEC_EXPAND(STDEXEC_CHECK_(__VA_ARGS__, 0, ))
#define STDEXEC_CHECK_(_XP, _NP, ...)                                     _NP
#define STDEXEC_PROBE(...)                                                STDEXEC_PROBE_(__VA_ARGS__, 1)
#define STDEXEC_PROBE_(_XP, _NP, ...)                                     _XP, _NP,

#define STDEXEC_NOT(_XP)                                                  STDEXEC_CHECK(STDEXEC_CAT(STDEXEC_NOT_, _XP))
#define STDEXEC_NOT_0                                                     STDEXEC_PROBE(~, 1)

#define STDEXEC_BOOL(_XP)                                                 STDEXEC_COMPL(STDEXEC_NOT(_XP))
#define STDEXEC_IF(_XP, _YP, ...)                                         STDEXEC_IIF(STDEXEC_BOOL(_XP), _YP, __VA_ARGS__)

#define STDEXEC_WHEN(_XP, ...)                                            STDEXEC_IF(_XP, STDEXEC_EXPAND, STDEXEC_EAT)(__VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
// STDEXEC_FOR_EACH
//   Inspired by "Recursive macros with C++20 __VA_OPT__", by David Mazières
//   https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define STDEXEC_EXPAND_R(...)                                                                      \
  STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(__VA_ARGS__))))          \
  /**/
#define STDEXEC_EXPAND_R1(...)                                                                     \
  STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(__VA_ARGS__))))          \
  /**/
#define STDEXEC_EXPAND_R2(...)                                                                     \
  STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(__VA_ARGS__))))          \
  /**/
#define STDEXEC_EXPAND_R3(...)                                                                     \
  STDEXEC_EXPAND(STDEXEC_EXPAND(STDEXEC_EXPAND(STDEXEC_EXPAND(__VA_ARGS__))))                      \
  /**/

#define STDEXEC_PARENS ()
#define STDEXEC_FOR_EACH(_MACRO, ...)                                                              \
  __VA_OPT__(STDEXEC_EXPAND_R(STDEXEC_FOR_EACH_HELPER(_MACRO, __VA_ARGS__)))                       \
  /**/
#define STDEXEC_FOR_EACH_HELPER(_MACRO, _A1, ...)                                                  \
  _MACRO(_A1) __VA_OPT__(STDEXEC_FOR_EACH_AGAIN STDEXEC_PARENS(_MACRO, __VA_ARGS__)) /**/
#define STDEXEC_FOR_EACH_AGAIN()       STDEXEC_FOR_EACH_HELPER
////////////////////////////////////////////////////////////////////////////////////////////////////

#define STDEXEC_FRONT(...)             __VA_OPT__(STDEXEC_FRONT_HELPER(__VA_ARGS__))
#define STDEXEC_FRONT_HELPER(_A1, ...) _A1
#define STDEXEC_BACK(...)              __VA_OPT__(STDEXEC_EXPAND_R(STDEXEC_BACK_HELPER(__VA_ARGS__)))
#define STDEXEC_BACK_HELPER(_A1, ...)                                                              \
  STDEXEC_FRONT(__VA_OPT__(, ) _A1, ) __VA_OPT__(STDEXEC_BACK_AGAIN STDEXEC_PARENS(__VA_ARGS__))
#define STDEXEC_BACK_AGAIN()             STDEXEC_BACK_HELPER

#define STDEXEC_TAIL(_, ...)             __VA_ARGS__

#define STDEXEC_REPEAT(_N, _MACRO, ...)  STDEXEC_REPEAT_(_N, _MACRO, __VA_ARGS__)
#define STDEXEC_REPEAT_(_N, _MACRO, ...) STDEXEC_REPEAT_##_N(_MACRO, __VA_ARGS__)
#define STDEXEC_REPEAT_0(_MACRO, ...)
#define STDEXEC_REPEAT_1(_MACRO, ...) _MACRO(0 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_2(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_1(_MACRO, __VA_ARGS__) _MACRO(1 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_3(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_2(_MACRO, __VA_ARGS__) _MACRO(2 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_4(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_3(_MACRO, __VA_ARGS__) _MACRO(3 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_5(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_4(_MACRO, __VA_ARGS__) _MACRO(4 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_6(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_5(_MACRO, __VA_ARGS__) _MACRO(5 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_7(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_6(_MACRO, __VA_ARGS__) _MACRO(6 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_8(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_7(_MACRO, __VA_ARGS__) _MACRO(7 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_9(_MACRO, ...)                                                              \
  STDEXEC_REPEAT_8(_MACRO, __VA_ARGS__) _MACRO(8 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_REPEAT_10(_MACRO, ...)                                                             \
  STDEXEC_REPEAT_9(_MACRO, __VA_ARGS__) _MACRO(9 __VA_OPT__(, ) __VA_ARGS__)
