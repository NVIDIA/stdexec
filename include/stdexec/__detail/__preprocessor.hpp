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

#define STDEXEC_PP_STRINGIZE_I(...) #__VA_ARGS__
#define STDEXEC_PP_STRINGIZE(...)   STDEXEC_PP_STRINGIZE_I(__VA_ARGS__)

#define STDEXEC_PP_LPAREN (
#define STDEXEC_PP_RPAREN )
#define STDEXEC_PP_PARENS            ()
#define STDEXEC_PP_COMMA             ,

#define STDEXEC_PP_CAT_I(_XP, ...)   _XP##__VA_ARGS__
#define STDEXEC_PP_CAT(_XP, ...)     STDEXEC_PP_CAT_I(_XP, __VA_ARGS__)

#define STDEXEC_PP_EXPAND(...)       __VA_ARGS__
#define STDEXEC_PP_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)
#define STDEXEC_PP_EAT(...)

#define STDEXEC_PP_IS_EMPTY_I(_BIT, ...) _BIT
#define STDEXEC_PP_IS_EMPTY(...)         STDEXEC_PP_IS_EMPTY_I(__VA_OPT__(0, ) 1)

#define STDEXEC_PP_IIF_0(_YP, ...)       __VA_ARGS__
#define STDEXEC_PP_IIF_1(_YP, ...)       _YP
#define STDEXEC_PP_IIF_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)
#define STDEXEC_PP_IIF(_XP, _YP, ...)                                                              \
  STDEXEC_PP_IIF_EVAL(STDEXEC_PP_CAT(STDEXEC_PP_IIF_, _XP), _YP, __VA_ARGS__)

#define STDEXEC_PP_COMPL_0             1 // NOLINT(modernize-macro-to-enum)
#define STDEXEC_PP_COMPL_1             0 // NOLINT(modernize-macro-to-enum)
#define STDEXEC_PP_COMPL_CAT(_XP, ...) _XP##__VA_ARGS__
#define STDEXEC_PP_COMPL(_BIT)         STDEXEC_PP_COMPL_CAT(STDEXEC_PP_COMPL_, _BIT)

#define STDEXEC_PP_COUNT(...)                                                                      \
  STDEXEC_PP_EXPAND(STDEXEC_PP_COUNT_I(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define STDEXEC_PP_COUNT_I(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _NP, ...) _NP

// Used to check various properties of arguments
#define STDEXEC_PP_CHECK_I(_XP, _NP, ...) _NP
#define STDEXEC_PP_CHECK(...)             STDEXEC_PP_EXPAND(STDEXEC_PP_CHECK_I(__VA_ARGS__, 0, ))
#define STDEXEC_PP_PROBE_I(_XP, _NP, ...) _XP, _NP,
#define STDEXEC_PP_PROBE(...)             STDEXEC_PP_PROBE_I(__VA_ARGS__, 1)

// Boolean logic
#define STDEXEC_PP_NOT(_XP)          STDEXEC_PP_CHECK(STDEXEC_PP_CAT(STDEXEC_PP_NOT_, _XP))
#define STDEXEC_PP_NOT_0             STDEXEC_PP_PROBE(~, 1)

#define STDEXEC_PP_BOOL(_XP)         STDEXEC_PP_COMPL(STDEXEC_PP_NOT(_XP))
#define STDEXEC_PP_IF(_XP, _YP, ...) STDEXEC_PP_IIF(STDEXEC_PP_BOOL(_XP), _YP, __VA_ARGS__)

#define STDEXEC_PP_WHEN(_XP, ...)    STDEXEC_PP_IF(_XP, STDEXEC_PP_EXPAND, STDEXEC_PP_EAT)(__VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
// STDEXEC_PP_FOR_EACH
//   Inspired by "Recursive macros with C++20 __VA_OPT__", by David Mazières
//   https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define STDEXEC_PP_EXPAND_R3(...)                                                                  \
  STDEXEC_PP_EXPAND(STDEXEC_PP_EXPAND(STDEXEC_PP_EXPAND(STDEXEC_PP_EXPAND(__VA_ARGS__))))
#define STDEXEC_PP_EXPAND_R2(...)                                                                  \
  STDEXEC_PP_EXPAND_R3(                                                                            \
    STDEXEC_PP_EXPAND_R3(STDEXEC_PP_EXPAND_R3(STDEXEC_PP_EXPAND_R3(__VA_ARGS__))))
#define STDEXEC_PP_EXPAND_R1(...)                                                                  \
  STDEXEC_PP_EXPAND_R2(                                                                            \
    STDEXEC_PP_EXPAND_R2(STDEXEC_PP_EXPAND_R2(STDEXEC_PP_EXPAND_R2(__VA_ARGS__))))
#define STDEXEC_PP_EXPAND_R(...)                                                                   \
  STDEXEC_PP_EXPAND_R1(                                                                            \
    STDEXEC_PP_EXPAND_R1(STDEXEC_PP_EXPAND_R1(STDEXEC_PP_EXPAND_R1(__VA_ARGS__))))

#define STDEXEC_PP_FOR_EACH_AGAIN() STDEXEC_PP_FOR_EACH_HELPER
#define STDEXEC_PP_FOR_EACH_HELPER(_MACRO, _A1, ...)                                               \
  _MACRO(_A1) __VA_OPT__(STDEXEC_PP_FOR_EACH_AGAIN STDEXEC_PP_PARENS(_MACRO, __VA_ARGS__)) /**/
#define STDEXEC_PP_FOR_EACH(_MACRO, ...)                                                           \
  __VA_OPT__(STDEXEC_PP_EXPAND_R(STDEXEC_PP_FOR_EACH_HELPER(_MACRO, __VA_ARGS__)))

////////////////////////////////////////////////////////////////////////////////////////////////////

#define STDEXEC_PP_FRONT_I(_A1, ...) _A1
#define STDEXEC_PP_FRONT(...)        __VA_OPT__(STDEXEC_PP_FRONT_I(__VA_ARGS__))
#define STDEXEC_PP_BACK_AGAIN()      STDEXEC_PP_BACK_I
#define STDEXEC_PP_BACK_I(_A1, ...)                                                                \
  STDEXEC_PP_FRONT(__VA_OPT__(, ) _A1, )                                                           \
  __VA_OPT__(STDEXEC_PP_BACK_AGAIN STDEXEC_PP_PARENS(__VA_ARGS__))
#define STDEXEC_PP_BACK(...)                 __VA_OPT__(STDEXEC_PP_EXPAND_R(STDEXEC_PP_BACK_I(__VA_ARGS__)))

#define STDEXEC_PP_TAIL(_IGN, ...)           __VA_ARGS__

#define STDEXEC_PP_REPEAT_I(_N, _MACRO, ...) STDEXEC_PP_REPEAT_##_N(_MACRO, __VA_ARGS__)
#define STDEXEC_PP_REPEAT_0(_MACRO, ...)
#define STDEXEC_PP_REPEAT_1(_MACRO, ...) _MACRO(0 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_2(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_1(_MACRO, __VA_ARGS__) _MACRO(1 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_3(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_2(_MACRO, __VA_ARGS__) _MACRO(2 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_4(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_3(_MACRO, __VA_ARGS__) _MACRO(3 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_5(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_4(_MACRO, __VA_ARGS__) _MACRO(4 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_6(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_5(_MACRO, __VA_ARGS__) _MACRO(5 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_7(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_6(_MACRO, __VA_ARGS__) _MACRO(6 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_8(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_7(_MACRO, __VA_ARGS__) _MACRO(7 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_9(_MACRO, ...)                                                           \
  STDEXEC_PP_REPEAT_8(_MACRO, __VA_ARGS__) _MACRO(8 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT_10(_MACRO, ...)                                                          \
  STDEXEC_PP_REPEAT_9(_MACRO, __VA_ARGS__) _MACRO(9 __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_PP_REPEAT(_N, _MACRO, ...) STDEXEC_PP_REPEAT_I(_N, _MACRO, __VA_ARGS__)
