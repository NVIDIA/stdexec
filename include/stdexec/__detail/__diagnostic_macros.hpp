/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

// ensure consumers have STDEXEC and STDEXEC_PP_STRINGIZE
#include "__config.hpp"
#include "__preprocessor.hpp"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE                                                     \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The given type is not a sender because " STDEXEC_PP_STRINGIZE(STDEXEC) "::enable_sender<Sender>"\
  "is false. Either:\n"                                                                            \
  "\n"                                                                                             \
  "1. Give the type a nested '::sender_concept' type that is an alias for '"                       \
  STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag',\n"                                                 \
  "   as in:\n"                                                                                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept = " STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag;\n"                 \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. Specialize the '" STDEXEC_PP_STRINGIZE(STDEXEC) "::enable_sender' boolean trait for this "   \
  "type to true, as follows:\n"                                                                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "     template <>\n"                                                                             \
  "     inline constexpr bool " STDEXEC_PP_STRINGIZE(STDEXEC) "::enable_sender<MySender> = true;\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES                                       \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The sender type was not able to report its completion signatures when asked.\n"                 \
  "This is either because it lacks the necessary member function, or because the\n"                \
  "member function was ill-formed.\n"                                                              \
  "\n"                                                                                             \
  "A sender can declare its completion signatures in one of two ways:\n"                           \
  "\n"                                                                                             \
  "1. By defining a nested type alias named 'completion_signatures' that is a\n"                   \
  "  specialization of '" STDEXEC_PP_STRINGIZE(STDEXEC) "::completion_signatures<...>', "          \
  "as follows:\n"                                                                                  \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = " STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag;\n"          \
  "       using completion_signatures = " STDEXEC_PP_STRINGIZE(STDEXEC)                            \
  "::completion_signatures<\n"                                                                     \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_value_t(int, float),\n"                         \
  "         // ... or in error with an exception_ptr\n"                                            \
  "         " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_error_t(std::exception_ptr)>;\n"                \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. By defining a member function named 'get_completion_signatures' that returns\n"              \
  "   a specialization of '" STDEXEC_PP_STRINGIZE(STDEXEC) "::completion_signatures<...>', as "    \
  "follows:\n"                                                                                     \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = " STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag;\n"          \
  "\n"                                                                                             \
  "       template <class Self, class... Env>\n"                                                   \
  "       static consteval auto get_completion_signatures() -> " STDEXEC_PP_STRINGIZE(STDEXEC)     \
  "::completion_signatures<\n"                                                                     \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_value_t(int, float),\n"                         \
  "         // ... or in error with a std::exception_ptr.\n"                                       \
  "         " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_error_t(std::exception_ptr)>\n"                 \
  "       {\n"                                                                                     \
  "        return {};\n"                                                                           \
  "       }\n"                                                                                     \
  "       ...\n"                                                                                   \
  "     };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_RETURNED_AN_ERROR                                \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Trying to compute the sender's completion signatures resulted in an error. See\n"               \
  "the rest of the compiler diagnostic for clues. Look for the string \"_ERROR_\".\n"

#define STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_HAS_INVALID_RETURN_TYPE                          \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The member function 'get_completion_signatures' of the sender returned an\n"                    \
  "invalid type.\n"                                                                                \
  "\n"                                                                                             \
  "A sender's 'get_completion_signatures' function must return a specialization of\n"              \
  "'" STDEXEC_PP_STRINGIZE(STDEXEC) "::completion_signatures<...>', as follows:\n"                 \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept = " STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag;\n"                    \
  "\n"                                                                                             \
  "    template <class Self, class... Env>\n"                                                      \
  "    static consteval auto get_completion_signatures() -> " STDEXEC_PP_STRINGIZE(STDEXEC)        \
  "::completion_signatures<\n"                                                                     \
  "      // This sender can complete successfully with an int and a float...\n"                    \
  "      " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_value_t(int, float),\n"                            \
  "      // ... or in error with a std::exception_ptr.\n"                                          \
  "      " STDEXEC_PP_STRINGIZE(STDEXEC) "::set_error_t(std::exception_ptr)>\n"                    \
  "    {\n"                                                                                        \
  "      return {};\n"                                                                             \
  "    }\n"                                                                                        \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER                                          \
  "\n"                                                                                             \
  "A sender must provide a 'connect' member function that takes a receiver as an\n"                \
  "argument and returns an object whose type satisfies '" STDEXEC_PP_STRINGIZE(STDEXEC)            \
  "::operation_state',\n"                                                                          \
  "as shown below:\n"                                                                              \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept        = " STDEXEC_PP_STRINGIZE(STDEXEC) "::sender_tag;\n"             \
  "    using completion_signatures = " STDEXEC_PP_STRINGIZE(STDEXEC) "::completion_signatures<"    \
  STDEXEC_PP_STRINGIZE(STDEXEC) "::set_value_t()>;\n"                                              \
  "\n"                                                                                             \
  "    template <class Receiver>\n"                                                                \
  "    struct MyOpState\n"                                                                         \
  "    {\n"                                                                                        \
  "      using operation_state_concept = " STDEXEC_PP_STRINGIZE(STDEXEC) "::operation_state_tag;\n"\
  "\n"                                                                                             \
  "      void start() noexcept\n"                                                                  \
  "      {\n"                                                                                      \
  "        // Start the operation, which will eventually complete and send its\n"                  \
  "        // result to rcvr_;\n"                                                                  \
  "      }\n"                                                                                      \
  "\n"                                                                                             \
  "      Receiver rcvr_;\n"                                                                        \
  "    };\n"                                                                                       \
  "\n"                                                                                             \
  "    template <" STDEXEC_PP_STRINGIZE(STDEXEC) "::receiver Receiver>\n"                          \
  "    auto connect(Receiver rcvr) -> MyOpState<Receiver>\n"                                       \
  "    {\n"                                                                                        \
  "      return MyOpState<Receiver>{std::move(rcvr)};\n"                                           \
  "    }\n"                                                                                        \
  "\n"                                                                                             \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_SYNC_WAIT_CANNOT_CONNECT_SENDER_TO_RECEIVER                                \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The sender passed to '" STDEXEC_PP_STRINGIZE(STDEXEC) "::sync_wait()' does not have a "         \
  "'connect'\n"                                                                                    \
  "member function that accepts sync_wait's receiver.\n"                                           \
  STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER
