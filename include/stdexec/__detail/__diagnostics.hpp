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

#include "__meta.hpp"

#include <exception> // IWYU pragma: keep for std::exception

namespace stdexec {
  struct sender_t;
  struct scheduler_t;

  namespace __errs {
    inline constexpr __mstring __unrecognized_sender_type_diagnostic =
      "The given type cannot be used as a sender with the given environment "
      "because the attempt to compute the completion signatures failed."_mstr;

    template <class _Sender>
    struct _WITH_SENDER_;

    template <class... _Senders>
    struct _WITH_SENDERS_;
  } // namespace __errs

  struct _WHERE_;

  struct _WHY_;

  struct _IN_ALGORITHM_;

  template <__mstring _Diagnostic = __errs::__unrecognized_sender_type_diagnostic>
  struct _UNRECOGNIZED_SENDER_TYPE_;

  template <class _Sender>
  using _WITH_SENDER_ = __errs::_WITH_SENDER_<__demangle_t<_Sender>>;

  template <class... _Senders>
  using _WITH_SENDERS_ = __errs::_WITH_SENDERS_<__demangle_t<_Senders>...>;

  template <class _Env>
  struct _WITH_ENVIRONMENT_;

  template <class _Ty>
  struct _WITH_TYPE_;

  template <class _Receiver>
  struct _WITH_RECEIVER_;

  template <class _Sig>
  struct _MISSING_COMPLETION_SIGNAL_;

  template <class _Sig>
  struct _WITH_COMPLETION_SIGNATURE_;

  template <class _Fun>
  struct _WITH_FUNCTION_;

  template <class... _Args>
  struct _WITH_ARGUMENTS_;

  template <class _Tag>
  struct _WITH_QUERY_;

  struct _TO_FIX_THIS_ERROR_;

  struct _SENDER_TYPE_IS_NOT_COPYABLE_;

  inline constexpr __mstring __not_callable_diag =
    "The specified function is not callable with the arguments provided."_mstr;

  template <__mstring _Context, __mstring _Diagnostic = __not_callable_diag>
  struct _NOT_CALLABLE_;

  template <auto _Reason = "You cannot pipe one sender into another."_mstr>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;

  template <__mstring _Context>
  struct __callable_error {
    template <class _Fun, class... _Args>
    using __f =
      __mexception<_NOT_CALLABLE_<_Context>, _WITH_FUNCTION_<_Fun>, _WITH_ARGUMENTS_<_Args...>>;
  };

#if __cpp_lib_constexpr_exceptions >= 202502L // constexpr exception types, https://wg21.link/p3378

  using __exception = ::std::exception;

#elif __cpp_constexpr >= 202411L // constexpr virtual functions

  struct __exception {
    constexpr __exception() noexcept = default;
    constexpr virtual ~__exception() = default;

    [[nodiscard]]
    constexpr virtual auto what() const noexcept -> const char* {
      return "<exception>";
    }
  };

#else // no constexpr virtual functions:

  struct __exception {
    constexpr __exception() noexcept = default;

    [[nodiscard]]
    constexpr auto what() const noexcept -> const char* {
      return "<exception>";
    }
  };

#endif // __cpp_lib_constexpr_exceptions >= 202502L

  template <class _Derived>
  struct __compile_time_error : __exception {
    __compile_time_error() = default; // NOLINT (bugprone-crtp-constructor-accessibility)

    [[nodiscard]]
    constexpr auto what() const noexcept -> const char* {
      return static_cast<_Derived const *>(this)->what();
    }
  };

  template <class _Data, class... _What>
  struct __sender_type_check_failure //
    : __compile_time_error<__sender_type_check_failure<_Data, _What...>> {
    static_assert(
      std::is_nothrow_move_constructible_v<_Data>,
      "The data member of sender_type_check_failure must be nothrow move constructible.");

    constexpr __sender_type_check_failure() noexcept = default;

    constexpr explicit __sender_type_check_failure(_Data data)
      : __data_(static_cast<_Data&&>(data)) {
    }

   private:
    friend struct __compile_time_error<__sender_type_check_failure>;

    [[nodiscard]]
    constexpr auto what() const noexcept -> const char* {
      return "This sender is not well-formed. It does not meet the requirements of a sender type.";
    }

    _Data __data_{};
  };

  struct dependent_sender_error : __compile_time_error<dependent_sender_error> {
    constexpr explicit dependent_sender_error(char const * what) noexcept
      : what_(what) {
    }

   private:
    friend struct __compile_time_error<dependent_sender_error>;

    [[nodiscard]]
    constexpr auto what() const noexcept -> char const * {
      return what_;
    }

    char const * what_;
  };

  template <class _Sender>
  struct __dependent_sender_error : dependent_sender_error {
    constexpr __dependent_sender_error() noexcept
      : dependent_sender_error{
          "This sender needs to know its execution environment before it can know how it will "
          "complete."} {
    }

    STDEXEC_ATTRIBUTE(host, device) auto operator+() -> __dependent_sender_error;

    template <class Ty>
    STDEXEC_ATTRIBUTE(host, device)
    auto operator,(Ty&) -> __dependent_sender_error&;

    template <class... What>
    STDEXEC_ATTRIBUTE(host, device)
    auto operator,(_ERROR_<What...>&) -> _ERROR_<What...>&;

    using __partitioned = __dependent_sender_error;

    template <class, class>
    using __value_types = __dependent_sender_error;

    template <class, class>
    using __error_types = __dependent_sender_error;

    template <class, class>
    using __stopped_types = __dependent_sender_error;
  };

  template <class _What, class... _With>
  struct __not_a_sender {
    using sender_concept = sender_t;

    template <class Self>
    constexpr auto get_completion_signatures(Self&&) const noexcept {
      return __mexception<_What, _With...>();
    }
  };

  template <class _What, class... _With>
  struct __not_a_scheduler {
    using scheduler_concept = scheduler_t;

    auto schedule() noexcept {
      return __not_a_sender<_What, _With...>{};
    }

    constexpr bool operator==(__not_a_scheduler) const noexcept {
      return true;
    }

    constexpr bool operator!=(__not_a_scheduler) const noexcept {
      return false;
    }
  };
} // namespace stdexec

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE                                                       \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The given type is not a sender because stdexec::enable_sender<Sender> is false. Either:\n"      \
  "\n"                                                                                             \
  "1. Give the type a nested `::sender_concept` type that is an alias for `stdexec::sender_t`,\n"  \
  "   as in:\n"                                                                                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept = stdexec::sender_t;\n"                                             \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. Specialize the `stdexec::enable_sender` boolean trait for this type to true, as follows:\n"  \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "     template <>\n"                                                                             \
  "     inline constexpr bool stdexec::enable_sender<MySender> = true;\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES                                         \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The sender type was not able to report its completion signatures when asked.\n"                 \
  "This is either because it lacks the necessary member function, or because the\n"                \
  "member function was ill-formed.\n"                                                              \
  "\n"                                                                                             \
  "A sender can declare its completion signatures in one of two ways:\n"                           \
  "\n"                                                                                             \
  "1. By defining a nested type alias named `completion_signatures` that is a\n"                   \
  "  specialization of `stdexec::completion_signatures<...>`, as follows:\n"                       \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = stdexec::sender_t;\n"                                      \
  "       using completion_signatures = stdexec::completion_signatures<\n"                         \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         stdexec::set_value_t(int, float),\n"                                                   \
  "         // ... or in error with an exception_ptr\n"                                            \
  "         stdexec::set_error_t(std::exception_ptr)>;\n"                                          \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. By defining a member function named `get_completion_signatures` that returns\n"              \
  "   a specialization of `stdexec::completion_signatures<...>`, as follows:\n"                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = stdexec::sender_t;\n"                                      \
  "\n"                                                                                             \
  "       template <class... _Env>\n"                                                              \
  "       auto get_completion_signatures(_Env&&...) -> stdexec::completion_signatures<\n"          \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         stdexec::set_value_t(int, float),\n"                                                   \
  "         // ... or in error with a std::exception_ptr.\n"                                       \
  "         stdexec::set_error_t(std::exception_ptr)>\n"                                           \
  "       {\n"                                                                                     \
  "        return {};\n"                                                                           \
  "       }\n"                                                                                     \
  "       ...\n"                                                                                   \
  "     };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_RETURNED_AN_ERROR                                  \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Trying to compute the sender's completion signatures resulted in an error. See\n"               \
  "the rest of the compiler diagnostic for clues. Look for the string \"_ERROR_\".\n"

#define STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_HAS_INVALID_RETURN_TYPE                            \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The member function `get_completion_signatures` of the sender returned an\n"                    \
  "invalid type.\n"                                                                                \
  "\n"                                                                                             \
  "A sender's `get_completion_signatures` function must return a specialization of\n"              \
  "`stdexec::completion_signatures<...>`, as follows:\n"                                           \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept = stdexec::sender_t;\n"                                                \
  "\n"                                                                                             \
  "    template <class... _Env>\n"                                                                 \
  "    auto get_completion_signatures(_Env&&...) -> stdexec::completion_signatures<\n"             \
  "      // This sender can complete successfully with an int and a float...\n"                    \
  "      stdexec::set_value_t(int, float),\n"                                                      \
  "      // ... or in error with a std::exception_ptr.\n"                                          \
  "      stdexec::set_error_t(std::exception_ptr)>\n"                                              \
  "    {\n"                                                                                        \
  "    return {};\n"                                                                               \
  "    }\n"                                                                                        \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER                                            \
  "\n"                                                                                             \
  "A sender must provide a `connect` member function that takes a receiver as an\n"                \
  "argument and returns an object whose type satisfies `stdexec::operation_state`,\n"              \
  "as shown below:\n"                                                                              \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept        = stdexec::sender_t;\n"                                         \
  "    using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;\n"    \
  "\n"                                                                                             \
  "    template <class Receiver>\n"                                                                \
  "    struct MyOpState\n"                                                                         \
  "    {\n"                                                                                        \
  "      using operation_state_concept = stdexec::operation_state_t;\n"                            \
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
  "    template <stdexec::receiver Receiver>\n"                                                    \
  "    auto connect(Receiver rcvr) -> MyOpState<Receiver>\n"                                       \
  "    {\n"                                                                                        \
  "      return MyOpState<Receiver>{std::move(rcvr)};\n"                                           \
  "    }\n"                                                                                        \
  "\n"                                                                                             \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_SYNC_WAIT_CANNOT_CONNECT_SENDER_TO_RECEIVER                                  \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The sender passed to `stdexec::sync_wait()` does not have a `connect`\n"                        \
  "member function that accepts sync_wait's "                                                      \
  "receiver.\n" STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER
