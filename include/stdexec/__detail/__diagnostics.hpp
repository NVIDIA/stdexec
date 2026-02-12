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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__meta.hpp"

#include <exception> // IWYU pragma: keep for std::exception

namespace STDEXEC {
  struct sender_t;
  struct scheduler_t;

  struct _WHAT_ { };

  struct _WHERE_ { };

  struct _WHY_ { };

  struct _IN_ALGORITHM_ { };

  struct _UNRECOGNIZED_SENDER_TYPE_;

  template <class _Sender>
  struct _WITH_SENDER_ { };

  template <class... _Senders>
  struct _WITH_SENDERS_ { };

  template <class _Sender>
  using _WITH_PRETTY_SENDER_ = _WITH_SENDER_<__demangle_t<_Sender>>;

  template <class... _Senders>
  using _WITH_PRETTY_SENDERS_ = _WITH_SENDERS_<__demangle_t<_Senders>...>;

  struct _WITH_ENVIRONMENT_ { };

  template <class _Ty>
  struct _WITH_TYPE_;

  struct _WITH_RECEIVER_ { };

  template <class _Sig>
  struct _UNHANDLED_COMPLETION_SIGNAL_;

  template <class _Sig>
  struct _WITH_COMPLETION_SIGNATURE_;

  struct _WITH_COMPLETION_SIGNATURES_ { };

  struct _WITH_FUNCTION_ { };

  struct _WITH_ARGUMENTS_ { };

  struct _WITH_QUERY_ { };

  struct _TO_FIX_THIS_ERROR_ { };

  struct _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_ { };

  struct _TYPE_IS_NOT_DECAY_COPYABLE_ { };

  struct _WITH_METAFUNCTION_ { };

  struct _INVALID_ARGUMENT_ { };

  struct _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_ { };

  struct _CANNOT_PIPE_ONE_SENDER_INTO_ANOTHER_ { };

  struct _DOMAIN_ERROR_ { };

  struct _INVALID_EXPRESSION_ { };

  struct _CONCEPT_CHECK_FAILURE_ { };

  struct _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<
    _WHAT_(_INVALID_EXPRESSION_),
    _WHY_(_CANNOT_PIPE_ONE_SENDER_INTO_ANOTHER_),
    _WITH_PRETTY_SENDER_<_Sender>
  >;

  template <class _Tag, class _Fun, class... _Args>
  using __callable_error_t = __mexception<
    _WHAT_(_INVALID_EXPRESSION_),
    _WHY_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
    _WHERE_(_IN_ALGORITHM_, _Tag),
    _WITH_FUNCTION_(_Fun),
    _WITH_ARGUMENTS_(_Args...)
  >;

  struct _UNABLE_TO_COMPUTE_THE_SENDER_COMPLETION_SIGNATURES_ { };

  template <class _Sender, class... _Env>
  using __unrecognized_sender_error_t = __mexception<
    _WHAT_(_UNRECOGNIZED_SENDER_TYPE_),
    _WHY_(_UNABLE_TO_COMPUTE_THE_SENDER_COMPLETION_SIGNATURES_),
    _WITH_PRETTY_SENDER_<_Sender>,
    _WITH_ENVIRONMENT_(_Env)...
  >;

#if __cpp_lib_constexpr_exceptions >= 2025'02L // constexpr exception types, https://wg21.link/p3378

  using __exception = ::std::exception;

#elif __cpp_constexpr >= 2024'11L // constexpr virtual functions

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

#endif // __cpp_lib_constexpr_exceptions >= 2025'02L

  template <class _Derived>
  struct __compile_time_error : __exception {
    constexpr __compile_time_error() = default; // NOLINT (bugprone-crtp-constructor-accessibility)

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

  // A specialization of _ERROR_ to be used to report dependent sender. It inherits
  // from dependent_sender_error.
  template <class... _What>
  struct _ERROR_<dependent_sender_error, _What...> : dependent_sender_error {
    using __t = _ERROR_;
    using __partitioned = _ERROR_;

    template <class, class>
    using __value_types = _ERROR_;

    template <class, class>
    using __error_types = _ERROR_;

    template <class, class>
    using __stopped_types = _ERROR_;

    using __decay_copyable = _ERROR_;
    using __nothrow_decay_copyable = _ERROR_;
    using __values = _ERROR_;
    using __errors = _ERROR_;
    using __all = _ERROR_;

    constexpr _ERROR_() noexcept
      : dependent_sender_error{
          "This sender needs to know its execution environment before it can know how it will "
          "complete."} {
    }

    STDEXEC_ATTRIBUTE(host, device) constexpr auto operator+() const -> _ERROR_;

    template <class _Ty>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator,(const _Ty&) const -> _ERROR_;

    template <class... Other>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator,(const _ERROR_<Other...>&) const -> _ERROR_<Other...>;
  };

  // By making __dependent_sender_error_t an alias for _ERROR_<...>, we ensure that
  // it will get propagated correctly through various metafunctions.
  template <class _Sender>
  using __dependent_sender_error_t = _ERROR_<dependent_sender_error, _WITH_PRETTY_SENDER_<_Sender>>;

  template <class... _What>
  struct __not_a_sender {
    using sender_concept = sender_t;

    template <class _Self>
    static consteval auto get_completion_signatures() {
      return STDEXEC::__throw_compile_time_error<_What...>();
    }
  };

  template <class... _What>
  struct __not_a_scheduler {
    using scheduler_concept = scheduler_t;

    constexpr auto schedule() noexcept {
      return __not_a_sender<_What...>{};
    }

    constexpr bool operator==(const __not_a_scheduler&) const noexcept = default;
  };
} // namespace STDEXEC

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE                                                       \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "The given type is not a sender because STDEXEC::enable_sender<Sender> is false. Either:\n"      \
  "\n"                                                                                             \
  "1. Give the type a nested `::sender_concept` type that is an alias for `STDEXEC::sender_t`,\n"  \
  "   as in:\n"                                                                                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept = STDEXEC::sender_t;\n"                                             \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. Specialize the `STDEXEC::enable_sender` boolean trait for this type to true, as follows:\n"  \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "     template <>\n"                                                                             \
  "     inline constexpr bool STDEXEC::enable_sender<MySender> = true;\n"

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
  "  specialization of `STDEXEC::completion_signatures<...>`, as follows:\n"                       \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = STDEXEC::sender_t;\n"                                      \
  "       using completion_signatures = STDEXEC::completion_signatures<\n"                         \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         STDEXEC::set_value_t(int, float),\n"                                                   \
  "         // ... or in error with an exception_ptr\n"                                            \
  "         STDEXEC::set_error_t(std::exception_ptr)>;\n"                                          \
  "       ...\n"                                                                                   \
  "     };\n"                                                                                      \
  "\n"                                                                                             \
  "   or,\n"                                                                                       \
  "\n"                                                                                             \
  "2. By defining a member function named `get_completion_signatures` that returns\n"              \
  "   a specialization of `STDEXEC::completion_signatures<...>`, as follows:\n"                    \
  "\n"                                                                                             \
  "     class MySender\n"                                                                          \
  "     {\n"                                                                                       \
  "     public:\n"                                                                                 \
  "       using sender_concept        = STDEXEC::sender_t;\n"                                      \
  "\n"                                                                                             \
  "       template <class Self, class... Env>\n"                                                   \
  "       static consteval auto get_completion_signatures() -> STDEXEC::completion_signatures<\n"  \
  "         // This sender can complete successfully with an int and a float...\n"                 \
  "         STDEXEC::set_value_t(int, float),\n"                                                   \
  "         // ... or in error with a std::exception_ptr.\n"                                       \
  "         STDEXEC::set_error_t(std::exception_ptr)>\n"                                           \
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
  "`STDEXEC::completion_signatures<...>`, as follows:\n"                                           \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept = STDEXEC::sender_t;\n"                                                \
  "\n"                                                                                             \
  "    template <class Self, class... Env>\n"                                                      \
  "    static consteval auto get_completion_signatures() -> STDEXEC::completion_signatures<\n"     \
  "      // This sender can complete successfully with an int and a float...\n"                    \
  "      STDEXEC::set_value_t(int, float),\n"                                                      \
  "      // ... or in error with a std::exception_ptr.\n"                                          \
  "      STDEXEC::set_error_t(std::exception_ptr)>\n"                                              \
  "    {\n"                                                                                        \
  "      return {};\n"                                                                             \
  "    }\n"                                                                                        \
  "    ...\n"                                                                                      \
  "  };\n"

////////////////////////////////////////////////////////////////////////////////
#define STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER                                            \
  "\n"                                                                                             \
  "A sender must provide a `connect` member function that takes a receiver as an\n"                \
  "argument and returns an object whose type satisfies `STDEXEC::operation_state`,\n"              \
  "as shown below:\n"                                                                              \
  "\n"                                                                                             \
  "  class MySender\n"                                                                             \
  "  {\n"                                                                                          \
  "  public:\n"                                                                                    \
  "    using sender_concept        = STDEXEC::sender_t;\n"                                         \
  "    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;\n"    \
  "\n"                                                                                             \
  "    template <class Receiver>\n"                                                                \
  "    struct MyOpState\n"                                                                         \
  "    {\n"                                                                                        \
  "      using operation_state_concept = STDEXEC::operation_state_t;\n"                            \
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
  "    template <STDEXEC::receiver Receiver>\n"                                                    \
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
  "The sender passed to `STDEXEC::sync_wait()` does not have a `connect`\n"                        \
  "member function that accepts sync_wait's "                                                      \
  "receiver.\n" STDEXEC_ERROR_CANNOT_CONNECT_SENDER_TO_RECEIVER
