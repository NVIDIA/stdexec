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

#include "__config.hpp"
#include "__diagnostic_macros.hpp"  // IWYU pragma: export

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__meta.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>  // IWYU pragma: keep for std::exception
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
  struct sender_tag;
  struct scheduler_tag;

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WHAT_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WHERE_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WHY_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _IN_ALGORITHM_
  {};

  struct _UNRECOGNIZED_SENDER_TYPE_;

  template <class _Sender>
  struct _WITH_SENDER_
  {};

  template <class... _Senders>
  struct _WITH_SENDERS_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sender>
  using _WITH_PRETTY_SENDER_ = _WITH_SENDER_<__demangle_t<_Sender>>;

  template <class... _Senders>
  using _WITH_PRETTY_SENDERS_ = _WITH_SENDERS_<__demangle_t<_Senders>...>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WITH_ENVIRONMENT_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Ty>
  struct _WITH_TYPE_;

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WITH_RECEIVER_
  {};

  template <class _Sig>
  struct _UNHANDLED_COMPLETION_SIGNAL_;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sig>
  struct _WITH_COMPLETION_SIGNATURE_;

  struct _WITH_COMPLETION_SIGNATURES_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WITH_FUNCTION_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WITH_ARGUMENTS_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _WITH_QUERY_
  {};

  struct _WITH_SCHEDULER_
  {};

  struct _WITH_ALLOCATOR_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _TO_FIX_THIS_ERROR_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_
  {};

  struct _TYPE_IS_NOT_DECAY_COPYABLE_
  {};

  struct _WITH_METAFUNCTION_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _INVALID_ARGUMENT_
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_
  {};

  struct _CANNOT_PIPE_ONE_SENDER_INTO_ANOTHER_
  {};

  struct _DOMAIN_ERROR_
  {};

  struct _INVALID_EXPRESSION_
  {};

  struct _CONCEPT_CHECK_FAILURE_
  {};

  struct _THE_CURRENT_EXECUTION_ENVIRONMENT_DOESNT_HAVE_A_SCHEDULER_
  {};

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_WHAT_(_INVALID_EXPRESSION_),
                                         _WHY_(_CANNOT_PIPE_ONE_SENDER_INTO_ANOTHER_),
                                         _WITH_PRETTY_SENDER_<_Sender>>;

  template <class _Tag, class _Fun, class... _Args>
  using __callable_error_t =
    __mexception<_WHAT_(_INVALID_EXPRESSION_),
                 _WHY_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
                 _WHERE_(_IN_ALGORITHM_, _Tag),
                 _WITH_FUNCTION_(_Fun),
                 _WITH_ARGUMENTS_(_Args...)>;

  struct _UNABLE_TO_COMPUTE_THE_SENDER_COMPLETION_SIGNATURES_
  {};

  template <class _Sender, class... _Env>
  using __unrecognized_sender_error_t =
    __mexception<_WHAT_(_UNRECOGNIZED_SENDER_TYPE_),
                 _WHY_(_UNABLE_TO_COMPUTE_THE_SENDER_COMPLETION_SIGNATURES_),
                 _WITH_PRETTY_SENDER_<_Sender>,
                 _WITH_ENVIRONMENT_(_Env)...>;

#  if __cpp_lib_constexpr_exceptions >= 202502L

  // constexpr stdlib exception types, https://wg21.link/p3378
  using __exception = ::std::exception;

#  elif __cpp_constexpr >= 201907L && !STDEXEC_MSVC() && !STDEXEC_NVHPC()

  // constexpr virtual functions
  struct __exception
  {
    constexpr __exception() noexcept = default;
    constexpr virtual ~__exception() = default;

    [[nodiscard]]
    constexpr virtual auto what() const noexcept -> char const *
    {
      return "<exception>";
    }
  };

#  else

  // no constexpr virtual functions
  struct __exception
  {
    constexpr __exception() noexcept = default;

    [[nodiscard]]
    constexpr auto what() const noexcept -> char const *
    {
      return "<exception>";
    }
  };

#  endif  // __cpp_lib_constexpr_exceptions >= 202502L

  STDEXEC_MODULE_EXPORT_AUTHORING
  struct __compile_time_error : __exception
  {};

  template <class _Data, class... _What>
  struct __sender_type_check_failure : __compile_time_error
  {
    static_assert(std::is_nothrow_move_constructible_v<_Data>,
                  "The data member of sender_type_check_failure must be nothrow move "
                  "constructible.");

    constexpr __sender_type_check_failure() noexcept = default;

    constexpr explicit __sender_type_check_failure(_Data data)
      : __data_(static_cast<_Data &&>(data))
    {}

    [[nodiscard]]
    constexpr auto what() const noexcept -> char const *  // NOLINT(modernize-use-override)
    {
      return "This sender is not well-formed. It does not meet the requirements of a sender type.";
    }

    // public so that __sender_type_check_failure is a structural type
    _Data __data_{};
  };

  struct dependent_sender_error : __compile_time_error
  {
    [[nodiscard]]
    constexpr auto what() const noexcept -> char const *  // NOLINT(modernize-use-override)
    {
      return "This sender needs to know its execution environment before it can "
             "know how it will complete.";
    }
  };

  // A specialization of _ERROR_ to be used to report dependent sender. It inherits
  // from dependent_sender_error.
  template <class... _What>
  struct _ERROR_<dependent_sender_error, _What...> : dependent_sender_error
  {
    using __t           = _ERROR_;
    using __partitioned = _ERROR_;

    template <class, class>
    using __value_types = _ERROR_;

    template <class, class>
    using __error_types = _ERROR_;

    template <class, class>
    using __stopped_types = _ERROR_;

    using __decay_copyable         = _ERROR_;
    using __nothrow_decay_copyable = _ERROR_;
    using __values                 = _ERROR_;
    using __errors                 = _ERROR_;
    using __all                    = _ERROR_;

    constexpr _ERROR_()  = default;
    constexpr ~_ERROR_() = default;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator+() const -> _ERROR_;

    template <class _Ty>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator,(_Ty const &) const -> _ERROR_
    {
      return *this;
    }

    template <class... Other>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator,(const _ERROR_<Other...> &__other) const -> _ERROR_<Other...>
    {
      return __other;
    }
  };

  static_assert(__structural<_ERROR_<dependent_sender_error>>);

  // By making __dependent_sender_error_t an alias for _ERROR_<...>, we ensure that
  // it will get propagated correctly through various metafunctions.
  template <class _Sender>
  using __dependent_sender_error_t = _ERROR_<dependent_sender_error, _WITH_PRETTY_SENDER_<_Sender>>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class... _What>
  struct __not_a_sender
  {
    using sender_concept = sender_tag;

    template <class _Self>
    static consteval auto get_completion_signatures()
    {
      return STDEXEC::__throw_compile_time_error<_What...>();
    }
  };

  template <class... _What>
  struct __not_a_scheduler
  {
    using scheduler_concept = scheduler_tag;

    constexpr auto schedule() noexcept
    {
      return __not_a_sender<_What...>{};
    }

    constexpr bool operator==(__not_a_scheduler const &) const noexcept = default;
  };
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
