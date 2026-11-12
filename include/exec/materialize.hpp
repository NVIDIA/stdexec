/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"
#include "completion_signatures.hpp"

namespace experimental::execution
{
  struct materialize_t;
  struct dematerialize_t;

  struct DEMATERIALIZE_EXPECTS_VALUE_COMPLETIONS_WHERE_THE_FIRST_VALUE_IS_A_COMPLETION_TAG;
  struct AN_ERROR_COMPLETION_REQUIRES_EXACTLY_ONE_ARGUMENT;
  struct A_STOPPED_COMPLETION_REQUIRES_EXACTLY_ZERO_ARGUMENTS;

  namespace _materialize
  {
    using namespace STDEXEC;

    template <class Receiver>
    struct _receiver
    {
      using receiver_concept = STDEXEC::receiver_tag;

      template <class... Args>
      constexpr void set_value(Args&&... args) noexcept
      {
        STDEXEC::set_value(static_cast<Receiver&&>(rcvr_),
                           set_value_t(),
                           static_cast<Args&&>(args)...);
      }

      template <class Error>
      constexpr void set_error(Error&& err) noexcept
      {
        STDEXEC::set_value(static_cast<Receiver&&>(rcvr_),
                           set_error_t(),
                           static_cast<Error&&>(err));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_value(static_cast<Receiver&&>(rcvr_), set_stopped_t());
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<Receiver>
      {
        return STDEXEC::get_env(rcvr_);
      }

      Receiver rcvr_;
    };

    template <class Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE _receiver(Receiver) -> _receiver<Receiver>;

    template <class Tag>
    static constexpr auto _materialize_cmplsigs = []<class... Args>() noexcept
    {
      return completion_signatures<set_value_t(Tag, Args...)>();
    };

    template <class Sender>
    struct _sender
    {
      using sender_concept = STDEXEC::sender_tag;

      template <__decays_to<_sender> Self, class Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        noexcept(__nothrow_connectable<__copy_cvref_t<Self, Sender>, _receiver<Receiver>>)
          -> connect_result_t<__copy_cvref_t<Self, Sender>, _receiver<Receiver>>
      {
        return STDEXEC::connect(static_cast<Self&&>(self).sndr_,
                                _receiver{static_cast<Receiver&&>(rcvr)});
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<_sender> Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        return exec::transform_completion_signatures(
          STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
          _materialize_cmplsigs<set_value_t>,
          _materialize_cmplsigs<set_error_t>,
          _materialize_cmplsigs<set_stopped_t>);
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<Sender>
      {
        return STDEXEC::get_env(sndr_);
      }

      Sender sndr_;
    };

    template <class Sender>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE _sender(Sender) -> _sender<Sender>;
  }  // namespace _materialize

  namespace _dematerialize
  {
    using namespace STDEXEC;

    template <class Receiver>
    struct _receiver
    {
      using receiver_concept = STDEXEC::receiver_tag;

      template <__completion_tag Tag, class... Args>
        requires __callable<Tag, Receiver, Args...>
      void set_value(Tag, Args&&... args) noexcept
      {
        Tag()(static_cast<Receiver&&>(rcvr_), static_cast<Args&&>(args)...);
      }

      template <class Error>
      constexpr void set_error(Error&& err) noexcept
      {
        STDEXEC::set_error(static_cast<Receiver&&>(rcvr_), static_cast<Error&&>(err));
      }

      constexpr void set_stopped() noexcept
      {
        STDEXEC::set_stopped(static_cast<Receiver&&>(rcvr_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<Receiver>
      {
        return STDEXEC::get_env(rcvr_);
      }

      Receiver rcvr_;
    };

    template <class Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE _receiver(Receiver) -> _receiver<Receiver>;

    static constexpr auto _dematerialize_cmplsigs = []<class Tag = void, class... Args>()
    {
      using tag_t = __decay_t<Tag>;
      if constexpr (!__completion_tag<tag_t>)
      {
        return exec::throw_compile_time_error<
          _WHAT_(_INVALID_ARGUMENT_),
          _WHY_(DEMATERIALIZE_EXPECTS_VALUE_COMPLETIONS_WHERE_THE_FIRST_VALUE_IS_A_COMPLETION_TAG),
          _WHERE_(_IN_ALGORITHM_, dematerialize_t),
          __if_c<__same_as<tag_t, void>, _WITH_ARGUMENTS_(), _WITH_ARGUMENTS_(Tag, Args...)>>();
      }
      else if constexpr (tag_t() == set_error && sizeof...(Args) != 1)
      {
        return exec::throw_compile_time_error<  //
          _WHAT_(_INVALID_ARGUMENT_),
          _WHY_(AN_ERROR_COMPLETION_REQUIRES_EXACTLY_ONE_ARGUMENT),
          _WHERE_(_IN_ALGORITHM_, dematerialize_t),
          _WITH_ARGUMENTS_(Args...)>();
      }
      else if constexpr (tag_t() == set_stopped && sizeof...(Args) != 0)
      {
        return exec::throw_compile_time_error<  //
          _WHAT_(_INVALID_ARGUMENT_),
          _WHY_(A_STOPPED_COMPLETION_REQUIRES_EXACTLY_ZERO_ARGUMENTS),
          _WHERE_(_IN_ALGORITHM_, dematerialize_t),
          _WITH_ARGUMENTS_(Args...)>();
      }
      else
      {
        return completion_signatures<tag_t(Args...)>();
      }
    };

    template <class Sender>
    struct _sender
    {
      using sender_concept = STDEXEC::sender_tag;

      template <__decays_to<_sender> Self, class Receiver>
        requires sender_to<__copy_cvref_t<Self, Sender>, _receiver<Receiver>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        noexcept(__nothrow_connectable<__copy_cvref_t<Self, Sender>, _receiver<Receiver>>)
          -> connect_result_t<__copy_cvref_t<Self, Sender>, _receiver<Receiver>>
      {
        return STDEXEC::connect(static_cast<Self&&>(self).sndr_,
                                _receiver{static_cast<Receiver&&>(rcvr)});
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<_sender> Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        return exec::transform_completion_signatures(
          STDEXEC::get_completion_signatures<__copy_cvref_t<Self, Sender>, Env...>(),
          _dematerialize_cmplsigs);
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<Sender>
      {
        return STDEXEC::get_env(sndr_);
      }

      Sender sndr_;
    };

    template <class Sender>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE _sender(Sender) -> _sender<Sender>;
  }  // namespace _dematerialize

  struct materialize_t
  {
    template <STDEXEC::sender Sender>
    constexpr auto operator()(Sender&& sndr) const  //
      noexcept(STDEXEC::__nothrow_decay_copyable<Sender>)
    {
      return _materialize::_sender{static_cast<Sender&&>(sndr)};
    }

    STDEXEC_ATTRIBUTE(always_inline)
    auto operator()() const noexcept
    {
      return STDEXEC::__closure(*this);
    }
  };

  inline constexpr materialize_t materialize{};

  struct dematerialize_t
  {
    template <STDEXEC::sender Sender>
    constexpr auto operator()(Sender&& sndr) const  //
      noexcept(STDEXEC::__nothrow_decay_copyable<Sender>)
    {
      return _dematerialize::_sender{static_cast<Sender&&>(sndr)};
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const noexcept
    {
      return STDEXEC::__closure(*this);
    }
  };

  inline constexpr dematerialize_t dematerialize{};
}  // namespace experimental::execution

namespace exec = experimental::execution;
