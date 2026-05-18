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
#include "__basic_sender.hpp"
#include "__env.hpp"
#include "__queries.hpp"
#include "__sender_adaptor_closure.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write
  {
    struct __write_env_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept
      {
        return __sync_attrs{__child};
      };

      static constexpr auto __get_env = []<class _State>(__ignore, _State const & __state) noexcept
        -> decltype(__env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_)))
      {
        return __env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_));
      };

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Self, __write_env_t>);
        return STDEXEC::get_completion_signatures<
          __child_of<_Self>,
          __minvoke_q<__join_env_t, __decay_t<__data_of<_Self>> const &, _Env>...>();
      }
    };
  }  // namespace __write

  struct __write_env_t
  {
    template <sender _Sender, class _Env>
    constexpr auto operator()(_Sender&& __sndr, _Env __env) const
    {
      return __make_sexpr<__write_env_t>(static_cast<_Env&&>(__env),
                                         static_cast<_Sender&&>(__sndr));
    }

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Env __env) const
    {
      return __closure(*this, static_cast<_Env&&>(__env));
    }
  };

  //! @brief A pipeable sender adaptor that augments the environment seen by
  //!        a predecessor sender with additional queries.
  //!
  //! @c write_env is the inverse of @ref read_env. Where @c read_env *reads*
  //! a value from the receiver's environment and exposes it on the value
  //! channel, @c write_env *injects* values into the environment a child
  //! sender sees — overriding or augmenting what the eventual receiver
  //! exposes.
  //!
  //! You give it a sender and an environment (typically built with
  //! @c stdexec::env and @c stdexec::prop); you get back a sender that,
  //! when connected, presents the *union* of the supplied environment and
  //! the connected receiver's environment to its predecessor. Anything the
  //! predecessor reaches for via @c get_env / @c read_env sees the merged
  //! view.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::write_env(sndr, env);
  //! auto s2 = sndr | stdexec::write_env(env);
  //! @endcode
  //!
  //! The supplied environment shadows the receiver's environment for any
  //! query the supplied environment can answer; queries it cannot answer
  //! fall through to the receiver's environment unchanged.
  //!
  //! **Common uses.**
  //!
  //! - Injecting a stop token: <tt>sndr | write_env(prop{get_stop_token, my_token})</tt>
  //!   so a sub-pipeline observes a different cancellation signal than the
  //!   outer pipeline.
  //! - Supplying an allocator: <tt>sndr | write_env(prop{get_allocator, my_alloc})</tt>
  //!   so child operations allocate via @c my_alloc.
  //! - Hooking domain customization: a custom scheduler may inject its
  //!   domain into the environment for senders that don't have a scheduler
  //!   in their chain.
  //!
  //! **Completion signatures.**
  //!
  //! @c write_env preserves the predecessor's completion signatures
  //! unchanged. (The predecessor may compute different signatures
  //! depending on what's in its environment — so the supplied env may
  //! influence which signatures the framework computes — but
  //! @c write_env does not itself add or remove any.)
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! using namespace stdexec;
  //!
  //! auto inner_sndr = read_env(get_stop_token)
  //!                 | then([](auto tok) { return tok.stop_requested(); });
  //!
  //! stop_source src;
  //! auto pipeline = inner_sndr
  //!               | write_env(prop{get_stop_token, src.get_token()});
  //!
  //! auto [requested] = sync_wait(std::move(pipeline)).value();
  //! // requested == src.stop_requested(), regardless of the outer
  //! // pipeline's stop token.
  //! @endcode
  //!
  //! @see stdexec::read_env  — read a value from the environment
  //! @see stdexec::env       — construct an environment from properties
  //! @see stdexec::prop      — bind a query CPO to a value
  //! @see stdexec::get_env   — the CPO that exposes the merged environment
  //!
  //! @hideinitializer
  inline constexpr __write_env_t write_env{};

  template <>
  struct __sexpr_impl<__write_env_t> : __write::__write_env_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
