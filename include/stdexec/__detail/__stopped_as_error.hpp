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
#include "__concepts.hpp"
#include "__just.hpp"
#include "__let.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.stopped.err]

  //! @brief A pipeable sender adaptor that converts a predecessor's stopped
  //!        completion into an error completion carrying a user-supplied
  //!        error value.
  //!
  //! @c stopped_as_error is a "translator" adaptor: it doesn't change what
  //! happens in the value channel, but it rewrites a @c set_stopped
  //! completion into a @c set_error completion with a specific datum.
  //! This is useful when downstream code needs to *distinguish* a
  //! cancellation from a real error and you want the cancellation to be
  //! delivered as an error of a particular type (typically because the
  //! consumer can't or won't handle the stopped channel).
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::stopped_as_error(sndr, my_error);
  //! auto s2 = sndr | stdexec::stopped_as_error(my_error);
  //! @endcode
  //!
  //! **Equivalence.**
  //!
  //! <tt>stopped_as_error(sndr, err)</tt> is implemented (and is
  //! observationally equivalent to)
  //! <tt>let_stopped(sndr, [err]{ return just_error(err); })</tt>.
  //! Use this adaptor whenever you would have written that pattern by
  //! hand — it is shorter, clearer at the call site, and the
  //! implementation can be specialized more efficiently in the future.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)    // forwarded unchanged
  //! set_error_t(Es)...    // forwarded unchanged
  //! set_stopped_t()       // consumed
  //! @endcode
  //!
  //! the sender produced by <tt>stopped_as_error(sndr, err)</tt> has
  //! completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)              // forwarded unchanged
  //! set_error_t(Es)...              // forwarded unchanged
  //! set_error_t(std::decay_t<E>)    // the supplied error, decay-copied
  //! @endcode
  //!
  //! The original @c set_stopped_t completion is replaced; the resulting
  //! sender will never deliver @c set_stopped.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! using namespace stdexec;
  //!
  //! auto sndr = just_stopped()
  //!           | stopped_as_error(std::runtime_error{"cancelled"});
  //!
  //! try {
  //!   sync_wait(std::move(sndr));    // throws std::runtime_error
  //! } catch (std::runtime_error const& e) {
  //!   // e.what() == "cancelled"
  //! }
  //! @endcode
  //!
  //! @see stdexec::stopped_as_optional  — convert stopped into a value-channel @c std::nullopt
  //! @see stdexec::upon_stopped         — handle stopped synchronously
  //! @see stdexec::let_stopped          — handle stopped with a sender-returning callback
  struct stopped_as_error_t
  {
    //! @brief Construct a sender that translates @c __sndr's @c set_stopped
    //!        completion into a @c set_error completion carrying @c __err.
    //!
    //! @tparam _Sender A type satisfying @c stdexec::sender.
    //! @tparam _Error  A decayed, move-constructible error datum type
    //!                 (satisfying the internal <tt>__movable_value</tt> concept).
    //!
    //! @param __sndr   The predecessor sender. Forwarded into the result.
    //! @param __err    The error datum to deliver if @c __sndr is stopped.
    //!                 Decay-copied into the resulting sender.
    template <sender _Sender, __movable_value _Error>
    constexpr auto operator()(_Sender&& __sndr, _Error __err) const -> __well_formed_sender auto
    {
      return let_stopped(static_cast<_Sender&&>(__sndr),
                         [__err2 = static_cast<_Error&&>(__err)]() mutable noexcept(
                           __nothrow_move_constructible<_Error>)
                         { return just_error(static_cast<_Error&&>(__err2)); });
    }

    //! @brief Construct a sender-adaptor closure for the pipe form.
    //!
    //! <tt>sndr | stopped_as_error(__err)</tt> is equivalent to
    //! <tt>stopped_as_error(sndr, __err)</tt>.
    template <__movable_value _Error>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Error __err) const noexcept(__nothrow_move_constructible<_Error>)
    {
      return __closure(*this, static_cast<_Error&&>(__err));
    }
  };

  //! @brief The customization point object for the @c stopped_as_error sender adaptor.
  //!
  //! @c stopped_as_error is an instance of @ref stopped_as_error_t. See
  //! @ref stopped_as_error_t for the full description and a usage example.
  //!
  //! @hideinitializer
  inline constexpr stopped_as_error_t stopped_as_error{};
}  // namespace STDEXEC

#include "__epilogue.hpp"
