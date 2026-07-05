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
#include "__concepts.hpp"
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"  // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"
#include "__utility.hpp"

#include <exception>
#include <tuple>
#include <variant>  // IWYU pragma: keep

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.into.variant]
  namespace __into_variant
  {
    template <class _Sender, class _Env>
      requires sender_in<_Sender, _Env>
    using __into_variant_result_t = value_types_of_t<_Sender, _Env>;

    template <class _Sender, class... _Env>
    using __variant_t = __value_types_t<__completion_signatures_of_t<_Sender, _Env...>,
                                        __qq<__decayed_std_tuple>,
                                        __qq<__std_variant>>;

    template <class _Variant>
    using __variant_completions =
      completion_signatures<set_value_t(_Variant), set_error_t(std::exception_ptr)>;

    template <class _Sender, class... _Env>
    using __completions = __transform_completion_signatures_t<
      __completion_signatures_of_t<_Sender, _Env...>,
      __minvoke_q<__variant_completions, __variant_t<_Sender, _Env...>>,
      __mconst<completion_signatures<>>::__f>;

    template <class _Receiver, class _Variant>
    struct __state
    {
      using __variant_t = _Variant;
      _Receiver __rcvr_;
    };

    struct __into_variant_impl : __sexpr_defaults
    {
      static constexpr auto __get_state =
        []<class _Self, class _Receiver>(_Self&&, _Receiver&& __rcvr) noexcept
      {
        using __variant_t = value_types_of_t<__child_of<_Self>, env_of_t<_Receiver>>;
        return __state<_Receiver, __variant_t>{static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto __complete =
        []<class _State, class _Tag, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        if constexpr (__same_as<_Tag, set_value_t>)
        {
          using __variant_t = _State::__variant_t;
          STDEXEC_TRY
          {
            STDEXEC::set_value(static_cast<_State&&>(__state).__rcvr_,
                               __variant_t{std::in_place_type<__decayed_std_tuple<_Args...>>,
                                           std::tuple<_Args&&...>{
                                             static_cast<_Args&&>(__args)...}});
          }
          STDEXEC_CATCH_ALL
          {
            STDEXEC::set_error(static_cast<_State&&>(__state).__rcvr_, std::current_exception());
          }
        }
        else
        {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Self, into_variant_t>);
        return __completions<__child_of<_Self>, _Env...>{};
      };
    };
  }  // namespace __into_variant

  //! @brief A pipeable sender adaptor that collapses a sender's multiple
  //!        value-completion signatures into a single
  //!        @c std::variant-of-tuples value completion.
  //!
  //! @c into_variant takes a sender whose @c set_value_t completion can be
  //! one of several shapes — e.g. <tt>set_value_t(int)</tt> *or*
  //! <tt>set_value_t(std::string)</tt> — and produces a sender that always
  //! value-completes with exactly one shape: a single
  //! `std::variant<std::tuple<Vs1...>, std::tuple<Vs2...>, ...>` datum
  //! whose alternatives match the input's possible value completions.
  //!
  //! This is the building block behind @c when_all_with_variant: it lifts
  //! any sender into the *single-value-completion* category that
  //! @c when_all and @c sync_wait require.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::into_variant(sndr);    // direct invocation
  //! auto s2 = sndr | stdexec::into_variant(); // pipe syntax (no args)
  //! @endcode
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with value-completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs1...)              // possibly several
  //! set_value_t(Vs2...)
  //! set_error_t(Es)...               // forwarded unchanged
  //! set_stopped_t()                  // forwarded unchanged
  //! @endcode
  //!
  //! the sender produced by <tt>into_variant(sndr)</tt> has completion
  //! signatures
  //!
  //! @code{.cpp}
  //! set_value_t(std::variant<std::tuple<Vs1...>, std::tuple<Vs2...>, ...>)
  //! set_error_t(Es)...               // unchanged
  //! set_error_t(std::exception_ptr)  // added if variant construction may throw
  //! set_stopped_t()                  // unchanged
  //! @endcode
  //!
  //! When @c sndr value-completes with arguments matching @c Vsi... the
  //! resulting sender value-completes with a variant engaged on the
  //! corresponding alternative.
  //!
  //! **Exception behavior.**
  //!
  //! If constructing the variant alternative (which involves decay-copying
  //! the original value arguments) throws, the exception is delivered
  //! through @c set_error_t(std::exception_ptr).
  //!
  //! **Cancellation.**
  //!
  //! @c into_variant does not interact with the stop token; it only
  //! reshapes the value channel.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //!
  //! // Suppose sndr can value-complete with either int or std::string.
  //! auto wrapped = stdexec::into_variant(sndr);
  //! auto [v]     = stdexec::sync_wait(std::move(wrapped)).value();
  //! //  v: std::variant<std::tuple<int>, std::tuple<std::string>>
  //! std::visit([](auto&& tup) { use(tup); }, v);
  //! @endcode
  //!
  //! @see stdexec::when_all_with_variant  — applies @c into_variant to each input internally
  //! @see stdexec::sync_wait_with_variant — variant-aware top-level wait
  //! @see stdexec::sync_wait              — requires a single value-completion shape
  struct into_variant_t
  {
    //! @brief Construct a sender that value-completes with a
    //!        @c std::variant of the possible value-completion tuples of
    //!        @c __sndr.
    //!
    //! @tparam _Sender A type satisfying @c stdexec::sender.
    //! @param __sndr   The predecessor sender. Forwarded into the result.
    //!
    //! @returns A sender with a single @c set_value_t completion whose
    //!          argument is a @c std::variant of `std::tuple<Vs...>`
    //!          alternatives.
    template <sender _Sender>
    constexpr auto operator()(_Sender&& __sndr) const -> __well_formed_sender auto
    {
      return __make_sexpr<into_variant_t>(__(), static_cast<_Sender&&>(__sndr));
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a
    //!        sender, produces <tt>into_variant(sndr)</tt>.
    //!
    //! This overload enables the pipe syntax:
    //! <tt>sndr | into_variant()</tt> is equivalent to
    //! <tt>into_variant(sndr)</tt>.
    //!
    //! @returns A sender-adaptor closure object.
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const noexcept
    {
      return __closure(*this);
    }
  };

  //! @brief The customization point object for the @c into_variant sender adaptor.
  //!
  //! @c into_variant is an instance of @ref into_variant_t. See
  //! @ref into_variant_t for the full description and a usage example.
  //!
  //! @hideinitializer
  inline constexpr into_variant_t into_variant{};

  template <>
  struct __sexpr_impl<into_variant_t> : __into_variant::__into_variant_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
