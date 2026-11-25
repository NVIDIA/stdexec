/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "__completion_signatures.hpp"
#  include "__concepts.hpp"
#  include "__operation_states.hpp"
#  include "__optional.hpp"
#  include "__receivers.hpp"
#  include "__scope_concepts.hpp"
#  include "__sender_adaptor_closure.hpp"
#  include "__sender_concepts.hpp"
#  include "__sender_facade.hpp"
#  include "__sender_introspection.hpp"
#  include "__senders.hpp"
#  include "__type_traits.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <memory>
#    include <type_traits>
#    include <utility>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // [exec.associate]
  namespace __associate
  {
    template <scope_token _Token, sender _Sender>
    struct __associate_data
    {
      using __wrap_result_t = decltype(__declval<_Token&>().wrap(__declval<_Sender>()));
      using __wrap_sender_t = std::remove_cvref_t<__wrap_result_t>;

      using __assoc_t = decltype(__declval<_Token&>().try_associate());

      // NOTE: the spec says the deleter should be a lambda like so:
      //
      //         using __sender_ref = std::unique_ptr<
      //             __wrap_sender_t,
      //             // this decltype(<lamnda>) breaks things
      //             decltype([](auto* p) noexcept { std::destroy_at(p); })
      //         >;
      //
      //       but the above code ICEs gcc 11 and 12 (and maybe MSVC)
      //       so we declare a named callable
      struct __deleter
      {
        constexpr void operator()(__wrap_sender_t* __p) const noexcept
        {
          std::destroy_at(__p);
        }
      };

      using __sender_ref = std::unique_ptr<__wrap_sender_t, __deleter>;

      // BUGBUG: should the spec require __token to be declared as a const
      // _Token, or should this be changed to declare __token as a mutable
      // _Token?
      explicit __associate_data(_Token const __token, _Sender&& __sndr)
        noexcept(__nothrow_constructible_from<__wrap_sender_t, __wrap_result_t>
                 && noexcept(__token.wrap(static_cast<_Sender&&>(__sndr)))
                 && noexcept(__token.try_associate()))
        : __sndr_(__token.wrap(static_cast<_Sender&&>(__sndr)))
        , __assoc_(
            [&]
            {
              __sender_ref guard{std::addressof(__sndr_)};

              auto assoc = __token.try_associate();

              if (assoc)
              {
                (void) guard.release();
              }

              return assoc;
            }())
      {}

      __associate_data(__associate_data const & __other)
        noexcept(__nothrow_copy_constructible<__wrap_sender_t>
                 && noexcept(__other.__assoc_.try_associate()))
        requires __std::copy_constructible<__wrap_sender_t>
        : __assoc_(__other.__assoc_.try_associate())
      {
        if (__assoc_)
        {
          std::construct_at(std::addressof(__sndr_), __other.__sndr_);
        }
      }

      __associate_data(__associate_data&& __other)
        noexcept(__nothrow_move_constructible<__wrap_sender_t>)
        : __associate_data(std::move(__other).release())
      {}

      ~__associate_data()
      {
        if (__assoc_)
        {
          std::destroy_at(&__sndr_);
        }
      }

      std::pair<__assoc_t, __sender_ref> release() && noexcept
      {
        __sender_ref __u(__assoc_ ? std::addressof(__sndr_) : nullptr);
        return {std::move(__assoc_), std::move(__u)};
      }

     private:
      explicit __associate_data(std::pair<__assoc_t, __sender_ref> __parts)
        : __assoc_(std::move(__parts.first))
      {
        if (__assoc_)
        {
          std::construct_at(std::addressof(__sndr_), std::move(*__parts.second));
        }
      }

      union
      {
        __wrap_sender_t __sndr_;
      };
      __assoc_t __assoc_;
    };

    template <scope_token _Token, sender _Sender>
    __associate_data(_Token, _Sender&&) -> __associate_data<_Token, _Sender>;

    ////////////////////////////////////////////////////////////////////////////
    STDEXEC_MODULE_EXPORT
    struct associate_t
    {
      template <sender _Sender, scope_token _Token>
      auto operator()(_Sender&& __sndr, _Token __token) const
        noexcept(__nothrow_constructible_from<__associate_data<_Token, _Sender>, _Token, _Sender>)
          -> __well_formed_sender auto
      {
        return __facade::__sender{associate_t(),
                                  __associate_data(static_cast<_Token&&>(__token),
                                                   static_cast<_Sender&&>(__sndr))};
      }

      template <scope_token _Token>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Token __token) const noexcept
      {
        return __closure(*this, static_cast<_Token&&>(__token));
      }
    };

    // The key of the operation of the wrapped sender
    struct __wrapped_op
    {};

    struct __algo;

    // NOTE: the spec declares this class template inside the get_state function
    //       but I couldn't get that to build with Clang 21 so I moved it out to
    //       this namespace-scoped template and __uglified all the symbols
    template <class _Sender, class _Receiver>
    struct __state
    {
      using __associate_data_t = std::remove_cvref_t<__data_of<_Sender>>;
      using __assoc_t          = __associate_data_t::__assoc_t;
      using __sender_ref_t     = __associate_data_t::__sender_ref;
      using __wrap_sender_t    = __associate_data_t::__wrap_sender_t;
      using __wrapped_rcvr_t =
        __facade::__receiver_t<__algo, __wrapped_op, __state, _Sender, _Receiver>;
      using __op_t = connect_result_t<__wrap_sender_t, __wrapped_rcvr_t>;

      // The wrapped sender's operation is connected only if the association succeeded.
      explicit __state(std::pair<__assoc_t, __sender_ref_t> __parts, _Receiver&& __rcvr)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __assoc_(STDEXEC::__allocator_aware_forward(std::move(__parts.first), __rcvr_))
      {
        if (__assoc_)
        {
          __op_.__emplace_from(STDEXEC::connect,
                               std::move(*__parts.second),
                               __wrapped_rcvr_t{this});
        }
      }

      explicit __state(__associate_data_t&& __ad, _Receiver&& __rcvr)
        : __state(std::move(__ad).release(), static_cast<_Receiver&&>(__rcvr))
      {}

      explicit __state(__associate_data_t const & __ad, _Receiver&& __rcvr)
        requires __std::copy_constructible<__associate_data_t>
        : __state(__associate_data_t(__ad).release(), static_cast<_Receiver&&>(__rcvr))
      {}

      STDEXEC_IMMOVABLE(__state);

      _Receiver          __rcvr_;
      __assoc_t          __assoc_;
      __optional<__op_t> __op_{};
    };

    // associate(sndr, token): if the token associated the wrapped sender with its scope,
    // the wrapped sender's operation runs, and its completions are the operation's.
    // Otherwise the operation completes with set_stopped_t() from start.
    struct __algo : __facade::__defaults
    {
      using __facade::__defaults::__nested_sender;

      // The wrapped sender, which the data holds only if the association succeeded, so
      // only its type is known
      template <class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __nested_sender(__wrapped_op, _Sender const &, _Env const &...) noexcept
        -> __mtype<typename __decay_t<__facade::__data_of_t<_Sender>>::__wrap_sender_t>
      {
        return {};
      }

      template <class _Sender, class _Receiver>
      static constexpr auto __get_state(_Sender&& __sndr, _Receiver __rcvr)
        noexcept((!std::is_lvalue_reference_v<_Sender>
                  || __nothrow_copy_constructible<std::remove_cvref_t<__data_of<_Sender>>>)
                 && __nothrow_connectable<typename __state<_Sender, _Receiver>::__wrap_sender_t,
                                          typename __state<_Sender, _Receiver>::__wrapped_rcvr_t>)
          -> __state<_Sender, _Receiver>
      {
        return __state<_Sender, _Receiver>(__facade::__get_data(static_cast<_Sender&&>(__sndr)),
                                           static_cast<_Receiver&&>(__rcvr));
      }

      template <class _State, class _Children>
      static constexpr auto __start(_State& __state, _Children&) noexcept
      {
        return __facade::__choose(
          static_cast<bool>(__state.__assoc_),
          [&]() noexcept { return __facade::__start_nested<__wrapped_op>(*__state.__op_); },
          [&]() noexcept { return __facade::__send(__state.__rcvr_, set_stopped_t()); });
      }
    };
  }  // namespace __associate

  STDEXEC_MODULE_EXPORT
  using __associate::associate_t;

  //! @brief The associate sender adaptor, which associates a sender with the
  //!        async scope referred to by the given token
  //! @hideinitializer
  STDEXEC_MODULE_EXPORT
  inline constexpr associate_t associate{};

  template <>
  struct __facade::__algorithm<associate_t> : __associate::__algo
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
