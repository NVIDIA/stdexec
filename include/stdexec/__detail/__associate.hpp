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

#include "__execution_fwd.hpp"

#include "__basic_sender.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__scope_concepts.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__sender_concepts.hpp"
#include "__sender_introspection.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"
#include "__type_traits.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.associate]
  namespace __associate {
    template <scope_token _Token, sender _Sender>
    struct __associate_data {
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
      struct __deleter {
        constexpr void operator()(__wrap_sender_t* __p) const noexcept {
          std::destroy_at(__p);
        }
      };

      using __sender_ref = std::unique_ptr<__wrap_sender_t, __deleter>;

      // BUGBUG: should the spec require __token to be declared as a const _Token, or should this be
      //         changed to declare __token as a mutable _Token?
      explicit __associate_data(const _Token __token, _Sender&& __sndr) noexcept(
        __nothrow_constructible_from<__wrap_sender_t, __wrap_result_t>
        && noexcept(__token.wrap(static_cast<_Sender&&>(__sndr)))
        && noexcept(__token.try_associate()))
        : __sndr_(__token.wrap(static_cast<_Sender&&>(__sndr)))
        , __assoc_([&] {
          __sender_ref guard{std::addressof(__sndr_)};

          auto assoc = __token.try_associate();

          if (assoc) {
            (void) guard.release();
          }

          return assoc;
        }()) {
      }

      __associate_data(const __associate_data& __other) noexcept(
        __nothrow_copy_constructible<__wrap_sender_t> && noexcept(__other.__assoc_.try_associate()))
        requires __std::copy_constructible<__wrap_sender_t>
        : __assoc_(__other.__assoc_.try_associate()) {
        if (__assoc_) {
          std::construct_at(std::addressof(__sndr_), __other.__sndr_);
        }
      }

      __associate_data(__associate_data&& __other)
        noexcept(__nothrow_move_constructible<__wrap_sender_t>)
        : __associate_data(std::move(__other).release()) {
      }

      ~__associate_data() {
        if (__assoc_) {
          std::destroy_at(&__sndr_);
        }
      }

      std::pair<__assoc_t, __sender_ref> release() && noexcept {
        __sender_ref __u(__assoc_ ? std::addressof(__sndr_) : nullptr);
        return {std::move(__assoc_), std::move(__u)};
      }

     private:
      explicit __associate_data(std::pair<__assoc_t, __sender_ref> __parts)
        : __assoc_(std::move(__parts.first)) {
        if (__assoc_) {
          std::construct_at(std::addressof(__sndr_), std::move(*__parts.second));
        }
      }

      union {
        __wrap_sender_t __sndr_;
      };
      __assoc_t __assoc_;
    };

    template <scope_token _Token, sender _Sender>
    __associate_data(_Token, _Sender&&) -> __associate_data<_Token, _Sender>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct associate_t {
      template <sender _Sender, scope_token _Token>
      auto operator()(_Sender&& __sndr, _Token&& __token) const
        noexcept(__nothrow_constructible_from<
                 __associate_data<std::remove_cvref_t<_Token>, _Sender>,
                 _Token,
                 _Sender
        >) -> __well_formed_sender auto {
        return __make_sexpr<associate_t>(
          __associate_data(static_cast<_Token&&>(__token), static_cast<_Sender&&>(__sndr)));
      }

      template <scope_token _Token>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Token&& __token) const noexcept {
        return __closure(*this, static_cast<_Token&&>(__token));
      }
    };

    // NOTE: the spec declares this class template inside the get_state function
    //       but I couldn't get that to build with Clang 21 so I moved it out to
    //       this namespace-scoped template and __uglified all the symbols
    template <class _Sender, class _Receiver>
    struct __op_state {
      using __associate_data_t = std::remove_cvref_t<__data_of<_Sender>>;
      using __assoc_t = __associate_data_t::__assoc_t;
      using __sender_ref_t = __associate_data_t::__sender_ref;

      using __op_t = connect_result_t<typename __sender_ref_t::element_type, _Receiver>;

      __assoc_t __assoc_;
      union {
        _Receiver __rcvr_;
        __op_t __op_;
      };

      explicit __op_state(std::pair<__assoc_t, __sender_ref_t> __parts, _Receiver&& __rcvr)
        : __assoc_(std::move(__parts.first)) {
        if (__assoc_) {
          ::new ((void*) std::addressof(__op_))
            __op_t(connect(std::move(*__parts.second), std::move(__rcvr)));
        } else {
          std::construct_at(std::addressof(__rcvr_), std::move(__rcvr));
        }
      }

      explicit __op_state(__associate_data_t&& __ad, _Receiver&& __rcvr)
        : __op_state(std::move(__ad).release(), std::move(__rcvr)) {
      }

      explicit __op_state(const __associate_data_t& __ad, _Receiver&& __rcvr)
        requires __std::copy_constructible<__associate_data_t>
        : __op_state(__associate_data_t(__ad).release(), std::move(__rcvr)) {
      }

      ~__op_state() {
        if (__assoc_) {
          std::destroy_at(std::addressof(__op_));
        } else {
          std::destroy_at(std::addressof(__rcvr_));
        }
      }

      void __run() noexcept {
        if (__assoc_) {
          STDEXEC::start(__op_);
        } else {
          STDEXEC::set_stopped(std::move(__rcvr_));
        }
      }
    };

    struct __associate_impl : __sexpr_defaults {
#if 0 // TODO: I don't know how to implement this correctly
      static constexpr auto get_attrs = []<class _Child>(__ignore, const _Child& __child) noexcept {
        return __sync_attrs{__child};
      };
#endif

      template <class _Sender>
      using __wrap_sender_of_t =
        __copy_cvref_t<_Sender, typename __data_of<std::remove_cvref_t<_Sender>>::__wrap_sender_t>;

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() //
        -> transform_completion_signatures<
          __completion_signatures_of_t<__wrap_sender_of_t<_Sender>, _Env...>,
          completion_signatures<set_stopped_t()>
        > {
        static_assert(sender_expr_for<_Sender, associate_t>);
        return {};
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver __rcvr) noexcept(
          (__std::same_as<_Self, std::remove_cvref_t<_Self>> || __nothrow_decay_copyable<_Self>) &&
            __nothrow_callable<
              connect_t,
              typename std::remove_cvref_t<__data_of<_Self>>::__wrap_sender_t,
              _Receiver
            >) {
          auto& [__tag, __data] = __self;

          using op_state_t = __op_state<std::remove_cvref_t<_Self>, _Receiver>;
          return op_state_t{__forward_like<_Self>(__data), std::move(__rcvr)};
        };

      static constexpr auto start = [](auto& __state) noexcept -> void {
        __state.__run();
      };
    };
  } // namespace __associate

  using __associate::associate_t;

  /// @brief The associate sender adaptor, which associates a sender with the
  ///        async scope referred to by the given token
  /// @hideinitializer
  inline constexpr associate_t associate{};

  template <>
  struct __sexpr_impl<associate_t> : __associate::__associate_impl { };
} // namespace STDEXEC
