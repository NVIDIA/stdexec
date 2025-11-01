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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "exec/sequence_senders.hpp"
#include "stdexec/__detail/__concepts.hpp"
#include "stdexec/__detail/__config.hpp"
#include "stdexec/__detail/__tuple.hpp"
#include <concepts>
#include <string>
#include <exception>
#include <system_error>

namespace std {
  inline std::string to_string(const std::error_code __error) noexcept {
    return __error.message();
  }
  inline std::string to_string(const std::exception_ptr __ex) noexcept {
    try {
      std::rethrow_exception(__ex);
    } catch (const std::exception& __ex) {
      return __ex.what();
    }
  }
} // namespace std

namespace stdexec::__rcvrs {
  inline std::string to_string(set_value_t) noexcept {
    return {"set_value"};
  }
  inline std::string to_string(set_error_t) noexcept {
    return {"set_error"};
  }
  inline std::string to_string(set_stopped_t) noexcept {
    return {"set_stopped"};
  }
} // namespace stdexec::__rcvrs

namespace exec::__sequence_sender {
  inline std::string to_string(set_next_t) noexcept {
    return {"set_next"};
  }
} // namespace exec::__sequence_sender

namespace exec {
  namespace __notification {
    using namespace stdexec;

    //
    // notification_t provides storage for any one of the
    // completions in the specified completion_signatures<>
    //
    // notification_t can be compared, visited by a receiver
    // to emit the stored signal, and provide a sender that will
    // complete with the stored signal
    //

    template <class... _Ts>
    using __nothrow_decay_copyable_and_move_constructible_t = __mbool<(
      (__nothrow_decay_copyable<_Ts> && __nothrow_move_constructible<__decay_t<_Ts>>) && ...)>;

    template <class... Args>
    using __as_rvalues = set_value_t (*)(__decay_t<Args>...);

    template <class... E>
    using __as_error = set_error_t (*)(E...);

    // Here we convert all set_value(Args...) to set_value(__decay_t<Args>...). Note, we keep all
    // error types as they are and unconditionally add set_stopped(). The indirection through the
    // __completions_fn is to avoid a pack expansion bug in nvc++.
    struct __completions_fn {
      template <class _CompletionSignatures>
      using __all_value_args_nothrow_decay_copyable = __value_types_t<
        _CompletionSignatures,
        __qq<__nothrow_decay_copyable_and_move_constructible_t>,
        __qq<__mand_t>
      >;

      template <class _CompletionSignatures>
      using __f = __mtry_q<__concat_completion_signatures>::__f<
        __eptr_completion_if_t<__all_value_args_nothrow_decay_copyable<_CompletionSignatures>>,
        completion_signatures<set_stopped_t()>,
        __transform_completion_signatures<
          _CompletionSignatures,
          __as_rvalues,
          __as_error,
          set_stopped_t (*)(),
          __completion_signature_ptrs
        >
      >;
    };

    template <class _CompletionSignatures>
    using __notification_storage_t = __for_each_completion_signature<
      __minvoke<__completions_fn, _CompletionSignatures>,
      __decayed_tuple,
      std::variant
    >;

    template <class _CompletionSignatures>
    struct __notification_sender;

    template <class _CompletionSignatures>
    struct notification_t {
      using __notification_t = __notification_storage_t<_CompletionSignatures>;
      using __notification_sender_t = stdexec::__t<__notification_sender<_CompletionSignatures>>;

      __notification_t __notification_{};

      template <class _Notification>
      using __tag_of_t =
        stdexec::__mapply<stdexec::__q<stdexec::__mfront>, STDEXEC_REMOVE_REFERENCE(_Notification)>;

      template <class _Tag, class... _Args>
      notification_t(_Tag __tag, _Args&&... __args) noexcept(noexcept(
        __notification_.template emplace<__decayed_tuple<_Tag, STDEXEC_REMOVE_REFERENCE(_Args)...>>(
          __tag,
          static_cast<_Args&&>(__args)...))) {
        __notification_.template emplace<__decayed_tuple<_Tag, STDEXEC_REMOVE_REFERENCE(_Args)...>>(
          __tag, static_cast<_Args&&>(__args)...);
      }

      template <class _Fn>
      auto visit(_Fn&& __fn) const noexcept {
        return std::visit(
          [&__fn](auto&& __tuple) noexcept {
            return __tuple.apply(
              [&__fn](auto __tag, auto&&... __args) noexcept {
                return static_cast<_Fn&&>(__fn)(__tag, __args...);
              },
              __tuple);
          },
          __notification_);
      }
      template <class _Receiver>
      void visit_receiver(_Receiver&& __receiver) noexcept {
        std::visit(
          [&__receiver]<class _Tuple>(_Tuple&& __tuple) noexcept {
            __tuple.apply(
              [&__receiver]<class _Tag, class... _Args>(_Tag __tag, _Args&&... __args) noexcept {
                __tag(
                  static_cast<_Receiver&&>(__receiver), static_cast<_Args&&>(__args)...);
              },
              static_cast<_Tuple&&>(__tuple));
          },
          static_cast<__notification_t&&>(__notification_));
      }
      auto visit_sender() noexcept -> __notification_sender_t;

      [[nodiscard]]
      bool value() const noexcept {
        return std::visit(
          []<class _Tuple>(const _Tuple&) noexcept {
            return stdexec::__decays_to<stdexec::set_value_t, __tag_of_t<_Tuple>>;
          },
          __notification_);
      }
      [[nodiscard]]
      bool error() const noexcept {
        return std::visit(
          []<class _Tuple>(const _Tuple&) noexcept {
            return stdexec::__decays_to<stdexec::set_error_t, __tag_of_t<_Tuple>>;
          },
          __notification_);
      }
      [[nodiscard]]
      bool stopped() const noexcept {
        return std::visit(
          []<class _Tuple>(const _Tuple&) noexcept {
            return stdexec::__decays_to<stdexec::set_stopped_t, __tag_of_t<_Tuple>>;
          },
          __notification_);
      }

      friend auto
        operator==(const notification_t& __lhs, const notification_t& __rhs) noexcept -> bool {
        return std::visit(
          []<class _Lhs, class _Rhs>(const _Lhs& __lhs, const _Rhs& __rhs) noexcept -> bool {
            using __lhs_tag_t = notification_t::__tag_of_t<_Lhs>;
            using __rhs_tag_t = notification_t::__tag_of_t<_Rhs>;
            if constexpr (
              !std::same_as<__lhs_tag_t, __rhs_tag_t>
              || stdexec::__v<stdexec::__mapply<stdexec::__msize, _Lhs>>
                   != stdexec::__v<stdexec::__mapply<stdexec::__msize, _Rhs>>) {
              return false;
            } else {
              return __lhs.apply(
                [&__lhs,
                 &__rhs]<class _LTag, class... _LArgs>(_LTag, const _LArgs&...) noexcept -> bool {
                  return [&__lhs, &__rhs]<std::size_t... _Is>(__indices<_Is...>) {
                    if constexpr ((std::equality_comparable_with<
                                     const decltype(_Lhs::template __get<_Is+1>(__lhs))&,
                                     const decltype(_Rhs::template __get<_Is+1>(__rhs))&
                                   >
                                   && ... && true)) {
                      return (
                        ((_Lhs::template __get<_Is+1>(__lhs)) == (_Rhs::template __get<_Is+1>(__rhs)))
                        && ... && true);
                    } else {
                      return false;
                    }
                  }(__indices_for<_LArgs...>{});
                },
                __lhs);
            }
          },
          __lhs.__notification_,
          __rhs.__notification_);
      }

      friend std::string to_string(const notification_t& __self) noexcept {
        using std::to_string;
        return __self.visit([](auto __tag, const auto&... __args) {
          int count = 0;
          return to_string(__tag) + "("
               + (((count++ > 0 ? ", " : "") + to_string(__args)) + ... + std::string{}) + ")";
        });
      }
    };

    template <class _ReceiverId, class _CompletionSignatures>
    struct __notification_op {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __notification_t = notification_t<_CompletionSignatures>;

      struct __t {
        using __id = __notification_op;

        _Receiver __receiver_;
        __notification_t* __notification_;

        void start() & noexcept {
          __notification_->visit_receiver(static_cast<_Receiver&&>(__receiver_));
        }
      };
    };

    template <class _CompletionSignatures>
    struct __notification_sender {
      using __notification_t = notification_t<_CompletionSignatures>;
      struct __t {
        using __id = __notification_sender;
        using sender_concept = stdexec::sender_t;

        template <class _ReceiverId>
        using __notification_op_t =
          stdexec::__t<__notification_op<_ReceiverId, _CompletionSignatures>>;

        __notification_t* __notification_;

        template <stdexec::__decays_to<__t> _Self, class... _Env>
        static auto
          get_completion_signatures(_Self&&, _Env&&...) noexcept -> _CompletionSignatures {
          return {};
        }

        template <std::same_as<__t> _Self, receiver _Receiver>
        static auto connect(_Self&& __self, _Receiver&& __rcvr)
          noexcept(__nothrow_move_constructible<_Receiver>)
            -> __notification_op_t<stdexec::__id<_Receiver>> {
          return {static_cast<_Receiver&&>(__rcvr), __self.__notification_};
        }
      };
    };

    template <class _CompletionSignatures>
    auto notification_t<_CompletionSignatures>::visit_sender() noexcept
      -> notification_t<_CompletionSignatures>::__notification_sender_t {
      return {this};
    }

  } // namespace __notification
  template <class _CompletionSignatures>
  using notification_t = __notification::notification_t<_CompletionSignatures>;

  namespace __notification {

  } // namespace __notification
} // namespace exec
