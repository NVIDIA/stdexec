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

#include "../sequence_senders.hpp"

#include <ranges>

namespace exec {
  namespace __iterate {
    using namespace stdexec;
    using namespace exec;

    template <class _Rng>
    concept __nothrow_begin = noexcept(std::ranges::begin(std::declval<_Rng>()));

    template <class _Rng>
    concept __nothrow_end = noexcept(std::ranges::end(std::declval<_Rng>()));

    template <class _Range, class _ReceiverId>
    struct __operation_base {
      using _Receiver = stdexec::__t<_ReceiverId>;
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
      STDEXEC_NO_UNIQUE_ADDRESS std::ranges::iterator_t<_Range> __it_;
      STDEXEC_NO_UNIQUE_ADDRESS std::ranges::sentinel_t<_Range> __end_;

      void (*__do_next_)(__operation_base*) noexcept = nullptr;
    };

    template <class _Range, class _ReceiverId>
    struct __receiver {
      struct __t {
        using __id = __receiver;
        using _Receiver = stdexec::__t<_ReceiverId>;
        __operation_base<_Range, _ReceiverId>* __op;

        static constexpr bool __stoppable_token =
          !unstoppable_token<stop_token_of_t<env_of_t<_Receiver>>>;

        bool __upstream_is_stopped() const noexcept {
          return stdexec::get_stop_token(stdexec::get_env(__op->__rcvr_)).stop_requested();
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
          requires __callable<set_value_t, _Receiver&&> && __callable<set_stopped_t, _Receiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          if constexpr (__stoppable_token) {
            if (__self.__upstream_is_stopped()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op->__rcvr_));
              return;
            }
          }
          ++__self.__op->__it_;
          if (__self.__op->__it_ != __self.__op->__end_) {
            __self.__op->__do_next_(__self.__op);
          } else {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__op->__rcvr_));
          }
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<set_value_t, _Receiver&&> && __callable<set_stopped_t, _Receiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          if constexpr (__stoppable_token) {
            if (__self.__upstream_is_stopped()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op->__rcvr_));
            } else {
              stdexec::set_value(static_cast<_Receiver&&>(__self.__op->__rcvr_));
            }
          } else {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__op->__rcvr_));
          }
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend auto tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return stdexec::get_env(__self.__op->__rcvr_);
        }
      };
    };

    template <class _Range, class _ReceiverId>
    struct __operation {
      struct __t : __operation_base<_Range, _ReceiverId> {
        using __id = __operation;
        using _Receiver = stdexec::__t<_ReceiverId>;

        using __item_sender_t =
          decltype(stdexec::just(__declval<std::ranges::iterator_t<_Range>>()) | stdexec::then([](std::ranges::iterator_t<_Range> __it) {
                     return *__it;
                   }));

        static auto __set_next(__operation_base<_Range, _ReceiverId>* __self) {
          return exec::set_next(
            __self->__rcvr_,
            stdexec::just(__self->__it_)
              | stdexec::then([](std::ranges::iterator_t<_Range> __it) { return *__it; }));
        }

        using __next_sender_t = decltype(__set_next(__declval<__t*>()));

        using __receiver_t = stdexec::__t<__receiver<_Range, _ReceiverId>>;
        std::optional<connect_result_t<__next_sender_t, __receiver_t>> __item_;

        static void __do_next(__operation_base<_Range, _ReceiverId>* __base) noexcept {
          __t* __self = static_cast<__t*>(__base);
          try {
            auto& __op = __self->__item_.emplace(__conv{[__self] {
              return stdexec::connect(__set_next(__self), __receiver_t{__self});
            }});
            stdexec::start(__op);
          } catch (...) {
            stdexec::set_error(static_cast<_Receiver&&>(__self->__rcvr_), std::current_exception());
          }
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          if (__self.__it_ == __self.__end_) {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
          } else {
            __do_next(&__self);
          }
        }

        template <__decays_to<_Range> _Rng>
        explicit __t(_Rng&& __rng, _Receiver&& __rcvr)
          : __operation_base<_Range, _ReceiverId>{
            static_cast<_Receiver&&>(__rcvr),
            std::ranges::begin(__rng),
            std::ranges::end(__rng),
            &__do_next} {
        }
      };
    };

    template <class _Range>
    struct __sender {
      struct __t {
        using __id = __sender;
        using is_sequence_sender = void;

        template <class _Rcvr>
        using __receiver_t = stdexec::__t<__receiver<_Range, stdexec::__id<__decay_t<_Rcvr>>>>;

        STDEXEC_NO_UNIQUE_ADDRESS _Range __range_;

        template <class _Env>
        using __completion_sigs = __if_c<
          stdexec::unstoppable_token<stdexec::stop_token_of_t<_Env>>,
          completion_signatures<
            set_value_t(std::ranges::range_reference_t<_Range>),
            set_error_t(std::exception_ptr)>,
          completion_signatures<
            set_value_t(std::ranges::range_reference_t<_Range>),
            set_error_t(std::exception_ptr),
            set_stopped_t()>>;

        using __item_sender_t =
          decltype(just(__declval<std::ranges::iterator_t<_Range>>()) | then([](std::ranges::iterator_t<_Range> __it) {
                     return *__it;
                   }));

        template <class _Rcvr>
        using __next_sender_t = __next_sender_of_t<__decay_t<_Rcvr>&, __item_sender_t>;

        template <class _Rcvr>
        using __operation_t = stdexec::__t<__operation<_Range, stdexec::__id<__decay_t<_Rcvr>>>>;

        template <__decays_to<__t> _Self, class _Rcvr>
          requires sequence_receiver_of<_Rcvr, __completion_sigs<env_of_t<_Rcvr>>>
                && sender_to<__next_sender_t<_Rcvr>, __receiver_t<_Rcvr>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Rcvr&& __rcvr)
          -> __operation_t<_Rcvr> {
          return __operation_t<_Rcvr>{
            static_cast<_Self&&>(__self).__range_, static_cast<_Rcvr&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<_Env>;
      };
    };

    struct iterate_t {
      template <std::ranges::input_range _Range>
      __t<__sender<__decay_t<_Range>>> operator()(_Range&& __rng) const {
        return {static_cast<_Range&&>(__rng)};
      }
    };
  }

  using __iterate::iterate_t;
  inline constexpr iterate_t iterate{};
}