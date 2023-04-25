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

namespace exec {
  namespace __enumerate_each {
    using namespace stdexec;

    template <class _Int, class _Receiver>
    struct __operation_base {
      [[no_unique_address]] _Receiver __rcvr_;
      std::atomic<_Int> __count_{};

      template <class _Rcvr>
      __operation_base(_Rcvr&& __rcvr, _Int __initial_val) noexcept(
        std::is_nothrow_constructible_v<_Receiver, _Rcvr>)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __count_{__initial_val} {
      }
    };

    template <class _Int>
    struct __increase_count {
      std::atomic<_Int>* __count_;

      template <class... _Args>
      auto operator()(_Args&&... __args) const {
        const _Int __count = __count_->fetch_add(1, std::memory_order_relaxed);
        return just(__count, static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Int, class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;

        explicit __t(__operation_base<_Int, _Receiver>* __op) noexcept
          : __op_(__op) {
        }

        friend void tag_invoke(set_value_t, __t&& __self) noexcept {
          set_value(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        template <class _Item>
        friend auto tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return set_next(
            __self.__op_->__rcvr_,
            let_value((_Item&&) __item, __increase_count<_Int>{&__self.__op_->__count_}));
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& error) noexcept {
          set_error((_Receiver&&) __self.__op_->__rcvr_, (_Error&&) error);
        }

        friend void tag_invoke(set_stopped_t, __t&& __self) noexcept {
          set_stopped((_Receiver&&) __self.__op_->__rcvr_);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__op_->__rcvr_);
        }

        __operation_base<_Int, _Receiver>* __op_;
      };
    };
    template <class _Int, class _Receiver>
    using __receiver_t = __t<__receiver<_Int, __id<_Receiver>>>;

    template <class _Int, class _SenderId, class _ReceiverId>
    struct __operation {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Int, _Receiver> {
        template <class Sndr, class Rcvr>
        __t(Sndr&& __sndr, Rcvr&& __rcvr, _Int __initial_val)
          : __operation_base<_Int, _Receiver>((Rcvr&&) __rcvr, __initial_val)
          , __op_(sequence_connect(
              (Sndr&&) __sndr,
              stdexec::__t<__receiver<_Int, _ReceiverId>>{this})) {
        }

        sequence_connect_result_t<_Sender, stdexec::__t<__receiver<_Int, _ReceiverId>>> __op_;

        friend void tag_invoke(start_t, __t& self) noexcept {
          start(self.__op_);
        }
      };
    };
    template <class _Int, class _Sender, class _Receiver>
    using __operation_t = __t<__operation<_Int, __id<_Sender>, __id<_Receiver>>>;

    template <class _Int, class... _Args>
    using __add_counter = completion_signatures<set_value_t(_Int, _Args...)>;

    template <class _Sender, class _Env, class _Int>
    using __completion_sigs = __try_make_completion_signatures<
      _Sender,
      _Env,
      completion_signatures<>,
      __mbind_front_q<__add_counter, _Int>>;

    template <std::integral _Int, class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;

        [[no_unique_address]] _Sender __sndr_;
        _Int __initial_val_;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sndr, _Int __initial_val = _Int{}) noexcept(
          std::is_nothrow_constructible_v<_Sender, _Sndr>)
          : __sndr_{(_Sndr&&) __sndr}
          , __initial_val_{__initial_val} {
        }

        template <__decays_to<__t> _Self, class _Rcvr>
          requires sequence_receiver_of<
                     _Rcvr,
                     __completion_sigs<__copy_cvref_t<_Self, _Sender>, env_of_t<_Rcvr>, _Int>>
                && sequence_sender_to<
                     __copy_cvref_t<_Self, _Sender>,
                     __receiver_t<_Int, __decay_t<_Rcvr>>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Rcvr&& __rcvr)
          -> __operation_t<_Int, __copy_cvref_t<_Self, _Sender>, __decay_t<_Rcvr>> {
          return __operation_t<_Int, __copy_cvref_t<_Self, _Sender>, __decay_t<_Rcvr>>(
            static_cast<_Self&&>(__self).__sndr_, (_Rcvr&&) __rcvr, __self.__initial_val_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<__copy_cvref_t<_Self, _Sender>, _Env, _Int>;
      };
    };

    struct enumerate_each_t {
      template <sender _Sender, std::integral _Int = int>
      auto operator()(_Sender&& __sndr, _Int __initial_val = _Int{}) const noexcept {
        return __t<__sender<_Int, __id<__decay_t<_Sender>>>>{
          static_cast<_Sender&&>(__sndr), __initial_val};
      }

      template <std::integral _Int = int>
      auto operator()(_Int __initial_val = _Int{}) const noexcept
        -> __binder_back<enumerate_each_t, _Int> {
        return {{}, {}, {__initial_val}};
      }
    };
  } // namespace __enumerate_each

  inline constexpr __enumerate_each::enumerate_each_t enumerate_each{};
} // namespace exec