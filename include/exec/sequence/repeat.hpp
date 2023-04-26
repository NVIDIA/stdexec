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

#include "../trampoline_scheduler.hpp"

namespace exec {
  namespace __repeat {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __op_base {
      [[no_unique_address]] __t<_ReceiverId> __rcvr_;
      void (*__repeat_)(__op_base*) noexcept = nullptr;
    };

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        __op_base<_ReceiverId>* __op_;

        static constexpr bool __unstoppable_token =
          unstoppable_token<stop_token_of_t<env_of_t<_Receiver>>>;

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          STDEXEC_ASSERT(__self.__op_->__repeat_);
          __self.__op_->__repeat_(__self.__op_);
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<set_error_t, _Receiver&&, _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error e) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(e));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          if constexpr (__unstoppable_token) {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          } else {
            auto token = stdexec::get_stop_token(stdexec::get_env(__self.__op_->__rcvr_));
            if (token.stop_requested()) {
              stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
            } else {
              stdexec::set_value(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
            }
          }
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept(
          __nothrow_callable<get_env_t, const _Receiver&>) {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    // Takes a sender and creates a sequence sender by repeating the sender as item to the set_next
    // operation.
    template <class _SourceSender, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __next_sender_t = __call_result_t<set_next_t, _Receiver&, _SourceSender>;
      using __next_on_scheduler_sender =
        __call_result_t<on_t, exec::trampoline_scheduler, __next_sender_t&&>;

      using __receiver_t = stdexec::__t<__receiver<_ReceiverId>>;

      struct __t : __op_base<_ReceiverId> {
        using __id = __operation;
        [[no_unique_address]] _SourceSender __source_;
        exec::trampoline_scheduler __trampoline_;
        std::optional<connect_result_t<__next_on_scheduler_sender, __receiver_t>> __next_op_;

        static void __repeat(__op_base<_ReceiverId>* __base) noexcept {
          __t* __self = static_cast<__t*>(__base);
          auto __token = stdexec::get_stop_token(stdexec::get_env(__self->__rcvr_));
          if (__token.stop_requested()) {
            stdexec::set_stopped((_Receiver&&) __self->__rcvr_);
            return;
          }
          try {
            auto& __next = __self->__next_op_.emplace(__conv{[&] {
              return stdexec::connect(
                stdexec::on(
                  __self->__trampoline_, exec::set_next(__self->__rcvr_, __self->__source_)),
                __receiver_t{__base});
            }});
            stdexec::start(__next);
          } catch (...) {
            stdexec::set_error((_Receiver&&) __self->__rcvr_, std::current_exception());
          }
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __repeat(&__self);
        }

        template <__decays_to<_SourceSender> _Sndr, __decays_to<_Receiver> _Rcvr>
        explicit __t(_Sndr&& __source, _Rcvr&& __rcvr)
          : __op_base<_ReceiverId>{(_Rcvr&&) __rcvr, &__repeat}
          , __source_{(_Sndr&&) __source} {
        }
      };
    };

    template <class _SourceSender, class _Receiver>
    using __operation_t = __t<__operation<_SourceSender, __id<__decay_t<_Receiver>>>>;

    template <class _Source, class _Env>
    using __compl_sigs = make_completion_signatures<
      _Source,
      _Env,
      completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>>;

    template <class _SourceId>
    struct __sender {
      using _Source = stdexec::__t<__decay_t<_SourceId>>;

      template <class _Rcvr>
      using __next_sender = __call_result_t<set_next_t, __decay_t<_Rcvr>&, _Source>;

      template <class _Rcvr>
      using __next_on_scheduler_sender =
        __call_result_t<on_t, exec::trampoline_scheduler, __next_sender<_Rcvr>&&>;

      template <class _Rcvr>
      using __recveiver_t = stdexec::__t<__receiver<__id<__decay_t<_Rcvr>>>>;

      class __t {
        [[no_unique_address]] _Source __source_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sequence_receiver_of<_Receiver, __compl_sigs<_Source&, env_of_t<_Receiver>>>
                && __valid<__next_on_scheduler_sender, _Receiver>
                && sender_to<__next_on_scheduler_sender<_Receiver>, __recveiver_t<_Receiver>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Source, _Receiver> {
          return __operation_t<_Source, _Receiver>{
            ((_Self&&) __self).__source_, (_Receiver&&) __rcvr};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __compl_sigs<_Source&, _Env>;

       public:
        using __id = __sender;

        template <__decays_to<_Source> _Sndr>
        explicit __t(_Sndr&& __source)
          : __source_((_Sndr&&) __source) {
        }
      };
    };

    struct repeat_t {
      template <sender Sender>
      auto operator()(Sender&& source) const {
        return __t<__sender<__id<__decay_t<Sender>>>>{static_cast<Sender&&>(source)};
      }
    };
  } // namespace __repeat

  using __repeat::repeat_t;
  inline constexpr repeat_t repeat;
} // namespace exec