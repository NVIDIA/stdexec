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
#include "../variant_sender.hpp"

namespace exec {
  namespace __take_while {
    using namespace stdexec;

    template <class _BaseEnv>
    using __env_t = __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _ReceiverId, class _Predicate>
    struct __operation_base {
      using _Receiver = __t<_ReceiverId>;
      using __on_stop =
        typename stop_token_of_t<env_of_t<_Receiver&>>::template callback_type<__on_stop_requested>;

      [[no_unique_address]] _Receiver __rcvr_;
      [[no_unique_address]] _Predicate __pred_;
      std::mutex __mutex_;
      in_place_stop_source __stop_source_{};
      std::optional<__on_stop> __on_stop_{};
    };

    template <class _ReceiverId, class _Predicate>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        __operation_base<_ReceiverId, _Predicate>* __op_;

        using __just_stopped_t = decltype(stdexec::just_stopped());

        template <class... _Args>
        using __just_t = decltype(stdexec::just(__declval<_Args>()...));

        template <class... _Args>
        using __next_t = __next_sender_of_t<_Receiver&, __just_t<_Args...>>;

        template <same_as<set_next_t> _Tag, __decays_to<__t> _Self, sender _Item>
          requires __callable<_Tag, _Receiver&, _Item>
        friend auto tag_invoke(_Tag, _Self&& __self, _Item&& __item) noexcept {
          return let_value(
            (_Item&&) __item,
            [__op = __self.__op_]<class... _Args>(
              _Args&&... __args) -> variant_sender<__just_stopped_t, __next_t<_Args...>> {
              std::scoped_lock __lock{__op->__mutex_};
              if (__op->__stop_source_.stop_requested()) {
                return stdexec::just_stopped();
              }
              if (std::invoke(__op->__pred_, __args...)) {
                return exec::set_next(__op->__rcvr_, just((_Args&&) __args...));
              } else {
                __op->__stop_source_.request_stop();
                return stdexec::just_stopped();
              }
            });
        }

        template <same_as<set_value_t> _Tag, __decays_to<__t> _Self>
          requires __callable<_Tag, _Receiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          __self.__op_->__on_stop_.reset();
          _Tag{}((_Receiver&&) __self.__op_->__rcvr_);
        }

        template <same_as<set_stopped_t> _Tag, __decays_to<__t> _Self>
          requires __callable<_Tag, _Receiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          __self.__op_->__on_stop_.reset();
          if (get_stop_token(__self.__op_->__rcvr_).stop_requested()) {
            stdexec::set_stopped((_Receiver&&) __self.__op_->__rcvr_);
          } else {
            stdexec::set_value((_Receiver&&) __self.__op_->__rcvr_);
          }
        }

        template <same_as<set_error_t> _Tag, __decays_to<__t> _Self, class _Error>
          requires __callable<_Tag, _Receiver&&, _Error>
        friend void tag_invoke(_Tag, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__on_stop_.reset();
          _Tag{}((_Receiver&&) __self.__op_->__rcvr_, (_Error&&) __error);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Predicate>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _Predicate>>;
      using __op_base_t = __operation_base<_ReceiverId, _Predicate>;

      struct __t : __op_base_t {
        sequence_connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver&& __rcvr, _Predicate __pred)
          : __op_base_t{(_Receiver&&) __rcvr, (_Predicate&&) __pred}
          , __op_{sequence_connect((_Sender&&) __sndr, __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_stop_.emplace(get_stop_token(__self.__rcvr_), __on_stop_requested{__self.__stop_source_});
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _SenderId, class _Predicate>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Self, class _Receiver>
      using __operation_t = stdexec::__t<
        __operation<__copy_cvref_t<_Self, _Sender>, __id<__decay_t<_Receiver>>, _Predicate>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>, _Predicate>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        _Sender __sndr_;
        _Predicate __pred_;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return __operation_t<_Self, _Receiver>(
            ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr, (_Predicate&&) __self.__pred_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    using namespace stdexec;

    struct take_while_t {
      template <class _Sender, class _Predicate>
        requires tag_invocable<take_while_t, _Sender, _Predicate>
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        noexcept(nothrow_tag_invocable<take_while_t, _Sender, _Predicate>)
          -> tag_invoke_result_t<take_while_t, _Sender, _Predicate> {
        return tag_invoke(*this, (_Sender&&) __sndr, (_Predicate&&) __pred);
      }

      template <sender _Sender, class _Predicate>
        requires(!tag_invocable<take_while_t, _Sender, _Predicate>)
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        -> __t<__sender<__id<__decay_t<_Sender>>, _Predicate>> {
        return {(_Sender&&) __sndr, (_Predicate&&) __pred};
      }

      template <class _Predicate>
      auto operator()(_Predicate __pred) const -> __binder_back<take_while_t, _Predicate> {
        return {{}, {}, {(_Predicate&&) __pred}};
      }
    };
  }

  using __take_while::take_while_t;
  inline constexpr take_while_t take_while{};
}