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
  namespace __filter_each {
    using namespace stdexec;

    template <class _ReceiverId, class _Predicate>
    struct __operation_base {
      [[no_unique_address]] __t<_ReceiverId> __rcvr_;
      [[no_unique_address]] _Predicate __pred_;
    };

    template <class _ReceiverId, class _Predicate>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        __operation_base<_ReceiverId, _Predicate>* __op_;

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
              _Args&&... __args) -> variant_sender<__just_t<>, __next_t<_Args...>> {
              if (std::invoke(__op->__pred_, __args...)) {
                return exec::set_next(__op->__rcvr_, just((_Args&&) __args...));
              }
              return stdexec::just();
            });
        }

        template <same_as<set_value_t> _Tag, __decays_to<__t> _Self>
          requires __callable<_Tag, _Receiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          _Tag{}((_Receiver&&) __self.__op_->__rcvr_);
        }

        template <same_as<set_stopped_t> _Tag, __decays_to<__t> _Self>
          requires __callable<_Tag, _Receiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          _Tag{}((_Receiver&&) __self.__op_->__rcvr_);
        }

        template <same_as<set_error_t> _Tag, __decays_to<__t> _Self, class _Error>
          requires __callable<_Tag, _Receiver&&, _Error>
        friend void tag_invoke(_Tag, _Self&& __self, _Error&& __error) noexcept {
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

    struct filter_each_t {
      template <class _Sender, class _Predicate>
        requires tag_invocable<filter_each_t, _Sender, _Predicate>
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        noexcept(nothrow_tag_invocable<filter_each_t, _Sender, _Predicate>)
          -> tag_invoke_result_t<filter_each_t, _Sender, _Predicate> {
        return tag_invoke(*this, (_Sender&&) __sndr, (_Predicate&&) __pred);
      }

      template <sender _Sender, class _Predicate>
        requires(!tag_invocable<filter_each_t, _Sender, _Predicate>)
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        -> __t<__sender<__id<__decay_t<_Sender>>, _Predicate>> {
        return {(_Sender&&) __sndr, (_Predicate&&) __pred};
      }

      template <class _Predicate>
      auto operator()(_Predicate __pred) const -> __binder_back<filter_each_t, _Predicate> {
        return {{}, {}, {(_Predicate&&) __pred}};
      }
    };
  }

  using __filter_each::filter_each_t;
  inline constexpr filter_each_t filter_each{};
}