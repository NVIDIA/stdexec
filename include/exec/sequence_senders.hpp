/*
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

#include "../stdexec/execution.hpp"

#include "./trampoline_scheduler.hpp"

namespace exec {
  namespace __sequence_sender {
    using namespace stdexec;

    struct set_next_t {
      template <class _Receiver, class _Item>
        requires tag_invocable<set_next_t, _Receiver&, _Item>
      auto operator()(_Receiver& __rcvr, _Item&& __item) const noexcept
        -> tag_invoke_result_t<set_next_t, _Receiver&, _Item> {
        static_assert(sender<tag_invoke_result_t<set_next_t, _Receiver&, _Item>>);
        static_assert(nothrow_tag_invocable<set_next_t, _Receiver&, _Item>);
        return tag_invoke(*this, __rcvr, (_Item&&) __item);
      }
    };
  }

  using __sequence_sender::set_next_t;
  inline constexpr set_next_t set_next;

  namespace __sequence_sender {
    template <class _Signature>
    struct _MISSING_NEXT_SIGNATURE_;

    template <class _Item>
    struct _MISSING_NEXT_SIGNATURE_<set_next_t(_Item)> {
      template <class _Receiver>
      struct _WITH_RECEIVER_ : std::false_type { };

      struct _ {
        _(...) {
        }
      };

      friend auto operator,(_MISSING_NEXT_SIGNATURE_, _) -> _MISSING_NEXT_SIGNATURE_ {
        return {};
      }
    };

    struct __found_next_signature {
      template <class _Receiver>
      using _WITH_RECEIVER_ = std::true_type;
    };

    template <class... _Args>
    using __just_t = decltype(just(__declval<_Args>()...));

    template <class _Receiver, class... _Args>
    using __missing_next_signature_t = __if<
      __mbool<nothrow_tag_invocable<set_next_t, _Receiver&, __just_t<_Args...>>>,
      __found_next_signature,
      _MISSING_NEXT_SIGNATURE_<set_next_t(__just_t<_Args...>)>>;

    template <class _Receiver, class... _Args>
    auto __has_sequence_signature(set_value_t (*)(_Args...))
      -> __missing_next_signature_t<_Receiver, _Args...>;

    template <__one_of<set_error_t, set_stopped_t> _Tag, class _Receiver, class... _Args>
    auto __has_sequence_signature(_Tag (*)(_Args...))
      -> __receiver_concepts::__missing_completion_signal_t<_Receiver, _Tag, _Args...>;

    template <class _Receiver, class... _Sigs>
    auto __has_sequence_signatures(completion_signatures<_Sigs...>*)
      -> decltype((__has_sequence_signature<_Receiver>(static_cast<_Sigs*>(nullptr)), ...));

    template <class _Signatures, class _Receiver>
    concept is_valid_next_completions = _Signatures::template _WITH_RECEIVER_<_Receiver>::value;
  }

  template <class _Receiver, class Signatures>
  concept sequence_receiver_of =
    stdexec::receiver_of<_Receiver, stdexec::completion_signatures<stdexec::set_value_t()>>
    && requires(Signatures* sigs) {
         {
           __sequence_sender::__has_sequence_signatures<stdexec::decay_t<_Receiver>>(sigs)
         } -> __sequence_sender::is_valid_next_completions<stdexec::decay_t<_Receiver>>;
       };

  template <class _Receiver, class _Sender>
  concept sequence_receiver_from = sequence_receiver_of<
    _Receiver,
    stdexec::completion_signatures_of_t<_Sender, stdexec::env_of_t<_Receiver>>>;

  namespace __sequence_sender {
    struct sequence_connect_t;

    template <class _Sender, class _Receiver>
    concept __sequence_connectable_with_tag_invoke =
      receiver<_Receiver> &&                        //
      sender_in<_Sender, env_of_t<_Receiver>> &&    //
      // sequence_receiver_from<_Receiver, _Sender> && //
      tag_invocable<sequence_connect_t, _Sender, _Receiver>;

    struct sequence_connect_t {
      template <class _Sender, class _Receiver>
        requires __sequence_connectable_with_tag_invoke<_Sender, _Receiver>
      auto operator()(_Sender&& __sender, _Receiver&& __rcvr) const
        noexcept(nothrow_tag_invocable<sequence_connect_t, _Sender, _Receiver>)
          -> tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver> {
        static_assert(
          operation_state<tag_invoke_result_t<sequence_connect_t, _Sender, _Receiver>>,
          "exec::sequence_connect(sender, receiver) must return a type that "
          "satisfies the operation_state concept");
        return tag_invoke(*this, (_Sender&&) __sender, (_Receiver&&) __rcvr);
      }
    };

    template <class _Sender, class _Receiver>
    using sequence_connect_result_t = __call_result_t<sequence_connect_t, _Sender, _Receiver>;
  }

  using __sequence_sender::sequence_connect_t;
  inline constexpr sequence_connect_t sequence_connect;

  using __sequence_sender::sequence_connect_result_t;

  template <class _Sender, class _Receiver>
  concept sequence_sender_to =
    stdexec::sender_in<_Sender, stdexec::env_of_t<_Receiver>>
    && sequence_receiver_from<_Receiver, _Sender>
    && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
         { stdexec::connect((_Sender&&) __sndr, (_Receiver&&) __rcvr) } -> stdexec::operation_state;
       };

  namespace __repeat_effect {
    using namespace stdexec;

    // Takes a sender and creates a sequence sender by repeating the sender as item to the set_next
    // operation.
    template <class _SourceSenderId, class _ReceiverId>
    struct __operation {
      using _SourceSender = stdexec::__t<_SourceSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __next_sender_t = __call_result_t<set_next_t, _Receiver&, _SourceSender>;
      using __next_on_scheduler_sender =
        __call_result_t<on_t, exec::trampoline_scheduler, __next_sender_t&&>;

      struct __t;

      struct __receiver {
        __t* __op_;

        template <class... _Args>
        friend void tag_invoke(set_value_t, __receiver&& __self, _Args&&...) noexcept {
          static_assert(sizeof...(_Args) == 0);
          __self.__op_->repeat();
        }

        template <__decays_to<__receiver> _Self, class _Error>
          requires __callable<set_error_t, _Receiver&&, _Error&&>
        friend void tag_invoke(set_error_t, _Self&& __self, _Error e) noexcept {
          set_error((_Receiver&&) __self.__op_->__rcvr_, (_Error&&) e);
        }

        friend void tag_invoke(set_stopped_t, __receiver&& __self) noexcept {
          auto token = get_stop_token(get_env(__self.__op_->__rcvr_));
          if (token.stop_requested()) {
            set_stopped((_Receiver&&) __self.__op_->__rcvr_);
          } else {
            set_value((_Receiver&&) __self.__op_->__rcvr_);
          }
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __receiver& __self) noexcept(
          __nothrow_callable<get_env_t, const _Receiver&>) {
          return get_env(__self.__op_->__rcvr_);
        }
      };

      struct __t {
        [[no_unique_address]] _Receiver __rcvr_;
        _SourceSender __source_;
        exec::trampoline_scheduler __trampoline_;
        std::optional<connect_result_t<__next_on_scheduler_sender, __receiver>> __next_op_;

        void repeat() noexcept {
          auto __token = get_stop_token(get_env(__rcvr_));
          if (__token.stop_requested()) {
            set_stopped((_Receiver&&) __rcvr_);
            return;
          }
          try {
            auto& next = __next_op_.emplace(__conv{[&] {
              return connect(
                stdexec::on(__trampoline_, set_next(__rcvr_, _SourceSender{__source_})),
                __receiver{this});
            }});
            start(next);
          } catch (...) {
            set_error((_Receiver&&) __rcvr_, std::current_exception());
          }
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.repeat();
        }

        template <__decays_to<_SourceSender> _Sndr, __decays_to<_Receiver> _Rcvr>
        explicit __t(_Sndr&& __source, _Rcvr&& __rcvr)
          : __rcvr_{(_Rcvr&&) __rcvr}
          , __source_{(_Sndr&&) __source} {
        }
      };
    };

    template <class _SourceSender, class _Receiver>
    using __operation_t = __t<__operation<__id<_SourceSender>, __id<decay_t<_Receiver>>>>;

    template <class _Source, class _Env>
    using __compl_sigs = make_completion_signatures<
      _Source,
      _Env,
      completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>>;

    template <class _SourceId>
    struct __sender {
      using _Source = stdexec::__t<decay_t<_SourceId>>;

      template <class _Rcvr>
      using __next_sender = __call_result_t<set_next_t, decay_t<_Rcvr>&, _Source>;

      template <class _Rcvr>
      using __next_on_scheduler_sender =
        __call_result_t<on_t, exec::trampoline_scheduler, __next_sender<_Rcvr>&&>;

      template <class _Rcvr>
      using __recveiver = typename __operation<__id<_Source>, __id<decay_t<_Rcvr>>>::__receiver;

      class __t {
        _Source __source_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__next_on_scheduler_sender<_Receiver>, __recveiver<_Receiver>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Source, _Receiver> {
          return __operation_t<_Source, _Receiver>{
            ((_Self&&) __self).__source_, (_Receiver&&) __rcvr};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __compl_sigs<__copy_cvref_t<_Self, _Source>, _Env>;

       public:
        template <__decays_to<_Source> _Sndr>
        explicit __t(_Sndr&& __source)
          : __source_((_Sndr&&) __source) {
        }
      };
    };

    struct __repeat_effectt {
      template <sender_of<set_value_t()> Sender>
        requires __single_typed_sender<Sender> && copy_constructible<Sender>
      auto operator()(Sender&& source) const {
        return __t<__sender<__id<decay_t<Sender>>>>{static_cast<Sender&&>(source)};
      }
    };
  } // namespace repeat_each_

  using __repeat_effect::__repeat_effectt;
  inline constexpr __repeat_effectt repeat_effect;

  namespace __join_all {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        _Receiver __rcvr_;

        template <class _Item>
        friend _Item&& tag_invoke(set_next_t, __t&, _Item&& __item) noexcept {
          return static_cast<_Item&&>(__item);
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__rcvr_), static_cast<_Error&&>(__error));
        }

        template <__one_of<set_stopped_t, set_value_t> _Tag>
        friend void tag_invoke(_Tag complete, __t&& __self) noexcept {
          complete(static_cast<_Receiver&&>(__self.__rcvr_));
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__rcvr_);
        }
      };
    };

    template <class _Rcvr>
    using __receiver_t = __t<__receiver<__id<decay_t<_Rcvr>>>>;

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<decay_t<_SenderId>>;

      struct __t {
        _Sender __sndr_;

        template <__decays_to<__t> _Self, class _Receiver>
        // requires sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend sequence_connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
          return sequence_connect(
            __copy_cvref_t<_Self, _Sender>(__self.__sndr_),
            __receiver_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    struct join_all_t {
      template <class _Sender>
      constexpr auto operator()(_Sender&& __sndr) const {
        return __t<__sender<__id<decay_t<_Sender>>>>{static_cast<_Sender&&>(__sndr)};
      }

      constexpr auto operator()() const noexcept -> __binder_back<join_all_t> {
        return {};
      }
    };
  } // namespace __join_all

  using __join_all::join_all_t;
  inline constexpr join_all_t join_all;
}