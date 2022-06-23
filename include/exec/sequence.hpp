/*
 * Copyright (c) NVIDIA
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

#include <stdexec/execution.hpp>

namespace _P0TBD::execution {

  using namespace stdexec;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.set_next]
  namespace __set_next {
    struct set_next_t {
      template <class _Receiver, class _Sender>
      auto operator()(_Receiver& __rcvr, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<set_next_t, _Receiver&, _Sender>)
        -> tag_invoke_result_t<set_next_t, _Receiver&, _Sender> {
        static_assert(
          sender<tag_invoke_result_t<set_next_t, _Receiver&, _Sender>>,
          "execution::set_next(receiver, sender) must return a type that "
          "satisfies the sender concept");
        return tag_invoke(set_next_t{}, __rcvr, (_Sender&&) __sndr);
      }
    };
  } // namespace __set_next

  using __set_next::set_next_t;
  inline constexpr __set_next::set_next_t set_next {};

  template <class _Receiver, class _Sender>
    using set_next_result_t = __call_result_t<set_next_t, _Receiver, _Sender>;

  template <class _SequenceReceiver, class _ValueSender>
    concept __sequence_receiver_of =
      requires (_SequenceReceiver&& __seq_rcvr, _ValueSender&& __val_sndr) {
        { set_next((_SequenceReceiver&&) __seq_rcvr, (_ValueSender&&) __val_sndr) } -> sender<>;
      };

  template <class _SequenceReceiver, class _ValueSender>
    concept sequence_receiver_of =
      receiver<_SequenceReceiver> &&
      __sequence_receiver_of<_SequenceReceiver, _ValueSender>;


  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __iotas {

    template <class _CPO, class _V>
    using __completion_signatures_ = completion_signatures<_CPO(_V)>;

    template <class _CPO, class _V>
    struct __value_sender {
        template <__decays_to<__value_sender<_CPO, _V>> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
          __completion_signatures_<_CPO, _V>;

        template <class _ReceiverId>
          struct __operation : __immovable {
            using _Receiver = __t<_ReceiverId>;
            _Receiver __rcvr_;
            _V __last_;

            friend void tag_invoke(start_t, __operation& __op_state) noexcept {
              static_assert(__nothrow_callable<_CPO, _Receiver, _V>);
              _CPO{}(std::move(__op_state.__rcvr_), std::move(__op_state.__last_));
            }
          };

        _V __last_;

        template <class _Receiver>
          requires copy_constructible<_V>
        friend auto tag_invoke(connect_t, const __value_sender& __sndr, _Receiver&& __rcvr)
          noexcept(std::is_nothrow_copy_constructible_v<_V>)
          -> __operation<__x<remove_cvref_t<_Receiver>>> {
          return {{}, (_Receiver&&) __rcvr, ((__value_sender&&) __sndr).__last_};
        }

        template <class _Receiver>
        friend auto tag_invoke(connect_t, __value_sender&& __sndr, _Receiver&& __rcvr)
          noexcept(std::is_nothrow_move_constructible_v<_V>)
          -> __operation<__x<remove_cvref_t<_Receiver>>> {
          return {{}, (_Receiver&&) __rcvr, ((__value_sender&&) __sndr).__last_};
        }
    };

    template <class _CPO, class _V, class _B>
      struct __sender {
        template <__decays_to<__sender<_CPO, _V, _B>> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
          completion_signatures<set_value_t()>;

        template <class _SequenceReceiverId>
          struct __operation : __immovable {
            using _SequenceReceiver = __t<_SequenceReceiverId>;

            struct __value_receiver {
                __operation* __op_state_;

                template<class... _Args>
                friend void tag_invoke(set_value_t, __value_receiver&& __value_rcvr, _Args&&...) noexcept try {
                    auto __op_state = ((__value_receiver&&) __value_rcvr).__op_state_;
                    auto& __value_op = __op_state->__value_op_;
                    __value_op.reset();
                    if (__op_state->__bound_ == __op_state->__next_) {
                      execution::set_value(std::move(__op_state->__rcvr_));
                      return;
                    }
                    auto __adapted_value{_P0TBD::execution::set_next(__op_state->__rcvr_, __value_sender<_CPO, _V>{__op_state->__next_++})};
                    __value_op.emplace(
                        __conv{
                            [&](){
                                return execution::connect(on(__op_state->__schd_, std::move(__adapted_value)), __value_receiver{__op_state});
                            }
                        });
                    execution::start(*__value_op);
                } catch(...) {
                    execution::set_error(
                        (_SequenceReceiver&&) ((__value_receiver&&) __value_rcvr).__op_state_->__rcvr_,
                        std::current_exception());
                }

                template <__one_of<set_error_t, set_stopped_t> _Tag, class... _Args>
                  requires __callable<_Tag, _SequenceReceiver, _Args...>
                friend void tag_invoke(_Tag, __value_receiver&& __value_rcvr, _Args&&... __args) noexcept {
                  auto __op_state = ((__value_receiver&&) __value_rcvr).__op_state_;
                  __op_state->__value_op_.reset();
                  _Tag{}((_SequenceReceiver&&) __op_state->__rcvr_, (_Args&&) __args...);
                }

                friend auto tag_invoke(get_env_t, const __value_receiver& __value_rcvr)
                  -> env_of_t<_SequenceReceiver> {
                  return get_env(__value_rcvr.__op_state_->__rcvr_);
                }
            };

            using scheduler_t = std::invoke_result_t<get_scheduler_t, env_of_t<_SequenceReceiver>>;
            using __adapted_sender_t=std::invoke_result_t<_P0TBD::execution::set_next_t, _SequenceReceiver&, __value_sender<_CPO, _V>>;
            using __value_sender_t=std::invoke_result_t<execution::on_t, scheduler_t, __adapted_sender_t>;
            using __value_op_t=connect_result_t<__value_sender_t, __value_receiver>;

            scheduler_t __schd_;
            _SequenceReceiver __rcvr_;
            [[no_unique_address]] _V __next_;
            [[no_unique_address]] _B __bound_;

            std::optional<__value_op_t> __value_op_;

            friend void tag_invoke(start_t, __operation& __op_state) noexcept {
              static_assert(__nothrow_callable<set_value_t, _SequenceReceiver>);
              if (__op_state.__bound_ == __op_state.__next_) {
                execution::set_value(std::move(__op_state.__rcvr_));
                return;
              }
              auto __adapted_value{_P0TBD::execution::set_next(__op_state.__rcvr_, __value_sender<_CPO, _V>{__op_state.__next_++})};
              __op_state.__value_op_.emplace(
                  __conv{
                      [&](){
                          return execution::connect(on(__op_state.__schd_, std::move(__adapted_value)), __value_receiver{&__op_state});
                      }
                  });
              execution::start(*__op_state.__value_op_);
            }
          };

        [[no_unique_address]] _V __value_;
        [[no_unique_address]] _B __bound_;

        template <class _SequenceReceiver>
          requires copy_constructible<_V> && copy_constructible<_B> && __scheduler_provider<env_of_t<_SequenceReceiver>>
        friend auto tag_invoke(connect_t, const __sender& __sndr, _SequenceReceiver&& __rcvr)
          noexcept(std::is_nothrow_copy_constructible_v<_V> && std::is_nothrow_copy_constructible_v<_B>)
          -> __operation<__x<remove_cvref_t<_SequenceReceiver>>> {
          return {{}, get_scheduler(get_env(__rcvr)), (_SequenceReceiver&&) __rcvr, ((__sender&&) __sndr).__value_, ((__sender&&) __sndr).__bound_};
        }

        template <class _SequenceReceiver>
          requires __scheduler_provider<env_of_t<_SequenceReceiver>>
        friend auto tag_invoke(connect_t, __sender&& __sndr, _SequenceReceiver&& __rcvr)
          noexcept(std::is_nothrow_move_constructible_v<_V> && std::is_nothrow_move_constructible_v<_B>)
          -> __operation<__x<remove_cvref_t<_SequenceReceiver>>> {
          return {{}, get_scheduler(get_env(__rcvr)), (_SequenceReceiver&&) __rcvr, ((__sender&&) __sndr).__value_, ((__sender&&) __sndr).__bound_};
        }
      };

    inline constexpr struct __iotas_t {
      template <copy_constructible _V, __equality_comparable_with<_V> _B>
      __sender<set_value_t, decay_t<_V>, decay_t<_B>> operator()(_V&& __v, _B&& __b) const
        noexcept(std::is_nothrow_constructible_v<decay_t<_V>, _V> && std::is_nothrow_constructible_v<decay_t<_B>, _B>) {
        return {(_V&&) __v, (_B&&) __b};
      }
    } iotas {};
  }
  using __iotas::iotas;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ignore_all]
  namespace __ignore_all {

    template<class _ReceiverId>
    struct __sequence_receiver {
        using _Receiver = __t<_ReceiverId>;

        [[no_unique_address]] _Receiver __rcvr_;

        template<class _ValueSender>
        friend auto tag_invoke(set_next_t, __sequence_receiver&, _ValueSender&& __vs) noexcept {
          return (_ValueSender&&)__vs;
        }

        template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        friend void tag_invoke(_Tag, __sequence_receiver&& __seq_rcvr, _Args&&... __args) noexcept {
          _Tag{}((_Receiver&&) __seq_rcvr.__rcvr_, (_Args&&) __args...);
        }

        friend auto tag_invoke(get_env_t, const __sequence_receiver& __seq_rcvr)
          -> env_of_t<_Receiver> {
          return get_env(__seq_rcvr.__rcvr_);
        }
    };

    template <class _SequenceSenderId>
      struct __sequence_sender {
        using _SequenceSender = __t<_SequenceSenderId>;

        [[no_unique_address]] _SequenceSender __seq_sndr_;

      template <class _Receiver>
        using __sequence_receiver = __sequence_receiver<__x<remove_cvref_t<_Receiver>>>;

        template <__decays_to<__sequence_sender> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>>)
          //recursive template instantiation
          {//-> connect_result_t<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>> {
          return connect(((_Self&&) __self).__seq_sndr_, __sequence_receiver<_Receiver>{(_Receiver&&) __rcvr});
        }

        template <__decays_to<__sequence_sender> _Self, receiver _Receiver, class _ValueAdaptor>
          requires sender_to<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr, _ValueAdaptor&&)
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>>)
          //recursive template instantiation
          {//-> connect_result_t<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver<_Receiver>> {
          return connect(((_Self&&) __self).__seq_sndr_, __sequence_receiver<_Receiver>{(_Receiver&&) __rcvr});
        }

        template <__decays_to<__sequence_sender> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
          completion_signatures_of_t<_SequenceSender, _Env>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As>
            requires __callable<_Tag, const _SequenceSender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sequence_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _SequenceSender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _SequenceSender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }
      };

    struct ignore_all_t {
      template <class _SequenceSender>
        using __sequence_sender = __sequence_sender<__x<remove_cvref_t<_SequenceSender>>>;

      template <sender _SequenceSender>
        requires __tag_invocable_with_completion_scheduler<ignore_all_t, set_value_t, _SequenceSender>
      sender auto operator()(_SequenceSender&& __seq_sndr) const
        noexcept(nothrow_tag_invocable<ignore_all_t, __completion_scheduler_for<_SequenceSender, set_value_t>, _SequenceSender>) {
        auto __sched = get_completion_scheduler<set_value_t>(__seq_sndr);
        return tag_invoke(ignore_all_t{}, std::move(__sched), (_SequenceSender&&) __seq_sndr);
      }
      template <sender _SequenceSender>
        requires (!__tag_invocable_with_completion_scheduler<ignore_all_t, set_value_t, _SequenceSender>) &&
          tag_invocable<ignore_all_t, _SequenceSender>
      sender auto operator()(_SequenceSender&& __seq_sndr) const
        noexcept(nothrow_tag_invocable<ignore_all_t, _SequenceSender>) {
        return tag_invoke(ignore_all_t{}, (_SequenceSender&&) __seq_sndr);
      }
      template <sender _SequenceSender>
        requires
          (!__tag_invocable_with_completion_scheduler<ignore_all_t, set_value_t, _SequenceSender>) &&
          (!tag_invocable<ignore_all_t, _SequenceSender>) &&
          sender<__sequence_sender<_SequenceSender>>
      __sequence_sender<_SequenceSender> operator()(_SequenceSender&& __seq_sndr) const {
        return __sequence_sender<_SequenceSender>{(_SequenceSender&&) __seq_sndr};
      }
      __binder_back<ignore_all_t> operator()() const {
        return {{}, {}};
      }
    };
  }
  using __ignore_all::ignore_all_t;
  inline constexpr ignore_all_t ignore_all{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then_each]
  namespace __then_each {

    template <class _SequenceSenderId, class _FunId>
      struct __sequence_sender {
        using _SequenceSender = __t<_SequenceSenderId>;
        using _Fun = __t<_FunId>;

        [[no_unique_address]] _SequenceSender __seq_sndr_;
        [[no_unique_address]] _Fun __fun_;

        template<class _SequenceReceiverId>
        struct __sequence_receiver {
            using _SequenceReceiver = __t<_SequenceReceiverId>;

            [[no_unique_address]] _SequenceReceiver __rcvr_;
            [[no_unique_address]] _Fun __fun_;

            template<class _ValueSender>
            friend auto tag_invoke(set_next_t, __sequence_receiver& __self, _ValueSender&& __vs) noexcept {
              return set_next(__self.__rcvr_, execution::then((_ValueSender&&)__vs, __self.__fun_));
            }

            template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _Args>
              requires __callable<_Tag, _SequenceReceiver, _Args...>
            friend void tag_invoke(_Tag, __sequence_receiver&& __seq_rcvr, _Args&&... __args) noexcept {
              _Tag{}((_SequenceReceiver&&) __seq_rcvr.__rcvr_, (_Args&&) __args...);
            }

            friend auto tag_invoke(get_env_t, const __sequence_receiver& __seq_rcvr)
              -> env_of_t<_SequenceReceiver> {
              return get_env(__seq_rcvr.__rcvr_);
            }
        };

        template <class _SequenceReceiver>
          using __sequence_receiver_t = __sequence_receiver<__x<remove_cvref_t<_SequenceReceiver>>>;

        template <class...>
          using __set_value =
            completion_signatures<set_value_t()>;

        template <class _Self, class _Env>
          using __completion_signatures =
            make_completion_signatures<
              __copy_cvref_t<_Self, _SequenceSender>, _Env, __with_exception_ptr, __set_value>;

        template <__decays_to<__sequence_sender> _Self, receiver _SequenceReceiver>
          requires sender_to<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver_t<_SequenceReceiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _SequenceReceiver&& __rcvr)
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver_t<_SequenceReceiver>>)
          // recursive template instantiation
          {//-> connect_result_t<__copy_cvref_t<_Self, _SequenceSender>, __sequence_receiver_t<_SequenceReceiver>> {
          return connect(
              ((_Self&&) __self).__seq_sndr_,
              __sequence_receiver_t<_SequenceReceiver>{(_SequenceReceiver&&) __rcvr, __self.__fun_});
        }

        template <__decays_to<__sequence_sender> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) ->
          __completion_signatures<_Self, _Env>;

        // forward sender queries:
        template <tag_category<forwarding_sender_query> _Tag, class... _As>
            requires __callable<_Tag, const _SequenceSender&, _As...>
          friend auto tag_invoke(_Tag __tag, const __sequence_sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _SequenceSender&, _As...>)
            -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _SequenceSender&, _As...> {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }
      };

    struct then_each_t {
      template <class _SequenceSender, class _Fun>
        using __sequence_sender = __sequence_sender<__x<remove_cvref_t<_SequenceSender>>, __x<remove_cvref_t<_Fun>>>;

      template <sender _SequenceSender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<then_each_t, set_value_t, _SequenceSender, _Fun>
      sender auto operator()(_SequenceSender&& __seq_sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_each_t, __completion_scheduler_for<_SequenceSender, set_value_t>, _SequenceSender, _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(__seq_sndr);
        return tag_invoke(then_each_t{}, std::move(__sched), (_SequenceSender&&) __seq_sndr, (_Fun&&) __fun);
      }
      template <sender _SequenceSender, __movable_value _Fun>
        requires (!__tag_invocable_with_completion_scheduler<then_each_t, set_value_t, _SequenceSender, _Fun>) &&
          tag_invocable<then_each_t, _SequenceSender, _Fun>
      sender auto operator()(_SequenceSender&& __seq_sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_each_t, _SequenceSender, _Fun>) {
        return tag_invoke(then_each_t{}, (_SequenceSender&&) __seq_sndr, (_Fun&&) __fun);
      }
      template <sender _SequenceSender, __movable_value _Fun>
        requires
          (!__tag_invocable_with_completion_scheduler<then_each_t, set_value_t, _SequenceSender, _Fun>) &&
          (!tag_invocable<then_each_t, _SequenceSender, _Fun>) &&
          sender<__sequence_sender<_SequenceSender, _Fun>>
      __sequence_sender<_SequenceSender, _Fun> operator()(_SequenceSender&& __seq_sndr, _Fun __fun) const {
        return __sequence_sender<_SequenceSender, _Fun>{(_SequenceSender&&) __seq_sndr, (_Fun&&) __fun};
      }
      template <class _Fun>
      __binder_back<then_each_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }
  using __then_each::then_each_t;
  inline constexpr then_each_t then_each{};

} // namespace _P0TBD::execution
