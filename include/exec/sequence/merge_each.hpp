#pragma once

#include "../../stdexec/execution.hpp"

#include "../sequence_senders.hpp"

namespace exec {
  namespace __merge_each {
    using namespace stdexec;

    struct __on_stop_requested {
      stdexec::in_place_stop_source& __source_;

      void operator()() const noexcept {
        __source_.request_stop();
      }
    };

    template <class _Env>
    using __env_t = __make_env_t<_Env, __with<get_stop_token_t, in_place_stop_token>>;

    enum class __state_t {
      __success,
      __emplace,
      __error
    };

    template <class _Receiver, class _ErrorsVariant>
    struct __operation_base {
      _Receiver __rcvr_;
      stdexec::in_place_stop_source __source_{};
      std::atomic<__state_t> __state_{__state_t::__success};
      _ErrorsVariant __errors_{};
      std::optional<
        typename stop_token_of_t<env_of_t<_Receiver>>::template callback_type< __on_stop_requested>>
        __on_rcvr_stop_;

      template <class _Error>
      void __notify_error(_Error&& __error) noexcept {
        __state_t expected = __state_t::__success;
        if (__state_.compare_exchange_strong(
              expected, __state_t::__emplace, std::memory_order_relaxed)) {
          __errors_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__error));
          __state_.store(__state_t::__error, std::memory_order_release);
          __source_.request_stop();
        }
      }
    };

    template <class _ItemReceiver, class _Receiver, class _ErrorsVariant>
    struct __item_operation_base {
      __operation_base<_Receiver, _ErrorsVariant>* __parent_op_;
      _ItemReceiver __item_rcvr_;
      std::optional<stdexec::in_place_stop_callback<__on_stop_requested>> __on_parent_stop_{};
      std::optional<typename stop_token_of_t<env_of_t<_ItemReceiver>>::template callback_type<
        __on_stop_requested>>
        __on_item_rcvr_stop_{};
    };

    template <class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using __id = __subsequence_receiver;
        __item_operation_base<_ItemReceiver, _Receiver, _ErrorsVariant>* __op_;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Sender>
          requires __callable<_SetNext, _Receiver&, _Sender>
        friend auto tag_invoke(_SetNext, _Self& __self, _Sender&& __sndr) noexcept {
          return exec::set_next(
            __self.__op_->__parent_op_->__rcvr_, static_cast<_Sender&&>(__sndr));
        }

        template <__one_of<set_value_t, set_stopped_t> _Tag, same_as<__t> _Self>
          requires __callable<_Tag, _ItemReceiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          __self.__op_->__on_parent_stop_.reset();
          __self.__op_->__on_item_rcvr_stop_.reset();
          _Tag{}(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__on_parent_stop_.reset();
          __self.__op_->__on_item_rcvr_stop_.reset();
          __self.__op_->__parent_op_->__notify_error(static_cast<_Error&&>(__error));
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
          requires __callable<_GetEnv, const _ItemReceiver&>
        friend __env_t<env_of_t<_ItemReceiver>> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return _GetEnv{}(__self.__op_->__item_rcvr_);
        }
      };
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_operation {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      struct __t;
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __item_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using __id = __item_receiver;
        stdexec::__t<__subsequence_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>>*
          __op_;

        template <same_as<set_value_t> _Tag, same_as<__t> _Self, class _Subsequence>
        friend void tag_invoke(_Tag, _Self&& __self, _Subsequence&& __subsequence) noexcept {
          __self.__op_->__start_subsequence(static_cast<_Subsequence&&>(__subsequence));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__parent_op_->__notify_error(static_cast<_Error&&>(__error));
          __self.__op_->__on_parent_stop_.reset();
          __self.__op_->__on_item_rcvr_stop_.reset();
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<_SetStopped, _ItemReceiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          __self.__op_->__on_parent_stop_.reset();
          __self.__op_->__on_item_rcvr_stop_.reset();
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
          requires __callable<_GetEnv, const _ItemReceiver&>
        friend __env_t<env_of_t<_ItemReceiver>> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return __make_env(
            _GetEnv{}(__self.__op_->__item_rcvr_),
            __with_(get_stop_token, __self.__op_->__source_.get_token()));
        }
      };
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>::__t
      : __item_operation_base<_ItemReceiver, _Receiver, _ErrorsVariant> {

      using __item_receiver_t =
        stdexec::__t<__item_receiver<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>>;
      using __subsequence_receiver_t =
        stdexec::__t<__subsequence_receiver<_ItemReceiverId, _ReceiverId, _ErrorsVariant>>;

      using __receive_subsequence_op_t = connect_result_t<_Item, __item_receiver_t>;

      template <class _Subseq>
      using __subsequence_operation_t =
        sequence_connect_result_t< _Subseq, __subsequence_receiver_t>;

      using __subsequence_operation_variant = __mapply<
        __transform<__q<__subsequence_operation_t>, __nullable_variant_t>,
        __value_types_of_t<_Item, env_of_t<_Receiver>, __q<__mfront>, __q<__types>>>;

      explicit __t(
        _Item&& __sndr,
        _ItemReceiver&& __rcvr,
        __operation_base<_Receiver, _ErrorsVariant>* __op)
        : __item_operation_base<
          _ItemReceiver,
          _Receiver,
          _ErrorsVariant>{__op, static_cast<_ItemReceiver&&>(__rcvr)}
        , __sub_op_{std::in_place_index<0>, __conv{[&] {
                      return stdexec::connect(
                        static_cast<_Item&&>(__sndr), __item_receiver_t{this});
                    }}} {
      }

      std::variant<__receive_subsequence_op_t, __subsequence_operation_variant> __sub_op_;
      stdexec::in_place_stop_source __source_{};

      template <class _Subsequence>
      void __start_subsequence(_Subsequence __subsequence) noexcept {
        __subsequence_operation_variant& __ops_variant = __sub_op_.template emplace<1>();
        try {
          auto& __second_op =
            __ops_variant.template emplace<__subsequence_operation_t<_Subsequence>>(__conv{[&] {
              return exec::sequence_connect(
                static_cast<_Subsequence&&>(__subsequence), __subsequence_receiver_t{this});
            }});
          stdexec::start(__second_op);
        } catch (...) {
          this->__parent_op_->__notify_error(std::current_exception());
        }
      }

      friend void tag_invoke(start_t, __t& __self) noexcept {
        __receive_subsequence_op_t* __first = std::get_if<0>(&__self.__sub_op_);
        STDEXEC_ASSERT(__first);
        __self.__on_parent_stop_.emplace(
          __self.__parent_op_->__source_.get_token(), __on_stop_requested{__self.__source_});
        __self.__on_item_rcvr_stop_.emplace(
          stdexec::get_stop_token(stdexec::get_env(__self.__item_rcvr_)),
          __on_stop_requested{__self.__source_});
        stdexec::start(*__first);
      }
    };

    template <class _ItemId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_sender {
      using _Item = stdexec::__t<_ItemId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __subsequence_sender;
        using is_sender = void;
        _Item __item_;
        __operation_base<_Receiver, _ErrorsVariant>* __op_;

        template <class _Self, class _ItemReceiver>
        using __sub_op_t = stdexec::__t<__subsequence_operation<
          __copy_cvref_t<_Self, _Item>,
          stdexec::__id<__decay_t<_ItemReceiver>>,
          _ReceiverId,
          _ErrorsVariant>>;

        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        template <class _Self, class _ItemReceiver>
        using __item_receiver_t = stdexec::__t<__item_receiver<
          __copy_cvref_t<_Self, _Item>,
          stdexec::__id<__decay_t<_ItemReceiver>>,
          _ReceiverId,
          _ErrorsVariant>>;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _ItemReceiver>
          requires sender_to<_Item, __item_receiver_t<_Self, _ItemReceiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _ItemReceiver&& __item_rcvr) noexcept
          -> __sub_op_t<_Self, _ItemReceiver> {
          return __sub_op_t<_Self, _ItemReceiver>{
            static_cast<_Self&&>(__self).__item_,
            static_cast<_ItemReceiver&&>(__item_rcvr),
            __self.__op_};
        }
      };
    };

    template <class _ReceiverId, class _ErrorsVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        __operation_base<_Receiver, _ErrorsVariant>* __op_;

        template <class _Item>
        using __subseq_sender_t = stdexec::__t<
          __subsequence_sender<stdexec::__id<__decay_t<_Item>>, _ReceiverId, _ErrorsVariant>>;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Item>
        friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __sender) noexcept
          -> __subseq_sender_t<_Item> {
          return {static_cast<_Item&&>(__sender), __self.__op_};
        }

        template <same_as<set_value_t> _Tag, same_as<__t> _Self>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          __self.__op_->__on_rcvr_stop_.reset();
          __state_t __state = __self.__op_->__state_.load(std::memory_order_acquire);
          switch (__state) {
          case __state_t::__success:
            _Tag{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
            break;
          case __state_t::__error:
            std::visit(
              [&]<class _Error>(_Error&& __error) {
                if constexpr (__not_decays_to<_Error, std::monostate>) {
                  stdexec::set_error(
                    static_cast<_Receiver&&>(__self.__op_->__rcvr_),
                    static_cast<_Error&&>(__error));
                }
              },
              static_cast<_ErrorsVariant&&>(__self.__op_->__errors_));
          case __state_t::__emplace:
            [[fallthrough]];
          default:
            STDEXEC_ASSERT(false);
          }
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          __self.__op_->__on_rcvr_stop_.reset();
          _SetStopped{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__on_rcvr_stop_.reset();
          _SetError{}(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend __env_t<env_of_t<_Receiver>> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return __make_env(
            _GetEnv{}(__self.__op_->__rcvr_),
            __with_(get_stop_token, __self.__op_->__source_.get_token()));
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _ErrorsVariant>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Receiver, _ErrorsVariant> {
        using __id = __operation;

        using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _ErrorsVariant>>;

        sequence_connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver&& __rcvr)
          : __operation_base<_Receiver, _ErrorsVariant>{static_cast<_Receiver&&>(__rcvr)}
          , __op_{exec::sequence_connect(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_rcvr_stop_.emplace(
            stdexec::get_stop_token(stdexec::get_env(__self.__rcvr_)),
            __on_stop_requested{__self.__source_});
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _Sender, class _Env>
    using __sub_sequence_senders = __value_types_of_t<_Sender, _Env, __q<__mfront>, __q<__types>>;

    template <class _Sender, class _Env>
    concept __only_single_values = __valid<__sub_sequence_senders, _Sender, _Env>;

    template <class _Sender, class _Env>
    using __completion_sigs = __concat_completion_signatures_t<
      __mapply<
        __q<__concat_completion_signatures_t>,
        __mapply<
          __transform<__mbind_back_q<completion_signatures_of_t, _Env>>,
          __sub_sequence_senders<_Sender, _Env>>>,
      completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>;

    template <class _Sender, class _Env>
    using __errors_variant_t = __gather_signal<
      set_error_t,
      __completion_sigs<_Sender, _Env>,
      __q<__decay_t>,
      __nullable_variant_t>;

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;

        using is_sequence_sender = void;

        template <class _Self, class _Rcvr>
        using __operation_t = stdexec::__t<__operation<
          __copy_cvref_t<_Self, _Sender>,
          stdexec::__id<__decay_t<_Rcvr>>,
          __errors_variant_t<__copy_cvref_t<_Self, _Sender>, env_of_t<_Rcvr>>>>;

        _Sender __sndr_;

        template <class _Self, class _Rcvr>
        using __receiver_t = stdexec::__t<__receiver<
          stdexec::__id<__decay_t<_Rcvr>>,
          __errors_variant_t<__copy_cvref_t<_Self, _Sender>, env_of_t<_Rcvr>> >>;

        template <__decays_to<__t> _Self, class _Receiver>
          requires __only_single_values<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>
                && sequence_receiver_of<
                     _Receiver,
                     __completion_sigs<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>>
                && sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return __operation_t<_Self, _Receiver>{
            static_cast<_Self&&>(__self).__sndr_, static_cast<_Receiver&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
          requires __only_single_values<__copy_cvref_t<_Self, _Sender>, _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, const _Env& __env)
          -> __completion_sigs<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    struct merge_each_sequence_t {
      template <class _Sender>
        requires(!tag_invocable<merge_each_sequence_t, _Sender>)
      __t<__sender<__id<__decay_t<_Sender>>>> operator()(_Sender&& __sender) const
        noexcept(__nothrow_decay_copyable<_Sender>) {
        return {static_cast<_Sender&&>(__sender)};
      }

      __binder_back<merge_each_sequence_t> operator()() const noexcept {
        return {{}, {}, {}};
      }
    };
  }

  using __merge_each::merge_each_sequence_t;
  inline constexpr merge_each_sequence_t merge_each_sequence{};
}