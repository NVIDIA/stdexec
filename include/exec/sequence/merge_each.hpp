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

#include <exception>

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../sequence_senders.hpp"

#include "../../stdexec/__detail/__completion_signatures_of.hpp"
#include "../../stdexec/__detail/__concepts.hpp"
#include "../../stdexec/__detail/__config.hpp"
#include "../../stdexec/__detail/__diagnostics.hpp"
#include "../../stdexec/__detail/__execution_fwd.hpp"
#include "../../stdexec/__detail/__meta.hpp"
#include "../../stdexec/__detail/__sender_introspection.hpp"
#include "../../stdexec/__detail/__stop_token.hpp"
#include "../../stdexec/__detail/__transform_completion_signatures.hpp"
#include "../../stdexec/__detail/__variant.hpp"
#include "../__detail/__basic_sequence.hpp"

#include <atomic>

namespace exec {
  namespace __merge_each {
    using namespace STDEXEC;
    struct merge_each_t;

    struct __env_with_inplace_stop_token_t {
      auto operator()(inplace_stop_token& __stop_token) const noexcept {
        return STDEXEC::prop{STDEXEC::get_stop_token, __stop_token};
      }
      template <class _Env>
      auto operator()(STDEXEC::inplace_stop_token& __stop_token, _Env&& __env) const noexcept {
        return __env::__join((*this)(__stop_token), static_cast<_Env&&>(__env));
      }
      auto operator()(inplace_stop_source& __stop_source) const noexcept {
        return STDEXEC::prop{STDEXEC::get_stop_token, __stop_source.get_token()};
      }
      template <class _Env>
      auto operator()(STDEXEC::inplace_stop_source& __stop_source, _Env&& __env) const noexcept {
        return __env::__join((*this)(__stop_source), static_cast<_Env&&>(__env));
      }
    };
    static constexpr inline __env_with_inplace_stop_token_t __env_with_inplace_stop_token;

    template <class _Env>
    using __env_with_inplace_stop_token_result_t = decltype(__env_with_inplace_stop_token(
      STDEXEC::__declval<STDEXEC::inplace_stop_source&>(),
      STDEXEC::__declval<_Env>()));

    template <class _Receiver>
    struct __nested_stop {

      struct __on_stop_request {
        inplace_stop_source& __stop_source_;

        void operator()() noexcept {
          __stop_source_.request_stop();
        }
      };
      using __env_t = env_of_t<_Receiver>;
      using __env_result_t = __env_with_inplace_stop_token_result_t<__env_t>;
      using __stop_token_t = stop_token_of_t<__env_t>;
      using __stop_callback_t = stop_callback_for_t<__stop_token_t, __on_stop_request>;
      struct no_callback { };
      using __callback_t =
        __if_c<unstoppable_token<__stop_token_t>, no_callback, __optional<__stop_callback_t>>;

      inplace_stop_source __stop_source_{};
      __callback_t __on_stop_{};

      void register_token(_Receiver& __receiver) noexcept {
        if constexpr (!unstoppable_token<__stop_token_t>) {
          // register stop callback:
          __on_stop_.emplace(
            get_stop_token(STDEXEC::get_env(__receiver)), __on_stop_request{__stop_source_});
        }
      }

      void unregister_token() noexcept {
        if constexpr (!unstoppable_token<__stop_token_t>) {
          __on_stop_.reset();
        }
      }

      bool stop_requested() const noexcept {
        if constexpr (!unstoppable_token<__stop_token_t>) {
          return __stop_source_.stop_requested();
        }
        return false;
      }

      bool request_stop() noexcept {
        if constexpr (!unstoppable_token<__stop_token_t>) {
          return __stop_source_.request_stop();
        }
        return false;
      }

      inplace_stop_token get_token() const & noexcept {
        return __stop_source_.get_token();
      }

      static auto __env_from(__nested_stop* __self, __env_t&& __env) noexcept -> __env_result_t {
        return __env_with_inplace_stop_token(__self->__stop_source_, static_cast<__env_t&&>(__env));
      }
      auto env_from(__env_t&& __env) noexcept {
        return __env_from(this, static_cast<__env_t&&>(__env));
      }
      auto env_from(_Receiver& __receiver) noexcept {
        return env_from(get_env(__receiver));
      }

      using env_t = decltype(__env_from(std::declval<__nested_stop*>(), __declval<__env_t>()));
    };

    template <class... _Args>
    using drop = __mlist<>;

    enum class __completion_t {
      __started,
      __error,
      __stopped
    };

    //
    // __operation.. coordinates all the nested operation completions, creates
    // a nested inplace_stop_source and stores the first error to arrive and
    // delays the error to be emmitted after all nested operations have completed
    //
    // The first error to arrive will request_stop on the nested inplace_stop_source
    //

    template <class _ErrorStorage>
    struct __operation_base_interface {
      virtual void nested_value_started() noexcept = 0;
      virtual void nested_value_complete() noexcept = 0;
      virtual bool nested_value_fail() noexcept = 0;
      virtual void nested_value_break() noexcept = 0;
      virtual void error_complete() noexcept = 0;

      _ErrorStorage* __error_storage_;
      inplace_stop_token __token_;

      __operation_base_interface(_ErrorStorage* __error_storage) noexcept
        : __error_storage_{__error_storage}
        , __token_{} {
      }

      template <class _Env>
      auto env_from(_Env&& __env) noexcept -> __env_with_inplace_stop_token_result_t<_Env> {
        return __env_with_inplace_stop_token(__token_, static_cast<_Env&&>(__env));
      }

      template <class _Error>
        requires(!__std::same_as<_Error, _ErrorStorage>)
      void
        store_error(_Error&& __error) noexcept(__nothrow_callable<
                                               decltype(&_ErrorStorage::template emplace<_Error>),
                                               _ErrorStorage&,
                                               _Error
        >) {
        if (this->nested_value_fail()) {
          // We are the first child to complete with an error, so we must save the error. (Any
          // subsequent errors are ignored.)
          if constexpr (noexcept(__error_storage_
                                   ->template emplace<_Error>(static_cast<_Error&&>(__error)))) {
            __error_storage_->template emplace<_Error>(static_cast<_Error&&>(__error));
          } else {
            STDEXEC_TRY {
              __error_storage_->template emplace<_Error>(static_cast<_Error&&>(__error));
            }
            STDEXEC_CATCH_ALL {
              __error_storage_->template emplace<std::exception_ptr>(std::current_exception());
            }
          }
        }
      }
    };

    //
    // __error_.. provides the delayed error from a nested sequence
    // or nested value as the final item in the merged output sequence
    //

    template <class _ErrorReceiver, class _ErrorStorage>
    struct __error_op {
      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;

      void start() & noexcept {
        // emit delayed error into the sequence
        STDEXEC::__visit(
          [this](auto&& __error) noexcept {
            STDEXEC::set_error(
              static_cast<_ErrorReceiver&&>(__rcvr_), static_cast<decltype(__error)&&>(__error));
          },
          static_cast<_ErrorStorage&&>(*__op_->__error_storage_));
      }

      _ErrorReceiver __rcvr_;
      __operation_base_interface_t* __op_;
    };

    template <class _ErrorStorage>
    struct __error_sender {
      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;
      using sender_concept = STDEXEC::sender_t;

      template <class _ErrorReceiver>
      using __error_op_t = __error_op<_ErrorReceiver, _ErrorStorage>;

      template <class _Error>
      using __error_signature_t = STDEXEC::set_error_t(_Error);

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        using __result_t = STDEXEC::__mapply<
          STDEXEC::__mtransform<
            STDEXEC::__q<__error_signature_t>,
            STDEXEC::__qq<STDEXEC::completion_signatures>
          >,
          _ErrorStorage
        >;
        if constexpr (__ok<__result_t>) {
          return __result_t();
        } else {
          return STDEXEC::__throw_compile_time_error(__result_t());
        }
      }

      template <class _Self, receiver _ErrorReceiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _ErrorReceiver&& __rcvr)
        noexcept(__nothrow_move_constructible<_ErrorReceiver>) -> __error_op_t<_ErrorReceiver> {
        return __error_op_t<_ErrorReceiver>{static_cast<_ErrorReceiver&&>(__rcvr), __self.__op_};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      __operation_base_interface_t* __op_;
    };

    template <class _ErrorStorage>
    using __error_sender_t = __minvoke_q<__error_sender, _ErrorStorage>;

    template <class _ErrorStorage, class _EnvFn>
    struct __error_next_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;

      __operation_base_interface_t* __op_;
      _EnvFn __env_fn_;

      void set_value() noexcept {
        __op_->error_complete();
      }

      void set_stopped() noexcept {
        __op_->error_complete();
      }

      auto get_env() const noexcept -> __call_result_t<_EnvFn> {
        return __env_fn_();
      }
    };

    template <class _Receiver>
    struct __env_fn {
      using __nested_stop_t = __nested_stop<_Receiver>;
      using __nested_stop_env_t = typename __nested_stop_t::env_t;

      auto operator()() const noexcept -> __nested_stop_env_t {
        return __source_->env_from(*__rcvr_);
      }

      _Receiver* __rcvr_;
      __nested_stop_t* __source_;
    };

    template <class _Receiver, class _ErrorStorage>
    struct __operation_base : __operation_base_interface<_ErrorStorage> {
      using __nested_stop_t = __nested_stop<_Receiver>;
      using __nested_stop_env_t = typename __nested_stop_t::env_t;

      using __error_storage_t = _ErrorStorage;
      using __interface_t = __operation_base_interface<__error_storage_t>;

      using __error_sender_t = __merge_each::__error_sender_t<__error_storage_t>;
      using __error_next_sender_t = next_sender_of_t<_Receiver, __error_sender_t>;
      using __env_fn_t = __env_fn<_Receiver>;
      using __error_next_receiver_t = __error_next_receiver<_ErrorStorage, __env_fn_t>;
      using __error_op_t =
        STDEXEC::connect_result_t<__error_next_sender_t, __error_next_receiver_t>;

      explicit __operation_base(_Receiver __receiver) noexcept
        : __interface_t{&__error_storage_}
        , __rcvr_{static_cast<_Receiver&&>(__receiver)} {
        __interface_t::__token_ = __nested_stop_.get_token();
      }

      template <class _Error>
      void set_exception(_Error&& __error) noexcept {
        switch (__completion_.exchange(__completion_t::__error)) {
        case __completion_t::__started:
          // We must request stop. When the previous state is __error or __stopped, then stop has
          // already been requested.
          __nested_stop_.request_stop();
          [[fallthrough]];
        case __completion_t::__stopped:
          // We are the first child to complete with an error, so we must save the error. (Any
          // subsequent errors are ignored.)
          if constexpr (__nothrow_decay_copyable<_Error>) {
            __ex_ = std::make_exception_ptr(static_cast<_Error&&>(__error));
          } else {
            STDEXEC_TRY {
              __ex_ = std::make_exception_ptr(static_cast<_Error&&>(__error));
            }
            STDEXEC_CATCH_ALL {
              __ex_ = std::current_exception();
            }
          }
          break;
        case __completion_t::__error:; // We're already in the "error" state. Ignore the error.
        }
      }
      void set_break() noexcept {
        switch (__completion_.exchange(__completion_t::__stopped)) {
        case __completion_t::__started:
          // We must request stop. When the previous state is __error or __stopped, then stop has
          // already been requested.
          __nested_stop_.request_stop();
          break;
        case __completion_t::__stopped:
          [[fallthrough]];             // We're already in the "stopped" state. Ignore the break.
        case __completion_t::__error:; // We're already in the "error" state. Ignore the break.
        }
      }

      void sequence_started() noexcept {
        __active_ = 1;
      }
      void sequence_complete() noexcept {
        complete_if_none_active();
      }
      void sequence_break() noexcept {
        set_break();
        complete_if_none_active();
      }
      void nested_sequence_started() noexcept {
        ++__active_;
      }
      void nested_sequence_complete() noexcept {
        complete_if_none_active();
      }
      void nested_sequence_break() noexcept {
        set_break();
        complete_if_none_active();
      }
      void nested_value_started() noexcept override {
        ++__active_;
      }
      void nested_value_complete() noexcept override {
        complete_if_none_active();
      }
      bool nested_value_fail() noexcept override {
        switch (__completion_.exchange(__completion_t::__error)) {
        case __completion_t::__started:
          // We must request stop. When the previous state is __error or __stopped, then stop has
          // already been requested.
          __nested_stop_.request_stop();
          [[fallthrough]];
        case __completion_t::__stopped:
          // We are the first child to complete with an error, so we must save the error. (Any
          // subsequent errors are ignored.)
          return true;
          break;
        case __completion_t::__error:; // We're already in the "error" state. Ignore the error.
        }
        return false;
      }
      void nested_value_break() noexcept override {
        set_break();
        complete_if_none_active();
      }
      void error_complete() noexcept override {
        // do not double report error
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
      }

      void complete_if_none_active() noexcept {
        if (--__active_ == 0) {
          __nested_stop_.unregister_token();
          switch (__completion_.load(std::memory_order_relaxed)) {
          case __completion_t::__started:
            STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
            break;
          case __completion_t::__error:
            if (__ex_ != nullptr) {
              // forward error from the subscribed sequence of sequences
              STDEXEC::set_error(
                static_cast<_Receiver&&>(__rcvr_), static_cast<std::exception_ptr&&>(__ex_));
            } else {
              // forward error from the nested sequences as the last item
              if constexpr (
                __nothrow_callable<exec::set_next_t, _Receiver&, __error_sender_t>
                && __nothrow_connectable<__error_next_sender_t, __error_next_receiver_t>) {
                auto __next_sender = exec::set_next(__rcvr_, __error_sender_t{this});
                auto __next_receiver = __error_next_receiver_t{
                  this, __env_fn_t{&__rcvr_, &__nested_stop_}
                };
                __error_op_.__emplace_from([&]() {
                  return STDEXEC::connect(
                    static_cast<__error_next_sender_t&&>(__next_sender),
                    static_cast<__error_next_receiver_t&&>(__next_receiver));
                });
                STDEXEC::start(__error_op_.value());
              } else {
                STDEXEC_TRY {
                  auto __next_sender = exec::set_next(__rcvr_, __error_sender_t{this});
                  auto __next_receiver = __error_next_receiver_t{
                    this, __env_fn_t{&__rcvr_, &__nested_stop_}
                  };
                  __error_op_.__emplace_from([&]() {
                    return STDEXEC::connect(
                      static_cast<__error_next_sender_t&&>(__next_sender),
                      static_cast<__error_next_receiver_t&&>(__next_receiver));
                  });
                  STDEXEC::start(__error_op_.value());
                }
                STDEXEC_CATCH_ALL {
                  STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
                }
              }
            }
            break;
          case __completion_t::__stopped:
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            break;
          };
        }
      }

      _Receiver __rcvr_;
      _ErrorStorage __error_storage_{__no_init};
      std::exception_ptr __ex_ = nullptr;
      std::atomic_int32_t __active_ = 0;
      std::atomic<__completion_t> __completion_{__completion_t::__started};
      __nested_stop_t __nested_stop_{};
      STDEXEC::__optional<__error_op_t> __error_op_{};
    };

    //
    // __nested_value.. exists to store the an error signal from a
    // value if it is the first error. All error completions are
    // removed from the completion_signatures. when an error occurs
    // the stopped signal will be emitted here and the error will
    // be emitted as a separate item after all active operations
    // have completed. otherwise __nested_value.. is transparent.
    //

    template <class _NestedValueReceiver>
    struct __nested_value_operation_base {

      _NestedValueReceiver __rcvr_;
    };

    template <class _NestedValueReceiver, class _ErrorStorage>
    struct __receive_nested_value {
      using receiver_concept = receiver_t;

      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;

      __nested_value_operation_base<_NestedValueReceiver>* __nested_value_op_;
      __operation_base_interface_t* __op_;

      template <class... _Results>
      void set_value(_Results&&... __results) noexcept {
        auto __op = __op_;
        STDEXEC::set_value(
          static_cast<_NestedValueReceiver&&>(__nested_value_op_->__rcvr_),
          static_cast<_Results&&>(__results)...);
        __op->nested_value_complete();
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        auto __op = __op_;
        STDEXEC::set_stopped(static_cast<_NestedValueReceiver&&>(__nested_value_op_->__rcvr_));
        __op->store_error(static_cast<_Error&&>(__error));
        __op->nested_value_break();
      }

      void set_stopped() noexcept {
        auto __op = __op_;
        STDEXEC::set_stopped(static_cast<_NestedValueReceiver&&>(__nested_value_op_->__rcvr_));
        __op->nested_value_complete();
      }

      using __env_t = decltype(__op_->env_from(__declval<env_of_t<_NestedValueReceiver>>()));
      auto get_env() const noexcept -> __env_t {
        return __op_->env_from(STDEXEC::get_env(__nested_value_op_->__rcvr_));
      }
    };

    template <
      class _NestedValueSender,
      class _NestedValueReceiver,
      class _ErrorStorage,
      class _NestedValueOp
    >
    struct __nested_value_op : __nested_value_operation_base<_NestedValueReceiver> {
      using __base_t = __nested_value_operation_base<_NestedValueReceiver>;
      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;
      using __receiver = __receive_nested_value<_NestedValueReceiver, _ErrorStorage>;
      using __nested_value_op_t = STDEXEC::connect_result_t<_NestedValueSender, __receiver>;

      __nested_value_op(
        _NestedValueReceiver __rcvr,
        _NestedValueSender __result,
        __operation_base_interface_t* __op)
        noexcept(
          __nothrow_move_constructible<_NestedValueReceiver>
          && __nothrow_connectable<_NestedValueSender, __receiver>)
        : __base_t{static_cast<_NestedValueReceiver&&>(__rcvr)}
        , __nested_value_op_{STDEXEC::connect(
            static_cast<_NestedValueSender&&>(__result),
            __receiver{this, __op})}
        , __op_{__op} {
      }

      void start() & noexcept {
        __op_->nested_value_started();
        STDEXEC::start(__nested_value_op_);
      }

      __nested_value_op_t __nested_value_op_;
      __operation_base_interface_t* __op_;
    };

    template <class _NestedValueSender, class _NestedValueReceiver, class _ErrorStorage>
    using __nested_value_op_t = STDEXEC::__minvoke_q<
      __nested_value_op,
      _NestedValueSender,
      _NestedValueReceiver,
      _ErrorStorage,
      __minvoke_q<
        STDEXEC::connect_result_t,
        _NestedValueSender,
        __receive_nested_value<_NestedValueReceiver, _ErrorStorage>
      >
    >;

    template <class _NestedValueSender, class _ErrorStorage>
    struct __nested_value_sender {
      using sender_concept = STDEXEC::sender_t;
      using __operation_base_interface_t = __operation_base_interface<_ErrorStorage>;

      template <class _NestedValueReceiver>
      using __nested_value_op_t =
        __nested_value_op_t<_NestedValueSender, _NestedValueReceiver, _ErrorStorage>;

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() noexcept {
        using __result_t = STDEXEC::transform_completion_signatures<
          STDEXEC::__completion_signatures_of_t<_NestedValueSender, _Env...>,
          STDEXEC::completion_signatures<STDEXEC::set_stopped_t()>,
          STDEXEC::__cmplsigs::__default_set_value,
          drop
        >;
        if constexpr (__ok<__result_t>) {
          return __result_t();
        } else {
          return STDEXEC::__throw_compile_time_error(__result_t());
        }
      }

      template <class _Self, receiver _NestedValueReceiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _NestedValueReceiver&& __rcvr)
        noexcept(__nothrow_constructible_from<
                 __nested_value_op_t<_NestedValueReceiver>,
                 _NestedValueReceiver,
                 _NestedValueSender,
                 __operation_base_interface_t*
        >) -> __nested_value_op_t<_NestedValueReceiver> {
        if constexpr (__ok<__nested_value_op_t<_NestedValueReceiver>>) {
          return {
            static_cast<_NestedValueReceiver&&>(__rcvr),
            static_cast<_NestedValueSender&&>(__self.__nested_value_),
            __self.__op_};
        } else {
          return {};
        }
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _NestedValueSender __nested_value_;
      __operation_base_interface_t* __op_;
    };

    template <class _NestedValue, class _ErrorStorage>
    using __nested_value_sender_t = __minvoke_q<__nested_value_sender, _NestedValue, _ErrorStorage>;

    //
    // __next_.. is returned from set_next. Unlike the rest of the
    // receivers here, the completion signals to the next receiver
    // travel to the producer.
    // Only set_value() and set_stopped() are allowed.
    // - set_value() will signal the producer to emit the next
    //     sequence and cleanup the storage for the previous
    //     sequence.
    // - set_stopped() will signal the producer to break out and
    //     send no more sequences.
    //

    struct __next_operation_interface {
      virtual ~__next_operation_interface() = default;
      virtual void nested_sequence_complete() noexcept = 0;
      virtual void nested_sequence_break() noexcept = 0;
    };

    template <class _NextReceiver, class _OperationBase, class _NestedSeqOp>
    struct __next_operation_base : __next_operation_interface {
      explicit __next_operation_base(_NextReceiver __receiver, _OperationBase* __op)
        noexcept(__nothrow_move_constructible<_NextReceiver>)
        : __next_operation_interface{}
        , __rcvr_{static_cast<_NextReceiver&&>(__receiver)}
        , __op_{__op} {
      }

      void nested_sequence_complete() noexcept override {
        auto __op = __op_;
        STDEXEC::set_value(static_cast<_NextReceiver&&>(this->__rcvr_));
        __op->nested_sequence_complete();
      }

      void nested_sequence_break() noexcept override {
        auto __op = __op_;
        STDEXEC::set_stopped(static_cast<_NextReceiver&&>(this->__rcvr_));
        __op->nested_sequence_break();
      }

      _NextReceiver __rcvr_;
      _OperationBase* __op_;
      _NestedSeqOp __nested_seq_op_{__no_init};
    };

    //
    // __receive_nested_values is subscribed to each nested sequence
    // This forwards all the nested value senders to the output sequence.
    // This wraps the nested value senders to capture and delay any
    // error signals emitted by the nested value sender.
    // This captures and delays any error signals received directly.
    // set_stopped is ignored. This allows nested sequences to be
    // stopped individually without stopping all the other nested
    // sequences or the merge_each operation.
    //

    template <class _OperationBase>
    struct __receive_nested_values {
      using receiver_concept = receiver_t;

      using __error_storage_t = _OperationBase::__error_storage_t;
      using __receiver_t = decltype(_OperationBase::__rcvr_);

      template <class _NestedValue>
      using __nested_value_sender_t =
        __merge_each::__nested_value_sender_t<_NestedValue, __error_storage_t>;


      template <class _NestedValue>
      auto set_next(_NestedValue&& __nested_value) noexcept(
        __nothrow_callable<exec::set_next_t, __receiver_t&, __nested_value_sender_t<_NestedValue>>) {
        return exec::set_next(
          __op_->__rcvr_,
          __nested_value_sender_t<_NestedValue>{
            static_cast<_NestedValue&&>(__nested_value), __op_});
      }

      void set_value() noexcept {
        __next_seq_op_->nested_sequence_complete();
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __op_->store_error(static_cast<_Error&&>(__error));
        __next_seq_op_->nested_sequence_break();
      }

      void set_stopped() noexcept {
        __next_seq_op_->nested_sequence_complete();
      }

      using __env_t = _OperationBase::__nested_stop_env_t;
      auto get_env() const noexcept -> __env_t {
        return __op_->__nested_stop_.env_from(__op_->__rcvr_);
      }

      __next_operation_interface* __next_seq_op_;
      _OperationBase* __op_;
    };

    template <class _OperationBase>
    using __receive_nested_values_t = __minvoke_q<__receive_nested_values, _OperationBase>;

    struct _MERGE_WITH_REQUIRES_A_SEQUENCE_OF_SEQUENCES_ { };

    template <class _Sequence, class _Sender, class... _Env>
    struct __value_completions_error {
      template <class... _Args>
      using __f = std::conditional_t<
        sizeof...(_Env) == 0 && sizeof...(_Args) == 1 && (dependent_sender<_Args> && ...),
        STDEXEC::__mexception<dependent_sender_error, _WITH_PRETTY_SENDER_<_Args>...>,
        STDEXEC::__mexception<
          _WHAT_(_INVALID_ARGUMENT_),
          _WHERE_(_IN_ALGORITHM_, merge_each_t),
          _WHY_(_MERGE_WITH_REQUIRES_A_SEQUENCE_OF_SEQUENCES_),
          _WITH_PRETTY_SEQUENCE_<_Sequence>,
          _WITH_PRETTY_SENDER_<_Sender>,
          __fn_t<_WITH_ENVIRONMENT_, _Env>...,
          _WITH_ARGUMENTS_(_Args...)
        >
      >;
    };

    struct __compute {
      template <class... _Env>
      struct __arg_of_base_t {
        template <class... _Senders>
        static constexpr bool __valid_args() noexcept {
          return sizeof...(_Senders) == 1 && (has_sequence_item_types<_Senders, _Env...> && ...);
        }
      };

      //
      // __nested_sequences extracts the types of all the nested sequences.
      //

      template <class _Sequence, class _Sender, class... _Env>
      struct __arg_of_t : __arg_of_base_t<_Env...> {
        template <class... _Senders>
        using __f = __minvoke_if_c<
          __arg_of_base_t<_Env...>::template __valid_args<_Senders...>(),
          __mconst<STDEXEC::__mlist<_Senders...>>,
          __value_completions_error<_Sequence, _Sender, _Env...>,
          _Senders...
        >;
      };

      template <class _Sequence, class _Sender, class... _Env>
      struct __gather_sequences_t {
        template <class _Completions>
        using __f = STDEXEC::__gather_completion_signatures_t<
          _Completions,
          STDEXEC::set_value_t,
          // if set_value
          __arg_of_t<_Sequence, _Sender, _Env...>::template __f,
          // else remove
          STDEXEC::__mconst<STDEXEC::__mlist<>>::__f,
          // concat to __mlist result
          STDEXEC::__mconcat<STDEXEC::__qq<STDEXEC::__mlist>>::__f
        >;
      };

      template <class _Sequence, class _Sender, class... _Env>
      using __nested_sequences_from_item_type_t = STDEXEC::__minvoke<
        __mtry<__gather_sequences_t<_Sequence, _Sender, _Env...>>,
        STDEXEC::__completion_signatures_of_t<_Sender, _Env...>
      >;

      template <class _Sequence, class... _Env>
      struct __nested_sequences_fn {
        template <class... _Senders>
        using __f = __mapply<
          STDEXEC::__munique<STDEXEC::__qq<__mlist>>,
          STDEXEC::__minvoke<
            STDEXEC::__mconcat<STDEXEC::__qq<__mlist>>,
            __minvoke_q<__nested_sequences_from_item_type_t, _Sequence, _Senders, _Env...>...
          >
        >;
      };

      template <class _Sequence, class... _Env>
      using __nested_sequences_t =
        __mapply<__nested_sequences_fn<_Sequence, _Env...>, __item_types_of_t<_Sequence, _Env...>>;

      //
      // __all_nested_values extracts the types of all the nested value senders.
      //

      template <class... _Env>
      struct __all_nested_values_fn {
        template <class... _Sequences>
        using __f = STDEXEC::__minvoke<
          STDEXEC::__mconcat<STDEXEC::__qq<STDEXEC::__mlist>>,
          __item_types_of_t<_Sequences, _Env...>...
        >;
      };

      template <class _Sequence, class... _Env>
      using __all_nested_values_t =
        __mapply<__all_nested_values_fn<_Env...>, __nested_sequences_t<_Sequence, _Env...>>;

      //
      // __error_types extracts the types of all the errors emitted by all the senders in the list.
      //

      template <class... _Env>
      struct __error_types_fn {
        template <class _Sender>
        using __f = STDEXEC::__error_types_t<
          __completion_signatures_of_t<_Sender, _Env...>,
          STDEXEC::__qq<STDEXEC::__mlist>
        >;
      };

      template <class _Senders, class... _Env>
      using __error_types_t = __mapply<
        STDEXEC::__mtransform<
          __error_types_fn<_Env...>,
          STDEXEC::__mconcat<STDEXEC::__qq<STDEXEC::__mlist>>
        >,
        _Senders
      >;

      //
      // __errors extracts the types of all the errors emitted by:
      // - all the senders of nested sequences
      // - all the nested sequences
      // - all the nested values senders
      // This represents all the errors that are emitted on the
      // output sequence.
      //

      template <class _Sequence, class... _Env>
      using __errors_t = STDEXEC::__minvoke<
        STDEXEC::__mconcat<STDEXEC::__qq<STDEXEC::__mlist>>,
        // always include std::exception_ptr
        STDEXEC::__mlist<std::exception_ptr>,
        // include errors from senders of the nested sequences
        __error_types_t<__item_types_of_t<_Sequence, _Env...>, _Env...>,
        // include errors from the nested sequences
        __error_types_t<__nested_sequences_t<_Sequence, _Env...>, _Env...>,
        // include errors from all the item type senders of all the nested sequences
        __error_types_t<__all_nested_values_t<_Sequence, _Env...>, _Env...>
      >;

      //
      // __error_variant_t makes a variant type of all the errors
      // that are emitted on the output sequence. This is used
      // to store the first error and emit it as an item after
      // all active operations are completed.
      //

      template <class _Sequence, class... _Env>
      using __error_variant_t =
        __mapply_q<STDEXEC::__uniqued_variant, __errors_t<_Sequence, _Env...>>;

      //
      // __nested_values extracts the types of all the nested value senders and
      // builds the item_types list for the merge_each sequence sender.
      //

      template <class _ErrorStorage, class... _Env>
      struct __nested_values_fn {
        template <class... _AllItems>
        using __f = __minvoke<
          __munique<STDEXEC::__q<item_types>>,
          __merge_each::__nested_value_sender_t<_AllItems, _ErrorStorage>...,
          __merge_each::__error_sender_t<_ErrorStorage>
        >;
      };

      template <class _Sequence, class... _Env>
      using __nested_values_t = __mapply<
        __nested_values_fn<__error_variant_t<_Sequence, _Env...>, _Env...>,
        __all_nested_values_t<_Sequence, _Env...>
      >;

      //
      // __nested_sequence_ops_variant makes a variant that contains the
      // types of all the nested sequence operations.
      //

      template <class _OperationBase>
      struct __nested_sequence_op_fn {
        template <class _Sequence>
        using __f =
          __minvoke_q<subscribe_result_t, _Sequence, __receive_nested_values_t<_OperationBase>>;
      };

      template <class _Sequence, class _OperationBase>
      using __nested_sequence_op_t = __minvoke<__nested_sequence_op_fn<_OperationBase>, _Sequence>;

      template <class _Sequence, class _Receiver>
      using __operation_base_t = __minvoke<
        __mtry_q<__operation_base>,
        _Receiver,
        __error_variant_t<_Sequence, __env_with_inplace_stop_token_result_t<env_of_t<_Receiver>>>
      >;

      template <class _Sequence, class _Receiver>
      using __nested_sequence_ops_variant = __mapply<
        STDEXEC::__mtransform<
          __nested_sequence_op_fn<__operation_base_t<_Sequence, _Receiver>>,
          STDEXEC::__qq<STDEXEC::__uniqued_variant>
        >,
        __nested_sequences_t<
          _Sequence,
          __env_with_inplace_stop_token_result_t<STDEXEC::env_of_t<_Receiver>>
        >
      >;
    };

    //
    // __receive_nested_sequence is connected to each sender of a nested sequence.
    // The nested sequence is then subscribed and the operation is stored in a
    // variant of all possible nested sequence operations.
    //

    template <class _NextReceiver, class _OperationBase, class _NestedSeqOp>
    struct __receive_nested_sequence {
      using receiver_concept = receiver_t;

      using __next_op_base_t = __next_operation_base<_NextReceiver, _OperationBase, _NestedSeqOp>;

      __next_op_base_t* __next_seq_op_;
      _OperationBase* __op_;

      template <class _NestedSequence>
      auto set_value(_NestedSequence&& __sequence) noexcept {
        using __receiver_t = __receive_nested_values_t<_OperationBase>;
        using __nested_op_t = __compute::__nested_sequence_op_t<_NestedSequence, _OperationBase>;
        static constexpr bool __is_nothrow =
          __nothrow_subscribable<_NestedSequence, __receiver_t>
          && STDEXEC::__nothrow_constructible_from<_NestedSeqOp, __nested_op_t>;
        STDEXEC_TRY {
          auto& __nested_seq_op = __next_seq_op_->__nested_seq_op_.__emplace_from(
            subscribe,
            static_cast<_NestedSequence&&>(__sequence),
            __receiver_t{__next_seq_op_, __op_});
          STDEXEC::start(__nested_seq_op);
        }
        STDEXEC_CATCH_ALL {
          if constexpr (!__is_nothrow) {
            __op_->store_error(std::current_exception());
            __op_->nested_sequence_break();
          }
        }
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __op_->store_error(static_cast<_Error&&>(__error));
        __op_->nested_sequence_break();
      }

      void set_stopped() noexcept {
        __op_->nested_sequence_complete();
      }

      using __env_t = typename _OperationBase::__nested_stop_env_t;
      auto get_env() const noexcept -> __env_t {
        return __op_->__nested_stop_.env_from(__op_->__rcvr_);
      }
    };

    template <
      class _NestedSequenceSender,
      class _NextReceiver,
      class _OperationBase,
      class _NestedSeqOp
    >
    struct __next_sequence_op : __next_operation_base<_NextReceiver, _OperationBase, _NestedSeqOp> {
      using __base_t = __next_operation_base<_NextReceiver, _OperationBase, _NestedSeqOp>;
      using __receiver = __receive_nested_sequence<_NextReceiver, _OperationBase, _NestedSeqOp>;
      using __nested_sequence_op_t = STDEXEC::connect_result_t<_NestedSequenceSender, __receiver>;

      __next_sequence_op(
        _NextReceiver __rcvr,
        _OperationBase* __op,
        _NestedSequenceSender __nested_sequence)
        noexcept(
          __nothrow_move_constructible<_NextReceiver>
          && __nothrow_connectable<_NestedSequenceSender, __receiver>)
        : __base_t{static_cast<_NextReceiver&&>(__rcvr), __op}
        , __op_(__op)
        , __nested_sequence_op_{STDEXEC::connect(
            static_cast<_NestedSequenceSender&&>(__nested_sequence),
            __receiver{this, __op_})} {
      }

      void start() & noexcept {
        __op_->nested_sequence_started();
        STDEXEC::start(__nested_sequence_op_);
      }

      _OperationBase* __op_;
      __nested_sequence_op_t __nested_sequence_op_;
    };

    template <class _NestedSequenceSender, class _OperationBase, class _NestedSeqOp>
    struct __next_sequence_sender {
      using sender_concept = STDEXEC::sender_t;

      template <class _NextReceiver>
      using __next_sequence_op_t =
        __next_sequence_op<_NestedSequenceSender, _NextReceiver, _OperationBase, _NestedSeqOp>;

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() noexcept {
        return STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>();
      }

      template <class _Self, receiver _NextReceiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _NextReceiver&& __rcvr)
        noexcept(__nothrow_constructible_from<
                 __next_sequence_op_t<_NextReceiver>,
                 _NextReceiver,
                 _OperationBase*,
                 _NestedSequenceSender
        >) -> __next_sequence_op_t<_NextReceiver> {
        return {
          static_cast<_NextReceiver&&>(__rcvr),
          __self.__op_,
          static_cast<_NestedSequenceSender&&>(__self.__nested_sequence_)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _OperationBase* __op_;
      _NestedSequenceSender __nested_sequence_;
    };

    //
    // __receive_nested_sequences is subscribed to the input sequence of sequences
    // each new sender of a nested sequence is placed in a __next_sequence_sender
    // that is returned from set_next()
    //

    template <class _OperationBase, class _NestedSeqOp>
    struct __receive_nested_sequences {
      using receiver_concept = receiver_t;

      template <class _NestedSequenceSender>
      using __next_sequence_sender_t =
        __next_sequence_sender<_NestedSequenceSender, _OperationBase, _NestedSeqOp>;

      _OperationBase* __op_;

      template <STDEXEC::sender _NestedSequenceSender>
      auto set_next(_NestedSequenceSender&& __nested_sequence)
        noexcept(__nothrow_constructible_from<
                 __next_sequence_sender_t<_NestedSequenceSender>,
                 _OperationBase*,
                 _NestedSequenceSender
        >) -> next_sender auto {
        return __next_sequence_sender_t<_NestedSequenceSender>{
          __op_, static_cast<_NestedSequenceSender>(__nested_sequence)};
      }

      void set_value() noexcept {
        __op_->sequence_complete();
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __op_->set_exception(static_cast<_Error&&>(__error));
        __op_->sequence_break();
      }

      void set_stopped() noexcept {
        __op_->sequence_break();
      }

      using __env_t = typename _OperationBase::__nested_stop_env_t;
      auto get_env() const noexcept -> __env_t {
        return __op_->__nested_stop_.env_from(__op_->__rcvr_);
      }
    };

    template <class _Receiver, class _Sequence, class _ErrorStorage>
    struct __operation : __operation_base<_Receiver, _ErrorStorage> {
      using __base_t = __operation_base<_Receiver, _ErrorStorage>;
      using __nested_seq_op_t = __compute::__nested_sequence_ops_variant<_Sequence, _Receiver>;
      using __receiver = __receive_nested_sequences<__base_t, __nested_seq_op_t>;
      using __op_t = subscribe_result_t<_Sequence, __receiver>;

      explicit __operation(_Receiver __rcvr, _Sequence __sequence)
        noexcept(__nothrow_subscribable<_Sequence, __receiver>)
        : __base_t{static_cast<_Receiver&&>(__rcvr)}
        , __op_{subscribe(static_cast<_Sequence&&>(__sequence), __receiver{this})} {
      }

      void start() & noexcept {
        this->__nested_stop_.register_token(this->__rcvr_);
        if (this->__nested_stop_.stop_requested()) {
          // Stop has already been requested. Don't bother starting
          // the child operations.
          STDEXEC::set_stopped(static_cast<_Receiver&&>(this->__rcvr_));
        } else {
          this->sequence_started();
          STDEXEC::start(__op_);
        }
      }

      __op_t __op_;
    };

    template <class _Receiver, class _Sequence>
    using __operation_t = __minvoke_q<
      __operation,
      _Receiver,
      _Sequence,
      __compute::__error_variant_t<
        _Sequence,
        __env_with_inplace_stop_token_result_t<env_of_t<_Receiver>>
      >
    >;


    template <class _Receiver>
    struct __subscribe_fn {
      template <class _Sequence>
      using __opstate_t = __operation_t<_Receiver, _Sequence>;

      template <class _Sequence>
      auto operator()(__ignore, __ignore, _Sequence __sequence) noexcept(
        __nothrow_constructible_from<__operation_t<_Receiver, _Sequence>, _Receiver, _Sequence>)
        -> __opstate_t<_Sequence> {
        if constexpr (__ok<__opstate_t<_Sequence>>) {
          return __opstate_t<_Sequence>{
            static_cast<_Receiver&&>(__rcvr_), static_cast<_Sequence&&>(__sequence)};
        } else {
          return {};
        }
      }

      _Receiver& __rcvr_;
    };

    //
    // merge_each is a sequence adaptor that takes a sequence of nested
    // sequences and merges all the nested values from all the nested
    // sequences into a single output sequence.
    //
    // the first error encountered will trigger a stop request for all
    // active operations. The error is stored and is emitted only after
    // all the active operations have completed.
    // If the error was emitted from an item, a new item is emitted
    // at the end to deliver the stored error.
    //
    // any nested sequence or nested value that completes with
    // set_stopped will not cause any other operations to be stopped.
    // This allows individual nested sequences to be stopped without
    // breaking the merge of the remaining sequences.
    //

    struct merge_each_t {
      template <class _Sequence>
      auto operator()(_Sequence&& __sequence) const noexcept(__nothrow_decay_copyable<_Sequence>)
        -> __well_formed_sequence_sender auto {
        return make_sequence_expr<merge_each_t>(__(), static_cast<_Sequence&&>(__sequence));
      }

      template <sender_expr_for<merge_each_t> _Self, class... _Env>
      static consteval auto get_item_types() {
        using __result_t = __compute::__nested_values_t<
          __child_of<_Self>,
          __env_with_inplace_stop_token_result_t<_Env>...
        >;
        if constexpr (__ok<__result_t>) {
          return __result_t();
        } else {
          return exec::__invalid_item_types(__result_t());
        }
      }

      template <class _Self, class... _Env>
      struct __completions_fn {
        template <class... _Sequences>
        using __f = __minvoke_q<
          __concat_completion_signatures_t,
          completion_signatures<set_stopped_t()>,
          __completion_signatures_of_t<__child_of<_Self>, _Env...>,
          __completion_signatures_of_t<_Sequences, _Env...>...
        >;
      };

      template <class _Self, class... _Env>
      using __completions_t = __mapply<
        __completions_fn<_Self, _Env...>,
        __compute::__nested_sequences_t<__child_of<_Self>, _Env...>
      >;

      template <sender_expr_for<merge_each_t> _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: update this to use constant evaluation:
        using __result_t = __minvoke<
          __mtry_q<__completions_t>,
          _Self,
          __env_with_inplace_stop_token_result_t<_Env>...
        >;
        if constexpr (__ok<__result_t>) {
          static_assert(__valid_completion_signatures<__result_t>);
          return __result_t();
        } else {
          return STDEXEC::__throw_compile_time_error(__result_t());
        }
      }

      static constexpr auto subscribe =
        []<class _Sequence, receiver _Receiver>(_Sequence&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_applicable<__subscribe_fn<_Receiver>, _Sequence>)
        -> __apply_result_t<__subscribe_fn<_Receiver>, _Sequence> {
        static_assert(sender_expr_for<_Sequence, merge_each_t>);
        return __apply(__subscribe_fn<_Receiver>{__rcvr}, static_cast<_Sequence&&>(__sndr));
      };
    };
  } // namespace __merge_each

  using __merge_each::merge_each_t;
  inline constexpr merge_each_t merge_each{};
} // namespace exec
