/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../../stdexec/execution.hpp"
#include <type_traits>
#include <ranges>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "algorithm_base.cuh"
#include "common.cuh"
#include "../detail/throw_on_cuda_error.cuh"
#include "../stream_context.cuh"

template <typename T>
struct type_printer;

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    namespace reduce_ {

      template <class _Receiver>
      struct __connect_fn;
      struct reduce_t;

      template <class _InitT, class _Fun>
      struct __data {
        _InitT __initT_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;
        static constexpr auto __mbrs_ = __mliterals<&__data::__initT_, &__data::__fun_>();
      };
      template <class _InitT, class _Fun>
      __data(_InitT, _Fun) -> __data<_InitT, _Fun>;

      template <class SenderId, class ReceiverId, class InitT, class Fun>
      struct receiver_t
        : public __algo_range_init_fun::receiver_t<
            SenderId,
            ReceiverId,
            InitT,
            Fun,
            receiver_t<SenderId, ReceiverId, InitT, Fun>> {
        using base = __algo_range_init_fun::
          receiver_t<SenderId, ReceiverId, InitT, Fun, receiver_t<SenderId, ReceiverId, InitT, Fun>>;

        template <class Range>
        using result_t = typename __algo_range_init_fun::binary_invoke_result_t<Range, InitT, Fun>;

        template <class Range>
        static void set_value_impl(base::__t&& self, Range&& range) noexcept {
          cudaError_t status{cudaSuccess};
          cudaStream_t stream = self.op_state_.get_stream();

          // `range` is produced asynchronously, so we need to wait for it to be ready
          if (status = STDEXEC_DBG_ERR(cudaStreamSynchronize(stream)); status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          using value_t = result_t<Range>;
          value_t* d_out = static_cast<value_t*>(self.op_state_.temp_storage_);

          void* d_temp_storage{};
          std::size_t temp_storage_size{};

          auto first = begin(range);
          auto last = end(range);

          std::size_t num_items = std::distance(first, last);

          if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                d_temp_storage,
                temp_storage_size,
                first,
                d_out,
                num_items,
                self.fun_,
                self.init_,
                stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          if (status = STDEXEC_DBG_ERR( //
                cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                d_temp_storage,
                temp_storage_size,
                first,
                d_out,
                num_items,
                self.fun_,
                self.init_,
                stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          status = STDEXEC_DBG_ERR(cudaFreeAsync(d_temp_storage, stream));
          self.op_state_.defer_temp_storage_destruction(d_out);

          if (status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <class Sender, class Receiver>
        auto connect(Sender&& sndr, Receiver rcvr) {
          return __sender_apply((Sender&&) sndr, __connect_fn<Receiver>{rcvr});
        }

        template <__lazy_sender_for<reduce_::reduce_t> _Sender, class _Receiver>
        static auto connect(_Sender&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_callable< __sender_apply_fn, _Sender, __connect_fn<_Receiver>>)
          -> __call_result_t< __sender_apply_fn, _Sender, __connect_fn<_Receiver>> {
          return __sender_apply((_Sender&&) __sndr, __connect_fn<_Receiver>{__rcvr});
        }
      };

      template <class _CvrefSenderId, class _ReceiverId, class _InitT, class _Fun>
      struct __operation {
        using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __receiver_id = receiver_t<_CvrefSender, _ReceiverId, _InitT, _Fun>;
        using __receiver_t = stdexec::__t<__receiver_id>;
        _CvrefSender sender;
        _Receiver rcvr;

        struct __t : __immovable {
          using __id = __operation;
          using __data_t = __data<_InitT, _Fun>;

          typename __receiver_id::__op_state __state_;
          connect_result_t<_CvrefSender, __receiver_t> __op_;

          __t(_CvrefSender&& __sndr, _Receiver __rcvr, __data_t __data) //
            noexcept(__nothrow_decay_copyable<_Receiver>                //
                       && __nothrow_decay_copyable<__data_t>            //
                         && __nothrow_connectable<_CvrefSender, __receiver_t>)
            : __state_{(__data_t&&) __data, (_Receiver&&) __rcvr}
            , __op_(connect((_CvrefSender&&) __sndr, __receiver_t{&__state_})) {
          }

          friend void tag_invoke(start_t, __t& __self) noexcept {
            start(__self.__op_);
          }

          template <__decays_to<__t> _Self, receiver _Receivr>
            requires sender_to<_CvrefSender&, __receiver_t >
          friend __operation tag_invoke(connect_t, __t& __self, _Receivr __rcvr) noexcept {
            return {((_Self&&) __self).sender, (_Receivr&&) __rcvr};
          }

          template <__decays_to<__t> _Self, class _Receivr>
          friend __operation tag_invoke(connect_t, _Self&& __self, _Receivr __rcvr) noexcept {
            return {((_Self&&) __self).sender, (_Receivr&&) __rcvr};
          }
        };
      };

      template <class _Receiver>
      struct __connect_fn {
        _Receiver& __rcvr_;

        template <class _Child, class _Data>
        using __operation_t = //
          __t<__operation<
            __cvref_id<_Child>,
            __id<_Receiver>,
            decltype(_Data::__initT_),
            decltype(_Data::__fun_)>>;

        template <class _Data, class _Child>
        auto operator()(__ignore, _Data __data, _Child&& __child) const noexcept(
          __nothrow_constructible_from<__operation_t<_Child, _Data>, _Child, _Receiver, _Data>)
          -> __operation_t<_Child, _Data> {
          return __operation_t<_Child, _Data>{
            (_Child&&) __child, (_Receiver&&) __rcvr_, (_Data&&) __data};
        }
      };

      template <class SenderId, class InitT, class Fun>
      struct sender_t
        : public __algo_range_init_fun::
            sender_t<SenderId, InitT, Fun, sender_t<SenderId, InitT, Fun>> {
        using Sender = SenderId;

        template <class Receiver>
        using receiver_t =
          stdexec::__t<reduce_::receiver_t< SenderId, stdexec::__id<Receiver>, InitT, Fun>>;


        using is_sender = void;
        template <class _Ty>
        using _set_error_t = stdexec::completion_signatures<set_error_t(__decay_t<cudaError>&&)>;
        template <class Range>
        using _set_value_t = stdexec::completion_signatures<set_value_t(
          ::std::add_lvalue_reference_t< __force_minvoke<__q<__algo_range_init_fun::binary_invoke_result_t> ,Range, InitT, Fun> >
        )>;


        // The signatures used to be here. Compiler doesn't look in this struct though, so was moved to reduce_t
        template <class Self, class Env>
        using _completion_signatures_t = //
          __try_make_completion_signatures<
            __copy_cvref_t<Self, SenderId>, // ?
            Env,
            stdexec::completion_signatures< //
              set_stopped_t() >,
            __q<_set_value_t>,
            __q<_set_error_t> >; //

        friend auto tag_invoke(get_env_t, const sender_t& self) noexcept
          -> env_of_t<const Sender&> {
          return get_env(self.sndr_);
        }
      };
    }

    struct reduce_t {
      template <class Sender, class InitT, class Fun>
      using __sender =
        stdexec::__t<reduce_::sender_t<stdexec::__id<__decay_t<Sender>>, InitT, Fun>>;

      template <class Range, class InitT, class Fun = cub::Sum>
      using _set_value_t = stdexec::completion_signatures<set_value_t(
          ::std::add_lvalue_reference_t< __force_minvoke<__q<__algo_range_init_fun::binary_invoke_result_t> , Range, InitT, Fun> >
        )>;
      template <class _Ty>
      using _set_error_t =
        stdexec::completion_signatures<stdexec::set_error_t(__decay_t<cudaError>&&)>;
      using is_sender = void;

      template <class CvRefSender, class _Env, class InitT, class Fun>
      using compl_sigs =
        __try_make_completion_signatures<
          CvRefSender, 
          _Env,
          stdexec::completion_signatures< set_stopped_t() >,
          __q<_set_value_t>,
          __q<_set_error_t> >;

      template < sender Sender, __movable_value InitT, __movable_value Fun = cub::Sum>
      auto operator()(Sender&& sndr, InitT init, Fun fun) const {
        return __make_basic_sender(reduce_t(), reduce_::__data{init, (Fun&&) fun}, (Sender&&) sndr);
      }

      template <__lazy_sender_for<reduce_t> _Sender>
      static auto get_env(const _Sender&) noexcept {
        return empty_env{};
      }

      template <__lazy_sender_for<reduce_t> _Sender, class _Env>
      static auto get_completion_signatures(_Sender&& __sndr, _Env&& env) {
        return __sender_apply(
          (_Sender&&) __sndr,
          []<class _Data, class _Child>(reduce_t tag, _Data data, _Child&& child) {
            using _InitT = decltype(_Data::__initT_);
            using _Fun = decltype(_Data::__fun_);

            return compl_sigs<_Child, _Env, _InitT, _Fun>();
          });
      }

  

      template <stdexec::__decays_to<reduce_t> Sender, class Env, class InitT, class Fun> 
      friend auto tag_invoke(get_completion_signatures_t, Sender&&, Env&&, InitT, Fun) { 
        return compl_sigs<Sender, Env, InitT, Fun>{};
      }
      template <class InitT, class Fun = cub::Sum>
      __binder_back<reduce_t, InitT, Fun> operator()(InitT init, Fun fun = {}) const {
        return {
          {},
          {},
          {(InitT&&) init, (Fun&&) fun}
        };
      }
    };
  }

  inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};

}
