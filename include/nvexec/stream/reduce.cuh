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

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    namespace reduce_ {

      template <class _Receiver>
      struct __connect_fn;

      template <class _Init, class _Fun>
      struct __data {
        _Init __init_;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Fun __fun_;
        static constexpr auto __mbrs_ = __mliterals<&__data::__init_, &__data::__fun_>();
      };
      template <class _Init, class _Fun>
      __data(_Init, _Fun) -> __data<_Init, _Fun>;

      template <class SenderId, class ReceiverId, class Init, class Fun>
      struct receiver_t
        : public __algo_range_init_fun::receiver_t<
            SenderId,
            ReceiverId,
            Init,
            Fun,
            receiver_t<SenderId, ReceiverId, Init, Fun>> {
        using base = __algo_range_init_fun::
          receiver_t<SenderId, ReceiverId, Init, Fun, receiver_t<SenderId, ReceiverId, Init, Fun>>;

        template <class Range>
        using result_t = typename __algo_range_init_fun::binary_invoke_result_t<Range, Init, Fun>;

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

        receiver_t(__data<Init, Fun>& _data)
          : _data_(_data) {
        }

        __data<Init, Fun>& _data_;
      };

      template <class _CvrefSenderId, class _ReceiverId, class _Init, class _Fun>
      struct __operation {
        using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __receiver_id = receiver_t<_CvrefSender, _ReceiverId, _Init, _Fun>;
        using __receiver_t = stdexec::__t<__receiver_id>;

        struct __t : __immovable {
          using __id = __operation;
          using __data_t = __data<_Init, _Fun>;

          __data<_Init, _Fun> __state_;
          _Receiver __rcvr_;
          connect_result_t<_CvrefSender, __receiver_t> __op_;

          __t(_CvrefSender&& __sndr, _Receiver __rcvr, __data_t __data) //
            noexcept(__nothrow_decay_copyable<_Receiver>                //
                       && __nothrow_decay_copyable<__data_t>            //
                         && __nothrow_connectable<_CvrefSender, __receiver_t>)
            : __state_{(__data_t&&) __data}
            , __rcvr_{(_Receiver&&) __rcvr}
            , __op_(connect((_CvrefSender&&) __sndr, __receiver_t{&__state_})) {
          }

          friend void tag_invoke(start_t, __t& __self) noexcept {
            start(__self.__op_);
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
            decltype(_Data::__init_),
            decltype(_Data::__fun_)>>;

        template <class _Data, class _Child>
        auto operator()(__ignore, _Data __data, _Child&& __child) const noexcept(
          __nothrow_constructible_from<__operation_t<_Child, _Data>, _Child, _Receiver, _Data>)
          -> __operation_t<_Child, _Data> {
          return __operation_t<_Child, _Data>{
            (_Child&&) __child, (_Receiver&&) __rcvr_, (_Data&&) __data};
        }
      };
    }

    struct reduce_t {
      // idk if needed
      // #if STDEXEC_FRIENDSHIP_IS_LEXICAL()
      //      private:
      //       template <class...>
      //       friend struct stdexec::__sexpr;
      // #endif

      template < sender Sender, __movable_value Init, __movable_value Fun = cub::Sum>
      auto operator()(Sender&& sndr, Init init, Fun fun) const {
        auto __domain = __get_sender_domain(sndr);
        return __domain.transform_sender(make_sender_expr<reduce_t>(
          reduce_::__data{(Init&&) init, (Fun&&) fun}, (Sender&&) sndr));
      }

      template <sender_expr_for<reduce_t> _Sender>
      static auto get_env(const _Sender&) noexcept {
        return empty_env{};
      }

      template <class _Sender>
      static auto get_env(const _Sender&) noexcept {
        return empty_env{};
      }

      struct op {
        friend void tag_invoke(start_t, op&) noexcept {
        }
      };

      template <sender_expr_for<reduce_t> _Sender, receiver _Receiver>
      //requires SOME CONSTRAINT HERE
      static auto connect(_Sender&& __sndr, _Receiver __rcvr) {
        return op{}; // return a dummy operation state to see if it compiles
      }

      template <class Range, class Init, class Fun>
      using _set_value_t = completion_signatures<set_value_t(
        __algo_range_init_fun::binary_invoke_result_t<Range, Init, Fun>&)>;

      template <class _CvrefSender, class _Env, class _Init, class _Fun>
      using __completion_signaturesss = //
        __try_make_completion_signatures<
          _CvrefSender,
          _Env,
          completion_signatures<set_stopped_t()>,
          __mbind_back_q<_set_value_t, _Init, _Fun>>;

      template <sender_expr_for<reduce_t> _Sender, class _Env>
      static auto get_completion_signatures(_Sender&& __sndr, _Env&& env) {
        // what's the relationship(if it exists) between the lambdas types and the lambda types in `stream_domain::transform_sender`
        // apply_sender?
        return stdexec::apply_sender(
          (_Sender&&) __sndr, [&]<class _Data, class _Child>(reduce_t, _Data, _Child&&) {
            using _Init = decltype(_Data::__init_);
            using _Fun = decltype(_Data::__fun_);
            if constexpr (__mvalid<__completion_signaturesss, _Child, _Env, _Init, _Fun>) {
              return __completion_signaturesss< _Child, _Env, _Init, _Fun>();
            } else if constexpr (__decays_to<_Env, no_env>) {
              // not sure i need this
              return dependent_completion_signatures<no_env>();
            } else {
              // BUGBUG improve this error message
              return __mexception<_WHAT_<"unknown error in nvexec::reduce"__csz>>();
            }
            STDEXEC_UNREACHABLE();
          });
      }

      using _Sender = __1;
      using _Init = __nth_member<0>(__0);
      using _Fun = __nth_member<1>(__0);
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          reduce_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Init,
          _Fun),
        tag_invoke_t(reduce_t, _Sender, _Init, _Fun)>;

      template <sender_expr_for<reduce_t> _Sender, receiver _Receiver>
      static auto connect(_Sender&& __sndr, _Receiver __rcvr) noexcept(
        __nothrow_callable< apply_sender_t, _Sender, reduce_::__connect_fn<_Receiver>>)
        -> __call_result_t< apply_sender_t, _Sender, reduce_::__connect_fn<_Receiver>> {
        return apply_sender((_Sender&&) __sndr, reduce_::__connect_fn<_Receiver>{__rcvr});
      }

      template <class Init, class Fun = cub::Sum>
      __binder_back<reduce_t, Init, Fun> operator()(Init init, Fun fun = {}) const {
        return {
          {},
          {},
          {(Init&&) init, (Fun&&) fun}
        };
      }
    };

    namespace reduce_ {
      // moved this below so i can use reduce_t as a Tag type to algorithm_base sender
      template <class SenderId, class Init, class Fun>
      struct sender_t
        : public __algo_range_init_fun::
            sender_t<reduce_t, SenderId, Init, Fun, sender_t<SenderId, Init, Fun>> {

        template <class Range>
        using _set_value_t = completion_signatures<set_value_t(
          __algo_range_init_fun::binary_invoke_result_t<Range, Init, Fun>&)>;

        template <class Receiver>
        using receiver_t =
          stdexec::__t<reduce_::receiver_t< SenderId, stdexec::__id<Receiver>, Init, Fun>>;
      };
    }
  }

  inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};
}

namespace stdexec::__detail {
  template <class SenderId, class Init, class Fun>
  extern __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::reduce_::sender_t<__name_of<__t<SenderId>>, Init, Fun>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::reduce_::sender_t<SenderId, Init, Fun>>;
}
