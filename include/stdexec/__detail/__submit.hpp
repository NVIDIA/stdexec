/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__senders.hpp"
#include "__meta.hpp"
#include "__type_traits.hpp"

#include <memory>

namespace stdexec {
  namespace {
    inline constexpr auto __ref = []<class _Ty>(_Ty& __ty) noexcept {
      return [__ty = &__ty]() noexcept -> decltype(auto) {
        return (*__ty);
      };
    };
  } // namespace

  template <class _Ty>
  using __ref_t = decltype(__ref(__declval<_Ty&>()));

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __submit
  namespace __submit_ {
    template <class _OpRef>
    struct __receiver {
      using receiver_concept = receiver_t;
      using __t = __receiver;
      using __id = __receiver;

      using _Operation = __decay_t<__call_result_t<_OpRef>>;
      using _Receiver = stdexec::__t<__mapply<__q<__msecond>, _Operation>>;

      _OpRef __opref_;

      // Forward all the receiver ops, and delete the operation state.
      template <__completion_tag _Tag, class... _As>
        requires __callable<_Tag, _Receiver, _As...>
      friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept {
        __tag(static_cast<_Receiver&&>(__self.__opref_().__rcvr_), static_cast<_As&&>(__as)...);
        __self.__delete_op();
      }

      void __delete_op() noexcept {
        _Operation* __op = &__opref_();
        if constexpr (__callable<get_allocator_t, env_of_t<_Receiver>>) {
          auto&& __env = stdexec::get_env(__op->__rcvr_);
          auto __alloc = stdexec::get_allocator(__env);
          using _Alloc = decltype(__alloc);
          using _OpAlloc =
            typename std::allocator_traits<_Alloc>::template rebind_alloc<_Operation>;
          _OpAlloc __op_alloc{__alloc};
          std::allocator_traits<_OpAlloc>::destroy(__op_alloc, __op);
          std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __op, 1);
        } else {
          delete __op;
        }
      }

      // Forward all receiever queries.
      auto get_env() const noexcept -> env_of_t<_Receiver&> {
        return stdexec::get_env(__opref_().__rcvr_);
      }
    };

    template <class _SenderId, class _ReceiverId>
    struct __operation {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = __receiver<__ref_t<__operation>>;

      STDEXEC_ATTRIBUTE((no_unique_address))
      _Receiver __rcvr_;
      connect_result_t<_Sender, __receiver_t> __op_state_;

      __operation(_Sender&& __sndr, _Receiver __rcvr)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __op_state_(connect(static_cast<_Sender&&>(__sndr), __receiver_t{__ref(*this)})) {
      }
    };

    struct __submit_t {
      template <receiver _Receiver, sender_to<_Receiver> _Sender>
      void operator()(_Sender&& __sndr, _Receiver __rcvr) const noexcept(false) {
        if constexpr (__callable<get_allocator_t, env_of_t<_Receiver>>) {
          auto&& __env = get_env(__rcvr);
          auto __alloc = get_allocator(__env);
          using _Alloc = decltype(__alloc);
          using _Op = __operation<__id<_Sender>, __id<_Receiver>>;
          using _OpAlloc = typename std::allocator_traits<_Alloc>::template rebind_alloc<_Op>;
          _OpAlloc __op_alloc{__alloc};
          auto __op = std::allocator_traits<_OpAlloc>::allocate(__op_alloc, 1);
          try {
            std::allocator_traits<_OpAlloc>::construct(
              __op_alloc, __op, static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
            stdexec::start(__op->__op_state_);
          } catch (...) {
            std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __op, 1);
            throw;
          }
        } else {
          start((new __operation<__id<_Sender>, __id<_Receiver>>{
                   static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)})
                  ->__op_state_);
        }
      }
    };
  } // namespace __submit_

  using __submit_::__submit_t;
  inline constexpr __submit_t __submit{};
} // namespace stdexec
