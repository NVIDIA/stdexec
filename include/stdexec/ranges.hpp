/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <cassert>
#include <concepts>

#ifdef __cpp_lib_ranges
#include <ranges>

#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__scope.hpp"
#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  namespace __view_adaptor {
    template <class _ReceiverId, class _Closure>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __data {
        _Receiver __rcvr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Closure __closure_;
      };

      struct __t {
        using __id = __receiver;
        __data* __op_;

        // Customize set_value by piping the view and passing the result
        // to the downstream receiver
        template <__same_as<set_value_t> _Tag, ::std::ranges::input_range _View>
        friend void tag_invoke(_Tag, __t&& __self, _View&& __view) noexcept {
          stdexec::set_value(
            (_Receiver &&) __self.__op_->__rcvr_, __view | __self.__op_->__closure_);
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, class... _As>
          requires __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          __tag((_Receiver &&) __self.__op_->__rcvr_, (_As &&) __as...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self)
          -> __call_result_t<get_env_t, const _Receiver&> {
          return get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Closure>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_id = __receiver<_ReceiverId, _Closure>;
      using __receiver_t = stdexec::__t<__receiver_id>;

      struct __t : __immovable {
        using __id = __operation;
        typename __receiver_id::__data __data_;
        connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver __rcvr, _Closure __fun) //
          noexcept(__nothrow_decay_copyable<_Receiver>          //
                     && __nothrow_decay_copyable<_Closure>      //
                       && __nothrow_connectable<_Sender, __receiver_t>)
          : __data_{(_Receiver &&) __rcvr, (_Closure &&) __fun}
          , __op_(connect((_Sender &&) __sndr, __receiver_t{&__data_})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_);
        }
      };
    };

    template <class _SenderId, class _Closure, class _Callable_error>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;
      template <class _Receiver>
      using __receiver = stdexec::__t<__receiver<stdexec::__id<_Receiver>, _Closure>>;
      template <class _Self, class _Receiver>
      using __operation = stdexec::__t<
        __operation<__copy_cvref_t<_Self, _Sender>, stdexec::__id<_Receiver>, _Closure>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Closure __closure_;

        template <class _Self, class _Env>
        using __completion_signatures = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __with_error_invoke_t<
              set_value_t,
              _Closure,
              __copy_cvref_t<_Self, _Sender>,
              _Env,
              _Callable_error>,
            __mbind_front<__mtry_catch_q<__set_value_invoke_t, _Callable_error>, _Closure>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_constructible_from<
                   __operation<_Self, _Receiver>,
                   __copy_cvref_t<_Self, _Sender>,
                   _Receiver&&,
                   __copy_cvref_t<_Self, _Closure>>) -> __operation<_Self, _Receiver> {
          return {
            ((_Self &&) __self).__sndr_, (_Receiver &&) __rcvr, ((_Self &&) __self).__closure_};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
          -> __completion_signatures<_Self, _Env>
          requires true;

        friend auto tag_invoke(get_env_t, const __t& __self) //
          noexcept(__nothrow_callable<get_env_t, const _Sender&>)
            -> __call_result_t<get_env_t, const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    template <class _Callable_error, class _Cpo, class DerivedClosure>
    struct __pipe {
      template <class... _Args>
      using __closure = ::std::invoke_result_t<_Cpo, _Args...>;

      template <class _Sender, class... _Args>
      using __sender =
        __t< __view_adaptor::
               __sender<stdexec::__id<__decay_t<_Sender>>, __closure<_Args...>, _Callable_error>>;

      // We want to allow all calls to the CPO that return a closure
      template <sender _Sender, class... _Args>
        requires invocable<_Cpo, _Args...> && (!::std::ranges::viewable_range<__closure<_Args...>>)
      __sender<_Sender, _Args...> operator()(_Sender&& __sndr, _Args&&... __args) const {
        return __sender<_Sender, _Args...>{
          (_Sender &&) __sndr, __closure<_Args...>((_Args &&) __args...)};
      }

      template <class... _Args>
      __binder_back<DerivedClosure, _Args...> operator()(_Args&&... __args) const {
        return {{}, {}, {(_Args &&) __args...}};
      }
    };
  } // namespace __view_adaptor

  namespace views {
    namespace __transform {
      inline constexpr __mstring __transform_context =
        "In stdexec::views::transform(Sender, Function)..."__csz;

      struct transform_t
        : public __view_adaptor::__pipe<
            __callable_error<__transform_context>,
            decltype(::std::views::transform),
            transform_t> { };
    } // __transform

    using __transform::transform_t;
    inline constexpr transform_t transform{};

    namespace __filter {
      inline constexpr __mstring __filter_context =
        "In stdexec::views::filter(Sender, Function)..."__csz;

      struct filter_t
        : public __view_adaptor::
            __pipe< __callable_error<__filter_context>, decltype(::std::views::filter), filter_t> {
      };
    } // __filter

    using __filter::filter_t;
    inline constexpr filter_t filter{};

  } // namespace views

} // namespace stdexec

#endif // __cpp_lib_ranges
