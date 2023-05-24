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

#include "../stdexec/execution.hpp"

#ifdef __EDG__
#  pragma diagnostic push
#  pragma diag_suppress 1302
#endif

namespace exec {
  template <class... _TagValue>
  using with_t = stdexec::__with<_TagValue...>;

  namespace __detail {
    struct __with_t {
      template <class _Tag, class _Value>
      with_t<_Tag, _Value> operator()(_Tag, _Value&& __val) const {
        return stdexec::__with_(_Tag(), (_Value&&) __val);
      }

      template <class _Tag>
      with_t<_Tag> operator()(_Tag) const {
        return stdexec::__with_(_Tag());
      }
    };
  } // namespace __detail

  inline constexpr __detail::__with_t with{};

  inline constexpr stdexec::__env::__make_env_t make_env{};

  template <class... _Ts>
  using make_env_t = stdexec::__make_env_t<_Ts...>;

  namespace __read_with_default {
    using namespace stdexec;

    struct read_with_default_t;

    template <class _Tag, class _DefaultId, class _ReceiverId>
    struct __operation : __immovable {
      using _Default = __t<_DefaultId>;
      using _Receiver = __t<_ReceiverId>;

      STDEXEC_NO_UNIQUE_ADDRESS _Default __default_;
      _Receiver __rcvr_;

      STDEXEC_DEFINE_CUSTOM(void start)(this __operation& __self, start_t) noexcept {
        try {
          if constexpr (__callable<_Tag, env_of_t<_Receiver>>) {
            const auto& __env = get_env(__self.__rcvr_);
            stdexec::set_value(std::move(__self.__rcvr_), _Tag{}(__env));
          } else {
            stdexec::set_value(std::move(__self.__rcvr_), std::move(__self.__default_));
          }
        } catch (...) {
          stdexec::set_error(std::move(__self.__rcvr_), std::current_exception());
        }
      }
    };

    template <class _Tag, class _DefaultId>
    struct __sender {
      using _Default = __t<_DefaultId>;
      using is_sender = void;
      STDEXEC_NO_UNIQUE_ADDRESS _Default __default_;

      template <class _Env>
      using __value_t =
        __minvoke<__with_default<__mbind_back_q<__call_result_t, _Env>, _Default>, _Tag>;
      template <class _Env>
      using __default_t = __if_c<__callable<_Tag, _Env>, __ignore, _Default>;
      template <class _Env>
      using __completions_t =
        completion_signatures< set_value_t(__value_t<_Env>), set_error_t(std::exception_ptr)>;

      template <__decays_to<__sender> _Self, class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __operation<_Tag, __x<__default_t<env_of_t<_Receiver>>>, __x<_Receiver>> {
        return {{}, ((_Self&&) __self).__default_, (_Receiver&&) __rcvr};
      }

      STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(
        this __sender,
        get_completion_signatures_t,
        no_env) -> dependent_completion_signatures<no_env>;

      template <__none_of<no_env> _Env>
      STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(
        this __sender,
        get_completion_signatures_t,
        _Env&&) -> __completions_t<_Env>;
    };

    struct __read_with_default_t {
      template <class _Tag, class _Default>
      constexpr auto operator()(_Tag, _Default&& __default) const
        -> __sender<_Tag, __x<__decay_t<_Default>>> {
        return {(_Default&&) __default};
      }
    };
  } // namespace __read_with_default

  inline constexpr __read_with_default::__read_with_default_t read_with_default{};

  namespace __write {
    using namespace stdexec;

    struct __write_t;

    template <class _ReceiverId, class _Env>
    struct __operation_base {
      using _Receiver = __t<_ReceiverId>;
      _Receiver __rcvr_;
      const _Env __env_;
    };

    template <class _ReceiverId, class _Env>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : receiver_adaptor<__t> {
        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_->__rcvr_;
        }

        const _Receiver& base() const & noexcept {
          return __op_->__rcvr_;
        }

        auto get_env() const noexcept -> __env::__env_join_t<const _Env&, env_of_t<_Receiver>> {
          return __env::__join_env(__op_->__env_, stdexec::get_env(base()));
        }

        __operation_base<_ReceiverId, _Env>* __op_;
      };
    };

    template <class _SenderId, class _ReceiverId, class _Env>
    struct __operation : __operation_base<_ReceiverId, _Env> {
      using _Sender = __t<_SenderId>;
      using __base_t = __operation_base<_ReceiverId, _Env>;
      using __receiver_t = __t<__receiver<_ReceiverId, _Env>>;
      connect_result_t<_Sender, __receiver_t> __state_;

      __operation(_Sender&& __sndr, auto&& __rcvr, auto&& __env)
        : __base_t{(decltype(__rcvr)) __rcvr, (decltype(__env)) __env}
        , __state_{connect((_Sender&&) __sndr, __receiver_t{{}, this})} {
      }

      STDEXEC_DEFINE_CUSTOM(void start)(this __operation& __self, start_t) noexcept {
        stdexec::start(__self.__state_);
      }
    };

    template <class _SenderId, class _Env>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;
      using is_sender = void;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<_Receiver>, _Env>>;
      template <class _Self, class _Receiver>
      using __operation_t =
        __operation<__id<__copy_cvref_t<_Self, _Sender>>, __id<_Receiver>, _Env>;

      struct __t {
        using __id = __sender;
        _Sender __sndr_;
        _Env __env_;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return {((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr, ((_Self&&) __self).__env_};
        }

        STDEXEC_DEFINE_CUSTOM(auto get_env)(this const __sender& __self, stdexec::get_env_t) //
          noexcept -> stdexec::__call_result_t<stdexec::get_env_t, const _Sender&> {
          return stdexec::get_env(__self.__sndr_);
        }

        template <__decays_to<__t> _Self, class _BaseEnv>
        STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(
          this _Self&&,
          get_completion_signatures_t,
          _BaseEnv&&)
          -> stdexec::__completion_signatures_of_t<
            __copy_cvref_t<_Self, _Sender>,
            __env::__env_join_t<_Env, _BaseEnv>>;
      };
    };

    struct __write_t {
      template <class _Sender, class... _Funs>
      using __sender_t =
        __t<__sender<__id<__decay_t<_Sender>>, __env::__env_join_t<__env::__env_fn<_Funs>...>>>;

      template <__is_not_instance_of<__env::__env_fn> _Sender, class... _Funs>
        requires sender<_Sender>
      auto operator()(_Sender&& __sndr, __env::__env_fn<_Funs>... __withs) const
        -> __sender_t<_Sender, _Funs...> {
        return {(_Sender&&) __sndr, __env::__join_env(std::move(__withs)...)};
      }

      template <class... _Funs>
      auto operator()(__env::__env_fn<_Funs>... __withs) const
        -> __binder_back<__write_t, __env::__env_fn<_Funs>...> {
        return {{}, {}, {std::move(__withs)...}};
      }
    };
  } // namespace __write

  inline constexpr __write::__write_t write{};
} // namespace exec

#ifdef __EDG__
#  pragma diagnostic pop
#endif
