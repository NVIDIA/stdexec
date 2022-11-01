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
#pragma diagnostic push
#pragma diag_suppress 1302
#endif

namespace exec {
  template <class _Tag, class _Value = stdexec::__none_such>
    using with_t = stdexec::__with_t<_Tag, _Value>;

  namespace __detail {
    struct __with_t {
      template <class _Tag, class _Value>
        with_t<_Tag, _Value> operator()(_Tag, _Value&& __val) const {
          return {{(_Value&&) __val}};
        }

      template <class _Tag>
        with_t<_Tag> operator()(_Tag) const {
          return {{}};
        }
    };
  } // namespace __detail

  inline constexpr __detail::__with_t with{};

  inline constexpr stdexec::__env::__make_env_t make_env{};

  template <class... _Ts>
    using make_env_t =
      stdexec::__make_env_t<_Ts...>;

  namespace __read_with_default {
    using namespace stdexec;

    struct read_with_default_t;

    template <class _Tag, class _DefaultId, class _ReceiverId>
      struct __operation : __immovable {
        using _Default = __t<_DefaultId>;
        using _Receiver = __t<_ReceiverId>;

        [[no_unique_address]] _Default __default_;
        _Receiver __rcvr_;

        friend void tag_invoke(start_t, __operation& __self) noexcept try {
          if constexpr (__callable<_Tag, env_of_t<_Receiver>>) {
            const auto& __env = get_env(__self.__rcvr_);
            set_value(std::move(__self.__rcvr_), _Tag{}(__env));
          } else {
            set_value(std::move(__self.__rcvr_), std::move(__self.__default_));
          }
        } catch(...) {
          set_error(std::move(__self.__rcvr_), std::current_exception());
        }
      };

    template <class _Tag, class _DefaultId>
      struct __sender {
        using _Default = __t<_DefaultId>;
        [[no_unique_address]] _Default __default_;

        template <class _Env>
          using __value_t =
            __minvoke<__with_default<__mbind_back_q<__call_result_t, _Env>, _Default>, _Tag>;
        template <class _Env>
          using __default_t =
            __if_c<__callable<_Tag, _Env>, __ignore, _Default>;
        template <class _Env>
          using __completions_t =
            completion_signatures<
              set_value_t(__value_t<_Env>),
              set_error_t(std::exception_ptr)>;

        template <__decays_to<__sender> _Self, class _Receiver>
          requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
          noexcept(std::is_nothrow_constructible_v<decay_t<_Receiver>, _Receiver>)
          -> __operation<_Tag, __x<__default_t<env_of_t<_Receiver>>>, __x<decay_t<_Receiver>>> {
          return {{}, ((_Self&&) __self).__default_, (_Receiver&&) __rcvr};
        }

        friend auto tag_invoke(get_completion_signatures_t, __sender, no_env)
          -> dependent_completion_signatures<no_env>;
        template <__none_of<no_env> _Env>
          friend auto tag_invoke(get_completion_signatures_t, __sender, _Env)
            -> __completions_t<_Env>;
      };

    struct __read_with_default_t {
      template <class _Tag, class _Default>
      constexpr auto operator()(_Tag, _Default&& __default) const
        -> __sender<_Tag, __x<decay_t<_Default>>> {
        return {(_Default&&) __default};
      }
    };
  } // namespace __read_with_default

  inline constexpr __read_with_default::__read_with_default_t read_with_default {};

  namespace __write {
    using namespace stdexec;

    struct __write_t;

    template <class _ReceiverId, class... _Withs>
      struct __operation_base {
        using _Receiver = __t<_ReceiverId>;
        _Receiver __rcvr_;
        std::tuple<_Withs...> __withs_;
      };

    template <class _ReceiverId, class... _Withs>
      struct __receiver
        : receiver_adaptor<__receiver<_ReceiverId, _Withs...>> {
        using _Receiver = stdexec::__t<_ReceiverId>;

        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_->__rcvr_;
        }
        const _Receiver& base() const & noexcept {
          return __op_->__rcvr_;
        }

        auto get_env() const
          -> make_env_t<env_of_t<_Receiver>, _Withs...> {
          return std::apply(
            [this](auto&... __withs) {
              return make_env(stdexec::get_env(base()), __withs...);
            },
            __op_->__withs_);
        }

        __operation_base<_ReceiverId, _Withs...>* __op_;
      };

    template <class _SenderId, class _ReceiverId, class... _Withs>
      struct __operation : __operation_base<_ReceiverId, _Withs...> {
        using _Sender = __t<_SenderId>;
        using __base_t = __operation_base<_ReceiverId, _Withs...>;
        using __receiver_t = __receiver<_ReceiverId, _Withs...>;
        connect_result_t<_Sender, __receiver_t> __state_;

        __operation(_Sender&& __sndr, auto&& __rcvr, auto&& __withs)
          : __base_t{(decltype(__rcvr)) __rcvr, (decltype(__withs)) __withs}
          , __state_{connect((_Sender&&) __sndr, __receiver_t{{}, this})}
        {}

        friend void tag_invoke(start_t, __operation& __self) noexcept {
          start(__self.__state_);
        }
      };

    template <class _SenderId, class... _Withs>
      struct __sender {
        using _Sender = __t<_SenderId>;
        template <class _ReceiverId>
          using __receiver_t =
            __receiver<_ReceiverId, _Withs...>;
        template <class _Self, class _ReceiverId>
          using __operation_t =
            __operation<__x<__member_t<_Self, _Sender>>, _ReceiverId, _Withs...>;

        _Sender __sndr_;
        std::tuple<_Withs...> __withs_;

        template <__decays_to<__sender> _Self, receiver _Receiver>
          requires sender_to<__member_t<_Self, _Sender>, __receiver_t<__x<decay_t<_Receiver>>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Self, __x<decay_t<_Receiver>>> {
          return {((_Self&&) __self).__sndr_,
                  (_Receiver&&) __rcvr,
                  ((_Self&&) __self).__withs_};
        }

        template <tag_category<forwarding_sender_query> _Tag, class... _As>
          requires __callable<_Tag, const _Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
          noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>)
          -> __call_result_if_t<tag_category<_Tag, forwarding_sender_query>, _Tag, const _Sender&, _As...> {
          return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
        }

        template <__decays_to<__sender> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
            -> make_completion_signatures<
                __member_t<_Self, _Sender>,
                make_env_t<_Env, _Withs...>>;
      };

    struct __write_t {
      template <__is_not_instance_of<__env::__with_> _Sender, class... _Withs>
          requires sender<_Sender>
        auto operator()(_Sender&& __sndr, __env::__with_<_Withs>... __withs) const
          -> __sender<__x<decay_t<_Sender>>, __env::__with_<_Withs>...> {
          return {(_Sender&&) __sndr, {std::move(__withs)...}};
        }

      template <class... _Withs>
        auto operator()(__env::__with_<_Withs>... __withs) const
          -> __binder_back<__write_t, __env::__with_<_Withs>...> {
          return {{}, {}, {std::move(__withs)...}};
        }
    };
  } // namespace __write

  inline constexpr __write::__write_t write {};
} // namespace exec

#ifdef __EDG__
#pragma diagnostic pop
#endif
