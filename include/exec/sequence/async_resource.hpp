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

#include "./let_each.hpp"
#include "./ignore_all.hpp"
#include "./zip.hpp"

#include "../finally.hpp"

namespace exec {
  namespace async_resource {
    namespace __run {
      using namespace stdexec;

      struct open_t {
        template <class _Resource>
          requires tag_invocable<open_t, _Resource&>
                && sender<tag_invoke_result_t<open_t, _Resource&>>
        auto operator()(_Resource& __resource) const
          noexcept(nothrow_tag_invocable<open_t, _Resource&>)
            -> tag_invoke_result_t<open_t, _Resource&> {
          return tag_invoke(*this, __resource);
        }
      };

      struct close_t {
        template <class _Token>
          requires tag_invocable<close_t, _Token> && sender<tag_invoke_result_t<close_t, _Token>>
        auto operator()(_Token&& __resource) const noexcept(nothrow_tag_invocable<close_t, _Token>)
          -> tag_invoke_result_t<close_t, _Token> {
          return tag_invoke(*this, static_cast<_Token&&>(__resource));
        }
      };

      template <class _ReceiverId>
      struct __close_receiver {
        using _Receiver = stdexec::__t<_ReceiverId>;

        struct __t {
          using __id = __close_receiver;
          _Receiver __rcvr_;

          friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
            return stdexec::get_env(__self.__rcvr_);
          }

          friend void tag_invoke(set_value_t, __t&& __self) noexcept {
            stdexec::set_value(static_cast<_Receiver&&>(__self.__rcvr_));
          }

          template <class _Error>
          friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
            stdexec::set_error(
              static_cast<_Receiver&&>(__self.__rcvr_), static_cast<_Error&&>(__error));
          }

          friend void tag_invoke(set_stopped_t, __t&& __self) noexcept {
            stdexec::set_stopped(static_cast<_Receiver&&>(__self.__rcvr_));
          }
        };
      };

      template <class _OpenId, class _ReceiverId>
      struct __operation {
        struct __t;
      };

      template <class _OpenId, class _ReceiverId>
      struct __open_receiver {
        using _Receiver = stdexec::__t<_ReceiverId>;

        struct __t {
          using __id = __open_receiver;

          stdexec::__t<__operation<_OpenId, _ReceiverId>>* __op_;

          friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
            return stdexec::get_env(__self.__op_->__receiver);
          }

          template <class _Token>
          friend void tag_invoke(set_value_t, __t&& __self, _Token&& __token) noexcept {
            __self.__op_->__notify_open(static_cast<_Token&&>(__token));
          }

          template <__one_of<set_stopped_t, set_error_t> _Tag, class... _Error>
          friend void tag_invoke(_Tag __tag, __t&& __self, _Error&&... __error) noexcept {
            __tag(
              static_cast<_Receiver&&>(__self.__op_->__receiver),
              static_cast<_Error&&>(__error)...);
          }
        };
      };

      template <class _SequenceReceiver, class _Token>
      auto __run_sender(_SequenceReceiver& __receiver, _Token& __token) {
        return exec::finally(
          exec::set_next(__receiver, stdexec::just(__token)), close_t{}(__token));
      }

      template <class _OpenId, class _ReceiverId>
      struct __operation<_OpenId, _ReceiverId>::__t : __immovable {
        using _Open = stdexec::__t<_OpenId>;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using _Env = env_of_t<_Receiver>;
        using _Token = __single_sender_value_t<_Open, _Env>;
        using _RunAndFinallyClose =
          decltype(__run_sender(__declval<_Receiver&>(), __declval<_Token&>()));
        using __id = __operation;
        using __open_receiver_t = stdexec::__t<__open_receiver<_OpenId, _ReceiverId>>;
        using __close_receiver_t = stdexec::__t<__close_receiver<_ReceiverId>>;

        [[no_unique_address]] _Receiver __receiver;
        std::optional<_Token> __token_;
        std::variant<
          connect_result_t<_Open, __open_receiver_t>,
          connect_result_t<_RunAndFinallyClose, __close_receiver_t>>
          __op_states_;

        void __notify_open(_Token&& __token) noexcept {
          if (stdexec::get_stop_token(stdexec::get_env(__receiver)).stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(__receiver));
            return;
          }
          try {
            __token_.emplace(static_cast<_Token&&>(__token));
            connect_result_t<_RunAndFinallyClose, __close_receiver_t>& __op =
              __op_states_.template emplace<1>(__conv{[&] {
                return stdexec::connect(
                  __run_sender(__receiver, *__token_),
                  __close_receiver_t{static_cast<_Receiver&&>(__receiver)});
              }});
            stdexec::start(__op);
          } catch (...) {
            stdexec::set_error(static_cast<_Receiver&&>(__receiver), std::current_exception());
          }
        }

        __t(_Open&& __open, _Receiver __receiver)
          : __receiver(static_cast<_Receiver&&>(__receiver))
          , __op_states_(std::in_place_index<0>, __conv{[&] {
                           return stdexec::connect(
                             static_cast<_Open&&>(__open), __open_receiver_t{this});
                         }}) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          connect_result_t<_Open, __open_receiver_t>* __op = std::get_if<0>(&__self.__op_states_);
          STDEXEC_ASSERT(__op);
          stdexec::start(*__op);
        }
      };

      template <class _Open, class _Receiver>
      using __operation_t = __t<__operation<__id<_Open>, __id<__decay_t<_Receiver>>>>;


      template <class _Resource, class _Env>
      concept __with_open = requires(_Resource& __resource) {
        { open_t{}(__resource) } -> stdexec::__single_typed_sender<_Env>;
      };

      template <class _Resource, class _Env>
        requires __with_open<_Resource, _Env>
      using __token_of_t = __single_sender_value_t<__call_result_t<open_t, _Resource&>, _Env>;

      template <class _Sndr>
      concept __sender_of_void = sender_of<_Sndr, set_value_t()> && __single_typed_sender<_Sndr>;

      template <class _Resource, class _Env>
      concept __with_open_and_close =
        __with_open<_Resource, _Env>
        && requires(_Resource& __resource, __token_of_t<_Resource, _Env>& __token) {
             { close_t{}(__token) } -> __sender_of_void;
           };

      template <class _Resource>
      struct __sequence_sender {
        struct __t {
          using __id = __sequence_sender;
          using __open_sender_t = __call_result_t<open_t, _Resource&>;

          _Resource* __resource;

          template <__decays_to<__t> _Self, receiver _Receiver>
            requires __with_open_and_close<_Resource, env_of_t<_Receiver>>
                  && sequence_receiver_of<
                       _Receiver,
                       completion_signatures<set_value_t(
                         __token_of_t<_Resource, env_of_t<_Receiver>>)>>
          friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __receiver)
            -> __operation_t<__open_sender_t&&, _Receiver> {
            return {open_t{}(*__self.__resource), static_cast<_Receiver&&>(__receiver)};
          }

          template <class _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
            -> make_completion_signatures<
              __copy_cvref_t<_Self, __open_sender_t>,
              _Env,
              completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>>;
        };
      };

      template <class _Resource>
      using __sequence_sender_t = __t<__sequence_sender<__decay_t<_Resource>>>;

      struct run_t {
        template <class _Resource>
          requires tag_invocable<run_t, _Resource> && sender<tag_invoke_result_t<run_t, _Resource>>
        auto operator()(_Resource&& __resource) const
          noexcept(nothrow_tag_invocable<run_t, _Resource>)
            -> tag_invoke_result_t<run_t, _Resource> {
          return tag_invoke(*this, static_cast<_Resource&&>(__resource));
        }

        template <class _Resource>
          requires(!tag_invocable<run_t, _Resource&>) && __with_open_and_close<_Resource, empty_env>
        __sequence_sender_t<_Resource> operator()(_Resource& __resource) const {
          return __sequence_sender_t<_Resource>{&__resource};
        }
      };

      template <class _Resource>
      struct resource_facade {
        struct __t {
          [[no_unique_address]] _Resource* __resource;

          template <__decays_to<__t> _Self>
          friend __sequence_sender_t<_Resource> tag_invoke(run_t, _Self&& __self) noexcept {
            return {__self.__resource};
          }
        };
      };

      template <class _Resource>
      using resource_facade_t = __t<resource_facade<__decay_t<_Resource>>>;
    }

    using __run::run_t;
    using __run::open_t;
    using __run::close_t;

    inline constexpr run_t run{};
    inline constexpr open_t open{};
    inline constexpr close_t close{};
  }

  struct use_resources_t {
    template <class _SenderFactory, class... _Resources>
    auto operator()(_SenderFactory&& __fn, _Resources&&... __resources) const {
      return ignore_all(let_value_each(
        zip(async_resource::run(static_cast<_Resources&&>(__resources))...),
        static_cast<_SenderFactory&&>(__fn)));
    }
  };

  inline constexpr use_resources_t use_resources{};
}
