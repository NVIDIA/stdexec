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

#include <stdexec/execution.hpp>

namespace exec {
  namespace __materialize {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;

        __t(_Receiver&& __upstream)
          : __upstream_{(_Receiver&&) __upstream} {
        }

       private:
        _Receiver __upstream_;

        template <__completion_tag _Tag, __decays_to<__t> _Self, class... _Args>
          requires tag_invocable<set_value_t, _Receiver&&, _Tag, _Args...>
        friend void tag_invoke(_Tag tag, _Self&& __self, _Args&&... __args) noexcept {
          set_value((_Receiver&&) __self.__upstream_, _Tag{}, (_Args&&) __args...);
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__upstream_);
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = __decay_t<stdexec::__t<_SenderId>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>>>;

      class __t {
       public:
        using sender_concept = stdexec::sender_t;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sender)
          : __sender_{(_Sndr&&) __sender} {
        }

        //  private:
        _Sender __sender_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver) noexcept(
            __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>) {
          return stdexec::connect(
            ((_Self&&) __self).__sender_, __receiver_t<_Receiver>{(_Receiver&&) __receiver});
        }

        template <class... _Args>
        using __materialize_value = completion_signatures<set_value_t(set_value_t, _Args...)>;

        template <class _Err>
        using __materialize_error = completion_signatures<set_value_t(set_error_t, _Err)>;

        template <class _Env>
        using __completion_signatures_for_t = make_completion_signatures<
          _Sender,
          _Env,
          completion_signatures<>,
          __materialize_value,
          __materialize_error,
          completion_signatures<set_value_t(set_stopped_t)>>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env)
          -> __completion_signatures_for_t<_Env> {
          return {};
        }
      };
    };

    struct __materialize_t {
      template <class _Sender>
      __t<__sender<__id<__decay_t<_Sender>>>> operator()(_Sender&& __sender) const
        noexcept(__nothrow_decay_copyable<_Sender>) {
        return {(_Sender&&) __sender};
      }

      __binder_back<__materialize_t> operator()() const noexcept {
        return {{}, {}, {}};
      }
    };
  }

  inline constexpr __materialize::__materialize_t materialize;

  namespace __dematerialize {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = __decay_t<stdexec::__t<_ReceiverId>>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;

        __t(_Receiver&& __upstream)
          : __upstream_{(_Receiver&&) __upstream} {
        }

       private:
        _Receiver __upstream_;

        template <
          same_as<set_value_t> _Tag,
          __completion_tag _Tag2,
          __decays_to<__t> _Self,
          class... _Args>
          requires tag_invocable<_Tag2, _Receiver&&, _Args...>
        friend void tag_invoke(_Tag, _Self&& __self, _Tag2 tag2, _Args&&... __args) noexcept {
          tag2((_Receiver&&) __self.__upstream_, (_Args&&) __args...);
        }

        template <__one_of<set_stopped_t, set_error_t> _Tag, __decays_to<__t> _Self, class... _Args>
          requires tag_invocable<_Tag, _Receiver&&, _Args...>
        friend void tag_invoke(_Tag tag, _Self&& __self, _Args&&... __args) noexcept {
          tag((_Receiver&&) __self.__upstream_, (_Args&&) __args...);
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__upstream_);
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = __decay_t<stdexec::__t<_SenderId>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>>>;

      class __t {
       public:
        using sender_concept = stdexec::sender_t;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sndr) noexcept(__nothrow_decay_copyable<_Sndr>)
          : __sender_{(_Sndr&&) __sndr} {
        }

       private:
        _Sender __sender_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __receiver) noexcept(
            __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>) {
          return stdexec::connect(
            ((_Self&&) __self).__sender_, __receiver_t<_Receiver>{(_Receiver&&) __receiver});
        }

        template <class _Tag, class... _Args>
          requires __completion_tag<__decay_t<_Tag>>
        using __dematerialize_value = completion_signatures<__decay_t<_Tag>(_Args...)>;

        template <class _Env>
        using __completion_signatures_for_t =
          make_completion_signatures<_Sender, _Env, completion_signatures<>, __dematerialize_value>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env)
          -> __completion_signatures_for_t<_Env> {
          return {};
        }
      };
    };

    struct __dematerialize_t {
      template <class _Sender>
      using __sender_t = __t<__sender<__id<_Sender>>>;

      template <sender _Sender>
      __sender_t<_Sender> operator()(_Sender&& __sndr) const
        noexcept(__nothrow_decay_copyable<_Sender>) {
        return __sender_t<_Sender>((_Sender&&) __sndr);
      }

      __binder_back<__dematerialize_t> operator()() const noexcept {
        return {{}, {}, {}};
      }
    };
  }

  inline constexpr __dematerialize::__dematerialize_t dematerialize;
}
