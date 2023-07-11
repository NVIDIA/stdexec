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
        using is_receiver = void;

        __t(_Receiver&& __upstream)
          : __upstream_{(_Receiver&&) __upstream} {
        }

       private:
        STDEXEC_CPO_ACCESS(set_value_t);
        STDEXEC_CPO_ACCESS(set_error_t);
        STDEXEC_CPO_ACCESS(set_stopped_t);
        STDEXEC_CPO_ACCESS(get_env_t);

        template <same_as<set_value_t> _Tag, same_as<__t> _Self, class... _Args>
          requires __callable<set_value_t, _Receiver, _Tag, _Args...>
        STDEXEC_DEFINE_CUSTOM(void set_value)(
          this _Self&& __self,
          _Tag,
          _Args&&... __args) noexcept {
          stdexec::set_value((_Receiver&&) __self.__upstream_, _Tag{}, (_Args&&) __args...);
        }

        template <same_as<set_error_t> _Tag, same_as<__t> _Self, class _Error>
          requires __callable<set_value_t, _Receiver, _Tag, _Error>
        STDEXEC_DEFINE_CUSTOM(void set_error)(this _Self&& __self, _Tag, _Error&& __err) noexcept {
          stdexec::set_value((_Receiver&&) __self.__upstream_, _Tag{}, (_Error&&) __err);
        }

        template <same_as<set_stopped_t> _Tag, same_as<__t> _Self>
          requires __callable<set_value_t, _Receiver, _Tag>
        STDEXEC_DEFINE_CUSTOM(void set_stopped)(this _Self&& __self, _Tag) noexcept {
          stdexec::set_value((_Receiver&&) __self.__upstream_, _Tag{});
        }

        template <std::same_as<__t> _Self>
        STDEXEC_DEFINE_CUSTOM(env_of_t<_Receiver> get_env)(
          this const _Self& __self,
          get_env_t) noexcept {
          return stdexec::get_env(__self.__upstream_);
        }

        _Receiver __upstream_;
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = __decay_t<stdexec::__t<_SenderId>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>>>;

      class __t {
       public:
        using is_sender = void;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sender)
          : __sender_{(_Sndr&&) __sender} {
        }

        //  private:
        _Sender __sender_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        STDEXEC_DEFINE_CUSTOM(auto connect)(this _Self&& __self, connect_t, _Receiver&& __receiver) noexcept(
          __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>)
          -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>> {
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
        STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(
          this _Self&&,
          get_completion_signatures_t,
          _Env&&) -> __completion_signatures_for_t<_Env>;
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
        using is_receiver = void;

        __t(_Receiver&& __upstream)
          : __upstream_{(_Receiver&&) __upstream} {
        }

       private:
        STDEXEC_CPO_ACCESS(set_value_t);
        STDEXEC_CPO_ACCESS(set_error_t);
        STDEXEC_CPO_ACCESS(set_stopped_t);
        STDEXEC_CPO_ACCESS(get_env_t);

        template <same_as<set_value_t> _Tag, __completion_tag _Tag2, class... _Args>
          requires tag_invocable<_Tag2, _Receiver, _Args...>
        STDEXEC_DEFINE_CUSTOM(void set_value)(
          this __t&& __self,
          _Tag,
          _Tag2,
          _Args&&... __args) noexcept {
          _Tag2()((_Receiver&&) __self.__upstream_, (_Args&&) __args...);
        }

        template <same_as<set_error_t> _Tag, class _Error>
          requires tag_invocable<_Tag, _Receiver, _Error>
        STDEXEC_DEFINE_CUSTOM(void set_error)(this __t&& __self, _Tag, _Error&& __err) noexcept {
          _Tag()((_Receiver&&) __self.__upstream_, (_Error&&) __err);
        }

        template <same_as<set_stopped_t> _Tag>
          requires tag_invocable<_Tag, _Receiver>
        STDEXEC_DEFINE_CUSTOM(void set_stopped)(this __t&& __self, _Tag) noexcept {
          _Tag()((_Receiver&&) __self.__upstream_);
        }

        template <std::same_as<__t> _Self>
        STDEXEC_DEFINE_CUSTOM(env_of_t<_Receiver> get_env)(
          this const _Self& __self,
          get_env_t) noexcept {
          return stdexec::get_env(__self.__upstream_);
        }

        _Receiver __upstream_;
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = __decay_t<stdexec::__t<_SenderId>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>>>;

      class __t {
       public:
        using is_sender = void;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sndr) noexcept(__nothrow_decay_copyable<_Sndr>)
          : __sender_{(_Sndr&&) __sndr} {
        }

       private:
        _Sender __sender_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>

        STDEXEC_DEFINE_CUSTOM(auto connect)(this _Self&& __self, connect_t, _Receiver&& __receiver) noexcept(
          __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>)
          -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>> {
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
        STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(
          this _Self&&,
          get_completion_signatures_t,
          _Env&&) -> __completion_signatures_for_t<_Env>;
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
