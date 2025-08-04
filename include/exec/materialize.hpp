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

#include "../stdexec/execution.hpp"

namespace exec {
  namespace __materialize {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;

        __t(_Receiver&& __upstream)
          : __upstream_{static_cast<_Receiver&&>(__upstream)} {
        }

        template <class... _As>
        void set_value(_As&&... __as) noexcept {
          stdexec::set_value(
            static_cast<_Receiver&&>(__upstream_), set_value_t(), static_cast<_As&&>(__as)...);
        }

        template <class _Error>
        void set_error(_Error __err) noexcept {
          stdexec::set_value(
            static_cast<_Receiver&&>(__upstream_), set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__upstream_), set_stopped_t());
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__upstream_);
        }

       private:
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
        using sender_concept = stdexec::sender_t;
        using __id = __sender;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sender)
          : __sndr_{static_cast<_Sndr&&>(__sender)} {
        }

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        static auto connect(_Self&& __self, _Receiver __receiver)
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>> {
          return stdexec::connect(
            static_cast<_Self&&>(__self).__sndr_,
            __receiver_t<_Receiver>{static_cast<_Receiver&&>(__receiver)});
        }

        template <class... _Args>
        using __materialize_value = completion_signatures<set_value_t(set_value_t, _Args...)>;

        template <class _Err>
        using __materialize_error = completion_signatures<set_value_t(set_error_t, _Err)>;

        template <class _Self, class... _Env>
        using __completions_t = __transform_completion_signatures<
          __completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...>,
          __materialize_value,
          __materialize_error,
          completion_signatures<set_value_t(set_stopped_t)>,
          __mconcat<__qq<completion_signatures>>::__f
        >;

        template <__decays_to<__t> _Self, class... _Env>
        static auto
          get_completion_signatures(_Self&&, _Env&&...) -> __completions_t<_Self, _Env...> {
          return {};
        }

       private:
        _Sender __sndr_;
      };
    };

    struct __materialize_t {
      template <class _Sender>
      auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
        -> __t<__sender<__id<__decay_t<_Sender>>>> {
        return {static_cast<_Sender&&>(__sndr)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept -> __binder_back<__materialize_t> {
        return {{}, {}, {}};
      }
    };
  } // namespace __materialize

  inline constexpr __materialize::__materialize_t materialize;

  namespace __dematerialize {
    using namespace stdexec;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = __decay_t<stdexec::__t<_ReceiverId>>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;

        __t(_Receiver&& __upstream)
          : __upstream_{static_cast<_Receiver&&>(__upstream)} {
        }

        template <__completion_tag _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        void set_value(_Tag, _Args&&... __args) noexcept {
          _Tag()(static_cast<_Receiver&&>(__upstream_), static_cast<_Args&&>(__args)...);
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
          stdexec::set_error(static_cast<_Receiver&&>(__upstream_), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept {
          stdexec::set_stopped(static_cast<_Receiver&&>(__upstream_));
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__upstream_);
        }

       private:
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
        using sender_concept = stdexec::sender_t;
        using __id = __sender;

        template <__decays_to<_Sender> _Sndr>
        __t(_Sndr&& __sndr) noexcept(__nothrow_decay_copyable<_Sndr>)
          : __sndr_{static_cast<_Sndr&&>(__sndr)} {
        }

        template <__decays_to<__t> _Self, class _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        static auto connect(_Self&& __self, _Receiver __receiver)
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>> {
          return stdexec::connect(
            static_cast<_Self&&>(__self).__sndr_,
            __receiver_t<_Receiver>{static_cast<_Receiver&&>(__receiver)});
        }

        template <class _Tag, class... _Args>
          requires __completion_tag<__decay_t<_Tag>>
        using __dematerialize_value = completion_signatures<__decay_t<_Tag>(_Args...)>;

        template <class _Self, class... _Env>
        using __completions_t = transform_completion_signatures<
          __completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...>,
          completion_signatures<>,
          __mtry_q<__dematerialize_value>::template __f
        >;

        template <__decays_to<__t> _Self, class... _Env>
        static auto
          get_completion_signatures(_Self&&, _Env&&...) -> __completions_t<_Self, _Env...> {
          return {};
        }

       private:
        _Sender __sndr_;
      };
    };

    struct __dematerialize_t {
      template <class _Sender>
      using __sender_t = __t<__sender<__id<_Sender>>>;

      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
        -> __sender_t<_Sender> {
        return __sender_t<_Sender>(static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept -> __binder_back<__dematerialize_t> {
        return {{}, {}, {}};
      }
    };
  } // namespace __dematerialize

  inline constexpr __dematerialize::__dematerialize_t dematerialize;
} // namespace exec
