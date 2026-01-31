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
    using namespace STDEXEC;

    template <class _Receiver>
    struct __receiver {
     public:
      using receiver_concept = STDEXEC::receiver_t;

      constexpr __receiver(_Receiver&& __upstream)
        : __upstream_{static_cast<_Receiver&&>(__upstream)} {
      }

      template <class... _As>
      constexpr void set_value(_As&&... __as) noexcept {
        STDEXEC::set_value(
          static_cast<_Receiver&&>(__upstream_), set_value_t(), static_cast<_As&&>(__as)...);
      }

      template <class _Error>
      constexpr void set_error(_Error __err) noexcept {
        STDEXEC::set_value(
          static_cast<_Receiver&&>(__upstream_), set_error_t(), static_cast<_Error&&>(__err));
      }

      constexpr void set_stopped() noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__upstream_), set_stopped_t());
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__upstream_);
      }

     private:
      _Receiver __upstream_;
    };

    template <class _Sender>
    struct __sender {
     public:
      using sender_concept = STDEXEC::sender_t;

      template <__decays_to<_Sender> _Sndr>
      constexpr explicit __sender(_Sndr&& __sender)
        : __sndr_{static_cast<_Sndr&&>(__sender)} {
      }

      template <__decays_to<__sender> _Self, class _Receiver>
        requires sender_to<__copy_cvref_t<_Self, _Sender>, __materialize::__receiver<_Receiver>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr) noexcept(
        __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __materialize::__receiver<_Receiver>>)
        -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __materialize::__receiver<_Receiver>> {
        return STDEXEC::connect(
          static_cast<_Self&&>(__self).__sndr_,
          __materialize::__receiver<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <class... _Args>
      using __materialize_value = completion_signatures<set_value_t(set_value_t, _Args...)>;

      template <class _Err>
      using __materialize_error = completion_signatures<set_value_t(set_error_t, _Err)>;

      template <class _Self, class... _Env>
      using __completions_t = __transform_completion_signatures_t<
        __completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...>,
        __materialize_value,
        __materialize_error,
        completion_signatures<set_value_t(set_stopped_t)>,
        __mconcat<__qq<completion_signatures>>::__f
      >;

      template <__decays_to<__sender> _Self, class... _Env>
      static consteval auto get_completion_signatures() -> __completions_t<_Self, _Env...> {
        return {};
      }

     private:
      _Sender __sndr_;
    };

    struct __materialize_t {
      template <class _Sender>
      constexpr auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
        -> __sender<__decay_t<_Sender>> {
        return __sender<__decay_t<_Sender>>{static_cast<_Sender&&>(__sndr)};
      }

      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()() const noexcept {
        return __closure(*this);
      }
    };
  } // namespace __materialize

  inline constexpr __materialize::__materialize_t materialize;

  namespace __dematerialize {
    using namespace STDEXEC;

    template <class _Receiver>
    struct __receiver {
     public:
      using receiver_concept = STDEXEC::receiver_t;

      constexpr __receiver(_Receiver&& __upstream)
        : __upstream_{static_cast<_Receiver&&>(__upstream)} {
      }

      template <__completion_tag _Tag, class... _Args>
        requires __callable<_Tag, _Receiver, _Args...>
      void set_value(_Tag, _Args&&... __args) noexcept {
        _Tag()(static_cast<_Receiver&&>(__upstream_), static_cast<_Args&&>(__args)...);
      }

      template <class Error>
      constexpr void set_error(Error&& err) noexcept {
        STDEXEC::set_error(static_cast<_Receiver&&>(__upstream_), static_cast<Error&&>(err));
      }

      constexpr void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__upstream_));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__upstream_);
      }

     private:
      _Receiver __upstream_;
    };

    template <class _Sender>
    struct __sender {
     public:
      using sender_concept = STDEXEC::sender_t;

      template <__decays_to<_Sender> _Sndr>
      constexpr explicit __sender(_Sndr&& __sndr) noexcept(__nothrow_decay_copyable<_Sndr>)
        : __sndr_{static_cast<_Sndr&&>(__sndr)} {
      }

      template <__decays_to<__sender> _Self, class _Receiver>
        requires sender_to<__copy_cvref_t<_Self, _Sender>, __dematerialize::__receiver<_Receiver>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr) noexcept(
        __nothrow_connectable<
          __copy_cvref_t<_Self, _Sender>,
          __dematerialize::__receiver<_Receiver>
        >)
        -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __dematerialize::__receiver<_Receiver>> {
        return STDEXEC::connect(
          static_cast<_Self&&>(__self).__sndr_,
          __dematerialize::__receiver<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <class _Tag, class... _Args>
        requires __completion_tag<__decay_t<_Tag>>
      using __dematerialize_value = completion_signatures<__decay_t<_Tag>(_Args...)>;

      template <class _Self, class... _Env>
      using __completions_t = transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env...>,
        completion_signatures<>,
        __mtry_q<__dematerialize_value>::template __f
      >;

      template <__decays_to<__sender> _Self, class... _Env>
      static consteval auto get_completion_signatures() -> __completions_t<_Self, _Env...> {
        return {};
      }

     private:
      _Sender __sndr_;
    };

    struct __dematerialize_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
        -> __sender<__decay_t<_Sender>> {
        return __sender<__decay_t<_Sender>>(static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const noexcept {
        return __closure(*this);
      }
    };
  } // namespace __dematerialize

  inline constexpr __dematerialize::__dematerialize_t dematerialize;
} // namespace exec
