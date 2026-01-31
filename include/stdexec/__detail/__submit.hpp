/*
 * Copyright (c) 2021-2025 NVIDIA Corporation
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

#include "__manual_lifetime.hpp"
#include "__operation_states.hpp"
#include "__optional.hpp"
#include "__senders.hpp"
#include "__type_traits.hpp"

namespace STDEXEC {
  namespace __submit {
    template <class _Sender, class _Receiver>
    concept __has_memfn = requires(_Sender && (*__sndr)(), _Receiver && (*__rcvr)()) {
      __sndr().submit(__rcvr());
    };

    template <class _Sender, class _Receiver>
    concept __has_static_memfn = requires(_Sender && (*__sndr)(), _Receiver && (*__rcvr)()) {
      __decay_t<_Sender>::submit(__sndr(), __rcvr());
    };

    // submit is a combination of connect and start. it is customizable for times when it
    // can be done more efficiently than by calling connect and start directly.
    struct __submit_t {
      struct __void { };

      // This implementation is used if the sender has a non-static submit member function.
      template <class _Sender, class _Receiver, class _Default = __void>
        requires sender_to<_Sender, _Receiver> && __submit::__has_memfn<_Sender, _Receiver>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto
        operator()(_Sender&& __sndr, _Receiver __rcvr, [[maybe_unused]] _Default __def = {}) const
        noexcept(noexcept(static_cast<_Sender&&>(__sndr)
                            .submit(static_cast<_Receiver&&>(__rcvr)))) {
        using __result_t = decltype(static_cast<_Sender&&>(__sndr)
                                      .submit(static_cast<_Receiver&&>(__rcvr)));
        if constexpr (__same_as<__result_t, void> && !__same_as<_Default, __void>) {
          static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr));
          return __def;
        } else {
          return static_cast<_Sender&&>(__sndr).submit(static_cast<_Receiver&&>(__rcvr));
        }
      }

      // This implementation is used if the sender has a static submit member function.
      template <class _Sender, class _Receiver, class _Default = __void>
        requires sender_to<_Sender, _Receiver> && __submit::__has_static_memfn<_Sender, _Receiver>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto
        operator()(_Sender&& __sndr, _Receiver __rcvr, [[maybe_unused]] _Default __def = {}) const
        noexcept(noexcept(
          __sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)))) {
        using __result_t =
          decltype(__sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)));
        if constexpr (__same_as<__result_t, void> && !__same_as<_Default, __void>) {
          __sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
          return __def;
        } else {
          return __sndr.submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
        }
      }
    };

    inline constexpr __submit_t __submit{};
  } // namespace __submit

  template <class _Sender, class _Receiver, class _Default = __submit::__submit_t::__void>
  using __submit_result_t = __call_result_t<__submit::__submit_t, _Sender, _Receiver, _Default>;

  template <class _Sender, class _Receiver>
  concept __submittable = requires(_Sender&& __sndr, _Receiver&& __rcvr) {
    __submit::__submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
  };

  template <class _Sender, class _Receiver>
  concept __nothrow_submittable =
    __submittable<_Sender, _Receiver> && requires(_Sender&& __sndr, _Receiver&& __rcvr) {
      {
        __submit::__submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr))
      } noexcept;
    };

  enum class __submit_result_kind {
    __connect,
    __submit,
    __submit_void,
    __submit_nothrow,
  };

  template <class _Sender, class _Receiver>
  constexpr auto __get_submit_result_kind() noexcept -> __submit_result_kind {
    if constexpr (__submittable<_Sender, _Receiver>) {
      using __result_t = __submit_result_t<_Sender, _Receiver>;
      constexpr std::size_t __opstate_size = sizeof(connect_result_t<_Sender, _Receiver>);

      if constexpr (std::is_void_v<__result_t>) {
        return __submit_result_kind::__submit_void;
      } else if constexpr (__nothrow_submittable<_Sender, _Receiver>) {
        return __opstate_size > sizeof(__result_t) ? __submit_result_kind::__submit_nothrow
                                                   : __submit_result_kind::__connect;
      } else {
        return __opstate_size > sizeof(__optional<__result_t>) ? __submit_result_kind::__submit
                                                               : __submit_result_kind::__connect;
      }
    }
    return __submit_result_kind::__connect;
  }

  template <
    class _Sender,
    class _Receiver,
    __submit_result_kind _Kind = __get_submit_result_kind<_Sender, _Receiver>()
  >
    // requires sender_to<_Sender, _Receiver>
  struct submit_result {
    using __result_t = connect_result_t<_Sender, _Receiver>;

    explicit submit_result(_Sender&& __sndr, _Receiver&& __rcvr)
      noexcept(__nothrow_connectable<_Sender, _Receiver>)
      : __result_(connect(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr))) {
    }

    void submit(_Sender&&, _Receiver&&) noexcept {
      STDEXEC::start(__result_);
    }

    __result_t __result_;
  };

  template <class _Sender, class _Receiver>
  struct submit_result<_Sender, _Receiver, __submit_result_kind::__submit> {
    using __result_t = __submit_result_t<_Sender, _Receiver>;

    constexpr submit_result(_Sender&&, _Receiver&&) noexcept {
    }

    constexpr void submit(_Sender&& __sndr, _Receiver&& __rcvr) {
      __result_.__emplace_from(
        __submit::__submit, static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
    }

    __optional<__result_t> __result_;
  };

  template <class _Sender, class _Receiver>
  struct submit_result<_Sender, _Receiver, __submit_result_kind::__submit_void> {
    using __result_t = __submit_result_t<_Sender, _Receiver>;

    explicit submit_result(_Sender&&, _Receiver&&) noexcept {
    }

    void submit(_Sender&& __sndr, _Receiver&& __rcvr)
      noexcept(__nothrow_submittable<_Sender, _Receiver>) {
      __submit::__submit(static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
    }
  };

  template <class _Sender, class _Receiver>
  struct submit_result<_Sender, _Receiver, __submit_result_kind::__submit_nothrow> {
    using __result_t = __submit_result_t<_Sender, _Receiver>;

    submit_result(_Sender&&, _Receiver&&) noexcept {
    }

    void submit(_Sender&& __sndr, _Receiver&& __rcvr) noexcept {
      __result_.__construct_from(
        __submit::__submit, static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
    }

    ~submit_result() {
      __result_.__destroy();
    }

    __manual_lifetime<__result_t> __result_;
  };

  template <class _Sender, class _Receiver>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    submit_result(_Sender&&, _Receiver) -> submit_result<_Sender, _Receiver>;

} // namespace STDEXEC
