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
#include "env.hpp"

#include <variant>

namespace exec {
  using namespace stdexec;

  template <typename... _TailSenderN>
  struct __variant_tail_sender : private std::variant<_TailSenderN...> {
    static_assert(sizeof...(_TailSenderN) >= 1, "variant_tail_sender requires at least one sender");
    static_assert(
      (tail_sender<_TailSenderN> && ...),
      "variant_tail_sender requires all senders to be tail_sender");

    using __senders_t = std::variant<_TailSenderN...>;

    using __senders_t::__senders_t;
    using __senders_t::operator=;
    using __senders_t::index;
    using __senders_t::emplace;
    using __senders_t::swap;

    // __variant_tail_sender() = default;
    // template<class... _OtherTailSenders>
    //   __variant_tail_sender(const __variant_tail_sender<_OtherTailSenders...>& __o)
    //     : __senders_t(static_cast<const std::variant<_OtherTailSenders...>&>(__o)) {}
    // template<class... _OtherTailSenders>
    //   __variant_tail_sender(__variant_tail_sender<_OtherTailSenders...>&& __o)
    //     : __senders_t(static_cast<std::variant<_OtherTailSenders...>&&>(__o)) {}
    // template<class... _OtherTailSenders>
    // __variant_tail_sender& operator=(const __variant_tail_sender<_OtherTailSenders...>& __o) {
    //   __senders_t::operator=(__o);
    //   return *this;
    // }
    // template<class... _OtherTailSenders>
    // __variant_tail_sender& operator=(__variant_tail_sender<_OtherTailSenders...>&& __o) {
    //   __senders_t::operator=(std::move(__o));
    //   return *this;
    // }

    // template<class... _An>
    //   requires (__is_not_instance_of<_An, __variant_tail_sender> && ...)
    // __variant_tail_sender(_An&&... __an) : __senders_t((_An&&)__an...) {}
    // template<class _A>
    //   requires __is_not_instance_of<_A, __variant_tail_sender>
    // __variant_tail_sender& operator=(_A&& __a) {
    //   __senders_t::operator=((_A&&)__a);
    //   return *this;
    // }

    template <class _TailReceiver>
    struct op {
      using __opn_t =
        __variant<std::monostate, std::optional<connect_result_t<_TailSenderN, _TailReceiver>>...>;
      using __start_result_t =
        __variant_tail_sender<next_tail_from_sender_to_t<_TailSenderN, _TailReceiver>...>;

      op(const op&) = delete;
      op(op&&) = delete;
      op& operator=(const op&) = delete;
      op& operator=(op&&) = delete;

      explicit op(__senders_t&& __t, _TailReceiver __r) {
        std::visit(
          [&, this](auto&& __t) -> void {
            using _T = std::remove_cvref_t<decltype(__t)>;
            if constexpr (tail_sender<_T>) {
              static_assert(
                tail_sender_to<_T, _TailReceiver>, "variant-tail-sender member cannot connect");
              using op_t = connect_result_t<_T, _TailReceiver>;
              using opt_t = std::optional<op_t>;
              __opn_.template emplace<opt_t>();
              opt_t& opt = std::get<opt_t>(__opn_);
              opt.~opt_t();
              new (&opt) opt_t{stdexec::__conv{[&]() -> op_t {
                return stdexec::connect((decltype(__t)&&) __t, __r);
              }}};
            } else {
              std::terminate();
            }
          },
          (__senders_t&&) __t);
      }

      operator bool() const noexcept {
        return std::visit(
          [&](auto&& __op) -> bool {
            using _Opt = std::decay_t<decltype(__op)>;
            if constexpr (__is_instance_of_<_Opt, std::optional>) {
              auto& op = *__op;
              using _Op = std::decay_t<decltype(op)>;
              if constexpr (__nullable_tail_operation_state<_Op>) {
                return !!op;
              }
              return true;
            } else {
              std::terminate();
            }
          },
          __opn_);
      }

      [[nodiscard]] friend __start_result_t tag_invoke(start_t, op& __self) noexcept {
        return std::visit(
          [&](auto&& __op) -> __start_result_t {
            using _Opt = std::decay_t<decltype(__op)>;
            if constexpr (__is_instance_of_<_Opt, std::optional>) {
              auto& op = *__op;
              using _Op = std::decay_t<decltype(op)>;
              if constexpr (__nullable_tail_operation_state<_Op>) {
                if (!op) {
                  return __start_result_t{};
                }
              }
              if constexpr (__terminal_tail_operation_state<_Op>) {
                stdexec::start(op);
                return __start_result_t{};
              } else {
                return result_from<__start_result_t>(stdexec::start(op));
              }
            } else {
              std::terminate();
            }
          },
          __self.__opn_);
      }

      friend void tag_invoke(unwind_t, op& __self) noexcept {
        return std::visit(
          [&](auto&& __op) -> void {
            using _Opt = std::decay_t<decltype(__op)>;
            if constexpr (__is_instance_of_<_Opt, std::optional>) {
              exec::unwind(*__op);
            } else {
              std::terminate();
            }
          },
          __self.__opn_);
      }

      __opn_t __opn_;
    };

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class _TailReceiver>
    [[nodiscard]] friend auto
      tag_invoke(connect_t, __variant_tail_sender&& __self, _TailReceiver&& __r) noexcept
      -> op<std::decay_t<_TailReceiver>> {
      return op<std::decay_t<_TailReceiver>>{
        (__variant_tail_sender&&) __self, (_TailReceiver&&) __r};
    }

    template <class _Env>
    friend constexpr bool tag_invoke(
      exec::always_completes_inline_t,
      exec::c_t<__variant_tail_sender>,
      exec::c_t<_Env>) noexcept {
      return true;
    }

   private:
    template <typename... _OtherTailSenderN>
    friend struct __variant_tail_sender;

    template <typename _To>
    friend constexpr _To variant_cast(__variant_tail_sender __f) noexcept {
      return std::visit(
        []<class _U>(_U&& __u) -> _To {
          if constexpr (stdexec::__v<__mapply<__contains<_U>, _To>>) {
            return _To{(_U&&) __u};
          } else {
            printf("variant_cast\n");
            fflush(stdout);
            std::terminate();
          }
        },
        std::move(static_cast<__senders_t&>(__f)));
    }

    template <tail_sender _To>
    friend constexpr std::decay_t<_To> get(__variant_tail_sender __f) noexcept {
      static_assert(
        stdexec::__v<__mapply<__contains<std::decay_t<_To>>, __variant_tail_sender>>,
        "get does not have _To as an alternative");
      if (!holds_alternative<std::decay_t<_To>>(__f)) {
        printf("get\n");
        fflush(stdout);
        std::terminate();
      }
      return std::get<std::decay_t<_To>>(std::move(static_cast<__senders_t&>(__f)));
    }

    template < class T>
    friend inline constexpr bool holds_alternative(const __variant_tail_sender& v) noexcept {
      return std::holds_alternative<T>(v);
    }
  };

  template <template <class...> class _T>
  struct __mflattener_of {

    template <class _Continuation>
    struct __push_back_flatten;

    template <class _Continuation = __q<__types>>
    struct __mflatten {
      template <class... _Ts>
      using __f = __mapply<
        _Continuation,
        __minvoke<__mfold_right<__types<>, __push_back_flatten<__q<__types>>>, _Ts...>>;
    };

    template <class _Continuation>
    struct __push_back_flatten {

      template <bool _IsInstance, class _List, class _Item>
      struct __f_;

      template <
        template <class...>
        class _List,
        class... _ListItems,
        template <class...>
        class _Instance,
        class... _InstanceItems>
      struct __f_<true, _List<_ListItems...>, _Instance<_InstanceItems...>> {
        using __t = __minvoke<__mflatten<_Continuation>, _ListItems..., _InstanceItems...>;
      };

      template <template <class...> class _List, class... _ListItems, class _Item>
      struct __f_<false, _List<_ListItems...>, _Item> {
        using __t = __minvoke<_Continuation, _ListItems..., _Item>;
      };
      template <class _List, class _Item>
      using __f = __t<__f_<__is_instance_of<_Item, _T>, _List, _Item>>;
    };
  };

  template <typename... _TailSenderN>
  using variant_tail_sender = __minvoke<
    __if_c<
      sizeof...(_TailSenderN) != 0,
      __transform<
        __q<decay_t>,
        __mflattener_of<__variant_tail_sender>::__mflatten< __munique<__q<__variant_tail_sender>>>>,
      __mconst<__not_a_variant>>,
    _TailSenderN...>;

} // namespace exec
