/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

namespace stdexec {
  namespace __detail {
    // Accessor for the "data" field of a sender
    struct __get_data {
      template <class _Data>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(__ignore, _Data&& __data, auto&&...) const noexcept -> _Data&& {
        return static_cast<_Data&&>(__data);
      }
    };

    // A function object that is to senders what std::apply is to tuples:
    struct __sexpr_apply_t {
      template <class _Sender, class _ApplyFn>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Sender&& __sndr, _ApplyFn&& __fun) const
        noexcept(noexcept(__sndr
                            .apply(static_cast<_Sender&&>(__sndr), static_cast<_ApplyFn&&>(__fun))))
          -> decltype(__sndr
                        .apply(static_cast<_Sender&&>(__sndr), static_cast<_ApplyFn&&>(__fun))) {
        return __sndr.apply(static_cast<_Sender&&>(__sndr), static_cast<_ApplyFn&&>(__fun));
      }
    };

    // A type that describes a sender's metadata
    template <class _Tag, class _Data, class... _Child>
    struct __desc {
      using __tag = _Tag;
      using __data = _Data;
      using __children = __types<_Child...>;

      template <class _Fn>
      using __f = __minvoke<_Fn, _Tag, _Data, _Child...>;
    };

    template <class _Fn>
    struct __sexpr_uncurry_fn {
      template <class _Tag, class _Data, class... _Child>
      constexpr auto operator()(_Tag, _Data&&, _Child&&...) const noexcept
        -> __minvoke<_Fn, _Tag, _Data, _Child...>;
    };

    template <class _CvrefSender, class _Fn>
    using __sexpr_uncurry = __call_result_t<__sexpr_apply_t, _CvrefSender, __sexpr_uncurry_fn<_Fn>>;

    template <class _Sender>
    using __desc_of = __sexpr_uncurry<_Sender, __q<__desc>>;

    using __get_desc = __sexpr_uncurry_fn<__q<__desc>>;
  } // namespace __detail

  using __detail::__sexpr_apply_t;
  inline constexpr __sexpr_apply_t __sexpr_apply{};

  template <class _Sender, class _ApplyFn>
  using __sexpr_apply_result_t = __call_result_t<__sexpr_apply_t, _Sender, _ApplyFn>;

  template <class _Sender>
  using tag_of_t = typename __detail::__desc_of<_Sender>::__tag;

  template <class _Sender>
  using __data_of = typename __detail::__desc_of<_Sender>::__data;

  template <class _Sender, class _Continuation = __q<__types>>
  using __children_of = __mapply<_Continuation, typename __detail::__desc_of<_Sender>::__children>;

  template <class _Ny, class _Sender>
  using __nth_child_of = __children_of<_Sender, __mbind_front_q<__m_at, _Ny>>;

  template <std::size_t _Ny, class _Sender>
  using __nth_child_of_c = __children_of<_Sender, __mbind_front_q<__m_at, __msize_t<_Ny>>>;

  template <class _Sender>
  using __child_of = __children_of<_Sender, __q<__mfront>>;

  template <class _Sender>
  inline constexpr std::size_t __nbr_children_of = __v<__children_of<_Sender, __msize>>;

  template <class _Tp>
    requires __mvalid<tag_of_t, _Tp>
  struct __muncurry_<_Tp> {
    template <class _Fn>
    using __f = __detail::__sexpr_uncurry<_Tp, _Fn>;
  };

  template <class _Sender>
  concept sender_expr = __mvalid<tag_of_t, _Sender>;

  template <class _Sender, class _Tag>
  concept sender_expr_for = sender_expr<_Sender> && same_as<tag_of_t<_Sender>, _Tag>;
} // namespace stdexec
