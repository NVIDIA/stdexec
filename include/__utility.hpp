/*
 * Copyright (c) NVIDIA
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

#include <type_traits>

namespace std {
  struct __ {};

  struct __ignore {
    __ignore() = default;
    __ignore(auto&&) noexcept {}
  };

  // For hiding a template type parameter from ADL
  template <class T>
    struct __id {
      struct __t {
        using type = T;
      };
    };
  template <class T>
    using __id_t = typename __id<T>::__t;

  template <class T>
    using __t = typename T::type;

  // Some utilities for manipulating lists of types at compile time
  template <class...>
    struct __types;

  template <class F, class... Args>
    using __meta_invoke = typename F::template __f<Args...>;

  template <template<class...> class List, template<class...> class Fn>
    struct __transform {
      template <class... Args>
        using __f = List<Fn<Args>...>;
    };

  template <template<class...> class List, class Old, class New>
    struct __replace {
      template <class... Args>
        using __f = List<conditional_t<is_same_v<Old, Args>, New, Old>...>;
    };

  template <class State, template <class, class> class Fn>
    struct __right_fold {
      template <class, class...>
        struct __f_ {};
      template <class State2, class Head, class... Tail>
          requires requires {typename Fn<State2, Head>;}
        struct __f_<State2, Head, Tail...> : __f_<Fn<State2, Head>, Tail...>
        {};
      template <class State2>
        struct __f_<State2> {
          using type = State2;
        };
      template <class... Args>
        using __f = __t<__f_<State, Args...>>;
    };

  template <template <class...> class List>
    struct __concat {
      template <class...>
        struct __f_ {};
      template <template <class...> class A, class... As,
                template <class...> class B, class... Bs,
                class... Tail>
        struct __f_<A<As...>, B<Bs...>, Tail...>
          : __f_<__types<As..., Bs...>, Tail...> {};
      template <template <class...> class A, class... As>
          requires requires {typename List<As...>;}
        struct __f_<A<As...>> {
          using type = List<As...>;
        };
      template <class... Args>
        using __f = __t<__f_<Args...>>;
    };

  template <bool>
    struct __if_ {
      template <class True, class>
        using __f = True;
    };
  template <>
    struct __if_<false> {
      template <class, class False>
        using __f = False;
    };
  template <template <class> class Pred, class True, class False>
    struct __if {
      template <class T>
        using __f = typename __if_<Pred<T>::value>::template __f<True, False>;
    };

  template <template<class...> class List>
    struct __curry {
      template <class... Ts>
        using __f = List<Ts...>;
    };

  template <template<class...> class List>
    struct __uncurry : __concat<List> {};

  template <class F, class List>
    using __meta_apply =
      __meta_invoke<__uncurry<F::template __f>, List>;

  struct __count {
    template <class... Ts>
      using __f = integral_constant<size_t, sizeof...(Ts)>;
  };

  template <class List, class Item>
    struct __push_back_unique_ {
      using type = List;
    };
  template <template <class...> class List, class... Ts, class Item>
      requires ((!is_same_v<Ts, Item>) &&...)
    struct __push_back_unique_<List<Ts...>, Item> {
      using type = List<Ts..., Item>;
    };
  template <class List, class Item>
    using __push_back_unique = __t<__push_back_unique_<List, Item>>;

  template <template <class...> class List>
    struct __unique {
      template <class... Ts>
        using __f =
          __meta_invoke<
            __uncurry<List>,
            __meta_invoke<__right_fold<__types<>, __push_back_unique>, Ts...>>;
    };

  template <template<class...> class First, template<class...> class Second>
    struct __compose {
      template <class...Args>
        using __f = Second<First<Args...>>;
    };

  template <template<class...> class List, class... Front>
    struct __bind_front {
      template <class...Args>
        using __f = List<Front..., Args...>;
    };

  template <template<class...> class List>
    struct __empty {
      template <class>
        using __f = List<>;
    };

  template <template<class> class F>
    struct __eval2 {
      template <class T>
        using __f = typename F<T>::template __f<T>;
    };

  // For copying cvref from one type to another:
  template <class T>
  T&& __declval() noexcept;

  template <class Member, class Self>
  Member Self::*__memptr(const Self&);

  template <typename Self, typename Member>
  using __member_t = decltype(
      (__declval<Self>() .* __memptr<Member>(__declval<Self>())));

  template <class... As>
      requires (sizeof...(As) != 0)
    struct __front;
  template <class A, class... As>
    struct __front<A, As...> {
      using type = A;
    };
  template <class... As>
      requires (sizeof...(As) == 1)
    using __single_t = __t<__front<As...>>;
  template <class... As>
      requires (sizeof...(As) <= 1)
    using __single_or_void_t = __t<__front<As..., void>>;

  // For emplacing non-movable types into optionals:
  template <class Fn>
      requires is_nothrow_move_constructible_v<Fn>
    struct __conv {
      Fn __fn_;
      operator invoke_result_t<Fn> () && {
        return ((Fn&&) __fn_)();
      }
    };
  template <class Fn>
    __conv(Fn) -> __conv<Fn>;
}
