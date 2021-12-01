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
    struct __x_ {
      struct __t {
        using type = T;
      };
    };
  template <class T>
    using __x = typename __x_<T>::__t;

  template <class T>
    using __t = typename T::type;

  template <bool B>
    using __bool = bool_constant<B>;

  // Some utilities for manipulating lists of types at compile time
  template <class...>
    struct __types;

  template <class T>
    using __id = T;

  template <class T>
    inline constexpr bool __v = T::value;

  template <class T, class U>
    inline constexpr bool __v<is_same<T, U>> = false;

  template <class T>
    inline constexpr bool __v<is_same<T, T>> = true;

  template <template <class...> class Fn>
    struct __q {
      template <class... Args>
        using __f = Fn<Args...>;
    };

  template <template <class> class Fn>
    struct __q1 {
      template <class Arg>
        using __f = Fn<Arg>;
    };

  template <template <class, class> class Fn>
    struct __q2 {
      template <class First, class Second>
        using __f = Fn<First, Second>;
    };

  template <class Fn, class... Args>
    using __minvoke = typename Fn::template __f<Args...>;

  template <class Fn, class Arg>
    using __minvoke1 = typename Fn::template __f<Arg>;

  template <class Fn, class First, class Second>
    using __minvoke2 = typename Fn::template __f<First, Second>;

  template <template <class...> class T, class... Args>
    concept __valid = requires { typename T<Args...>; };
  template <template<class...> class T>
    struct __defer {
      template <class... Args> requires __valid<T, Args...>
        struct __f_ { using type = T<Args...>; };
      template <class A> requires __valid<T, A>
        struct __f_<A> { using type = T<A>; };
      template <class A, class B> requires __valid<T, A, B>
        struct __f_<A, B> { using type = T<A, B>; };
      template <class A, class B, class C> requires __valid<T, A, B, C>
        struct __f_<A, B, C> { using type = T<A, B, C>; };
      template <class... Args>
        using __f = __t<__f_<Args...>>;
    };

  template <class Fn, class Continuation = __q<__types>>
    struct __transform {
      template <class... Args>
        using __f = __minvoke<Continuation, __minvoke1<Fn, Args>...>;
    };

  template <class Init, class Fn>
    struct __right_fold {
      template <class, class...>
        struct __f_ {};
      template <class State, class Head, class... Tail>
          requires requires {typename __minvoke2<Fn, State, Head>;}
        struct __f_<State, Head, Tail...>
          : __f_<__minvoke2<Fn, State, Head>, Tail...>
        {};
      template <class State>
        struct __f_<State> {
          using type = State;
        };
      template <class... Args>
        using __f = __t<__f_<Init, Args...>>;
    };

  template <class Continuation = __q<__types>>
    struct __concat {
      template <class...>
        struct __f_ {};
      template <template <class...> class A, class... As,
                template <class...> class B, class... Bs,
                class... Tail>
        struct __f_<A<As...>, B<Bs...>, Tail...>
          : __f_<__types<As..., Bs...>, Tail...> {};
      template <template <class...> class A, class... As>
          requires requires {typename __minvoke<Continuation, As...>;}
        struct __f_<A<As...>> {
          using type = __minvoke<Continuation, As...>;
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
  template <class Pred, class True, class False>
    using __if = __minvoke2<__if_<__v<Pred>>, True, False>;

  template <class Fn>
    struct __curry {
      template <class... Ts>
        using __f = __minvoke<Fn, Ts...>;
    };

  template <class Fn>
    struct __uncurry : __concat<Fn> {};

  template <class Fn, class List>
    using __mapply =
      __minvoke<__uncurry<Fn>, List>;

  struct __count {
    template <class... Ts>
      using __f = integral_constant<size_t, sizeof...(Ts)>;
  };

  struct __push_back_unique {
    template <class List, class Item>
      struct __f_ {
        using type = List;
      };
    template <template <class...> class List, class... Ts, class Item>
        requires ((!__v<is_same<Ts, Item>>) &&...)
      struct __f_<List<Ts...>, Item> {
        using type = List<Ts..., Item>;
      };
    template <class List, class Item>
      using __f = __t<__f_<List, Item>>;
  };

  template <class Continuation = __q<__types>>
    struct __unique {
      template <class... Ts>
        using __f =
          __mapply<
            Continuation,
            __minvoke<__right_fold<__types<>, __push_back_unique>, Ts...>>;
    };

  template <class Second, class First>
    struct __compose {
      template <class... Args>
        using __f = __minvoke<Second, __minvoke<First, Args...>>;
    };

  template <template<class...> class Fn, class... Front>
    struct __bind_front_q {
      template <class... Args>
        using __f = Fn<Front..., Args...>;
    };

  template <class Fn, class... Front>
    using __bind_front = __bind_front_q<Fn::template __f, Front...>;

  template <template<class...> class Fn, class... Back>
    struct __bind_back_q {
      template <class... Args>
        using __f = Fn<Args..., Back...>;
    };

  template <class Fn, class... Back>
    using __bind_back = __bind_back_q<Fn::template __f, Back...>;

  template <class Old, class New, class Continuation = __q<__types>>
    struct __replace {
      template <class... Args>
        using __f =
          __minvoke<Continuation, __if<is_same<Args, Old>, New, Args>...>;
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
