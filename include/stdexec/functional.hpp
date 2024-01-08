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

#include "__detail/__meta.hpp"
#include "__detail/__config.hpp"
#include "concepts.hpp"

#include <functional>
#include <tuple>

namespace stdexec::__std_concepts {
#if STDEXEC_HAS_STD_CONCEPTS_HEADER()
  using std::invocable;
#else
  template <class _Fun, class... _As>
  concept invocable = //
    requires(_Fun&& __f, _As&&... __as) { std::invoke((_Fun&&) __f, (_As&&) __as...); };
#endif
} // stdexec::__std_concepts

namespace std {
  using namespace stdexec::__std_concepts;
}

namespace stdexec {
  template <auto _Fun>
  struct __function_constant {
    using _FunT = decltype(_Fun);

    template <class... _Args>
      requires __callable<_FunT, _Args...>
    STDEXEC_ATTRIBUTE((always_inline)) //
      auto
      operator()(_Args&&... __args) const
      noexcept(noexcept(_Fun((_Args&&) __args...))) -> decltype(_Fun((_Args&&) __args...)) {
      return _Fun((_Args&&) __args...);
    }
  };

  template <class _Ty, class _Cl, _Ty _Cl::*_MemPtr>
  struct __function_constant<_MemPtr> {
    using _FunT = _Ty _Cl::*;

    template <class _Arg>
      requires requires(_Arg&& __arg) { ((_Arg&&) __arg).*_MemPtr; }
    STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto
      operator()(_Arg&& __arg) const noexcept -> decltype((((_Arg&&) __arg).*_MemPtr)) {
      return ((_Arg&&) __arg).*_MemPtr;
    }
  };

  template <auto _Fun>
  inline constexpr __function_constant<_Fun> __function_constant_v{};

  template <class _Fun0, class _Fun1>
  struct __composed {
    STDEXEC_ATTRIBUTE((no_unique_address)) _Fun0 __t0_;
    STDEXEC_ATTRIBUTE((no_unique_address)) _Fun1 __t1_;

    template <class... _Ts>
      requires __callable<_Fun1, _Ts...> && __callable<_Fun0, __call_result_t<_Fun1, _Ts...>>
    STDEXEC_ATTRIBUTE((always_inline)) //
      __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>>
      operator()(_Ts&&... __ts) && {
      return ((_Fun0&&) __t0_)(((_Fun1&&) __t1_)((_Ts&&) __ts...));
    }

    template <class... _Ts>
      requires __callable<const _Fun1&, _Ts...>
            && __callable<const _Fun0&, __call_result_t<const _Fun1&, _Ts...>>
    STDEXEC_ATTRIBUTE((always_inline)) //
      __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>>
      operator()(_Ts&&... __ts) const & {
      return __t0_(__t1_((_Ts&&) __ts...));
    }
  };

  inline constexpr struct __compose_t {
    template <class _Fun0, class _Fun1>
    STDEXEC_ATTRIBUTE((always_inline)) //
    __composed<_Fun0, _Fun1> operator()(_Fun0 __fun0, _Fun1 __fun1) const {
      return {(_Fun0&&) __fun0, (_Fun1&&) __fun1};
    }
  } __compose{};

  namespace __invoke_ {
    template <class>
    inline constexpr bool __is_refwrap = false;
    template <class _Up>
    inline constexpr bool __is_refwrap<std::reference_wrapper<_Up>> = true;

    struct __funobj {
      template <class _Fun, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Fun&& __fun, _Args&&... __args) const
        noexcept(noexcept(((_Fun&&) __fun)((_Args&&) __args...)))
          -> decltype(((_Fun&&) __fun)((_Args&&) __args...)) {
        return ((_Fun&&) __fun)((_Args&&) __args...);
      }
    };

    struct __memfn {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const
        noexcept(noexcept((((_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...)))
          -> decltype((((_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...)) {
        return (((_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...);
      }
    };

    struct __memfn_refwrap {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Memptr __mem_ptr, _Ty __ty, _Args&&... __args) const
        noexcept(noexcept((__ty.get().*__mem_ptr)((_Args&&) __args...)))
          -> decltype((__ty.get().*__mem_ptr)((_Args&&) __args...)) {
        return (__ty.get().*__mem_ptr)((_Args&&) __args...);
      }
    };

    struct __memfn_smartptr {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const
        noexcept(noexcept(((*(_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...)))
          -> decltype(((*(_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...)) {
        return ((*(_Ty&&) __ty).*__mem_ptr)((_Args&&) __args...);
      }
    };

    struct __memobj {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Mbr _Class::*__mem_ptr, _Ty&& __ty) const noexcept
        -> decltype((((_Ty&&) __ty).*__mem_ptr)) {
        return (((_Ty&&) __ty).*__mem_ptr);
      }
    };

    struct __memobj_refwrap {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Mbr _Class::*__mem_ptr, _Ty __ty) const noexcept
        -> decltype((__ty.get().*__mem_ptr)) {
        return (__ty.get().*__mem_ptr);
      }
    };

    struct __memobj_smartptr {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Mbr _Class::*__mem_ptr, _Ty&& __ty) const noexcept
        -> decltype(((*(_Ty&&) __ty).*__mem_ptr)) {
        return ((*(_Ty&&) __ty).*__mem_ptr);
      }
    };

    __funobj __invoke_selector(__ignore, __ignore) noexcept;

    template <class _Mbr, class _Class, class _Ty>
    auto __invoke_selector(_Mbr _Class::*, const _Ty&) noexcept {
      if constexpr (STDEXEC_IS_CONST(_Mbr) || STDEXEC_IS_CONST(_Mbr const)) {
        // member function ptr case
        if constexpr (STDEXEC_IS_BASE_OF(_Class, _Ty)) {
          return __memobj{};
        } else if constexpr (__is_refwrap<_Ty>) {
          return __memobj_refwrap{};
        } else {
          return __memobj_smartptr{};
        }
      } else {
        // member object ptr case
        if constexpr (STDEXEC_IS_BASE_OF(_Class, _Ty)) {
          return __memfn{};
        } else if constexpr (__is_refwrap<_Ty>) {
          return __memfn_refwrap{};
        } else {
          return __memfn_smartptr{};
        }
      }
    }

    struct __invoke_t {
      template <class _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Fun&& __fun) const noexcept(noexcept(((_Fun&&) __fun)()))
        -> decltype(((_Fun&&) __fun)()) {
        return ((_Fun&&) __fun)();
      }

      template <class _Fun, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto operator()(_Fun&& __fun, _Ty&& __ty, _Args&&... __args) const noexcept(
        noexcept(__invoke_selector(__fun, __ty)((_Fun&&) __fun, (_Ty&&) __ty, (_Args&&) __args...)))
        -> decltype(__invoke_selector(
          __fun,
          __ty)((_Fun&&) __fun, (_Ty&&) __ty, (_Args&&) __args...)) {
        return decltype(__invoke_selector(__fun, __ty))()(
          (_Fun&&) __fun, (_Ty&&) __ty, (_Args&&) __args...);
      }
    };
  }

  inline constexpr __invoke_::__invoke_t __invoke{};

  template <class _Fun, class... _As>
  concept __invocable = //
    requires(_Fun&& __f, _As&&... __as) { __invoke((_Fun&&) __f, (_As&&) __as...); };

  template <class _Fun, class... _As>
  concept __nothrow_invocable =  //
    __invocable<_Fun, _As...> && //
    requires(_Fun&& __f, _As&&... __as) {
      { __invoke((_Fun&&) __f, (_As&&) __as...) } noexcept;
    };

  template <class _Fun, class... _As>
  using __invoke_result_t = //
    decltype(__invoke(__declval<_Fun>(), __declval<_As>()...));

  namespace __apply_ {
    using std::get;

    template <std::size_t... _Is, class _Fn, class _Tup>
    STDEXEC_ATTRIBUTE((always_inline)) //
    constexpr auto __impl(__indices<_Is...>, _Fn&& __fn, _Tup&& __tup) noexcept(
      noexcept(__invoke((_Fn&&) __fn, get<_Is>((_Tup&&) __tup)...)))
      -> decltype(__invoke((_Fn&&) __fn, get<_Is>((_Tup&&) __tup)...)) {
      return __invoke((_Fn&&) __fn, get<_Is>((_Tup&&) __tup)...);
    }

    template <class _Tup>
    using __tuple_indices = __make_indices<std::tuple_size<std::remove_cvref_t<_Tup>>::value>;

    template <class _Fn, class _Tup>
    using __result_t =
      decltype(__apply_::__impl(__tuple_indices<_Tup>(), __declval<_Fn>(), __declval<_Tup>()));
  } // namespace __apply_

  template <class _Fn, class _Tup>
  concept __applicable = __mvalid<__apply_::__result_t, _Fn, _Tup>;

  template <class _Fn, class _Tup>
  concept __nothrow_applicable = __applicable<_Fn, _Tup> //
    && noexcept(
      __apply_::__impl(__apply_::__tuple_indices<_Tup>(), __declval<_Fn>(), __declval<_Tup>()));

  template <class _Fn, class _Tup>
    requires __applicable<_Fn, _Tup>
  using __apply_result_t = __apply_::__result_t<_Fn, _Tup>;

  struct __apply_t {
    template <class _Fn, class _Tup>
      requires __applicable<_Fn, _Tup>
    STDEXEC_ATTRIBUTE((always_inline)) //
      constexpr auto
      operator()(_Fn&& __fn, _Tup&& __tup) const
      noexcept(__nothrow_applicable<_Fn, _Tup>) -> __apply_result_t<_Fn, _Tup> {
      return __apply_::__impl(__apply_::__tuple_indices<_Tup>(), (_Fn&&) __fn, (_Tup&&) __tup);
    }
  };

  inline constexpr __apply_t __apply{};

  template <class _Tag, class _Ty>
  struct __field {
    STDEXEC_ATTRIBUTE((always_inline)) //
    _Ty operator()(_Tag) const noexcept(__nothrow_decay_copyable<const _Ty&>) {
      return __t_;
    }

    _Ty __t_;
  };

  template <class _Tag>
  struct __mkfield_ {
    template <class _Ty>
    STDEXEC_ATTRIBUTE((always_inline)) //
    __field<_Tag, __decay_t<_Ty>> operator()(_Ty&& __ty) const
      noexcept(__nothrow_decay_copyable<_Ty>) {
      return {(_Ty&&) __ty};
    }
  };

  template <class _Tag>
  inline constexpr __mkfield_<_Tag> __mkfield{};

  // [func.tag_invoke], tag_invoke
  namespace __tag_invoke {
    void tag_invoke();

    // NOT TO SPEC: Don't require tag_invocable to subsume invocable.
    // std::invoke is more expensive at compile time than necessary,
    // and results in diagnostics that are more verbose than necessary.
    template <class _Tag, class... _Args>
    concept tag_invocable = //
      requires(_Tag __tag, _Args&&... __args) { tag_invoke((_Tag&&) __tag, (_Args&&) __args...); };

    template <class _Ret, class _Tag, class... _Args>
    concept __tag_invocable_r = //
      requires(_Tag __tag, _Args&&... __args) {
        { static_cast<_Ret>(tag_invoke((_Tag&&) __tag, (_Args&&) __args...)) };
      };

    // NOT TO SPEC: nothrow_tag_invocable subsumes tag_invocable
    template <class _Tag, class... _Args>
    concept nothrow_tag_invocable =
      tag_invocable<_Tag, _Args...> && //
      requires(_Tag __tag, _Args&&... __args) {
        { tag_invoke((_Tag&&) __tag, (_Args&&) __args...) } noexcept;
      };

    template <class _Tag, class... _Args>
    using tag_invoke_result_t = decltype(tag_invoke(__declval<_Tag>(), __declval<_Args>()...));

    template <class _Tag, class... _Args>
    struct tag_invoke_result { };

    template <class _Tag, class... _Args>
      requires tag_invocable<_Tag, _Args...>
    struct tag_invoke_result<_Tag, _Args...> {
      using type = tag_invoke_result_t<_Tag, _Args...>;
    };

    struct tag_invoke_t {
      template <class _Tag, class... _Args>
        requires tag_invocable<_Tag, _Args...>
      STDEXEC_ATTRIBUTE((always_inline)) constexpr auto
        operator()(_Tag __tag, _Args&&... __args) const
        noexcept(nothrow_tag_invocable<_Tag, _Args...>) -> tag_invoke_result_t<_Tag, _Args...> {
        return tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
      }
    };

  } // namespace __tag_invoke

  using __tag_invoke::tag_invoke_t;

  namespace __ti {
    inline constexpr tag_invoke_t tag_invoke{};
  }

  using namespace __ti;

  template <auto& _Tag>
  using tag_t = __decay_t<decltype(_Tag)>;

  using __tag_invoke::tag_invocable;
  using __tag_invoke::__tag_invocable_r;
  using __tag_invoke::nothrow_tag_invocable;
  using __tag_invoke::tag_invoke_result_t;
  using __tag_invoke::tag_invoke_result;
} // namespace stdexec
