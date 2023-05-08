/*
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

#include "__config.hpp"
#include "../concepts.hpp"
#include "__type_traits.hpp"

#include <bit>

// nvc++ cannot yet handle lambda-based tuples
#if STDEXEC_NVHPC()

#include <tuple>

namespace stdexec {
  using std::tuple;
  using std::get;
  using std::apply;
  using std::make_tuple;
}

#else // STDEXEC_NVHPC()

namespace stdexec {

  template <std::size_t _Ny, class _Impl>
  struct __tuple;

  namespace __tup {
    template <std::size_t _Ny, class _Impl, class _Ty>
      requires same_as<__tuple<_Ny, _Impl>, _Ty>
    void __is_tuple_(const __tuple<_Ny, _Impl>&, const _Ty&);

    template <class _Ty>
    concept __is_tuple = requires(_Ty&& t) { __tup::__is_tuple_(t, t); };
  }

  template <__tup::__is_tuple _Tuple>
  extern const std::size_t tuple_size_v;

  template <std::size_t _Ny, class _Impl>
  struct __mexpand<__tuple<_Ny, _Impl>>;

  namespace __tup {
    struct __base { };

    template <class _Ty>
    struct __wrapper;

    template <class _Ty>
    struct __wrapper<_Ty&> {
      using type = _Ty;
      using reference_wrapper = __wrapper;
      _Ty& __val_;

      /*implicit*/ constexpr __wrapper(_Ty& t) noexcept
        : __val_(t) {
      }

      constexpr operator _Ty&() const noexcept {
        return __val_;
      }
    };

    // Also works with std::reference_wrapper:
    template <class _Ty>
    using __unwrapped_t = typename _Ty::reference_wrapper::type;

    struct __wrap_fn {
      template <class _Ty>
        requires __invalid<__unwrapped_t, _Ty>
      constexpr auto operator()(const _Ty&&) const = delete;

      template <class _Ty>
        requires __invalid<__unwrapped_t, _Ty>
      constexpr auto operator()(_Ty& __ty) const noexcept -> __wrapper<_Ty&> {
        return __ty;
      }

      template <class _Ty>
        requires __valid<__unwrapped_t, _Ty>
      constexpr auto operator()(_Ty __ty) const noexcept -> __wrapper<__unwrapped_t<_Ty>&> {
        return static_cast<__unwrapped_t<_Ty>&>(__ty);
      }
    };

    template <class _Ty>
    struct __any_convertible_to {
      virtual constexpr operator _Ty() = 0;

      friend constexpr _Ty __move_capture(__any_convertible_to& __self) {
        return static_cast<_Ty>(__self);
      }
    };
    template <class _Ty>
    struct __any_convertible_to<__any_convertible_to<_Ty>>;

    template <class _Ty, class _Uy>
      requires convertible_to<_Uy, _Ty>
    struct __convertible_from : __any_convertible_to<_Ty> {
      _Uy&& __src_;

      constexpr __convertible_from(_Uy&& __src) noexcept
        : __src_((_Uy&&) __src) {
      }

      constexpr operator _Ty() override {
        return static_cast<_Ty>(static_cast<_Uy&&>(__src_));
      }
    };

    struct __self_fn {
      template <class _Ty, class _Uy>
      using __value = __if_c<same_as<_Ty&&, _Uy&&>, _Uy&&, _Ty>;
      template <class _Ty>
      using __param = _Ty;
    };

    struct __tie_fn {
      template <class _Ty, class>
      using __value = __wrapper<_Ty>;
      template <class _Ty>
      using __param = __wrapper<_Ty>;
    };

    struct __convert_fn {
      template <class _Ty, class _Uy>
      using __value = __convertible_from<_Ty, _Uy>;
      template <class _Ty>
      using __param = __any_convertible_to<_Ty>&&;
    };

    template <class _Ty>
    extern const __self_fn __wrap;
    template <class _Ty>
    extern const __tie_fn __wrap<_Ty&>;
    template <class _Ty>
      requires(!move_constructible<_Ty>)
    extern const __convert_fn __wrap<_Ty>;

    template <class _Ty, class _Uy>
    using __value_t = typename decltype(__wrap<_Ty>)::template __value<_Ty, _Uy>;
    template <class _Ty>
    using __param_t = typename decltype(__wrap<_Ty>)::template __param<_Ty>;

    template <class _Ty>
    auto __unwrap(_Ty&&, long) -> _Ty&&;
    template <class _Ty>
    auto __unwrap(_Ty, int) -> __unwrapped_t<_Ty>&;
    template <class _Ty>
    using __unwrap_t = decltype(__tup::__unwrap(__declval<_Ty>(), 0));

    template <class _Ty>
    auto __unconst(_Ty&&) -> _Ty;
    template <class _Ty>
    auto __unconst(_Ty&) -> _Ty&;
    template <class _Ty>
    auto __unconst(const _Ty&&) -> _Ty;
    template <class _Ty>
    auto __unconst(const _Ty&) -> _Ty&;
    template <class _Ty>
    using __unconst_t = decltype(__tup::__unconst(__declval<_Ty>()));

    struct __access {
      template <__is_tuple _Tuple>
      static constexpr decltype(auto) __get_impl(_Tuple&& __tup) noexcept {
        return (((_Tuple&&) __tup).__fn_);
      }
    };

    template <class _Tuple>
    using __impl_of = decltype(__access::__get_impl(__declval<_Tuple>()));

    struct __apply_impl {
      template <class _Fun, class _Impl>
      constexpr auto operator()(_Fun&& __fn, _Impl&& __impl) const
        noexcept(__nothrow_callable<__unconst_t<_Impl>, _Impl, __unconst_t<_Fun>>)
          -> __call_result_t<__unconst_t<_Impl>, _Impl, _Fun> {
        return const_cast<__unconst_t<_Impl>&&>(__impl)((_Impl&&) __impl, (_Fun&&) __fn);
      }
    };

    template <class _Fun, __is_tuple _Tuple>
    constexpr auto apply(_Fun&& __fn, _Tuple&& __self) noexcept(
      __nothrow_callable<__apply_impl, _Fun, __impl_of<_Tuple>>)
      -> __call_result_t<__apply_impl, _Fun, __impl_of<_Tuple>> {
      return __apply_impl()((_Fun&&) __fn, __access::__get_impl((_Tuple&&) __self));
    }

    template <class _Fun, __is_tuple _Tuple>
    using __apply_result_t = //
      __call_result_t<__apply_impl, _Fun, __impl_of<_Tuple>>;

    template <class _Fun, class _Tuple>
    concept __applicable = //
      __callable<__apply_impl, _Fun, __impl_of<_Tuple>>;

    template <class _Fun, class _Tuple>
    concept __nothrow_applicable = //
      __nothrow_callable<__apply_impl, _Fun, __impl_of<_Tuple>>;

    struct __impl_types_ {
      template <class... _Ts>
      auto operator()(_Ts&&...) const -> __types<_Ts...>;
    };

    template <class _Impl>
    using __impl_types = //
      __call_result_t<__apply_impl, __impl_types_, _Impl>;

    template <class _Tuple>
    using __types_of = __impl_types<__impl_of<_Tuple>>;

    template <class _Ty>
    constexpr _Ty&& __move_capture(_Ty& __ty) noexcept {
      return (_Ty&&) __ty;
    }

    template <class _Ty>
    using __rvalue_t = decltype(__move_capture(__declval<_Ty&>()));

    template <class _Self, class _Ty>
    using __element_t = __unwrap_t<__copy_cvref_t<_Self, __rvalue_t<_Ty>>>;

    template <class _Ty>
    void __decay_copy(_Ty) noexcept;

    template <class _From, class _To>
    concept __decay_convertible_to = //
      requires(_From&& __from, __param_t<_To>& __w) {
        static_cast<__param_t<_To>>(static_cast<__value_t<_To, _From>>((_From&&) __from));
        __tup::__decay_copy(__move_capture(__w));
      };

    template <class... _Ts>
    constexpr auto
      __make_impl(_Ts&&... __ts) noexcept((__nothrow_decay_copyable<__rvalue_t<_Ts>> && ...)) {
      return [... __ts = __move_capture(__ts)]                                           //
            <class _Self, class _Fun>(_Self&&, _Fun && __fn) constexpr mutable           //
             noexcept(__nothrow_callable<__unconst_t<_Fun>, __element_t<_Self, _Ts>...>) //
             -> __call_result_t<__unconst_t<_Fun>, __element_t<_Self, _Ts>...> {
               return const_cast<__unconst_t<_Fun>&&>(__fn)(
                 static_cast<__element_t<_Self, _Ts>&&>(__ts)...);
             };
    }

    template <class _Ret, class... _Args>
    _Ret __impl_for_(_Ret (*)(_Args...));

    template <class... _Ts>
    using __impl_for = decltype(__tup::__impl_for_(&__make_impl<__param_t<_Ts>...>));

    template <class... _Ts>
    using __tuple_for = __tuple<sizeof...(_Ts), __impl_for<_Ts...>>;

    struct __make_tuple_fn {
      template <move_constructible... _Ts>
      constexpr __tuple_for<_Ts...> operator()(_Ts... __ts) const
        noexcept(__nothrow_constructible_from<__tuple_for<_Ts...>, _Ts...>) {
        return __tuple_for<_Ts...>((_Ts&&) __ts...);
      }
    };

    template <class _Ty>
    struct __make_default {
      operator _Ty() const noexcept(noexcept(_Ty())) {
        return (_Ty());
      }
    };

    template <class _Ty>
    using __default_init_for = __if_c<move_constructible<_Ty>, _Ty, __make_default<_Ty>>;

    template <class... _Ts>
    struct __construct_impl {
      template <class... _Us>
        requires(__decay_convertible_to<_Us, _Ts> && ...)
      constexpr auto operator()(_Us&&... us) const noexcept(noexcept(
        __tup::__make_impl<__param_t<_Ts>...>(static_cast<__value_t<_Ts, _Us>>((_Us&&) us)...)))
        -> __impl_for<_Ts...> {
        return __tup::__make_impl<__param_t<_Ts>...>(
          static_cast<__value_t<_Ts, _Us>>((_Us&&) us)...);
      };
    };

    template <class... _Ts>
    struct __default_init_impl {
      constexpr auto operator()() noexcept(
        __nothrow_callable<__construct_impl<_Ts...>, __default_init_for<_Ts>...>)
        requires(default_initializable<_Ts> && ...)
      {
        return __construct_impl<_Ts...>()(__default_init_for<_Ts>()...);
      }
    };

    template <class _Ty, class U>
    concept __nothrow_assignable_from = //
      requires(_Ty&& t, U&& u) {
        { ((_Ty&&) t) = ((U&&) u) } noexcept;
      };

    struct __assign_tuple {
      template <class... _Ts>
      constexpr auto operator()(_Ts&&... __ts) const noexcept {
        return [&]<class... _Us>(_Us && ... us) noexcept(
          (__nothrow_assignable_from<_Ts, _Us> && ...))
          requires(assignable_from<_Ts, _Us> && ...)
        {
          ((void) (((_Ts&&) __ts) = ((_Us&&) us)), ...);
        };
      }
    };

    template <class _To, class _From>
    concept __tuple_assignable_from = //
      __applicable<__apply_result_t<__assign_tuple, _To>, _From>;

    template <class _To, class _From>
    concept __nothrow_tuple_assignable_from = //
      __nothrow_applicable<__apply_result_t<__assign_tuple, _To>, _From>;

    template <std::size_t _Nn, __is_tuple _Tuple>
    constexpr auto get(_Tuple&& __tup) noexcept
      -> __apply_result_t<__nth_pack_element<_Nn>, _Tuple> {
      return __tup::apply(__nth_pack_element<_Nn>(), (_Tuple&&) __tup);
    }
  } // namespace __tup

  template <std::size_t _Size, class _Impl>
  struct __tuple : private __tup::__base {
   private:
    friend __tup::__access;
    using __types = __tup::__impl_types<_Impl>;
    using __construct_impl = __mapply<__q<__tup::__construct_impl>, __types>;
    using __default_init_impl = __mapply<__q<__tup::__default_init_impl>, __types>;

    template <std::size_t, class>
    friend struct __tuple;

    _Impl __fn_;

   public:
    constexpr __tuple() noexcept(__nothrow_callable<__default_init_impl>)
      requires __callable<__default_init_impl>
      : __fn_(__default_init_impl()()) {
    }

    __tuple(__tuple&&) = default;
    __tuple(__tuple const &) = default;
    __tuple& operator=(__tuple&&) = default;
    __tuple& operator=(__tuple const &) = default;

    template <class... _Us>
      requires(sizeof...(_Us) == _Size) && __callable<__construct_impl, _Us...>
    explicit(sizeof...(_Us) == 1) constexpr __tuple(_Us&&... __us) noexcept(
      __nothrow_callable<__construct_impl, _Us...>)
      : __fn_(__construct_impl()((_Us&&) __us...)) {
    }

    template <__tup::__is_tuple _Other>
      requires(!__decays_to<_Other, __tuple>) && __tup::__applicable<__construct_impl, _Other>
    explicit constexpr __tuple(_Other&& __other) noexcept(
      __tup::__nothrow_applicable<__construct_impl, _Other>)
      : __fn_(__tup::apply(__construct_impl(), (_Other&&) __other)) {
    }

    template <__tup::__is_tuple _Other>
      requires __tup::__tuple_assignable_from<__tuple, _Other>
    constexpr __tuple&& operator=(_Other&& __other) && noexcept(
      __tup::__nothrow_tuple_assignable_from<__tuple, _Other>) {
      __tup::apply(__tup::apply(__tup::__assign_tuple(), (__tuple&&) *this), (_Other&&) __other);
      return (__tuple&&) *this;
    }

    template <__tup::__is_tuple _Other>
      requires __tup::__tuple_assignable_from<__tuple&, _Other>
    constexpr __tuple& operator=(_Other&& __other) & noexcept(
      __tup::__nothrow_tuple_assignable_from<__tuple&, _Other>) {
      __tup::apply(__tup::apply(__tup::__assign_tuple(), *this), (_Other&&) __other);
      return *this;
    }

    template <__tup::__is_tuple _Other>
      requires __tup::__tuple_assignable_from<const __tuple&, _Other>
    constexpr const __tuple& operator=(_Other&& __other) const & noexcept(
      __tup::__nothrow_tuple_assignable_from<const __tuple&, _Other>) {
      __tup::apply(__tup::apply(__tup::__assign_tuple(), *this), (_Other&&) __other);
      return *this;
    }
  };

  using __tup::get;
  using __tup::apply;
  using __tup::__apply_result_t;
  using __tup::__applicable;
  using __tup::__nothrow_applicable;

  // From __meta.hpp:
  template <std::size_t _Size, class _Impl>
  struct __mexpand<__tuple<_Size, _Impl>> {
    template <class _MetaFn>
    using __f = __mapply<_MetaFn, __tup::__impl_types<_Impl>>;
  };

  template <std::size_t _Size, class _Impl>
  inline constexpr std::size_t tuple_size_v<__tuple<_Size, _Impl>> = _Size;

  inline constexpr __tup::__wrap_fn ref{};
  inline constexpr __tup::__make_tuple_fn make_tuple{};

  template <class... _Ts>
  using tuple = __tuple<sizeof...(_Ts), __tup::__impl_for<_Ts...>>;
}

STDEXEC_BEGIN_NAMESPACE_STD
  template <class>
  struct tuple_size;

  template <size_t _Size, class _Impl>
  struct tuple_size<stdexec::__tuple<_Size, _Impl>> : integral_constant<size_t, _Size> { };

  template <size_t _Size, class _Impl>
  struct tuple_size<const stdexec::__tuple<_Size, _Impl>> : integral_constant<size_t, _Size> { };

  template <size_t _Nn>
  struct __tuple_element_ {
    template <class... _Ts>
    using __f = stdexec::__m_at<_Nn, _Ts...>;
  };

  template <size_t _Nn, size_t _Size, class _Impl>
    requires(_Nn < _Size)
  struct tuple_element<_Nn, stdexec::__tuple<_Size, _Impl>> {
    using type = stdexec::__mapply<__tuple_element_<_Nn>, stdexec::__tuple<_Size, _Impl>>;
  };

STDEXEC_END_NAMESPACE_STD

#endif // !STDEXEC_NVHPC()
