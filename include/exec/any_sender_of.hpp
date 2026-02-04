/* Copyright (c) 2023 Maikel Nadolski
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

#include "sequence_senders.hpp"

#include <cstddef>
#include <utility>

namespace exec {
  namespace __any {
    using namespace STDEXEC;

    template <class _Sig>
    struct __rcvr_vfun;

    template <class _Tag, class... _Args>
    struct __rcvr_vfun<_Tag(_Args...)> {
      void (*__complete_)(void*, _Args...) noexcept;

      void operator()(void* __obj, _Tag, _Args... __args) const noexcept {
        __complete_(__obj, static_cast<_Args&&>(__args)...);
      }
    };

    template <class _GetReceiver = std::identity, class _Obj, class _Tag, class... _Args>
    constexpr auto __rcvr_vfun_fn(_Obj*, _Tag (*)(_Args...)) noexcept {
      return +[](void* __ptr, _Args... __args) noexcept {
        _Obj* __obj = static_cast<_Obj*>(__ptr);
        _Tag()(std::move(_GetReceiver()(*__obj)), static_cast<_Args&&>(__args)...);
      };
    }

    struct __create_vtable_t {
      template <class _VTable, class _Tp>
      constexpr auto operator()(__mtype<_VTable>, __mtype<_Tp>) const noexcept -> const _VTable* {
        return _VTable::__create_vtable(__mtype<_Tp>{});
      }
    };

    inline constexpr __create_vtable_t __create_vtable{};

    template <class _Sig>
    struct __query_vfun;

    template <class _Tag, class _Ret, class... _As>
    struct __query_vfun<_Tag (*const)(_Ret (*)(_As...))> {
      _Ret (*__fn_)(void*, _As...);

      auto operator()(_Tag, void* __rcvr, _As&&... __as) const -> _Ret {
        return __fn_(__rcvr, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Tag, class _Ret, class... _As>
    struct __query_vfun<_Tag (*)(_Ret (*)(_As...))> {
      _Ret (*__fn_)(void*, _As...);

      auto operator()(_Tag, void* __rcvr, _As&&... __as) const -> _Ret {
        return __fn_(__rcvr, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Tag, class _Ret, class... _As>
    struct __query_vfun<_Tag (*const)(_Ret (*)(_As...) noexcept)> {
      _Ret (*__fn_)(void*, _As...) noexcept;

      auto operator()(_Tag, void* __rcvr, _As&&... __as) const noexcept -> _Ret {
        return __fn_(__rcvr, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Tag, class _Ret, class... _As>
    struct __query_vfun<_Tag (*)(_Ret (*)(_As...) noexcept)> {
      _Ret (*__fn_)(void*, _As...) noexcept;

      auto operator()(_Tag, void* __rcvr, _As&&... __as) const noexcept -> _Ret {
        return __fn_(__rcvr, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Queryable, bool _IsEnvProvider = true>
    struct __query_vfun_fn;

    template <class _EnvProvider>
    struct __query_vfun_fn<_EnvProvider, true> {
      template <class _Tag, class _Ret, class... _As>
        requires __callable<_Tag, env_of_t<const _EnvProvider&>, _As...>
      constexpr auto
        operator()(_Tag (*)(_Ret (*)(_As...))) const noexcept -> _Ret (*)(void*, _As...) {
        return +[](void* __env_provider, _As... __as) -> _Ret {
          return _Tag{}(
            STDEXEC::get_env(*static_cast<const _EnvProvider*>(__env_provider)),
            static_cast<_As&&>(__as)...);
        };
      }

      template <class _Tag, class _Ret, class... _As>
        requires __callable<_Tag, env_of_t<const _EnvProvider&>, _As...>
      constexpr auto operator()(_Tag (*)(_Ret (*)(_As...) noexcept)) const noexcept
        -> _Ret (*)(void*, _As...) noexcept {
        return +[](void* __env_provider, _As... __as) noexcept -> _Ret {
          static_assert(__nothrow_callable<_Tag, const env_of_t<_EnvProvider>&, _As...>);
          return _Tag{}(
            STDEXEC::get_env(*static_cast<const _EnvProvider*>(__env_provider)),
            static_cast<_As&&>(__as)...);
        };
      }
    };

    template <class _Queryable>
    struct __query_vfun_fn<_Queryable, false> {
      template <class _Tag, class _Ret, class... _As>
        requires __callable<_Tag, const _Queryable&, _As...>
      constexpr auto
        operator()(_Tag (*)(_Ret (*)(_As...))) const noexcept -> _Ret (*)(void*, _As...) {
        return +[](void* __queryable, _As... __as) -> _Ret {
          return _Tag{}(*static_cast<const _Queryable*>(__queryable), static_cast<_As&&>(__as)...);
        };
      }

      template <class _Tag, class _Ret, class... _As>
        requires __callable<_Tag, const _Queryable&, _As...>
      constexpr auto operator()(_Tag (*)(_Ret (*)(_As...) noexcept)) const noexcept
        -> _Ret (*)(void*, _As...) noexcept {
        return +[](void* __env_provider, _As... __as) noexcept -> _Ret {
          static_assert(__nothrow_callable<_Tag, const _Queryable&, _As...>);
          return _Tag{}(
            *static_cast<const _Queryable*>(__env_provider), static_cast<_As&&>(__as)...);
        };
      }
    };

    template <class _Sig>
    struct __storage_vfun;

    template <class _Tag, class... _As>
    struct __storage_vfun<_Tag(void (*)(_As...))> {
      void (*__fn_)(void*, _As...) = [](void*, _As...) {
      };

      void operator()(_Tag, void* __storage, _As&&... __as) const {
        return __fn_(__storage, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Tag, class... _As>
    struct __storage_vfun<_Tag(void (*)(_As...) noexcept)> {
      void (*__fn_)(void*, _As...) noexcept = [](void*, _As...) noexcept {
      };

      void operator()(_Tag, void* __storage, _As&&... __as) const noexcept {
        return __fn_(__storage, static_cast<_As&&>(__as)...);
      }
    };

    template <class _Storage, class _Tp>
    struct __storage_vfun_fn {
      template <class _Tag, class... _As>
        requires __callable<_Tag, __mtype<_Tp>, _Storage&, _As...>
      constexpr auto
        operator()(_Tag (*)(void (*)(_As...))) const noexcept -> void (*)(void*, _As...) {
        return +[](void* __storage, _As... __as) -> void {
          return _Tag{}(
            __mtype<_Tp>{}, *static_cast<_Storage*>(__storage), static_cast<_As&&>(__as)...);
        };
      }

      template <class _Tag, class... _As>
        requires __callable<_Tag, __mtype<_Tp>, _Storage&, _As...>
      constexpr auto operator()(_Tag (*)(void (*)(_As...) noexcept)) const noexcept
        -> void (*)(void*, _As...) noexcept {
        return +[](void* __storage, _As... __as) noexcept -> void {
          static_assert(__nothrow_callable<_Tag, __mtype<_Tp>, _Storage&, _As...>);
          return _Tag{}(
            __mtype<_Tp>{}, *static_cast<_Storage*>(__storage), static_cast<_As&&>(__as)...);
        };
      }
    };

    struct __delete_t {
      template <class _Storage, class _Tp>
      void operator()(__mtype<_Tp>, _Storage& __storage) noexcept {
        static_assert(noexcept(__storage.__delete(__mtype<_Tp>{})));
        __storage.__delete(__mtype<_Tp>{});
      }
    };

    inline constexpr __delete_t __delete{};

    struct __move_construct_t {
      template <class _Storage, class _Tp>
      void operator()(__mtype<_Tp>, _Storage& __self, __midentity<_Storage&&> __from) noexcept {
        static_assert(
          noexcept(__self.__move_construct(__mtype<_Tp>{}, static_cast<_Storage&&>(__from))));
        __self.__move_construct(__mtype<_Tp>{}, static_cast<_Storage&&>(__from));
      }
    };

    inline constexpr __move_construct_t __move_construct{};

    struct __copy_construct_t {
      template <class _Storage, class _Tp>
      void operator()(__mtype<_Tp>, _Storage& __self, const _Storage& __from)
        noexcept(noexcept(__self.__copy_construct(__mtype<_Tp>{}, __from))) {
        __self.__copy_construct(__mtype<_Tp>{}, __from);
      }
    };

    inline constexpr __copy_construct_t __copy_construct{};

    template <class _ParentVTable, class... _StorageCPOs>
    struct __storage_vtable;

    template <class _ParentVTable, class... _StorageCPOs>
      requires requires { _ParentVTable::operator(); }
    struct __storage_vtable<_ParentVTable, _StorageCPOs...>
      : _ParentVTable
      , __storage_vfun<_StorageCPOs>... {
      using _ParentVTable::operator();
      using __storage_vfun<_StorageCPOs>::operator()...;
    };

    template <class _ParentVTable, class... _StorageCPOs>
      requires(!requires { _ParentVTable::operator(); })
    struct __storage_vtable<_ParentVTable, _StorageCPOs...>
      : _ParentVTable
      , __storage_vfun<_StorageCPOs>... {
      using __storage_vfun<_StorageCPOs>::operator()...;
    };

    template <class _ParentVTable, class... _StorageCPOs>
    inline constexpr __storage_vtable<_ParentVTable, _StorageCPOs...> __null_storage_vtbl{};

    template <class _ParentVTable, class... _StorageCPOs>
    constexpr auto
      __default_storage_vtable(__storage_vtable<_ParentVTable, _StorageCPOs...>*) noexcept
      -> const __storage_vtable<_ParentVTable, _StorageCPOs...>* {
      return &__null_storage_vtbl<_ParentVTable, _StorageCPOs...>;
    }

    template <class _Storage, class _Tp, class _ParentVTable, class... _StorageCPOs>
    static const __storage_vtable<_ParentVTable, _StorageCPOs...> __storage_vtbl{
      {*__create_vtable(__mtype<_ParentVTable>{}, __mtype<_Tp>{})},
      {__storage_vfun_fn<_Storage, _Tp>{}(static_cast<_StorageCPOs*>(nullptr))}...};

    template <
      class _Vtable,
      class _Allocator,
      bool _Copyable = false,
      std::size_t _InlineSize = 3 * sizeof(void*),
      std::size_t _Alignment = alignof(std::max_align_t)
    >
    class __storage;

    template <
      class _Vtable,
      class _Allocator,
      std::size_t _InlineSize = 3 * sizeof(void*),
      std::size_t _Alignment = alignof(std::max_align_t)
    >
    struct __immovable_storage : __immovable {
      static constexpr std::size_t __buffer_size = (std::max) (_InlineSize, sizeof(void*));
      static constexpr std::size_t __alignment = (std::max) (_Alignment, alignof(void*));
      using __with_delete = __delete_t(void() noexcept);
      using __vtable_t = __storage_vtable<_Vtable, __with_delete>;

      template <class _Tp>
      static constexpr bool __is_small = sizeof(_Tp) <= __buffer_size
                                      && alignof(_Tp) <= __alignment;

      template <class _Tp>
      static constexpr auto __get_vtable_of_type() noexcept -> const __vtable_t* {
        return &__storage_vtbl<__immovable_storage, __decay_t<_Tp>, _Vtable, __with_delete>;
      }
     public:
      __immovable_storage() = default;

      template <__not_decays_to<__immovable_storage> _Tp>
        requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<__decay_t<_Tp>>>
      __immovable_storage(_Tp&& __object)
        : __vtable_{__get_vtable_of_type<_Tp>()} {
        using _Dp = __decay_t<_Tp>;
        if constexpr (__is_small<_Dp>) {
          __construct_small<_Dp>(static_cast<_Tp&&>(__object));
        } else {
          __construct_large<_Dp>(static_cast<_Tp&&>(__object));
        }
      }

      template <class _Tp, class... _Args>
        requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<_Tp>>
      __immovable_storage(std::in_place_type_t<_Tp>, _Args&&... __args)
        : __vtable_{__get_vtable_of_type<_Tp>()} {
        if constexpr (__is_small<_Tp>) {
          __construct_small<_Tp>(static_cast<_Args&&>(__args)...);
        } else {
          __construct_large<_Tp>(static_cast<_Args&&>(__args)...);
        }
      }

      ~__immovable_storage() {
        __reset();
      }

      void __reset() noexcept {
        (*__vtable_)(__any::__delete, this);
        __object_pointer_ = nullptr;
        __vtable_ = __default_storage_vtable(static_cast<__vtable_t*>(nullptr));
      }

      [[nodiscard]]
      auto __get_vtable() const noexcept -> const _Vtable* {
        return __vtable_;
      }

      [[nodiscard]]
      auto __get_object_pointer() const noexcept -> void* {
        return __object_pointer_;
      }

     private:
      friend struct __any::__delete_t;

      template <class _Tp, class... _As>
      void __construct_small(_As&&... __args) {
        static_assert(sizeof(_Tp) <= __buffer_size && alignof(_Tp) <= __alignment);
        _Tp* __pointer = reinterpret_cast<_Tp*>(&__buffer_[0]);
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        std::allocator_traits<_Alloc>::construct(__alloc, __pointer, static_cast<_As&&>(__args)...);
        __object_pointer_ = __pointer;
      }

      template <class _Tp, class... _As>
      void __construct_large(_As&&... __args) {
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        _Tp* __pointer = std::allocator_traits<_Alloc>::allocate(__alloc, 1);
        STDEXEC_TRY {
          std::allocator_traits<_Alloc>::construct(
            __alloc, __pointer, static_cast<_As&&>(__args)...);
        }
        STDEXEC_CATCH_ALL {
          std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
          STDEXEC_THROW();
        }
        __object_pointer_ = __pointer;
      }

      template <class _Tp>
      void __delete(__mtype<_Tp>) noexcept {
        if (!__object_pointer_) {
          return;
        }
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        _Tp* __pointer = static_cast<_Tp*>(std::exchange(__object_pointer_, nullptr));
        std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
        if constexpr (!__is_small<_Tp>) {
          std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
        }
      }

     private:
      const __vtable_t* __vtable_{__default_storage_vtable(static_cast<__vtable_t*>(nullptr))};
      void* __object_pointer_{nullptr};
      alignas(__alignment) std::byte __buffer_[__buffer_size]{};
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Allocator __allocator_{};
    };

    template <
      class _Vtable,
      class _Allocator,
      bool _Copyable,
      std::size_t _InlineSize,
      std::size_t _Alignment
    >
    class __storage : __if_c<_Copyable, __empty, __move_only> {
      static_assert(
        STDEXEC_IS_CONVERTIBLE_TO(typename std::allocator_traits<_Allocator>::void_pointer, void*));

      static constexpr std::size_t __buffer_size = (std::max) (_InlineSize, sizeof(void*));
      static constexpr std::size_t __alignment = (std::max) (_Alignment, alignof(void*));
      using __with_copy = __copy_construct_t(void(const __storage&));
      using __with_move = __move_construct_t(void(__storage&&) noexcept);
      using __with_delete = __delete_t(void() noexcept);

      template <class _Tp>
      static constexpr bool __is_small = sizeof(_Tp) <= __buffer_size && alignof(_Tp) <= __alignment
                                      && std::is_nothrow_move_constructible_v<_Tp>;

      using __vtable_t = __if_c<
        _Copyable,
        __storage_vtable<_Vtable, __with_delete, __with_move, __with_copy>,
        __storage_vtable<_Vtable, __with_delete, __with_move>
      >;

      template <class _Tp>
      static constexpr auto __get_vtable_of_type() noexcept -> const __vtable_t* {
        if constexpr (_Copyable) {
          return &__storage_vtbl<
            __storage,
            __decay_t<_Tp>,
            _Vtable,
            __with_delete,
            __with_move,
            __with_copy
          >;
        } else {
          return &__storage_vtbl<__storage, __decay_t<_Tp>, _Vtable, __with_delete, __with_move>;
        }
      }

     public:
      __storage() = default;

      template <__not_decays_to<__storage> _Tp>
        requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<__decay_t<_Tp>>>
      __storage(_Tp&& __object)
        : __vtable_{__get_vtable_of_type<_Tp>()} {
        using _Dp = __decay_t<_Tp>;
        if constexpr (__is_small<_Dp>) {
          __construct_small<_Dp>(static_cast<_Tp&&>(__object));
        } else {
          __construct_large<_Dp>(static_cast<_Tp&&>(__object));
        }
      }

      template <class _Tp, class... _Args>
        requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<_Tp>>
      __storage(std::in_place_type_t<_Tp>, _Args&&... __args)
        : __vtable_{__get_vtable_of_type<_Tp>()} {
        if constexpr (__is_small<_Tp>) {
          __construct_small<_Tp>(static_cast<_Args&&>(__args)...);
        } else {
          __construct_large<_Tp>(static_cast<_Args&&>(__args)...);
        }
      }

      __storage(const __storage& __other)
        requires(_Copyable)
        : __vtable_(__other.__vtable_) {
        (*__other.__vtable_)(__any::__copy_construct, this, __other);
      }

      auto operator=(const __storage& __other) -> __storage&
        requires(_Copyable)
      {
        if (&__other != this) {
          __storage tmp(__other);
          *this = std::move(tmp);
        }
        return *this;
      }

      __storage(__storage&& __other) noexcept
        : __vtable_(__other.__vtable_) {
        (*__other.__vtable_)(__any::__move_construct, this, static_cast<__storage&&>(__other));
      }

      auto operator=(__storage&& __other) noexcept -> __storage& {
        __reset();
        (*__other.__vtable_)(__any::__move_construct, this, static_cast<__storage&&>(__other));
        return *this;
      }

      ~__storage() {
        __reset();
      }

      void __reset() noexcept {
        (*__vtable_)(__any::__delete, this);
        __object_pointer_ = nullptr;
        __vtable_ = __default_storage_vtable(static_cast<__vtable_t*>(nullptr));
      }

      auto __get_vtable() const noexcept -> const _Vtable* {
        return __vtable_;
      }

      [[nodiscard]]
      auto __get_object_pointer() const noexcept -> void* {
        return __object_pointer_;
      }

     private:
      friend struct __any::__delete_t;
      friend struct __any::__move_construct_t;
      friend struct __any::__copy_construct_t;

      template <class _Tp, class... _As>
      void __construct_small(_As&&... __args) {
        static_assert(sizeof(_Tp) <= __buffer_size && alignof(_Tp) <= __alignment);
        _Tp* __pointer = reinterpret_cast<_Tp*>(&__buffer_[0]);
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        std::allocator_traits<_Alloc>::construct(__alloc, __pointer, static_cast<_As&&>(__args)...);
        __object_pointer_ = __pointer;
      }

      template <class _Tp, class... _As>
      void __construct_large(_As&&... __args) {
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        _Tp* __pointer = std::allocator_traits<_Alloc>::allocate(__alloc, 1);
        STDEXEC_TRY {
          std::allocator_traits<_Alloc>::construct(
            __alloc, __pointer, static_cast<_As&&>(__args)...);
        }
        STDEXEC_CATCH_ALL {
          std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
          STDEXEC_THROW();
        }
        __object_pointer_ = __pointer;
      }

      template <class _Tp>
      void __delete(__mtype<_Tp>) noexcept {
        if (!__object_pointer_) {
          return;
        }
        using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        _Tp* __pointer = static_cast<_Tp*>(std::exchange(__object_pointer_, nullptr));
        std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
        if constexpr (!__is_small<_Tp>) {
          std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
        }
      }

      template <class _Tp>
      void __move_construct(__mtype<_Tp>, __storage&& __other) noexcept {
        if (!__other.__object_pointer_) {
          return;
        }
        _Tp* __pointer = static_cast<_Tp*>(std::exchange(__other.__object_pointer_, nullptr));
        if constexpr (__is_small<_Tp>) {
          _Tp& __other_object = *__pointer;
          this->template __construct_small<_Tp>(static_cast<_Tp&&>(__other_object));
          using _Alloc = std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
          _Alloc __alloc{__allocator_};
          std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
        } else {
          __object_pointer_ = __pointer;
        }
        __vtable_ = std::exchange(
          __other.__vtable_, __default_storage_vtable(static_cast<__vtable_t*>(nullptr)));
      }

      template <class _Tp>
        requires _Copyable
      void __copy_construct(__mtype<_Tp>, const __storage& __other) {
        if (!__other.__object_pointer_) {
          return;
        }
        const _Tp& __other_object = *static_cast<const _Tp*>(__other.__object_pointer_);
        if constexpr (__is_small<_Tp>) {
          this->template __construct_small<_Tp>(__other_object);
        } else {
          this->template __construct_large<_Tp>(__other_object);
        }
        __vtable_ = __other.__vtable_;
      }

      const __vtable_t* __vtable_{__default_storage_vtable(static_cast<__vtable_t*>(nullptr))};
      void* __object_pointer_{nullptr};
      alignas(__alignment) std::byte __buffer_[__buffer_size]{};
      STDEXEC_ATTRIBUTE(no_unique_address) _Allocator __allocator_ { };
    };

    struct __empty_vtable {
      template <class _Sender>
      static auto __create_vtable(__mtype<_Sender>) noexcept -> const __empty_vtable* {
        static const __empty_vtable __vtable_{};
        return &__vtable_;
      }
    };

    template <
      class _VTable = __empty_vtable,
      class _Allocator = std::allocator<std::byte>,
      std::size_t _InlineSize = 3 * sizeof(void*),
      std::size_t _Alignment = alignof(std::max_align_t)
    >
    using __immovable_storage_t = __immovable_storage<_VTable, _Allocator, _InlineSize, _Alignment>;

    template <class _VTable, class _Allocator = std::allocator<std::byte>>
    using __unique_storage_t = __storage<_VTable, _Allocator>;

    template <
      class _VTable,
      std::size_t _InlineSize = 3 * sizeof(void*),
      class _Allocator = std::allocator<std::byte>
    >
    using __copyable_storage_t = __storage<_VTable, _Allocator, true, _InlineSize>;

    template <class _Tag, class... _As>
    auto __tag_type(_Tag (*)(_As...)) -> _Tag;

    template <class _Tag, class... _As>
    auto __tag_type(_Tag (*)(_As...) noexcept) -> _Tag;

    template <class _Query>
    using __tag_type_t = decltype(__tag_type(static_cast<_Query>(nullptr)));

    template <class _Query>
    concept __is_stop_token_query = __same_as<__tag_type_t<_Query>, get_stop_token_t>;

    template <class _Query>
    concept __is_not_stop_token_query = !__is_stop_token_query<_Query>;

    template <class _Query>
    using __is_not_stop_token_query_t = __mbool<__is_not_stop_token_query<_Query>>;

    auto __test_never_stop_token(get_stop_token_t (*)(never_stop_token (*)() noexcept))
      -> __mbool<true>;

    template <class _Tag, class _Ret, class... _As>
    auto __test_never_stop_token(_Tag (*)(_Ret (*)(_As...) noexcept)) -> __mbool<false>;

    template <class _Tag, class _Ret, class... _As>
    auto __test_never_stop_token(_Tag (*)(_Ret (*)(_As...))) -> __mbool<false>;

    template <class _Query>
    using __is_never_stop_token_query_t = decltype(__test_never_stop_token(
      static_cast<_Query>(nullptr)));

    template <class _Query>
    concept __is_never_stop_token_query = __is_never_stop_token_query_t<_Query>::value;

    template <class _Query, class _Env>
    concept __satisfies_receiver_stop_token_query =
      __same_as<__decay_t<__query_result_t<_Env, __tag_type_t<_Query>>>, stop_token_of_t<_Env>>;

    template <class _Query, class... _Env>
    concept __satisfies_receiver_query =
      !__is_stop_token_query<_Query> || __is_never_stop_token_query<_Query>
      || (__satisfies_receiver_stop_token_query<_Query, _Env> || ...);

    namespace __rec {
      template <class _Sigs, class... _Queries>
      struct __vtable;

      template <class _Sigs, class... _Queries>
      struct __ref;

      template <class... _Sigs, class... _Queries>
      struct __vtable<completion_signatures<_Sigs...>, _Queries...>
        : __overload<__rcvr_vfun<_Sigs>...>
        , __query_vfun<_Queries>... {
        using __query_vfun<_Queries>::operator()...;

        template <class _Tag, class... _As>
          requires __one_of<_Tag(_As...), _Sigs...>
                || __callable<__overload<__rcvr_vfun<_Sigs>...>, void*, _Tag, _As...>
        void operator()(void* __rcvr, _Tag, _As&&... __as) const noexcept {
          if constexpr (__one_of<_Tag(_As...), _Sigs...>) {
            const __rcvr_vfun<_Tag(_As...)>& __vfun = *this;
            __vfun(__rcvr, _Tag(), static_cast<_As&&>(__as)...);
          } else {
            const __overload<__rcvr_vfun<_Sigs>...>& __vfun = *this;
            __vfun(__rcvr, _Tag(), static_cast<_As&&>(__as)...);
          }
        }

        template <class _Rcvr>
          requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        static auto __create_vtable(__mtype<_Rcvr>) noexcept -> const __vtable* {
          static const __vtable __vtable_{
            {{__rcvr_vfun_fn(static_cast<_Rcvr*>(nullptr), static_cast<_Sigs*>(nullptr))}...},
            {__query_vfun_fn<_Rcvr>{}(static_cast<_Queries>(nullptr))}...};
          return &__vtable_;
        }
      };

      template <class... _Sigs, class... _Queries>
        requires(__is_not_stop_token_query<_Queries> && ...)
      struct __ref<completion_signatures<_Sigs...>, _Queries...> {
#if !STDEXEC_MSVC()
        // MSVCBUG https://developercommunity.visualstudio.com/t/Private-member-inaccessible-when-used-in/10448363
       private:
#endif
        using __vtable_t = __vtable<completion_signatures<_Sigs...>, _Queries...>;

        struct __env_t {
          const __vtable_t* __vtable_;
          void* __rcvr_;
          inplace_stop_token __token_;

          template <class _Tag, class... _As>
            requires __callable<const __vtable_t&, _Tag, void*, _As...>
          auto query(_Tag, _As&&... __as) const
            noexcept(__nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
              -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
            return (*__vtable_)(_Tag{}, __rcvr_, static_cast<_As&&>(__as)...);
          }

          [[nodiscard]]
          auto query(get_stop_token_t) const noexcept -> inplace_stop_token {
            return __token_;
          }
        } __env_;
       public:
        using receiver_concept = STDEXEC::receiver_t;

        template <__none_of<__ref, const __ref, __env_t, const __env_t> _Rcvr>
          requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        /*implicit*/ __ref(_Rcvr& __rcvr) noexcept
          : __env_{
              __create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}),
              &__rcvr,
              STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr))} {
        }

        template <class... _As>
          requires __callable<__vtable_t, void*, set_value_t, _As...>
        void set_value(_As&&... __as) noexcept {
          (*__env_.__vtable_)(__env_.__rcvr_, set_value_t(), static_cast<_As&&>(__as)...);
        }

        template <class _Error>
          requires __callable<__vtable_t, void*, set_error_t, _Error>
        void set_error(_Error&& __err) noexcept {
          (*__env_.__vtable_)(__env_.__rcvr_, set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept
          requires __callable<__vtable_t, void*, set_stopped_t>
        {
          (*__env_.__vtable_)(__env_.__rcvr_, set_stopped_t());
        }

        auto get_env() const noexcept -> const __env_t& {
          return __env_;
        }
      };

      template <class... _Sigs, class... _Queries>
        requires(__is_stop_token_query<_Queries> || ...)
      struct __ref<completion_signatures<_Sigs...>, _Queries...> {
#if !STDEXEC_MSVC()
        // MSVCBUG https://developercommunity.visualstudio.com/t/Private-member-inaccessible-when-used-in/10448363

       private:
#endif
        using _FilteredQueries =
          __minvoke<__mremove_if<__q<__is_never_stop_token_query_t>>, _Queries...>;
        using __vtable_t =
          __mapply<__mbind_front_q<__vtable, completion_signatures<_Sigs...>>, _FilteredQueries>;

        struct __env_t {
          const __vtable_t* __vtable_;
          void* __rcvr_;

          template <class _Tag, class... _As>
            requires __callable<const __vtable_t&, _Tag, void*, _As...>
          auto query(_Tag, _As&&... __as) const
            noexcept(__nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
              -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
            return (*__vtable_)(_Tag{}, __rcvr_, static_cast<_As&&>(__as)...);
          }
        } __env_;
       public:
        using receiver_concept = STDEXEC::receiver_t;

        template <__none_of<__ref, const __ref, __env_t, const __env_t> _Rcvr>
          requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        __ref(_Rcvr& __rcvr) noexcept
          : __env_{__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}), &__rcvr} {
        }

        template <class... _As>
          requires __one_of<set_value_t(_As...), _Sigs...>
                || __callable<__overload<__rcvr_vfun<_Sigs>...>, void*, set_value_t, _As...>
        void set_value(_As&&... __as) noexcept {
          if constexpr (__one_of<set_value_t(_As...), _Sigs...>) {
            const __rcvr_vfun<set_value_t(_As...)>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_value_t(), static_cast<_As&&>(__as)...);
          } else {
            const __overload<__rcvr_vfun<_Sigs>...>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_value_t(), static_cast<_As&&>(__as)...);
          }
        }

        template <class _Error>
          requires __one_of<set_error_t(_Error), _Sigs...>
                || __callable<__overload<__rcvr_vfun<_Sigs>...>, void*, set_error_t, _Error>
        void set_error(_Error&& __err) noexcept {
          if constexpr (__one_of<set_error_t(_Error), _Sigs...>) {
            const __rcvr_vfun<set_error_t(_Error)>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_error_t(), static_cast<_Error&&>(__err));
          } else {
            const __overload<__rcvr_vfun<_Sigs>...>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_error_t(), static_cast<_Error&&>(__err));
          }
        }

        void set_stopped() noexcept
          requires __one_of<set_stopped_t(), _Sigs...>
        {
          const __rcvr_vfun<set_stopped_t()>& __vfun = *__env_.__vtable_;
          __vfun(__env_.__rcvr_, set_stopped_t());
        }

        auto get_env() const noexcept -> const __env_t& {
          return __env_;
        }
      };
    } // namespace __rec

    struct __operation_vtable {
      template <class _Op>
      static auto __create_vtable(__mtype<_Op>) noexcept -> const __operation_vtable* {
        static __operation_vtable __vtable{[](void* __object_pointer) noexcept -> void {
          STDEXEC_ASSERT(__object_pointer);
          _Op& __op = *static_cast<_Op*>(__object_pointer);
          static_assert(operation_state<_Op>);
          STDEXEC::start(__op);
        }};
        return &__vtable;
      }

      void (*__start_)(void*) noexcept;
    };

    using __immovable_operation_storage =
      __immovable_storage_t<__operation_vtable, std::allocator<std::byte>, 6 * sizeof(void*)>;

    template <class _Sigs, class _Queries>
    using __receiver_ref = __mapply<__mbind_front_q<__rec::__ref, _Sigs>, _Queries>;

    template <class _Receiver>
    struct __operation_base {
      STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;
      STDEXEC::inplace_stop_source __stop_source_{};
      using __stop_callback = typename STDEXEC::stop_token_of_t<
        STDEXEC::env_of_t<_Receiver>
      >::template callback_type<__forward_stop_request>;
      std::optional<__stop_callback> __on_stop_{};
    };

    template <class _Env>
    using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _Env>;

    template <class _Receiver>
    struct __stoppable_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      template <class _Item>
        requires __callable<set_next_t, _Receiver&, _Item>
      [[nodiscard]]
      auto set_next(_Item&& __item) & noexcept(__nothrow_callable<set_next_t, _Receiver&, _Item>)
        -> __call_result_t<set_next_t, _Receiver&, _Item> {
        return exec::set_next(__op_->__rcvr_, static_cast<_Item&&>(__item));
      }

      template <class... _Args>
        requires __callable<set_value_t, _Receiver, _Args...>
      void set_value(_Args&&... __args) noexcept {
        __op_->__on_stop_.reset();
        STDEXEC::set_value(
          static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Args&&>(__args)...);
      }

      template <class _Error>
        requires __callable<set_error_t, _Receiver, _Error>
      void set_error(_Error&& __err) noexcept {
        __op_->__on_stop_.reset();
        STDEXEC::set_error(static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Error&&>(__err));
      }

      void set_stopped() noexcept
        requires __callable<set_stopped_t, _Receiver>
      {
        __op_->__on_stop_.reset();
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      auto get_env() const noexcept -> __env_t<env_of_t<_Receiver>> {
        return __env::__join(
          prop{get_stop_token, __op_->__stop_source_.get_token()},
          STDEXEC::get_env(__op_->__rcvr_));
      }

      __operation_base<_Receiver>* __op_;
    };

    template <class _Receiver, bool>
    struct __operation : __operation_base<_Receiver> {
     public:
      template <class _Sender>
      explicit __operation(_Sender&& __sender, _Receiver&& __receiver)
        : __operation_base<_Receiver>{static_cast<_Receiver&&>(__receiver)}
        , __rec_{this}
        , __storage_{__sender.__connect(__rec_)} {
      }

      void start() & noexcept {
        this->__on_stop_.emplace(
          STDEXEC::get_stop_token(STDEXEC::get_env(this->__rcvr_)),
          __forward_stop_request{this->__stop_source_});
        STDEXEC_ASSERT(__storage_.__get_vtable()->__start_);
        __storage_.__get_vtable()->__start_(__storage_.__get_object_pointer());
      }

     private:
      __stoppable_receiver<_Receiver> __rec_;
      __immovable_operation_storage __storage_{};
    };

    template <class _Receiver>
    struct __operation<_Receiver, false> {
     public:
      template <class _Sender>
      explicit __operation(_Sender&& __sender, _Receiver&& __receiver)
        : __rec_{static_cast<_Receiver&&>(__receiver)}
        , __storage_{__sender.__connect(__rec_)} {
      }

      void start() & noexcept {
        STDEXEC_ASSERT(__storage_.__get_vtable()->__start_);
        __storage_.__get_vtable()->__start_(__storage_.__get_object_pointer());
      }

     private:
      STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rec_;
      __immovable_operation_storage __storage_{};
    };

    template <class _Queries, bool _IsEnvProvider = true>
    class __query_vtable;

    template <template <class...> class _List, class... _Queries, bool _IsEnvProvider>
    class __query_vtable<_List<_Queries...>, _IsEnvProvider> : public __query_vfun<_Queries>... {
     public:
      using __query_vfun<_Queries>::operator()...;

      template <class _Queryable>
        requires(__callable<__query_vfun_fn<_Queryable, _IsEnvProvider>, _Queries> && ...)
      static auto __create_vtable(__mtype<_Queryable>) noexcept -> const __query_vtable* {
        static const __query_vtable __vtable{
          {__query_vfun_fn<_Queryable, _IsEnvProvider>{}(static_cast<_Queries>(nullptr))}...};
        return &__vtable;
      }
    };

    template <class _Sigs, class _SenderQueries = __mlist<>, class _ReceiverQueries = __mlist<>>
    struct __sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = _Sigs;
      using __receiver_ref_t = __receiver_ref<_Sigs, _ReceiverQueries>;

      struct __vtable;
      struct __attrs;

      static constexpr bool __with_inplace_stop_token =
        __mapply<__mall_of<__q<__is_not_stop_token_query_t>>, _ReceiverQueries>::value;

      __sender(__sender&&) = default;
      __sender(const __sender&) = delete;

      auto operator=(__sender&&) -> __sender& = default;
      auto operator=(const __sender&) -> __sender& = delete;

      template <__not_decays_to<__sender> _Sender>
        requires sender_to<_Sender, __receiver_ref<_Sigs, _ReceiverQueries>>
      __sender(_Sender&& __sndr)
        : __storage_{static_cast<_Sender&&>(__sndr)} {
      }

      auto __connect(__receiver_ref_t __receiver) -> __immovable_operation_storage {
        return __storage_.__get_vtable()->__connect_(
          __storage_.__get_object_pointer(), static_cast<__receiver_ref_t&&>(__receiver));
      }

      auto get_env() const noexcept -> __attrs {
        return {__storage_.__get_vtable(), __storage_.__get_object_pointer()};
      }

      template <receiver_of<_Sigs> _Receiver>
      auto connect(_Receiver __rcvr) && -> __operation<_Receiver, __with_inplace_stop_token> {
        return __operation<_Receiver, __with_inplace_stop_token>{
          static_cast<__sender&&>(*this), static_cast<_Receiver&&>(__rcvr)};
      }

     private:
      __unique_storage_t<__vtable> __storage_;
    };

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender<_Sigs, _SenderQueries, _ReceiverQueries>::__vtable
      : __query_vtable<_SenderQueries> {
      auto __queries() const noexcept -> const __query_vtable<_SenderQueries>& {
        return *this;
      }

      template <sender_to<__receiver_ref_t> _Sender>
      static auto __create_vtable(__mtype<_Sender>) noexcept -> const __vtable* {
        static const __vtable __vtable_{
          {*__any::__create_vtable(__mtype<__query_vtable<_SenderQueries>>{}, __mtype<_Sender>{})},
          [](void* __object_pointer, __receiver_ref_t __receiver) -> __immovable_operation_storage {
            _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
            using __op_state_t = connect_result_t<_Sender, __receiver_ref_t>;
            return __immovable_operation_storage{
              std::in_place_type<__op_state_t>, __emplace_from{[&] {
                return STDEXEC::connect(
                  static_cast<_Sender&&>(__sender), static_cast<__receiver_ref_t&&>(__receiver));
              }}};
          }};
        return &__vtable_;
      }

      __immovable_operation_storage (*__connect_)(void*, __receiver_ref_t);
    };

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender<_Sigs, _SenderQueries, _ReceiverQueries>::__attrs {
      template <class _Tag, class... _As>
        requires __callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>
      auto query(_Tag, _As&&... __as) const
        noexcept(__nothrow_callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>)
          -> __call_result_t<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...> {
        return __vtable_->__queries()(_Tag{}, __sender_, static_cast<_As&&>(__as)...);
      }

      const __vtable* __vtable_;
      void* __sender_;
    };

    template <class _ScheduleSender, class _SchedulerQueries = __mlist<>>
    class __scheduler {
      static constexpr std::size_t __buffer_size = 4 * sizeof(void*);
      template <class _Ty>
      static constexpr bool __is_small = sizeof(_Ty) <= __buffer_size
                                      && alignof(_Ty) <= alignof(std::max_align_t);

     public:
      template <__not_decays_to<__scheduler> _Scheduler>
        requires scheduler<_Scheduler>
      __scheduler(_Scheduler&& __scheduler)
        : __storage_{static_cast<_Scheduler&&>(__scheduler)} {
        static_assert(
          __is_small<_Scheduler>,
          "any_scheduler<> must have a nothrow copy constructor, so the scheduler object must be "
          "small enough to be stored in the internal buffer to avoid dynamic allocation.");
      }

      __scheduler(__scheduler&&) noexcept = default;
      __scheduler(const __scheduler&) noexcept = default;
      auto operator=(__scheduler&&) noexcept -> __scheduler& = default;
      auto operator=(const __scheduler&) noexcept -> __scheduler& = default;

      using __sender_t = _ScheduleSender;

      auto schedule() const noexcept -> __sender_t {
        STDEXEC_ASSERT(__storage_.__get_vtable()->__schedule_);
        return __storage_.__get_vtable()->__schedule_(__storage_.__get_object_pointer());
      }

      template <class _Tag, class... _As>
        requires __callable<const __query_vtable<_SchedulerQueries, false>&, _Tag, void*, _As...>
      auto query(_Tag, _As&&... __as) const noexcept(
        __nothrow_callable<const __query_vtable<_SchedulerQueries, false>&, _Tag, void*, _As...>)
        -> __call_result_t<const __query_vtable<_SchedulerQueries, false>&, _Tag, void*, _As...> {
        return __storage_.__get_vtable()
          ->__queries()(_Tag{}, __storage_.__get_object_pointer(), static_cast<_As&&>(__as)...);
      }

     private:
      class __vtable : public __query_vtable<_SchedulerQueries, false> {
       public:
        __sender_t (*__schedule_)(void*) noexcept;
        bool (*__equal_to_)(const void*, const void* other) noexcept;

        auto __queries() const noexcept -> const __query_vtable<_SchedulerQueries, false>& {
          return *this;
        }

        template <scheduler _Scheduler>
        static auto __create_vtable(__mtype<_Scheduler>) noexcept -> const __vtable* {
          static const __vtable __vtable_{
            {*__any::__create_vtable(
              __mtype<__query_vtable<_SchedulerQueries, false>>{}, __mtype<_Scheduler>{})},
            [](void* __object_pointer) noexcept -> __sender_t {
              const _Scheduler& __scheduler = *static_cast<const _Scheduler*>(__object_pointer);
              return __sender_t{STDEXEC::schedule(__scheduler)};
            },
            [](const void* __self, const void* __other) noexcept -> bool {
              static_assert(
                noexcept(__declval<const _Scheduler&>() == __declval<const _Scheduler&>()));
              STDEXEC_ASSERT(__self && __other);
              const _Scheduler& __self_scheduler = *static_cast<const _Scheduler*>(__self);
              const _Scheduler& __other_scheduler = *static_cast<const _Scheduler*>(__other);
              return __self_scheduler == __other_scheduler;
            }};
          return &__vtable_;
        }
      };

      friend auto
        operator==(const __scheduler& __self, const __scheduler& __other) noexcept -> bool {
        if (__self.__storage_.__get_vtable() != __other.__storage_.__get_vtable()) {
          return false;
        }

        void* __p = __self.__storage_.__get_object_pointer();
        void* __o = __other.__storage_.__get_object_pointer();
        // if both object pointers are not null, use the virtual equal_to function
        return (__p && __o && __self.__storage_.__get_vtable()->__equal_to_(__p, __o))
            // if both object pointers are nullptrs, they are equal
            || (!__p && !__o);
      }

      friend auto
        operator!=(const __scheduler& __self, const __scheduler& __other) noexcept -> bool {
        return !(__self == __other);
      }

      __copyable_storage_t<__vtable, __buffer_size> __storage_{};
    };

    template <class _Tag>
    struct __ret_equals_to {
      template <class _Sig>
      using __f = STDEXEC::__mbool<STDEXEC_IS_SAME(_Tag, __detail::__tag_of_sig_t<_Sig>)>;
    };
  } // namespace __any

  template <auto... _Sigs>
  using queries = STDEXEC::__mlist<decltype(_Sigs)...>;

  template <class _Completions, auto... _ReceiverQueries>
  class any_receiver_ref {
    using __receiver_base = __any::__rec::__ref<_Completions, decltype(_ReceiverQueries)...>;
    using __env_t = STDEXEC::env_of_t<__receiver_base>;
    __receiver_base __rcvr_;

   public:
    using receiver_concept = STDEXEC::receiver_t;

    template <STDEXEC::__none_of<any_receiver_ref, const any_receiver_ref, __env_t, const __env_t>
                _Receiver>
      requires STDEXEC::receiver_of<_Receiver, _Completions>
    any_receiver_ref(_Receiver& __receiver)
      noexcept(STDEXEC::__nothrow_constructible_from<__receiver_base, _Receiver>)
      : __rcvr_(__receiver) {
    }

    template <class... _As>
      requires STDEXEC::__callable<STDEXEC::set_value_t, __receiver_base, _As...>
    void set_value(_As&&... __as) noexcept {
      STDEXEC::set_value(static_cast<__receiver_base&&>(__rcvr_), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
      requires STDEXEC::__callable<STDEXEC::set_error_t, __receiver_base, _Error>
    void set_error(_Error&& __err) noexcept {
      STDEXEC::set_error(static_cast<__receiver_base&&>(__rcvr_), static_cast<_Error&&>(__err));
    }

    void set_stopped() noexcept
      requires STDEXEC::__callable<STDEXEC::set_stopped_t, __receiver_base>
    {
      STDEXEC::set_stopped(static_cast<__receiver_base&&>(__rcvr_));
    }

    auto get_env() const noexcept -> STDEXEC::env_of_t<__receiver_base> {
      return STDEXEC::get_env(__rcvr_);
    }

    template <auto... _SenderQueries>
    class any_sender {
      using __base_t =
        __any::__sender<_Completions, queries<_SenderQueries...>, queries<_ReceiverQueries...>>;
      __base_t __sender_;

     public:
      using sender_concept = STDEXEC::sender_t;

      template <STDEXEC::__not_decays_to<any_sender> _Sender>
        requires STDEXEC::sender_to<_Sender, __receiver_base>
      any_sender(_Sender&& __sender)
        : __sender_(static_cast<_Sender&&>(__sender)) {
      }

      template <STDEXEC::__decays_to_derived_from<any_sender> _Self, class... _Env>
        requires(__any::__satisfies_receiver_query<decltype(_ReceiverQueries), _Env...> && ...)
      static consteval auto get_completion_signatures() -> __base_t::completion_signatures {
        return {};
      }

      template <STDEXEC::receiver_of<_Completions> _Receiver>
      auto connect(_Receiver __rcvr) && -> STDEXEC::connect_result_t<__base_t, _Receiver> {
        return static_cast<__base_t&&>(__sender_).connect(static_cast<_Receiver&&>(__rcvr));
      }

      auto get_env() const noexcept -> STDEXEC::env_of_t<__base_t> {
        return static_cast<const __base_t&>(__sender_).get_env();
      }

      template <auto... _SchedulerQueries>
      class any_scheduler {
        // Add the required set_value_t() completions to the schedule-sender.
        using __schedule_completions_t = STDEXEC::__concat_completion_signatures_t<
          _Completions,
          STDEXEC::completion_signatures<STDEXEC::set_value_t()>
        >;
        using __schedule_receiver_t =
          any_receiver_ref<__schedule_completions_t, _ReceiverQueries...>;

        template <class _BaseSender>
        class __schedule_sender : public _BaseSender {
          struct __attrs {
            template <class... _Env>
            STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
            constexpr auto query(
              STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>,
              const _Env&...) const noexcept {
              return __self_.__sch_;
            }

            template <STDEXEC::__forwarding_query _Tag, class... _Args>
              requires STDEXEC::__queryable_with<STDEXEC::env_of_t<_BaseSender>, _Tag, _Args...>
            constexpr auto query(_Tag __tag, _Args&&... __args) const noexcept(
              STDEXEC::__nothrow_queryable_with<STDEXEC::env_of_t<_BaseSender>, _Tag, _Args...>)
              -> STDEXEC::__query_result_t<STDEXEC::env_of_t<_BaseSender>, _Tag, _Args...> {
              return STDEXEC::get_env(static_cast<const _BaseSender&>(__self_))
                .query(__tag, static_cast<_Args&&>(__args)...);
            }

            const __schedule_sender& __self_;
          };

          friend struct __attrs;
          any_scheduler __sch_;

         public:
          __schedule_sender(any_scheduler __sch, _BaseSender&& __sender)
            : _BaseSender(static_cast<_BaseSender&&>(__sender))
            , __sch_(static_cast<any_scheduler&&>(__sch)) {
          }

          [[nodiscard]]
          constexpr auto get_env() const noexcept -> __attrs {
            return __attrs{*this};
          }

         private:
        };

        template <class... _ScheduleSenderQueries>
        using __any_sender_t =
          typename __schedule_receiver_t::template any_sender<_ScheduleSenderQueries{}...>;

        using __schedule_sender_base_t = STDEXEC::__minvoke<
          STDEXEC::__mremove_if<
            __any::__ret_equals_to<STDEXEC::get_completion_scheduler_t<STDEXEC::set_value_t>>,
            STDEXEC::__q<__any_sender_t>
          >,
          decltype(_SenderQueries)...
        >;

        using __schedule_sender_t = __schedule_sender<__schedule_sender_base_t>;

        using __scheduler_base =
          __any::__scheduler<__schedule_sender_base_t, queries<_SchedulerQueries...>>;

        __scheduler_base __scheduler_;

       public:
        using scheduler_concept = STDEXEC::scheduler_t;

        template <STDEXEC::__none_of<any_scheduler> _Scheduler>
          requires STDEXEC::scheduler<_Scheduler>
        any_scheduler(_Scheduler __scheduler)
          : __scheduler_{static_cast<_Scheduler&&>(__scheduler)} {
        }

        auto schedule() const noexcept -> __schedule_sender_t {
          return __schedule_sender_t(*this, __scheduler_.schedule());
        }

        template <class _Tag, class... _As>
          requires STDEXEC::__queryable_with<const __scheduler_base&, _Tag, _As...>
        auto query(_Tag, _As&&... __as) const
          noexcept(STDEXEC::__nothrow_queryable_with<const __scheduler_base&, _Tag, _As...>)
            -> STDEXEC::__query_result_t<const __scheduler_base&, _Tag, _As...> {
          return __scheduler_.query(_Tag{}, static_cast<_As&&>(__as)...);
        }

        auto operator==(const any_scheduler&) const noexcept -> bool = default;
      };
    };
  };
} // namespace exec
