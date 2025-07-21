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
#include "../stdexec/__detail/__any_receiver_ref.hpp"
#include "../stdexec/__detail/__transform_completion_signatures.hpp"

#include "sequence_senders.hpp"

#include <cstddef>
#include <utility>

namespace exec {
  namespace __any {
    using namespace stdexec;

    struct __create_vtable_t {
      template <class _VTable, class _Tp>
        requires __tag_invocable_r<const _VTable*, __create_vtable_t, __mtype<_VTable>, __mtype<_Tp>>
      constexpr auto operator()(__mtype<_VTable>, __mtype<_Tp>) const noexcept -> const _VTable* {
        return stdexec::tag_invoke(__create_vtable_t{}, __mtype<_VTable>{}, __mtype<_Tp>{});
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
            stdexec::get_env(*static_cast<const _EnvProvider*>(__env_provider)),
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
            stdexec::get_env(*static_cast<const _EnvProvider*>(__env_provider)),
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
        requires tag_invocable<__delete_t, __mtype<_Tp>, _Storage&>
      void operator()(__mtype<_Tp>, _Storage& __storage) noexcept {
        static_assert(nothrow_tag_invocable<__delete_t, __mtype<_Tp>, _Storage&>);
        stdexec::tag_invoke(__delete_t{}, __mtype<_Tp>{}, __storage);
      }
    };

    inline constexpr __delete_t __delete{};

    struct __copy_construct_t {
      template <class _Storage, class _Tp>
        requires tag_invocable<__copy_construct_t, __mtype<_Tp>, _Storage&, const _Storage&>
      void operator()(__mtype<_Tp>, _Storage& __self, const _Storage& __from) noexcept(
        nothrow_tag_invocable<__copy_construct_t, __mtype<_Tp>, _Storage&, const _Storage&>) {
        stdexec::tag_invoke(__copy_construct_t{}, __mtype<_Tp>{}, __self, __from);
      }
    };

    inline constexpr __copy_construct_t __copy_construct{};

    struct __move_construct_t {
      template <class _Storage, class _Tp>
        requires tag_invocable<__move_construct_t, __mtype<_Tp>, _Storage&, _Storage&&>
      void operator()(__mtype<_Tp>, _Storage& __self, __midentity<_Storage&&> __from) noexcept {
        static_assert(
          nothrow_tag_invocable<__move_construct_t, __mtype<_Tp>, _Storage&, _Storage&&>);
        stdexec::tag_invoke(
          __move_construct_t{}, __mtype<_Tp>{}, __self, static_cast<_Storage&&>(__from));
      }
    };

    inline constexpr __move_construct_t __move_construct{};

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
    struct __storage {
      class __t;
    };

    template <
      class _Vtable,
      class _Allocator,
      std::size_t _InlineSize = 3 * sizeof(void*),
      std::size_t _Alignment = alignof(std::max_align_t)
    >
    struct __immovable_storage {
      class __t : __immovable {
        static constexpr std::size_t __buffer_size = std::max(_InlineSize, sizeof(void*));
        static constexpr std::size_t __alignment = std::max(_Alignment, alignof(void*));
        using __with_delete = __delete_t(void() noexcept);
        using __vtable_t = __storage_vtable<_Vtable, __with_delete>;

        template <class _Tp>
        static constexpr bool __is_small = sizeof(_Tp) <= __buffer_size
                                        && alignof(_Tp) <= __alignment;

        template <class _Tp>
        static constexpr auto __get_vtable_of_type() noexcept -> const __vtable_t* {
          return &__storage_vtbl<__t, __decay_t<_Tp>, _Vtable, __with_delete>;
        }
       public:
        using __id = __immovable_storage;

        __t() = default;

        template <__not_decays_to<__t> _Tp>
          requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<__decay_t<_Tp>>>
        __t(_Tp&& __object)
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
        __t(std::in_place_type_t<_Tp>, _Args&&... __args)
          : __vtable_{__get_vtable_of_type<_Tp>()} {
          if constexpr (__is_small<_Tp>) {
            __construct_small<_Tp>(static_cast<_Args&&>(__args)...);
          } else {
            __construct_large<_Tp>(static_cast<_Args&&>(__args)...);
          }
        }

        ~__t() {
          __reset();
        }

        void __reset() noexcept {
          (*__vtable_)(__delete, this);
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
        template <class _Tp, class... _As>
        void __construct_small(_As&&... __args) {
          static_assert(sizeof(_Tp) <= __buffer_size && alignof(_Tp) <= __alignment);
          _Tp* __pointer = reinterpret_cast<_Tp*>(&__buffer_[0]);
          using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
          _Alloc __alloc{__allocator_};
          std::allocator_traits<_Alloc>::construct(
            __alloc, __pointer, static_cast<_As&&>(__args)...);
          __object_pointer_ = __pointer;
        }

        template <class _Tp, class... _As>
        void __construct_large(_As&&... __args) {
          using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
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
        STDEXEC_MEMFN_DECL(void __delete)(this __mtype<_Tp>, __t& __self) noexcept {
          if (!__self.__object_pointer_) {
            return;
          }
          using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
          _Alloc __alloc{__self.__allocator_};
          _Tp* __pointer = static_cast<_Tp*>(std::exchange(__self.__object_pointer_, nullptr));
          std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
          if constexpr (!__is_small<_Tp>) {
            std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
          }
        }
       private:
        const __vtable_t* __vtable_{__default_storage_vtable(static_cast<__vtable_t*>(nullptr))};
        void* __object_pointer_{nullptr};
        alignas(__alignment) std::byte __buffer_[__buffer_size]{};
        STDEXEC_ATTRIBUTE(no_unique_address) _Allocator __allocator_ { };
      };
    };

    template <
      class _Vtable,
      class _Allocator,
      bool _Copyable,
      std::size_t _InlineSize,
      std::size_t _Alignment
    >
    class __storage<_Vtable, _Allocator, _Copyable, _InlineSize, _Alignment>::__t
      : __if_c<_Copyable, __, __move_only> {
      static_assert(
        STDEXEC_IS_CONVERTIBLE_TO(typename std::allocator_traits<_Allocator>::void_pointer, void*));

      static constexpr std::size_t __buffer_size = std::max(_InlineSize, sizeof(void*));
      static constexpr std::size_t __alignment = std::max(_Alignment, alignof(void*));
      using __with_copy = __copy_construct_t(void(const __t&));
      using __with_move = __move_construct_t(void(__t&&) noexcept);
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
            __t,
            __decay_t<_Tp>,
            _Vtable,
            __with_delete,
            __with_move,
            __with_copy
          >;
        } else {
          return &__storage_vtbl<__t, __decay_t<_Tp>, _Vtable, __with_delete, __with_move>;
        }
      }

     public:
      using __id = __storage;

      __t() = default;

      template <__not_decays_to<__t> _Tp>
        requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<__decay_t<_Tp>>>
      __t(_Tp&& __object)
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
      __t(std::in_place_type_t<_Tp>, _Args&&... __args)
        : __vtable_{__get_vtable_of_type<_Tp>()} {
        if constexpr (__is_small<_Tp>) {
          __construct_small<_Tp>(static_cast<_Args&&>(__args)...);
        } else {
          __construct_large<_Tp>(static_cast<_Args&&>(__args)...);
        }
      }

      __t(const __t& __other)
        requires(_Copyable)
        : __vtable_(__other.__vtable_) {
        (*__other.__vtable_)(__copy_construct, this, __other);
      }

      auto operator=(const __t& __other) -> __t&
        requires(_Copyable)
      {
        if (&__other != this) {
          __t tmp(__other);
          *this = std::move(tmp);
        }
        return *this;
      }

      __t(__t&& __other) noexcept
        : __vtable_(__other.__vtable_) {
        (*__other.__vtable_)(__move_construct, this, static_cast<__t&&>(__other));
      }

      auto operator=(__t&& __other) noexcept -> __t& {
        __reset();
        (*__other.__vtable_)(__move_construct, this, static_cast<__t&&>(__other));
        return *this;
      }

      ~__t() {
        __reset();
      }

      void __reset() noexcept {
        (*__vtable_)(__delete, this);
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
      template <class _Tp, class... _As>
      void __construct_small(_As&&... __args) {
        static_assert(sizeof(_Tp) <= __buffer_size && alignof(_Tp) <= __alignment);
        _Tp* __pointer = reinterpret_cast<_Tp*>(&__buffer_[0]);
        using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__allocator_};
        std::allocator_traits<_Alloc>::construct(__alloc, __pointer, static_cast<_As&&>(__args)...);
        __object_pointer_ = __pointer;
      }

      template <class _Tp, class... _As>
      void __construct_large(_As&&... __args) {
        using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
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
      STDEXEC_MEMFN_DECL(void __delete)(this __mtype<_Tp>, __t& __self) noexcept {
        if (!__self.__object_pointer_) {
          return;
        }
        using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
        _Alloc __alloc{__self.__allocator_};
        _Tp* __pointer = static_cast<_Tp*>(std::exchange(__self.__object_pointer_, nullptr));
        std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
        if constexpr (!__is_small<_Tp>) {
          std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
        }
      }

      template <class _Tp>
      STDEXEC_MEMFN_DECL(
        void __move_construct)(this __mtype<_Tp>, __t& __self, __t&& __other) noexcept {
        if (!__other.__object_pointer_) {
          return;
        }
        _Tp* __pointer = static_cast<_Tp*>(std::exchange(__other.__object_pointer_, nullptr));
        if constexpr (__is_small<_Tp>) {
          _Tp& __other_object = *__pointer;
          __self.template __construct_small<_Tp>(static_cast<_Tp&&>(__other_object));
          using _Alloc = typename std::allocator_traits<_Allocator>::template rebind_alloc<_Tp>;
          _Alloc __alloc{__self.__allocator_};
          std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
        } else {
          __self.__object_pointer_ = __pointer;
        }
        __self.__vtable_ = std::exchange(
          __other.__vtable_, __default_storage_vtable(static_cast<__vtable_t*>(nullptr)));
      }

      template <class _Tp>
        requires _Copyable
      STDEXEC_MEMFN_DECL(
        void __copy_construct)(this __mtype<_Tp>, __t& __self, const __t& __other) {
        if (!__other.__object_pointer_) {
          return;
        }
        const _Tp& __other_object = *static_cast<const _Tp*>(__other.__object_pointer_);
        if constexpr (__is_small<_Tp>) {
          __self.template __construct_small<_Tp>(__other_object);
        } else {
          __self.template __construct_large<_Tp>(__other_object);
        }
        __self.__vtable_ = __other.__vtable_;
      }

      const __vtable_t* __vtable_{__default_storage_vtable(static_cast<__vtable_t*>(nullptr))};
      void* __object_pointer_{nullptr};
      alignas(__alignment) std::byte __buffer_[__buffer_size]{};
      STDEXEC_ATTRIBUTE(no_unique_address) _Allocator __allocator_ { };
    };

    struct __empty_vtable {
      template <class _Sender>
      STDEXEC_MEMFN_DECL(
        auto __create_vtable)(this __mtype<__empty_vtable>, __mtype<_Sender>) noexcept
        -> const __empty_vtable* {
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
    using __immovable_storage_t =
      __t<__immovable_storage<_VTable, _Allocator, _InlineSize, _Alignment>>;

    template <class _VTable, class _Allocator = std::allocator<std::byte>>
    using __unique_storage_t = __t<__storage<_VTable, _Allocator>>;

    template <
      class _VTable,
      std::size_t _InlineSize = 3 * sizeof(void*),
      class _Allocator = std::allocator<std::byte>
    >
    using __copyable_storage_t = __t<__storage<_VTable, _Allocator, true, _InlineSize>>;

    template <class _Tag, class... _As>
    auto __tag_type(_Tag (*)(_As...)) -> _Tag;

    template <class _Tag, class... _As>
    auto __tag_type(_Tag (*)(_As...) noexcept) -> _Tag;

    template <class _Query>
    concept __is_stop_token_query = requires {
      { __tag_type(static_cast<_Query>(nullptr)) } -> same_as<get_stop_token_t>;
    };

    template <class _Query>
    concept __is_not_stop_token_query = !__is_stop_token_query<_Query>;

    template <class _Query>
    using __is_not_stop_token_query_v = __mbool<__is_not_stop_token_query<_Query>>;

    namespace __rec {
      template <class _Sigs, class... _Queries>
      struct __vtable {
        class __t;
      };

      template <class _Sigs, class... _Queries>
      struct __ref;

      template <class... _Sigs, class... _Queries>
      struct __vtable<completion_signatures<_Sigs...>, _Queries...> {
        struct __t
          : __overload<__any_::__rcvr_vfun<_Sigs>...>
          , __query_vfun<_Queries>... {
          using __query_vfun<_Queries>::operator()...;

          template <class _Tag, class... _As>
            requires __one_of<_Tag(_As...), _Sigs...>
                  || __callable<__overload<__any_::__rcvr_vfun<_Sigs>...>, void*, _Tag, _As...>
          void operator()(void* __rcvr, _Tag, _As&&... __as) const noexcept {
            if constexpr (__one_of<_Tag(_As...), _Sigs...>) {
              const __any_::__rcvr_vfun<_Tag(_As...)>& __vfun = *this;
              __vfun(__rcvr, _Tag(), static_cast<_As&&>(__as)...);
            } else {
              const __overload<__any_::__rcvr_vfun<_Sigs>...>& __vfun = *this;
              __vfun(__rcvr, _Tag(), static_cast<_As&&>(__as)...);
            }
          }

         private:
          template <class _Rcvr>
            requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                  && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
          STDEXEC_MEMFN_DECL(auto __create_vtable)(this __mtype<__t>, __mtype<_Rcvr>) noexcept
            -> const __t* {
            static const __t __vtable_{
              {{__any_::__rcvr_vfun_fn(
                static_cast<_Rcvr*>(nullptr), static_cast<_Sigs*>(nullptr))}...},
              {__query_vfun_fn<_Rcvr>{}(static_cast<_Queries>(nullptr))}...};
            return &__vtable_;
          }
        };
      };

      template <class... _Sigs, class... _Queries>
        requires(__is_not_stop_token_query<_Queries> && ...)
      struct __ref<completion_signatures<_Sigs...>, _Queries...> {
#if !STDEXEC_MSVC()
        // MSVCBUG https://developercommunity.visualstudio.com/t/Private-member-inaccessible-when-used-in/10448363
       private:
#endif
        using __vtable_t = stdexec::__t<__vtable<completion_signatures<_Sigs...>, _Queries...>>;

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
        using receiver_concept = stdexec::receiver_t;
        using __id = __ref;
        using __t = __ref;

        template <__none_of<__ref, const __ref, __env_t, const __env_t> _Rcvr>
          requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        __ref(_Rcvr& __rcvr) noexcept
          : __env_{
              __create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}),
              &__rcvr,
              stdexec::get_stop_token(stdexec::get_env(__rcvr))} {
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

      auto __test_never_stop_token(get_stop_token_t (*)(never_stop_token (*)() noexcept))
        -> __mbool<true>;

      template <class _Tag, class _Ret, class... _As>
      auto __test_never_stop_token(_Tag (*)(_Ret (*)(_As...) noexcept)) -> __mbool<false>;

      template <class _Tag, class _Ret, class... _As>
      auto __test_never_stop_token(_Tag (*)(_Ret (*)(_As...))) -> __mbool<false>;

      template <class _Query>
      using __is_never_stop_token_query = decltype(__test_never_stop_token(
        static_cast<_Query>(nullptr)));

      template <class... _Sigs, class... _Queries>
        requires(__is_stop_token_query<_Queries> || ...)
      struct __ref<completion_signatures<_Sigs...>, _Queries...> {
#if !STDEXEC_MSVC()
        // MSVCBUG https://developercommunity.visualstudio.com/t/Private-member-inaccessible-when-used-in/10448363

       private:
#endif
        using _FilteredQueries =
          __minvoke<__mremove_if<__q<__is_never_stop_token_query>>, _Queries...>;
        using __vtable_t = stdexec::__t<
          __mapply<__mbind_front_q<__vtable, completion_signatures<_Sigs...>>, _FilteredQueries>
        >;

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
        using receiver_concept = stdexec::receiver_t;
        using __id = __ref;
        using __t = __ref;

        template <__none_of<__ref, const __ref, __env_t, const __env_t> _Rcvr>
          requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        __ref(_Rcvr& __rcvr) noexcept
          : __env_{__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}), &__rcvr} {
        }

        template <class... _As>
          requires __one_of<set_value_t(_As...), _Sigs...>
                || __callable<__overload<__any_::__rcvr_vfun<_Sigs>...>, void*, set_value_t, _As...>
        void set_value(_As&&... __as) noexcept {
          if constexpr (__one_of<set_value_t(_As...), _Sigs...>) {
            const __any_::__rcvr_vfun<set_value_t(_As...)>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_value_t(), static_cast<_As&&>(__as)...);
          } else {
            const __overload<__any_::__rcvr_vfun<_Sigs>...>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_value_t(), static_cast<_As&&>(__as)...);
          }
        }

        template <class _Error>
          requires __one_of<set_error_t(_Error), _Sigs...>
                || __callable<__overload<__any_::__rcvr_vfun<_Sigs>...>, void*, set_error_t, _Error>
        void set_error(_Error&& __err) noexcept {
          if constexpr (__one_of<set_error_t(_Error), _Sigs...>) {
            const __any_::__rcvr_vfun<set_error_t(_Error)>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_error_t(), static_cast<_Error&&>(__err));
          } else {
            const __overload<__any_::__rcvr_vfun<_Sigs>...>& __vfun = *__env_.__vtable_;
            __vfun(__env_.__rcvr_, set_error_t(), static_cast<_Error&&>(__err));
          }
        }

        void set_stopped() noexcept
          requires __one_of<set_stopped_t(), _Sigs...>
        {
          const __any_::__rcvr_vfun<set_stopped_t()>& __vfun = *__env_.__vtable_;
          __vfun(__env_.__rcvr_, set_stopped_t());
        }

        auto get_env() const noexcept -> const __env_t& {
          return __env_;
        }
      };
    } // namespace __rec

    class __operation_vtable {
     public:
      void (*__start_)(void*) noexcept;

     private:
      template <class _Op>
      STDEXEC_MEMFN_DECL(
        auto __create_vtable)(this __mtype<__operation_vtable>, __mtype<_Op>) noexcept
        -> const __operation_vtable* {
        static __operation_vtable __vtable{[](void* __object_pointer) noexcept -> void {
          STDEXEC_ASSERT(__object_pointer);
          _Op& __op = *static_cast<_Op*>(__object_pointer);
          static_assert(operation_state<_Op>);
          stdexec::start(__op);
        }};
        return &__vtable;
      }
    };

    using __immovable_operation_storage =
      __immovable_storage_t<__operation_vtable, std::allocator<std::byte>, 6 * sizeof(void*)>;

    template <class _Sigs, class _Queries>
    using __receiver_ref = __mapply<__mbind_front_q<__rec::__ref, _Sigs>, _Queries>;

    struct __on_stop_t {
      stdexec::inplace_stop_source& __source_;

      void operator()() const noexcept {
        __source_.request_stop();
      }
    };

    template <class _Receiver>
    struct __operation_base {
      STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;
      stdexec::inplace_stop_source __stop_source_{};
      using __stop_callback = typename stdexec::stop_token_of_t<
        stdexec::env_of_t<_Receiver>
      >::template callback_type<__on_stop_t>;
      std::optional<__stop_callback> __on_stop_{};
    };

    template <class _Env>
    using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _Env>;

    template <class _ReceiverId>
    struct __stoppable_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using receiver_concept = stdexec::receiver_t;
        __operation_base<_Receiver>* __op_;

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
          stdexec::set_value(
            static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
          requires __callable<set_error_t, _Receiver, _Error>
        void set_error(_Error&& __err) noexcept {
          __op_->__on_stop_.reset();
          stdexec::set_error(
            static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept
          requires __callable<set_stopped_t, _Receiver>
        {
          __op_->__on_stop_.reset();
          stdexec::set_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
        }

        auto get_env() const noexcept -> __env_t<env_of_t<_Receiver>> {
          return __env::__join(
            prop{get_stop_token, __op_->__stop_source_.get_token()},
            stdexec::get_env(__op_->__rcvr_));
        }
      };
    };

    template <class _ReceiverId>
    using __stoppable_receiver_t = stdexec::__t<__stoppable_receiver<_ReceiverId>>;

    template <class _ReceiverId, bool>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t : public __operation_base<_Receiver> {
       public:
        using __id = __operation;

        template <class _Sender>
        __t(_Sender&& __sender, _Receiver&& __receiver)
          : __operation_base<_Receiver>{static_cast<_Receiver&&>(__receiver)}
          , __rec_{this}
          , __storage_{__sender.__connect(__rec_)} {
        }

        void start() & noexcept {
          this->__on_stop_.emplace(
            stdexec::get_stop_token(stdexec::get_env(this->__rcvr_)),
            __on_stop_t{this->__stop_source_});
          STDEXEC_ASSERT(__storage_.__get_vtable()->__start_);
          __storage_.__get_vtable()->__start_(__storage_.__get_object_pointer());
        }

       private:
        __stoppable_receiver_t<_ReceiverId> __rec_;
        __immovable_operation_storage __storage_{};
      };
    };

    template <class _ReceiverId>
    struct __operation<_ReceiverId, false> {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        using __id = __operation;

        template <class _Sender>
        __t(_Sender&& __sender, _Receiver&& __receiver)
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
    };

    template <class _Queries, bool _IsEnvProvider = true>
    class __query_vtable;

    template <template <class...> class _List, class... _Queries, bool _IsEnvProvider>
    class __query_vtable<_List<_Queries...>, _IsEnvProvider> : public __query_vfun<_Queries>... {
     public:
      using __query_vfun<_Queries>::operator()...;
     private:
      template <class _Queryable>
        requires(__callable<__query_vfun_fn<_Queryable, _IsEnvProvider>, _Queries> && ...)
      STDEXEC_MEMFN_DECL(
        auto __create_vtable)(this __mtype<__query_vtable>, __mtype<_Queryable>) noexcept
        -> const __query_vtable* {
        static const __query_vtable __vtable{
          {__query_vfun_fn<_Queryable, _IsEnvProvider>{}(static_cast<_Queries>(nullptr))}...};
        return &__vtable;
      }
    };

    template <class _Sigs, class _SenderQueries = __types<>, class _ReceiverQueries = __types<>>
    struct __sender {
      using __receiver_ref_t = __receiver_ref<_Sigs, _ReceiverQueries>;
      static constexpr bool __with_inplace_stop_token =
        __v<__mapply<__mall_of<__q<__is_not_stop_token_query_v>>, _ReceiverQueries>>;

      class __vtable : public __query_vtable<_SenderQueries> {
       public:
        using __id = __vtable;

        auto __queries() const noexcept -> const __query_vtable<_SenderQueries>& {
          return *this;
        }

        __immovable_operation_storage (*__connect_)(void*, __receiver_ref_t);
       private:
        template <sender_to<__receiver_ref_t> _Sender>
        STDEXEC_MEMFN_DECL(auto __create_vtable)(this __mtype<__vtable>, __mtype<_Sender>) noexcept
          -> const __vtable* {
          static const __vtable __vtable_{
            {*__create_vtable(__mtype<__query_vtable<_SenderQueries>>{}, __mtype<_Sender>{})},
            [](void* __object_pointer, __receiver_ref_t __receiver)
              -> __immovable_operation_storage {
              _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
              using __op_state_t = connect_result_t<_Sender, __receiver_ref_t>;
              return __immovable_operation_storage{
                std::in_place_type<__op_state_t>, __emplace_from{[&] {
                  return stdexec::connect(
                    static_cast<_Sender&&>(__sender), static_cast<__receiver_ref_t&&>(__receiver));
                }}};
            }};
          return &__vtable_;
        }
      };

      struct __env_t {
        const __vtable* __vtable_;
        void* __sender_;

        template <class _Tag, class... _As>
          requires __callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>
        auto query(_Tag, _As&&... __as) const
          noexcept(__nothrow_callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>)
            -> __call_result_t<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...> {
          return __vtable_->__queries()(_Tag{}, __sender_, static_cast<_As&&>(__as)...);
        }
      };

      struct __t {
        using __id = __sender;
        using completion_signatures = _Sigs;
        using sender_concept = stdexec::sender_t;

        __t(const __t&) = delete;
        auto operator=(const __t&) -> __t& = delete;

        __t(__t&&) = default;
        auto operator=(__t&&) -> __t& = default;

        template <__not_decays_to<__t> _Sender>
          requires sender_to<_Sender, __receiver_ref<_Sigs, _ReceiverQueries>>
        __t(_Sender&& __sndr)
          : __storage_{static_cast<_Sender&&>(__sndr)} {
        }

        auto __connect(__receiver_ref_t __receiver) -> __immovable_operation_storage {
          return __storage_.__get_vtable()->__connect_(
            __storage_.__get_object_pointer(), static_cast<__receiver_ref_t&&>(__receiver));
        }

        explicit operator bool() const noexcept {
          return __get_object_pointer(__storage_) != nullptr;
        }

        auto get_env() const noexcept -> __env_t {
          return {__storage_.__get_vtable(), __storage_.__get_object_pointer()};
        }

        template <receiver_of<_Sigs> _Rcvr>
        auto connect(_Rcvr __rcvr) && -> stdexec::__t<
          __operation<stdexec::__id<_Rcvr>, __with_inplace_stop_token>
        > {
          return {static_cast<__t&&>(*this), static_cast<_Rcvr&&>(__rcvr)};
        }

       private:
        __unique_storage_t<__vtable> __storage_;
      };
    };

    template <class _ScheduleSender, class _SchedulerQueries = __types<>>
    class __scheduler {
      static constexpr std::size_t __buffer_size = 4 * sizeof(void*);
      template <class _Ty>
      static constexpr bool __is_small = sizeof(_Ty) <= __buffer_size
                                      && alignof(_Ty) <= alignof(std::max_align_t);

     public:
      template <class _Scheduler>
        requires(!__decays_to<_Scheduler, __scheduler>) && scheduler<_Scheduler>
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
       private:
        template <scheduler _Scheduler>
        STDEXEC_MEMFN_DECL(
          auto __create_vtable)(this __mtype<__vtable>, __mtype<_Scheduler>) noexcept
          -> const __vtable* {
          static const __vtable __vtable_{
            {*__create_vtable(
              __mtype<__query_vtable<_SchedulerQueries, false>>{}, __mtype<_Scheduler>{})},
            [](void* __object_pointer) noexcept -> __sender_t {
              const _Scheduler& __scheduler = *static_cast<const _Scheduler*>(__object_pointer);
              return __sender_t{stdexec::schedule(__scheduler)};
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
  } // namespace __any

  template <auto... _Sigs>
  using queries = stdexec::__types<decltype(_Sigs)...>;

  template <class _Completions, auto... _ReceiverQueries>
  class any_receiver_ref {
    using __receiver_base = __any::__rec::__ref<_Completions, decltype(_ReceiverQueries)...>;
    using __env_t = stdexec::env_of_t<__receiver_base>;
    __receiver_base __receiver_;

   public:
    using receiver_concept = stdexec::receiver_t;
    using __t = any_receiver_ref;
    using __id = any_receiver_ref;

    template <stdexec::__none_of<any_receiver_ref, const any_receiver_ref, __env_t, const __env_t>
                _Receiver>
      requires stdexec::receiver_of<_Receiver, _Completions>
    any_receiver_ref(_Receiver& __receiver)
      noexcept(stdexec::__nothrow_constructible_from<__receiver_base, _Receiver>)
      : __receiver_(__receiver) {
    }

    template <class... _As>
      requires stdexec::__callable<stdexec::set_value_t, __receiver_base, _As...>
    void set_value(_As&&... __as) noexcept {
      stdexec::set_value(static_cast<__receiver_base&&>(__receiver_), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
      requires stdexec::__callable<stdexec::set_error_t, __receiver_base, _Error>
    void set_error(_Error&& __err) noexcept {
      stdexec::set_error(static_cast<__receiver_base&&>(__receiver_), static_cast<_Error&&>(__err));
    }

    void set_stopped() noexcept
      requires stdexec::__callable<stdexec::set_stopped_t, __receiver_base>
    {
      stdexec::set_stopped(static_cast<__receiver_base&&>(__receiver_));
    }

    auto get_env() const noexcept -> stdexec::env_of_t<__receiver_base> {
      return stdexec::get_env(__receiver_);
    }

    template <auto... _SenderQueries>
    class any_sender {
      using __sender_base = stdexec::__t<
        __any::__sender<_Completions, queries<_SenderQueries...>, queries<_ReceiverQueries...>>
      >;
      __sender_base __sender_;

     public:
      using sender_concept = stdexec::sender_t;
      using completion_signatures = typename __sender_base::completion_signatures;
      using __t = any_sender;
      using __id = any_sender;

      template <stdexec::__not_decays_to<any_sender> _Sender>
        requires stdexec::sender_to<_Sender, __receiver_base>
      any_sender(_Sender&& __sender)
        noexcept(stdexec::__nothrow_constructible_from<__sender_base, _Sender>)
        : __sender_(static_cast<_Sender&&>(__sender)) {
      }

      template <stdexec::receiver_of<_Completions> _Receiver>
      auto connect(_Receiver __rcvr) && -> stdexec::connect_result_t<__sender_base, _Receiver> {
        return static_cast<__sender_base&&>(__sender_).connect(static_cast<_Receiver&&>(__rcvr));
      }

      auto get_env() const noexcept -> stdexec::env_of_t<__sender_base> {
        return static_cast<const __sender_base&>(__sender_).get_env();
      }

      template <auto... _SchedulerQueries>
      class any_scheduler {
        // Add the required set_value_t() completions to the schedule-sender.
        using __schedule_completions = stdexec::__concat_completion_signatures<
          _Completions,
          stdexec::completion_signatures<stdexec::set_value_t()>
        >;
        using __schedule_receiver = any_receiver_ref<__schedule_completions, _ReceiverQueries...>;

        template <typename _Tag, typename _Sig>
        static auto __ret_fn(_Tag (*const)(_Sig)) -> _Tag;

        template <class _Tag>
        struct __ret_equals_to {
          template <class _Sig>
          using __f =
            stdexec::__mbool<STDEXEC_IS_SAME(_Tag, decltype(__ret_fn(static_cast<_Sig>(nullptr))))>;
        };

        using __schedule_sender_queries = stdexec::__minvoke<
          stdexec::__mremove_if<
            __ret_equals_to<stdexec::get_completion_scheduler_t<stdexec::set_value_t>>
          >,
          decltype(_SenderQueries)...
        >;

#if STDEXEC_MSVC()
        // MSVCBUG https://developercommunity.visualstudio.com/t/ICE-and-non-ICE-bug-in-NTTP-argument-w/10361081

        static constexpr auto __any_scheduler_noexcept_signature =
          stdexec::get_completion_scheduler<stdexec::set_value_t>.signature<any_scheduler() noexcept>;
        template <class... _Queries>
        using __schedule_sender_fn =
          typename __schedule_receiver::template any_sender<__any_scheduler_noexcept_signature>;
#else
        template <class... _Queries>
        using __schedule_sender_fn = typename __schedule_receiver::template any_sender<
          stdexec::get_completion_scheduler<stdexec::set_value_t>.template signature<any_scheduler() noexcept>
        >;
#endif
        using __schedule_sender =
          stdexec::__mapply<stdexec::__q<__schedule_sender_fn>, __schedule_sender_queries>;

        using __scheduler_base =
          __any::__scheduler<__schedule_sender, queries<_SchedulerQueries...>>;

        __scheduler_base __scheduler_;

       public:
        using scheduler_concept = stdexec::scheduler_t;
        using __t = any_scheduler;
        using __id = any_scheduler;

        template <stdexec::__none_of<any_scheduler> _Scheduler>
          requires stdexec::scheduler<_Scheduler>
        any_scheduler(_Scheduler __scheduler)
          : __scheduler_{static_cast<_Scheduler&&>(__scheduler)} {
        }

        auto schedule() const noexcept -> __schedule_sender {
          return __scheduler_.schedule();
        }

        template <class _Tag, class... _As>
          requires stdexec::__env::__queryable<const __scheduler_base&, _Tag, _As...>
        auto query(_Tag, _As&&... __as) const
          noexcept(stdexec::__env::__nothrow_queryable<const __scheduler_base&, _Tag, _As...>)
            -> stdexec::__env::__query_result_t<const __scheduler_base&, _Tag, _As...> {
          return __scheduler_.query(_Tag{}, static_cast<_As&&>(__as)...);
        }

        auto operator==(const any_scheduler&) const noexcept -> bool = default;
      };
    };
  };
} // namespace exec
