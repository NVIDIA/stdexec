/* Copyright (C) 2023 Maikel Nadolski
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

#include <stdexec/execution.hpp>

namespace exec {
  namespace __any {
    using namespace stdexec;

    struct __create_vtable_t {
      template <class _VTable, class _T>
          requires __tag_invocable_r<const _VTable*, __create_vtable_t, __mtype<_VTable>, __mtype<_T>> 
        constexpr const _VTable* operator()(__mtype<_VTable>, __mtype<_T>) const noexcept 
        {
          return tag_invoke(__create_vtable_t{}, __mtype<_VTable>{}, __mtype<_T>{});
        }
    };

    inline constexpr __create_vtable_t __create_vtable{};

    template <class _Sig>
      struct __query_vfun;

    template <class _Tag, class _Ret, class... _As>
      struct __query_vfun<_Tag(_Ret(*)(_As...))> {
        _Ret (*__fn_)(void*, _As...);
        _Ret operator()(_Tag, void* __rcvr, _As&&... __as) const {
          return __fn_(__rcvr, (_As&&) __as...);
        }
      };

    template <class _Tag, class _Ret, class... _As>
      struct __query_vfun<_Tag(_Ret(*)(_As...) noexcept)> {
        _Ret (*__fn_)(void*, _As...) noexcept;
        _Ret operator()(_Tag, void* __rcvr, _As&&... __as) const noexcept {
          return __fn_(__rcvr, (_As&&) __as...);
        }
      };

    template <class _EnvProvider>
      struct __query_vfun_fn {
        template <class _Tag, class _Ret, class... _As>
            requires __callable<_Tag, env_of_t<const _EnvProvider&>, _As...>
          constexpr _Ret (*operator()(_Tag(*)(_Ret(*)(_As...))) const noexcept)(void*, _As...) {
          return +[](void* __env_provider, _As... __as) -> _Ret {
            return _Tag{}(get_env(*(const _EnvProvider*) __env_provider), (_As&&) __as...);
          };
        }
        template <class _Tag, class _Ret, class... _As>
            requires __callable<_Tag, env_of_t<const _EnvProvider&>, _As...>
          constexpr _Ret (*operator()(_Tag(*)(_Ret(*)(_As...) noexcept)) const noexcept)(void*, _As...) noexcept {
          return +[](void* __env_provider, _As... __as) noexcept -> _Ret {
            static_assert(__nothrow_callable<_Tag, const _EnvProvider&, _As...>);
            return _Tag{}(get_env(*(const _EnvProvider*) __env_provider), (_As&&) __as...);
          };
        }
      };

    
    template <class _Sig>
      struct __storage_vfun;

    template <class _Tag, class... _As>
      struct __storage_vfun<_Tag(void(*)(_As...))> {
        void (*__fn_)(void*, _As...) = [](void*, _As...) {};
        void operator()(_Tag, void* __storage, _As&&... __as) const {
          return __fn_(__storage, (_As&&) __as...);
        }
      };

    template <class _Tag, class... _As>
      struct __storage_vfun<_Tag(void(*)(_As...) noexcept)> {
        void (*__fn_)(void*, _As...) noexcept = [](void*, _As...) noexcept {};
        void operator()(_Tag, void* __storage, _As&&... __as) const noexcept {
          return __fn_(__storage, (_As&&) __as...);
        }
      };

    template <class _Storage, class _T>
      struct __storage_vfun_fn {
        template <class _Tag, class... _As>
            requires __callable<_Tag, __mtype<_T>, _Storage&, _As...>
          constexpr void (*operator()(_Tag(*)(void(*)(_As...))) const noexcept)(void*, _As...) {
          return +[](void* __storage, _As... __as) -> void {
            return _Tag{}(__mtype<_T>{}, *(_Storage*) __storage, (_As&&) __as...);
          };
        }
        template <class _Tag, class... _As>
            requires __callable<_Tag, __mtype<_T>, _Storage&, _As...>
          constexpr void (*operator()(_Tag(*)(void(*)(_As...) noexcept)) const noexcept)(void*, _As...) noexcept {
          return +[](void* __storage, _As... __as) noexcept -> void {
            static_assert(__nothrow_callable<_Tag, __mtype<_T>, _Storage&, _As...>);
            return _Tag{}(__mtype<_T>{}, *(_Storage*) __storage, (_As&&) __as...);
          };
        }
      };

    struct __get_vtable_t {
      template <class _Storage>
          requires tag_invocable<__get_vtable_t, const _Storage&>
        const auto operator()(const _Storage& __storage) const noexcept {
          return tag_invoke(__get_vtable_t{}, __storage);
        }
    };
    inline constexpr __get_vtable_t __get_vtable{};

    struct __get_object_pointer_t {
      template <class _Storage>
          requires __tag_invocable_r<void*, __get_object_pointer_t, const _Storage&>
        void* operator()(const _Storage& __storage) const noexcept {
          return tag_invoke(__get_object_pointer_t{}, __storage);
        }
    };
    inline constexpr __get_object_pointer_t __get_object_pointer{};

    struct __delete_t {
      template <class _Storage, class _T>
          requires tag_invocable<__delete_t, __mtype<_T>, _Storage&>
        void operator()(__mtype<_T>, _Storage& __storage) noexcept {
          tag_invoke(__delete_t{}, __mtype<_T>{}, __storage);
        }
    };
    inline constexpr __delete_t __delete{};

    struct __copy_construct_t {
      template <class _Storage, class _T>
          requires tag_invocable<__copy_construct_t, __mtype<_T>, _Storage&,  const _Storage&>
        void operator()(__mtype<_T>, _Storage& __self, const _Storage& __from) {
          tag_invoke(__copy_construct_t{}, __mtype<_T>{}, __self, __from);
        }
    };
    inline constexpr __copy_construct_t __copy_construct{};

    struct __move_construct_t {
      template <class _Storage, class _T>
          requires tag_invocable<__move_construct_t, __mtype<_T>, _Storage&,  _Storage&&>
        void operator()(__mtype<_T>, _Storage& __self, __midentity<_Storage&&> __from) noexcept {
          tag_invoke(__move_construct_t{}, __mtype<_T>{}, __self, (_Storage&&) __from);
        }
    };
    inline constexpr __move_construct_t __move_construct{};

    template <class _ParentVTable, class... _StorageCPOs>
      struct __storage_vtable;

    template <class _ParentVTable, class... _StorageCPOs>
        requires requires (_ParentVTable pv) { pv(); }
      struct __storage_vtable<_ParentVTable, _StorageCPOs...>
      : _ParentVTable, __storage_vfun<_StorageCPOs>... {
        using _ParentVTable::operator();
        using __storage_vfun<_StorageCPOs>::operator()...;
      };

    template <class _ParentVTable, class... _StorageCPOs>
        requires (!(requires (_ParentVTable pv) { pv(); }))
      struct __storage_vtable<_ParentVTable, _StorageCPOs...>
      : _ParentVTable, __storage_vfun<_StorageCPOs>... {
        using __storage_vfun<_StorageCPOs>::operator()...;
      };

    template <class _ParentVTable, class... _StorageCPOs>
      inline constexpr __storage_vtable<_ParentVTable, _StorageCPOs...> __null_storage_vtbl {};

    template <class _ParentVTable, class... _StorageCPOs>
      constexpr const __storage_vtable<_ParentVTable, _StorageCPOs...>*
        __default_storage_vtable(__storage_vtable<_ParentVTable, _StorageCPOs...>*) noexcept {
          return &__null_storage_vtbl<_ParentVTable, _StorageCPOs...>;
        }

    template <class _Storage, class _T, class _ParentVTable, class... _StorageCPOs>
      static const __storage_vtable<_ParentVTable, _StorageCPOs...>
        __storage_vtbl {
          {*__create_vtable(__mtype<_ParentVTable>{}, __mtype<_T>{})},
          {__storage_vfun_fn<_Storage, _T>{}((_StorageCPOs*) nullptr)}...
        };

    template <class _Allocator,
              bool _Copyable = false,
              std::size_t _Alignment = alignof(std::max_align_t), 
              std::size_t _InlineSize = 3*sizeof(void*)>
    struct __basic_storage {
      template <class _Vtable>
        struct __storage {
          class __t;
        };
    };

    template <class _Allocator = std::allocator<std::byte>>
      using __unique_storage = __basic_storage<_Allocator>;

    template <class _Allocator = std::allocator<std::byte>>
      using __copyable_storage = __basic_storage<_Allocator, true>;

    template <class _Allocator, bool _Copyable, std::size_t _Alignment, std::size_t _InlineSize>
    template <class _Vtable> 
      class __basic_storage<_Allocator, _Copyable, _Alignment, _InlineSize>::__storage<_Vtable>::__t {
        static constexpr std::size_t __buffer_size = std::max(_InlineSize, sizeof(void*));
        static constexpr std::size_t __alignment = std::max(_Alignment, alignof(void*));
        using __with_copy = __copy_construct_t(void(const __t&));
        using __with_move = __move_construct_t(void(__t&&) noexcept);
        using __with_delete = __delete_t(void() noexcept);

        template <class _T>
          static constexpr bool __is_small = sizeof(_T) <= __buffer_size 
                                             && alignof(_T) <= __alignment 
                                             && std::is_nothrow_move_constructible_v<_T>;

        using __vtable_t = __if_c<_Copyable, 
            __storage_vtable<_Vtable, __with_delete, __with_move, __with_copy>, 
            __storage_vtable<_Vtable, __with_delete, __with_move>>;

        template <class _T>
          static constexpr const __vtable_t* __get_vtable() noexcept {
            if constexpr (_Copyable) {
              return &__storage_vtbl<__t, decay_t<_T>, _Vtable, __with_delete, __with_move, __with_copy>;
            } else {
              return &__storage_vtbl<__t, decay_t<_T>, _Vtable, __with_delete, __with_move>;
            }
          }

       public:
        using __id = __basic_storage;

        __t() = default;

        template <__none_of<__t&, const __t&> _T>
            requires __callable<__create_vtable_t, __mtype<_Vtable>, __mtype<std::decay_t<_T>>>
          __t(_T&& __object)
          : __vtable_{__get_vtable<_T>()}
          {
            using _D = decay_t<_T>;
            if constexpr (__is_small<_D>) {
              __construct_small<_D>((_T&&) __object);
            } else {
              __construct_large<_D>((_T&&) __object);
            }
          }

        template <class _T, class... _Args>
            requires (__callable<__create_vtable_t, __mtype<_Vtable>, __mtype<_T>>)
          __t(std::in_place_type_t<_T>, _Args&&... __args)
          : __vtable_{__get_vtable<_T>()}
          {
            if constexpr (__is_small<_T>) {
              __construct_small<_T>((_Args&&) __args...);
            } else {
              __construct_large<_T>((_Args&&) __args...);
            }
          }

        __t(const __t&) requires (!_Copyable) = delete;
        __t& operator=(const __t&) requires (!_Copyable) = delete;

        __t(const __t& __other) requires (_Copyable) {
          (*__other.__vtable_)(__copy_construct, this, __other);
        }

        __t& operator=(const __t& __other) requires (_Copyable) {
          (*__other.__vtable_)(__copy_construct, this, __other);
          return *this;
        }

        __t(__t&& __other) noexcept
        {
          (*__other.__vtable_)(__move_construct, this, (__t&&) __other);
        }

        __t& operator=(__t&& __other) noexcept {
          (*__other.__vtable_)(__move_construct, this, (__t&&) __other);
          return *this;
        }

        ~__t() {
          __reset();
        }

        void __reset() noexcept {
          (*__vtable_)(__delete, this);
          __object_pointer_ = nullptr;
          __vtable_ = __default_storage_vtable((__vtable_t*) nullptr);
        }

       private:
        template <class _T, class... _As>
          void __construct_small(_As&&... __args) {
            static_assert(sizeof(_T) <= __buffer_size && alignof(_T) <= __alignment);
            _T* __pointer = static_cast<_T*>(static_cast<void*>(&__buffer_[0]));
            using _Alloc = typename  std::allocator_traits<_Allocator>::template rebind_alloc<_T>;
            _Alloc __alloc{__allocator_};
            std::allocator_traits<_Alloc>::construct(__alloc, __pointer, (_As&&) __args...);
            __object_pointer_ = __pointer;
          }

        template <class _T, class... _As>
          void __construct_large(_As&&... __args) {
            using _Alloc = typename  std::allocator_traits<_Allocator>::template rebind_alloc<_T>;
            _Alloc __alloc{__allocator_};
            _T* __pointer = std::allocator_traits<_Alloc>::allocate(__alloc, 1);
            try {
              std::allocator_traits<_Alloc>::construct(__alloc, __pointer, (_As&&) __args...);
            } catch (...) {
              std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
              throw;
            }
            __object_pointer_ = __pointer;
          }

        friend const _Vtable* tag_invoke(__get_vtable_t, const __t& __self) noexcept {
          return __self.__vtable_;
        }

        friend void* tag_invoke(__get_object_pointer_t, const __t& __self) noexcept {
          return __self.__object_pointer_;
        }

        template <class _T>
          friend void tag_invoke(__delete_t, __mtype<_T>, __t& __self) noexcept {
            if (!__self.__object_pointer_) { 
              return;
            }
            using _Alloc = typename  std::allocator_traits<_Allocator>::template rebind_alloc<_T>;
            _Alloc __alloc{__self.__allocator_};
            _T* __pointer = static_cast<_T*>(std::exchange(__self.__object_pointer_, nullptr));
            std::allocator_traits<_Alloc>::destroy(__alloc, __pointer);
            if (!__is_small<_T>) {
              std::allocator_traits<_Alloc>::deallocate(__alloc, __pointer, 1);
            }
          }

          template <class _T>
            friend void tag_invoke(__move_construct_t, __mtype<_T>, __t& __self, __t&& __other) noexcept {
              __self.__reset();
              if (!__other.__object_pointer_) {
                return;
              }
              _T* __pointer = static_cast<_T*>(std::exchange(__other.__object_pointer_, nullptr));
              if constexpr (__is_small<_T>) {
                _T& __other_object = *__pointer; 
                __self.template __construct_small<_T>((_T&&)__other_object);
              } else {
                __self.__object_pointer_ = __pointer;
              }
              __self.__vtable_ = std::exchange(__other.__vtable_, __default_storage_vtable((__vtable_t*) nullptr));
            }

        template <class _T>
            requires _Copyable
          friend void tag_invoke(__copy_construct_t, __mtype<_T>, __t& __self, const __t& __other) {
            __self.__reset();
            if (!__other.__object_pointer_) {
              return;
            }
            const _T& __other_object = *static_cast<const _T*>(__other.__object_pointer_); 
            if constexpr (__is_small<_T>) {
              __self.template __construct_small<_T>(__other_object);
            } else {
              __self.template __construct_large<_T>(__other_object);
            }
            __self.__vtable_ = __other.__vtable_;
          }

        const __vtable_t* __vtable_{__default_storage_vtable((__vtable_t*) nullptr)};
        void* __object_pointer_{nullptr};
        alignas(_Alignment) std::byte __buffer_[_InlineSize]{};
        [[no_unique_address]] _Allocator __allocator_{};
      };

    template <class _Storage, class _VTable>
      using __storage_t = typename _Storage::template __storage<_VTable>::__t;

    namespace __rec {
      template <class _Sig>
        struct __rcvr_vfun;

      template <class _Sigs, class... _Queries>
        struct __vtable { class __t; };

      template <class _Sigs, class... _Queries>
        struct __ref;

      template <class _Tag, class... _As>
        struct __rcvr_vfun<_Tag(_As...)> {
          void (*__fn_)(void*, _As...) noexcept;
        };


      template <class _Rcvr>
        struct __rcvr_vfun_fn {
          template <class _Tag, class... _As>
            constexpr void (*operator()(_Tag(*)(_As...)) const noexcept)(void*, _As...) noexcept {
            return +[](void* __rcvr, _As... __as) noexcept -> void {
              _Tag{}((_Rcvr&&) *(_Rcvr*) __rcvr, (_As&&) __as...);
            };
          }
        };

      template <class _Rcvr, class _Sigs, class... _Queries>
        constexpr const __t<__vtable<_Sigs, _Queries...>>* __vtbl_() noexcept;

      template <class... _Sigs, class... _Queries>
        struct __vtable<completion_signatures<_Sigs...>, _Queries...> {
          class __t : public __rcvr_vfun<_Sigs>...
                    , public __query_vfun<_Queries>... {
           public:
            using __query_vfun<_Queries>::operator()...;

           private:
            template <class _Rcvr>
                requires receiver_of<_Rcvr, completion_signatures<_Sigs...>> &&
                      (__callable<__query_vfun_fn<_Rcvr>, _Queries*> &&...)
              friend constexpr const __t*
              tag_invoke(__create_vtable_t, __mtype<__t>, __mtype<_Rcvr>) noexcept {
                return __vtbl_<_Rcvr, completion_signatures<_Sigs...>, _Queries...>();
              }
          };
        };

      template <class _Rcvr, class _Sigs, class... _Queries>
        extern __t<__vtable<_Sigs, _Queries...>> __vtbl;

      template <class _Rcvr, class... _Sigs, class... _Queries>
        inline constexpr __t<__vtable<completion_signatures<_Sigs...>, _Queries...>>
          __vtbl<_Rcvr, completion_signatures<_Sigs...>, _Queries...> {
            {__rcvr_vfun_fn<_Rcvr>{}((_Sigs*) nullptr)}...,
            {__query_vfun_fn<_Rcvr>{}((_Queries*) nullptr)}...
          };

      template <class _Rcvr, class _Sigs, class... _Queries>
        constexpr const __t<__vtable<_Sigs, _Queries...>>* __vtbl_() noexcept {
          return &__vtbl<_Rcvr, _Sigs, _Queries...>;
        }

      template <class... _Sigs, class... _Queries>
        struct __ref<completion_signatures<_Sigs...>, _Queries...> {
         private:
          using __vtable_t = __t<__vtable<completion_signatures<_Sigs...>, _Queries...>>;
          struct __env_t {
            const __vtable_t* __vtable_;
            void* __rcvr_;

            template <class _Tag, class... _As>
                requires __callable<const __vtable_t&, _Tag, void*, _As...>
              friend auto tag_invoke(_Tag, const __env_t& __self, _As&&... __as)
                noexcept(__nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
                -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
                return (*__self.__vtable_)(_Tag{}, __self.__rcvr_, (_As&&) __as...);
              }
          } __env_;
         public:
          template <__none_of<__ref, const __ref, __env_t, const __env_t> _Rcvr>
              requires receiver_of<_Rcvr, completion_signatures<_Sigs...>> &&
                (__callable<__query_vfun_fn<_Rcvr>, _Queries*> &&...)
            __ref(_Rcvr& __rcvr) noexcept
              : __env_{&__vtbl<_Rcvr, completion_signatures<_Sigs...>, _Queries...>, &__rcvr}
            {}
          template <__one_of<set_value_t, set_error_t, set_stopped_t> _Tag, class... _As>
              requires __one_of<_Tag(_As...), _Sigs...>
            friend void tag_invoke(_Tag, __ref&& __self, _As&&... __as) noexcept {
              (*static_cast<const __rcvr_vfun<_Tag(_As...)>*>(__self.__env_.__vtable_)->__fn_)(
                __self.__env_.__rcvr_,
                (_As&&) __as...);
            }
          friend const __env_t& tag_invoke(get_env_t, const __ref& __self) noexcept {
            return __self.__env_;
          }
        };
    } // __rec

    namespace __sender {
      struct __operation_vtable;
      
      template <class _Op>
        constexpr const __operation_vtable* __op_vtbl_();

      struct __operation_vtable {
        void (*__start_)(void*) noexcept;
        void operator()(start_t, void* __op) const noexcept {
          __start_(__op);
        }
       private:
        template <class _Op>
          friend constexpr const __operation_vtable*
          tag_invoke(__create_vtable_t, __mtype<__operation_vtable>, __mtype<_Op>) noexcept {
            return __op_vtbl_<_Op>();
          }
      };

      template <class _Op>
      inline constexpr __operation_vtable __op_vtbl{[](void* __object_pointer) noexcept -> void {
        _Op& __op = *static_cast<_Op*>(__object_pointer);
        static_assert(operation_state<_Op>);
        start(__op);
      }};

      template <class _Op>
        constexpr const __operation_vtable* __op_vtbl_() {
          return &__op_vtbl<_Op>;
        }

      using __unique_operation_storage = __storage_t<__unique_storage<>, __operation_vtable>;

      template <class _Sigs, class _Queries>
        using __receiver_ref = __mapply<__mbind_front<__q<__rec::__ref>, _Sigs>, _Queries>;

      template <class _Sender, class _Receiver, class _Queries>
        class __operation : __immovable {
          using _Sigs = completion_signatures_of_t<_Sender>;
          using __receiver_ref_t = __receiver_ref<_Sigs, _Queries>;

         public:
          __operation(_Sender&& sender, _Receiver&& receiver)
          : __receiver_{(_Receiver&&) receiver}
          , __storage_{connect((_Sender&&) sender, __receiver_ref_t{*this})}
          {
          }

         private:
          [[no_unique_address]] _Receiver __receiver_;
          __unique_operation_storage __storage_{};

          template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO, 
                    __decays_to<__operation> _Self, class... _Args>
            void take_invoke(_CPO, _Self&& __self, _Args&&... __args) noexcept {
              _CPO{}((_Receiver&&) __self.__receiver_, (_Args&&) __args...);
            }

          friend env_of_t<_Receiver> tag_invoke(get_env_t, const __operation& __self) noexcept {
            return get_env(__self.__receiver_);
          }

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            (*__get_vtable(__self.__storage_)->__start_)(__get_object_pointer(__self.__storage_));
          }
        };

      template <class _Sender, class _ReceiverQueries>
      struct __sender_vtable_fn {
        using _Sigs = completion_signatures_of_t<_Sender>;
        using __receiver_ref_t = __receiver_ref<_Sigs, _ReceiverQueries>;
        constexpr __unique_operation_storage
        (*operator()() const noexcept)(void*, __receiver_ref_t) {
          return +[](void* __object_pointer, __receiver_ref_t __receiver) -> __unique_operation_storage {
            _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
            static_assert(receiver_of<__receiver_ref_t, _Sigs>);
            using __op_state_t = connect_result_t<_Sender, __receiver_ref_t>;
            return __unique_operation_storage{std::in_place_type<__op_state_t>, 
                __conv{[&] { return connect((_Sender&&)__sender, (__receiver_ref_t&&)__receiver); }}};
          };
        }
      };

      template <class _Sigs, class _Queries>
      struct __sender_vtable;

      template <class _Sender, class _Queries>
        constexpr const __sender_vtable<completion_signatures_of_t<_Sender>, _Queries>*
        __sender_vtbl_();

      template <class _Sigs, class _Queries>
        struct __sender_vtable {
          using __receiver_ref_t = __receiver_ref<_Sigs, _Queries>;
          __unique_operation_storage (*__connect_)(void*, __receiver_ref_t);
          __unique_operation_storage operator()(connect_t, void* __sender, __receiver_ref_t __receiver) const {
            return __connect_(__sender, __receiver);
          }
         private:
          template <class _Sender>
            friend constexpr const __sender_vtable<completion_signatures_of_t<_Sender>, _Queries>*
            tag_invoke(__create_vtable_t, __mtype<__sender_vtable<_Sigs, _Queries>>, __mtype<_Sender>) noexcept {
              return __sender_vtbl_<_Sender, _Queries>();
            }
        };

      template <class _Sender, class _Queries>
        inline constexpr __sender_vtable<completion_signatures_of_t<_Sender>, _Queries>
        __sender_vtbl{__sender_vtable_fn<_Sender, _Queries>{}()};

        template <class _Sender, class _Queries>
        constexpr const __sender_vtable<completion_signatures_of_t<_Sender>, _Queries>*
        __sender_vtbl_() { 
          return &__sender_vtbl<_Sender, _Queries>;
        }

      template <class _Sigs, class _ReceiverQueries, class _SenderQueries>
        class __sender {
         public:
          using completion_signatures = _Sigs;

          __sender(const __sender&) = delete;
          __sender& operator=(const __sender&) = delete;

          __sender(__sender&&) = default;
          __sender& operator=(__sender&&) = default;

          template <__none_of<__sender&, const __sender&> _Sender>
            __sender(_Sender&& __sender)
              : __storage_{(_Sender&&) __sender} {}

         private:
          using __vtable_t = __sender_vtable<_Sigs, _ReceiverQueries>;

          __storage_t<__unique_storage<>, __vtable_t> __storage_;

          template <receiver_of<_Sigs> _Rcvr>
              requires sender_to<__sender, _Rcvr>
            friend __operation<__sender, std::decay_t<_Rcvr>, _ReceiverQueries> 
            tag_invoke(connect_t, __sender&& __self, _Rcvr&& __rcvr) {
              return {(_Rcvr&&) __rcvr, (__sender&&) __self};
            }
        };
    } // namespace __sender
  } // namepsace __any
} // namespace exec