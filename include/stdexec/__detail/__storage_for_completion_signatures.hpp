/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../functional.hpp"
#include "__config.hpp"
#include "__matching_completion_signature.hpp"
#include "__receivers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__variant.hpp"

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace STDEXEC
{

  template <typename>
  struct storage_for_completion_signature;

  template <typename _Tag, typename... _Args>
  struct storage_for_completion_signature<_Tag(_Args...)>
  {
    using tag_type                 = _Tag;
    using signature_type           = _Tag(_Args...);
    using __normalized_signature_t = _Tag(_Args&&...);

   private:
    using __tuple_t = __tuple<_Args...>;

    struct __forward_as_tuple_t
    {
      template <typename... _Ts>
      constexpr std::tuple<_Ts&&...> operator()(_Ts&&... __args) const noexcept
      {
        return {static_cast<_Ts&&>(__args)...};
      }
    };

    struct __forward_as_internal_tuple_t
    {
      template <typename... _Ts>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr __tuple<_Ts&&...> operator()(_Ts&&... __args) const noexcept
      {
        return {static_cast<_Ts&&>(__args)...};
      }
    };

   public:
    template <typename... _OtherArgs>
      requires(sizeof...(_OtherArgs) == sizeof...(_Args))
           && (std::is_constructible_v<_Args, _OtherArgs> && ...)
    STDEXEC_ATTRIBUTE(host, device)
    constexpr explicit storage_for_completion_signature(_Tag, _OtherArgs&&... __args)
      noexcept((std::is_nothrow_constructible_v<_Args, _OtherArgs> && ...))
      : __args_{static_cast<_OtherArgs&&>(__args)...}
    {}

    STDEXEC_ATTRIBUTE(host, device)
    static constexpr auto tag() noexcept -> _Tag
    {
      return {};
    }

    constexpr auto forward_arguments() & noexcept
    {
      return STDEXEC::__apply(__forward_as_tuple_t{}, __args_);
    }

    constexpr auto forward_arguments() const & noexcept
    {
      return STDEXEC::__apply(__forward_as_tuple_t{}, __args_);
    }

    constexpr auto forward_arguments() && noexcept
    {
      return STDEXEC::__apply(__forward_as_tuple_t{}, static_cast<__tuple_t&&>(__args_));
    }

    constexpr auto forward_arguments() const && noexcept
    {
      return STDEXEC::__apply(__forward_as_tuple_t{}, static_cast<__tuple_t const &&>(__args_));
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __forward_arguments() & noexcept
    {
      return STDEXEC::__apply(__forward_as_internal_tuple_t{}, __args_);
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __forward_arguments() const & noexcept
    {
      return STDEXEC::__apply(__forward_as_internal_tuple_t{}, __args_);
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __forward_arguments() && noexcept
    {
      return STDEXEC::__apply(__forward_as_internal_tuple_t{}, static_cast<__tuple_t&&>(__args_));
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __forward_arguments() const && noexcept
    {
      return STDEXEC::__apply(__forward_as_internal_tuple_t{},
                              static_cast<__tuple_t const &&>(__args_));
    }

   private:
    __tuple_t __args_;
  };

  namespace __completion_storage
  {

    template <typename _Signatures, typename _Tag, typename... _Args>
    using __storage_for_arrival_t = storage_for_completion_signature<
      matching_completion_signature_t<_Signatures, _Tag, _Args...>>;

    template <typename>
    struct __variant_for_signatures;

    template <typename... _Signatures>
    struct __variant_for_signatures<completion_signatures<_Signatures...>>
    {
      using __t = __uniqued_variant<storage_for_completion_signature<_Signatures>...>;
    };

    template <typename>
    struct __nothrow_storable;

    template <typename _Tag, typename... _Args>
    struct __nothrow_storable<_Tag(_Args...)>
    {
      static constexpr bool value = (std::is_nothrow_constructible_v<_Args, _Args> && ...);
    };

    template <typename, typename...>
    struct __arrival_storable;

    template <typename _Tag, typename... _StoredArgs, typename... _Args>
    struct __arrival_storable<storage_for_completion_signature<_Tag(_StoredArgs...)>, _Args...>
    {
      static constexpr bool value =
        sizeof...(_StoredArgs) == sizeof...(_Args)
        && ((std::is_reference_v<_StoredArgs> || !std::is_lvalue_reference_v<_Args>) && ...);
    };

    struct __ambiguous_normalized_completion_signatures;
    struct __non_persistable_completion_signature;

    template <typename>
    struct __persistable_completion_signature;

    template <typename _Tag, typename... _Args>
    struct __persistable_completion_signature<_Tag(_Args...)>
    {
      static constexpr bool value =
        ((std::is_reference_v<_Args> || std::is_move_constructible_v<_Args>) && ...);
    };

    template <typename>
    struct __persistable_completion_signatures;

    template <typename... _Signatures>
    struct __persistable_completion_signatures<completion_signatures<_Signatures...>>
    {
      static constexpr bool value = (__persistable_completion_signature<_Signatures>::value && ...);
    };

    template <typename>
    struct __normalized_completion_signatures;

    template <typename... _Signatures>
    struct __normalized_completion_signatures<completion_signatures<_Signatures...>>
    {
      using __t = transform_completion_signatures<completion_signatures<
        typename storage_for_completion_signature<_Signatures>::__normalized_signature_t...>>;
    };

    template <typename _Signatures>
    using __normalized_completion_signatures_t =
      typename __normalized_completion_signatures<_Signatures>::__t;

    template <typename _Signatures>
    inline constexpr bool __has_unique_normalized_completion_signatures =
      __mapply<__msize, _Signatures>::value
      == __mapply<__msize, __normalized_completion_signatures_t<_Signatures>>::value;

    template <typename _InputSignatures, typename _OutputSignatures>
    consteval auto __get_completion_signatures()
    {
      if constexpr (!__has_unique_normalized_completion_signatures<_InputSignatures>)
      {
        return __throw_compile_time_error<__ambiguous_normalized_completion_signatures>();
      }
      else if constexpr (!__persistable_completion_signatures<_InputSignatures>::value)
      {
        return __throw_compile_time_error<__non_persistable_completion_signature>();
      }
      else
      {
        return _OutputSignatures{};
      }
    }

    template <typename _Storage>
    using __variant_for_storage_t = decltype(std::declval<_Storage>().__get_variant());

    template <typename>
    struct __variant_alternatives;

    template <typename... _Alternatives>
    struct __variant_alternatives<__variant<_Alternatives...>>
    {
      template <typename _Variant>
      using __with_cvref_t = __mlist<STDEXEC::__copy_cvref_t<_Variant, _Alternatives>...>;
    };

    template <typename _Variant>
    using __variant_alternatives_t = typename __variant_alternatives<
      std::remove_cvref_t<_Variant>>::template __with_cvref_t<_Variant>;

    template <typename _Visitor, typename _Alternatives, typename... _Variants>
    struct __nothrow_visit_stored_completion;

    template <typename _Visitor, typename... _Alternatives>
    struct __nothrow_visit_stored_completion<_Visitor, __mlist<_Alternatives...>>
    {
      static constexpr bool value = __nothrow_invocable<_Visitor, _Alternatives...>;
    };

    template <typename _Visitor,
              typename... _Alternatives,
              typename... _CurrentAlternatives,
              typename... _Variants>
    struct __nothrow_visit_stored_completion<_Visitor,
                                             __mlist<_Alternatives...>,
                                             __mlist<_CurrentAlternatives...>,
                                             _Variants...>
    {
      static constexpr bool value =
        (__nothrow_visit_stored_completion<_Visitor,
                                           __mlist<_Alternatives..., _CurrentAlternatives>,
                                           _Variants...>::value
         && ...);
    };

    template <typename _Visitor, typename... _Variants>
    constexpr bool __nothrow_visit_stored_completion_for_variants_v =
      __nothrow_invocable<_Visitor>
      && __nothrow_visit_stored_completion<_Visitor,
                                           __mlist<>,
                                           __variant_alternatives_t<_Variants>...>::value;

    template <typename _Visitor, typename... _Storages>
    constexpr bool __nothrow_visit_stored_completion_v =
      __nothrow_visit_stored_completion_for_variants_v<_Visitor,
                                                       __variant_for_storage_t<_Storages>...>;

    template <typename _Result, typename _Visitor, typename _Alternatives, typename... _Variants>
    struct __nothrow_visit_stored_completion_r;

    template <typename _Result, typename _Visitor, typename... _Alternatives>
    struct __nothrow_visit_stored_completion_r<_Result, _Visitor, __mlist<_Alternatives...>>
    {
      static constexpr bool value =
        std::is_nothrow_invocable_r_v<_Result, _Visitor, _Alternatives...>;
    };

    template <typename _Result,
              typename _Visitor,
              typename... _Alternatives,
              typename... _CurrentAlternatives,
              typename... _Variants>
    struct __nothrow_visit_stored_completion_r<_Result,
                                               _Visitor,
                                               __mlist<_Alternatives...>,
                                               __mlist<_CurrentAlternatives...>,
                                               _Variants...>
    {
      static constexpr bool value =
        (__nothrow_visit_stored_completion_r<_Result,
                                             _Visitor,
                                             __mlist<_Alternatives..., _CurrentAlternatives>,
                                             _Variants...>::value
         && ...);
    };

    template <typename _Result, typename _Visitor, typename... _Variants>
    constexpr bool __nothrow_visit_stored_completion_r_for_variants_v =
      std::is_nothrow_invocable_r_v<_Result, _Visitor>
      && __nothrow_visit_stored_completion_r<_Result,
                                             _Visitor,
                                             __mlist<>,
                                             __variant_alternatives_t<_Variants>...>::value;

    template <typename _Result, typename _Visitor, typename... _Storages>
    constexpr bool __nothrow_visit_stored_completion_r_v =
      __nothrow_visit_stored_completion_r_for_variants_v<_Result,
                                                         _Visitor,
                                                         __variant_for_storage_t<_Storages>...>;

    template <typename _Visitor, typename _Variant>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr decltype(auto) __visit_variant_completion(_Visitor&& __visitor, _Variant&& __variant)
    {
      return STDEXEC::__visit(static_cast<_Visitor&&>(__visitor),
                              static_cast<_Variant&&>(__variant));
    }

    template <typename _Visitor, typename _Storage>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr decltype(auto)
      __visit_present_stored_completion(_Visitor&& __visitor, _Storage&& __storage)
    {
      return __completion_storage::__visit_variant_completion(
        static_cast<_Visitor&&>(__visitor),
        static_cast<_Storage&&>(__storage).__get_variant());
    }

    template <typename _Visitor, typename _Storage, typename... _Storages>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr decltype(auto) __visit_present_stored_completion(_Visitor&& __visitor,
                                                               _Storage&& __storage,
                                                               _Storages&&... __storages)
    {
      return __completion_storage::__visit_variant_completion(
        [&](auto&& __completion) -> decltype(auto)
        {
          return __completion_storage::__visit_present_stored_completion(
            [&](auto&&... __completions) -> decltype(auto)
            {
              return STDEXEC::__invoke(static_cast<_Visitor&&>(__visitor),
                                       static_cast<decltype(__completion)&&>(__completion),
                                       static_cast<decltype(__completions)&&>(__completions)...);
            },
            static_cast<_Storages&&>(__storages)...);
        },
        static_cast<_Storage&&>(__storage).__get_variant());
    }

  }  // namespace __completion_storage

  template <typename _Visitor, typename _Storage, typename... _Storages>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr decltype(auto)
    visit_stored_completion(_Visitor&& __visitor, _Storage&& __storage, _Storages&&... __storages)
      noexcept(
        __completion_storage::__nothrow_visit_stored_completion_v<_Visitor, _Storage, _Storages...>)
  {
    if (!(__storage.has_completion() && ... && __storages.has_completion()))
    {
      return STDEXEC::__invoke(static_cast<_Visitor&&>(__visitor));
    }
    return __completion_storage::__visit_present_stored_completion(
      static_cast<_Visitor&&>(__visitor),
      static_cast<_Storage&&>(__storage),
      static_cast<_Storages&&>(__storages)...);
  }

  template <typename _Result, typename _Visitor, typename _Storage, typename... _Storages>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr _Result
    visit_stored_completion(_Visitor&& __visitor, _Storage&& __storage, _Storages&&... __storages)
      noexcept(__completion_storage::__nothrow_visit_stored_completion_r_v<_Result,
                                                                           _Visitor,
                                                                           _Storage,
                                                                           _Storages...>)
  {
    if (!(__storage.has_completion() && ... && __storages.has_completion()))
    {
      if constexpr (std::is_void_v<_Result>)
      {
        STDEXEC::__invoke(static_cast<_Visitor&&>(__visitor));
        return;
      }
      else
      {
        return STDEXEC::__invoke(static_cast<_Visitor&&>(__visitor));
      }
    }
    if constexpr (std::is_void_v<_Result>)
    {
      __completion_storage::__visit_present_stored_completion(static_cast<_Visitor&&>(__visitor),
                                                              static_cast<_Storage&&>(__storage),
                                                              static_cast<_Storages&&>(
                                                                __storages)...);
      return;
    }
    else
    {
      return __completion_storage::__visit_present_stored_completion(
        static_cast<_Visitor&&>(__visitor),
        static_cast<_Storage&&>(__storage),
        static_cast<_Storages&&>(__storages)...);
    }
  }

  enum class storage_for_completion_signatures_error_policy
  {
    internalize,
    propagate
  };

  template <typename,
            storage_for_completion_signatures_error_policy =
              storage_for_completion_signatures_error_policy::internalize>
  class storage_for_completion_signatures;

  template <>
  class storage_for_completion_signatures<
    completion_signatures<>,
    storage_for_completion_signatures_error_policy::internalize>
  {
   public:
    using completion_signatures          = STDEXEC::completion_signatures<>;
    static constexpr bool nothrow_arrive = true;

    static consteval auto get_completion_signatures() noexcept
    {
      return completion_signatures{};
    }

    STDEXEC_ATTRIBUTE(host, device)
    static constexpr __variant<> __get_variant() noexcept
    {
      return __variant<>(__no_init);
    }

    STDEXEC_ATTRIBUTE(host, device)
    static constexpr bool has_completion() noexcept
    {
      return false;
    }

    template <receiver _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool complete(_Receiver&&) && noexcept
    {
      return false;
    }
  };

  template <>
  class storage_for_completion_signatures<completion_signatures<>,
                                          storage_for_completion_signatures_error_policy::propagate>
    : public storage_for_completion_signatures<
        completion_signatures<>,
        storage_for_completion_signatures_error_policy::internalize>
  {};

  template <typename... _Signatures>
  class storage_for_completion_signatures<
    completion_signatures<_Signatures...>,
    storage_for_completion_signatures_error_policy::internalize>
  {
    using __input_completion_signatures_t =
      transform_completion_signatures<STDEXEC::completion_signatures<_Signatures...>>;
    static constexpr auto __nothrow = (__completion_storage::__nothrow_storable<_Signatures>::value
                                       && ...);
    using __maybe_throwing_signature_t =
      std::conditional_t<__nothrow,
                         STDEXEC::completion_signatures<>,
                         STDEXEC::completion_signatures<set_error_t(std::exception_ptr)>>;
   public:
    using completion_signatures = transform_completion_signatures<__input_completion_signatures_t,
                                                                  __maybe_throwing_signature_t>;
    static constexpr bool nothrow_arrive = true;

    static consteval auto get_completion_signatures()
    {
      return __completion_storage::__get_completion_signatures<__input_completion_signatures_t,
                                                               completion_signatures>();
    }

   private:
    template <typename _Tag, typename... _Args>
    using __storage_for_arrival_t =
      __completion_storage::__storage_for_arrival_t<__input_completion_signatures_t, _Tag, _Args...>;
    using __storage_t =
      typename __completion_storage::__variant_for_signatures<completion_signatures>::__t;
    __storage_t __storage_;
   public:
    STDEXEC_ATTRIBUTE(host, device)
    constexpr storage_for_completion_signatures() noexcept
      : __storage_(__no_init)
    {}

    template <typename _Tag, typename... _Args>
      requires std::is_constructible_v<__storage_for_arrival_t<_Tag, _Args...>, _Tag, _Args...>
            && __completion_storage::__arrival_storable<__storage_for_arrival_t<_Tag, _Args...>,
                                                        _Args...>::value
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void arrive(_Tag __tag, _Args&&... __args) noexcept
    {
      STDEXEC_ASSERT(__storage_.__is_valueless());
      constexpr auto __nothrow =
        std::is_nothrow_constructible_v<__storage_for_arrival_t<_Tag, _Args...>, _Tag, _Args...>;
      auto const __impl = [&]() noexcept(__nothrow)
      {
        __storage_.template emplace<__storage_for_arrival_t<_Tag, _Args...>>((_Tag&&) __tag,
                                                                             (_Args&&) __args...);
      };
      if constexpr (__nothrow)
      {
        __impl();
      }
      else
      {
        STDEXEC_TRY
        {
          __impl();
        }
        STDEXEC_CATCH_ALL
        {
          __storage_
            .template emplace<storage_for_completion_signature<set_error_t(std::exception_ptr)>>(
              set_error,
              std::current_exception());
        }
      }
    }
    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool has_completion() const noexcept
    {
      return !__storage_.__is_valueless();
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t& __get_variant() & noexcept
    {
      return __storage_;
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t const & __get_variant() const & noexcept
    {
      return __storage_;
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t&& __get_variant() && noexcept
    {
      return static_cast<__storage_t&&>(__storage_);
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t const && __get_variant() const && noexcept
    {
      return static_cast<__storage_t const &&>(__storage_);
    }

    template <receiver_of<completion_signatures> _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool complete(_Receiver&& __rcvr) && noexcept
    {
      if (!has_completion())
      {
        return false;
      }
      STDEXEC::__visit(
        [&](auto&& __active_alternative) noexcept
        {
          using __completion_t = std::remove_cvref_t<decltype(__active_alternative)>;
          constexpr auto __nothrow =
            std::is_nothrow_constructible_v<__completion_t, __completion_t&&>;
          auto const __complete = [&]() noexcept(__nothrow)
          {
            auto __completion = __completion_t(std::move(__active_alternative));
            STDEXEC::__apply(
              [&](auto&&... __args) noexcept
              {
                typename __completion_t::tag_type{}((_Receiver&&) __rcvr,
                                                    (decltype(__args)&&) __args...);
              },
              //  Odds are this is inside an operation state, which means that
              //  sending the completion signal may end our lifetime, which means
              //  we shouldn't send references into ourselves, therefore we move
              //  all the non-references onto the stack
              std::move(__completion).__forward_arguments());
          };
          if constexpr (__nothrow)
          {
            __complete();
          }
          else
          {
            STDEXEC_TRY
            {
              __complete();
            }
            STDEXEC_CATCH_ALL
            {
              set_error((_Receiver&&) __rcvr, std::current_exception());
            }
          }
        },
        (__storage_t&&) __storage_);
      return true;
    }
  };

  template <typename... _Signatures>
  class storage_for_completion_signatures<completion_signatures<_Signatures...>,
                                          storage_for_completion_signatures_error_policy::propagate>
  {
    using __input_completion_signatures_t =
      transform_completion_signatures<STDEXEC::completion_signatures<_Signatures...>>;
   public:
    using completion_signatures = __input_completion_signatures_t;
    static constexpr bool nothrow_arrive =
      (__completion_storage::__nothrow_storable<_Signatures>::value && ...);

    static consteval auto get_completion_signatures()
    {
      return __completion_storage::__get_completion_signatures<__input_completion_signatures_t,
                                                               completion_signatures>();
    }

   private:
    template <typename _Tag, typename... _Args>
    using __storage_for_arrival_t =
      __completion_storage::__storage_for_arrival_t<__input_completion_signatures_t, _Tag, _Args...>;
    using __storage_t =
      typename __completion_storage::__variant_for_signatures<completion_signatures>::__t;
    __storage_t __storage_;

   public:
    STDEXEC_ATTRIBUTE(host, device)
    constexpr storage_for_completion_signatures() noexcept
      : __storage_(__no_init)
    {}

    template <typename _Tag, typename... _Args>
      requires std::is_constructible_v<__storage_for_arrival_t<_Tag, _Args...>, _Tag, _Args...>
            && __completion_storage::__arrival_storable<__storage_for_arrival_t<_Tag, _Args...>,
                                                        _Args...>::value
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void arrive(_Tag __tag, _Args&&... __args) noexcept(
      std::is_nothrow_constructible_v<__storage_for_arrival_t<_Tag, _Args...>, _Tag, _Args...>)
    {
      STDEXEC_ASSERT(__storage_.__is_valueless());
      STDEXEC_TRY
      {
        __storage_.template emplace<__storage_for_arrival_t<_Tag, _Args...>>((_Tag&&) __tag,
                                                                             (_Args&&) __args...);
      }
      STDEXEC_CATCH_ALL
      {
        STDEXEC_THROW();
      }
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool has_completion() const noexcept
    {
      return !__storage_.__is_valueless();
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t& __get_variant() & noexcept
    {
      return __storage_;
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t const & __get_variant() const & noexcept
    {
      return __storage_;
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t&& __get_variant() && noexcept
    {
      return static_cast<__storage_t&&>(__storage_);
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr __storage_t const && __get_variant() const && noexcept
    {
      return static_cast<__storage_t const &&>(__storage_);
    }

    template <receiver_of<completion_signatures> _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr bool complete(_Receiver&& __rcvr) && noexcept(nothrow_arrive)
    {
      if (!has_completion())
      {
        return false;
      }
      STDEXEC::__visit(
        [&](auto&& __active_alternative) noexcept(nothrow_arrive)
        {
          using __completion_t = std::remove_cvref_t<decltype(__active_alternative)>;
          auto __completion    = __completion_t(std::move(__active_alternative));
          STDEXEC::__apply(
            [&](auto&&... __args) noexcept(
              noexcept(typename __completion_t::tag_type{}((_Receiver&&) __rcvr,
                                                           (decltype(__args)&&) __args...)))
            {
              typename __completion_t::tag_type{}((_Receiver&&) __rcvr,
                                                  (decltype(__args)&&) __args...);
            },
            std::move(__completion).__forward_arguments());
        },
        (__storage_t&&) __storage_);
      return true;
    }
  };

}  // namespace STDEXEC
