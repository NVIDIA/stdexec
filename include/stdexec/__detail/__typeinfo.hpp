/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <compare>
#include <source_location>
#include <string_view>
#include <typeinfo>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-private-field")

//////////////////////////////////////////////////////////////////////////////////////////
// __type_info, __mtypeid, and __msplice

namespace STDEXEC
{
  template <class _Ty>
  struct __mtype
  {
    using __t = _Ty;
  };

  namespace __detail
  {
    template <class _Ty>
    extern __mtype<_Ty> __demangle_v;
  }  // namespace __detail

  // A utility for pretty-printing type names in diagnostics
  template <class _Ty>
  using __demangle_t = decltype(__detail::__demangle_v<_Ty>)::__t;

  namespace __detail
  {
    template <class _Ty>
    extern __mtype<__demangle_t<_Ty> &> __demangle_v<_Ty &>;

    template <class _Ty>
    extern __mtype<__demangle_t<_Ty> &&> __demangle_v<_Ty &&>;

    template <class _Ty>
    extern __mtype<__demangle_t<_Ty> const &> __demangle_v<_Ty const &>;
  }  // namespace __detail

  namespace __detail
  {
    //////////////////////////////////////////////////////////////////////////////////////////
    // __get_pretty_name
    template <class>
    struct __xyzzy
    {
      struct __plugh
      {};
    };

    inline constexpr char __type_name_prefix[] = "__xyzzy<";
    inline constexpr char __type_name_suffix[] = ">::__plugh";

    [[nodiscard]]
    consteval std::string_view __find_pretty_name(std::string_view __sv) noexcept
    {
      auto const __beg_pos = __sv.find(__type_name_prefix);
      auto const __end_pos = __sv.rfind(__type_name_suffix);

      auto const __start = __beg_pos + sizeof(__type_name_prefix) - 1;
      auto const __len   = __end_pos - __start;

      return __sv.substr(__start, __len);
    }

    template <class _Ty>
    [[nodiscard]]
    consteval std::string_view __get_pretty_name_helper() noexcept
    {
#if STDEXEC_EDG()
      return __detail::__find_pretty_name(std::string_view{STDEXEC_PRETTY_FUNCTION()});
#else
      return __detail::__find_pretty_name(std::source_location::current().function_name());
#endif
    }

    template <class _Ty>
    [[nodiscard]]
    consteval std::string_view __get_pretty_name() noexcept
    {
      return __detail::__get_pretty_name_helper<typename __xyzzy<_Ty>::__plugh>();
    }
  }  // namespace __detail

  ////////////////////////////////////////////////////////////////////////////////////////////
  // __mnameof: get the pretty name of a type _Ty as a string_view at compile time
  template <class _Ty>
  inline constexpr std::string_view __mnameof = __detail::__get_pretty_name<__demangle_t<_Ty>>();

  static_assert(__mnameof<void> == "void");

  //////////////////////////////////////////////////////////////////////////////////////////
  // __type_info
  struct __type_info
  {
    constexpr __type_info(__type_info &&)            = delete;
    constexpr __type_info &operator=(__type_info &&) = delete;

    constexpr explicit __type_info(std::string_view      __name,
                                   std::type_info const *__type = nullptr) noexcept
      : __name_(__name)
      , __type_(__type)
    {}

    [[nodiscard]]
    constexpr auto name() const noexcept -> std::string_view
    {
      return __name_;
    }

    [[nodiscard]]
    constexpr auto operator==(__type_info const &__other) const noexcept -> bool
    {
      return this == &__other || __name_ == __other.__name_;
    }

    constexpr auto operator<=>(__type_info const &__other) const noexcept -> std::strong_ordering
    {
      return __name_ <=> __other.__name_;
    }

#if !STDEXEC_NO_STDCPP_TYPEID()
    [[nodiscard]]
    constexpr auto type() const noexcept -> std::type_info const &
    {
      return *__type_;
    }

    [[nodiscard]]
    constexpr operator std::type_info const &() const noexcept
    {
      return *__type_;
    }
#endif

   private:
    std::string_view const      __name_;
    std::type_info const *const __type_     = nullptr;  // used only if !STDEXEC_NO_STDCPP_TYPEID()
    void const *const           __reserved_ = nullptr;  // reserved for future use
  };

  namespace __detail
  {
    template <class _Ty>
    inline constexpr __type_info __mtypeid_v{
      __mnameof<_Ty>                                                                 //
        STDEXEC_PP_WHEN(STDEXEC_PP_NOT(STDEXEC_NO_STDCPP_TYPEID()), , &typeid(_Ty))  //
    };

    template <class _Ty>
    inline constexpr __type_info const &__mtypeid_v<_Ty const> = __mtypeid_v<_Ty>;
  }  // namespace __detail

  //////////////////////////////////////////////////////////////////////////////////////////
  // __type_index
  struct __type_index
  {
    __type_index() = default;

    constexpr __type_index(__type_info const &__info) noexcept
      : __info_(&__info)
    {}

    [[nodiscard]]
    constexpr std::string_view name() const noexcept
    {
      return (*__info_).name();
    }

    [[nodiscard]]
    constexpr bool operator==(__type_index const &__other) const noexcept
    {
      return *__info_ == *__other.__info_;
    }

    [[nodiscard]]
    constexpr std::strong_ordering operator<=>(__type_index const &__other) const noexcept
    {
      return *__info_ <=> *__other.__info_;
    }

#if !STDEXEC_NO_STDCPP_TYPEID()
    [[nodiscard]]
    constexpr auto type() const noexcept -> std::type_info const &
    {
      return (*__info_).type();
    }

    [[nodiscard]]
    constexpr operator std::type_info const &() const noexcept
    {
      return (*__info_).type();
    }
#endif

    __type_info const *__info_ = &__detail::__mtypeid_v<void>;
  };

  namespace __detail
  {
    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wnon-template-friend")
    STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)

    // The following two classes use the stateful metaprogramming trick to create a spooky
    // association between a __type_index object and the type it represents.
    template <__type_index Id>
    struct __mtypeid_key
    {
      friend constexpr auto __typeid_lookup(__mtypeid_key<Id>) noexcept;
    };

    template <class _Ty>
    struct __mtypeid_value
    {
      using __t                          = _Ty;
      static constexpr __type_index __id = __type_index(__mtypeid_v<_Ty>);

      friend constexpr auto __typeid_lookup(__mtypeid_key<__id>) noexcept
      {
        return __mtypeid_value<_Ty>();
      }
    };

    STDEXEC_PRAGMA_POP()

#if STDEXEC_MSVC()
    template <auto _Id>
    struct __msplice_helper : decltype(__typeid_lookup(__mtypeid_key<_Id>())){};
#else
    // Cache the result of the lookup:
    template <auto _Id>
    extern decltype(__typeid_lookup(__mtypeid_key<_Id>())) __msplice_v;
#endif
  }  // namespace __detail

  // For a given type, return a __type_index object
  template <class _Ty>
  inline constexpr __type_index __mtypeid = __detail::__mtypeid_value<_Ty>::__id;

#if STDEXEC_MSVC()
  template <__type_index _Id>
  using __msplice = __detail::__msplice_helper<_Id>::__t;
#elif STDEXEC_GCC() && STDEXEC_GCC_VERSION < 1300
  template <auto _Id>
  using __msplice = decltype(__detail::__msplice_v<_Id>)::__t;
#else
  template <__type_index _Id>
  using __msplice = decltype(__detail::__msplice_v<_Id>)::__t;
#endif

  // Sanity check:
  static_assert(STDEXEC_IS_SAME(void, __msplice<__mtypeid<void>>));
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
