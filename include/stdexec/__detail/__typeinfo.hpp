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
#include "__meta.hpp"

#include <cstddef>

#include <compare>
#include <string_view>

//////////////////////////////////////////////////////////////////////////////////////////
// __type_info, __mtypeid, and __mtypeof

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////////////////
  // __type_info
  struct __type_info {
    constexpr __type_info(__type_info &&) = delete;
    constexpr __type_info &operator=(__type_info &&) = delete;

    constexpr explicit __type_info(std::string_view __name) noexcept
      : __name_(__name) {
    }

    [[nodiscard]]
    constexpr std::string_view name() const noexcept {
      return __name_;
    }

    [[nodiscard]]
    constexpr auto operator==(const __type_info &__other) const noexcept -> bool {
      return this == &__other || __name_ == __other.__name_;
    }

    constexpr auto
      operator<=>(const __type_info &) const noexcept -> std::strong_ordering = default;

   private:
    std::string_view __name_;
  };

  namespace __detail {
    template <class _Ty>
    inline constexpr __type_info __mtypeid_v{__mnameof<_Ty>};

    template <class _Ty>
    inline constexpr const __type_info &__mtypeid_v<_Ty const> = __mtypeid_v<_Ty>;
  } // namespace __detail

  //////////////////////////////////////////////////////////////////////////////////////////
  // __type_index
  struct __type_index {
    constexpr __type_index(const __type_info &info) noexcept
      : __info_(&info) {
    }

    [[nodiscard]]
    constexpr std::string_view name() const noexcept {
      return (*__info_).name();
    }

    [[nodiscard]]
    constexpr bool operator==(const __type_index &other) const noexcept {
      return *__info_ == *other.__info_;
    }

    [[nodiscard]]
    constexpr std::strong_ordering operator<=>(const __type_index &other) const noexcept {
      return *__info_ <=> *other.__info_;
    }

    const __type_info *__info_;
  };

  namespace __detail {
    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wnon-template-friend")
    STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)

    // The following two classes use the stateful metaprogramming trick to create a spooky
    // association between a __type_index object and the type it represents.
    template <__type_index Id>
    struct __mtypeid_key {
      friend constexpr auto __typeid_lookup(__mtypeid_key<Id>) noexcept;
    };

    template <class _Ty>
    struct __mtypeid_value {
      using __t = _Ty;
      static constexpr __type_index __id = __type_index(__mtypeid_v<_Ty>);

      friend constexpr auto __typeid_lookup(__mtypeid_key<__id>) noexcept {
        return __mtypeid_value<_Ty>();
      }
    };

    STDEXEC_PRAGMA_POP()

    // This specialization is what makes __mtypeof< Id > return the type associated with Id.
    template <auto _Index>
      requires __same_as<decltype(_Index) const, __type_index const>
    extern __fn_t<__t<decltype(__typeid_lookup(__detail::__mtypeid_key<_Index>()))>>
      *__mtypeof_v<_Index>;
  } // namespace __detail

  // For a given type, return a __type_index object
  template <class _Ty>
  inline constexpr __type_index __mtypeid = __detail::__mtypeid_value<_Ty>::__id;

  // Sanity check:
  static_assert(STDEXEC_IS_SAME(void, __mtypeof<__mtypeid<void>>));
} // namespace STDEXEC
