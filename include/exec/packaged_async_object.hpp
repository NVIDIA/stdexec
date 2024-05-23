/*
 * Copyright (c) 2024 Kirk Shoop
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

#include "stdexec/__detail/__execution_fwd.hpp"
#include "stdexec/__detail/__transform_completion_signatures.hpp"

#include "stdexec/concepts.hpp"
#include "stdexec/functional.hpp"

#include "async_object.hpp"

namespace exec {

//
// packaged_async_object
// A utility to allow async_using to take a pack of 
// async-objects that have a defaulted async_constructor
//
template<class _O, class... _An>
struct packaged_async_object {
  using object = typename _O::object;
  using handle = typename _O::handle;
  using storage = typename _O::storage;
  using arguments = stdexec::__decayed_tuple<_An...>;

  packaged_async_object() = delete;
  template<stdexec::__decays_to<_O> _T, class... _Tn>
  explicit packaged_async_object(_T&& __t, _Tn&&... __tn) : 
    __o_((_T&&)__t), 
    __an_((_Tn&&)__tn...) {
  }
private:
  _O __o_;
  arguments __an_;

public:

  auto async_construct(storage& stg) noexcept(noexcept(__o_.async_construct(std::declval<storage&>(), std::declval<_An&&>()...))) { 
    return stdexec::__apply(
      [&]<class... _Args>(_Args&&... __args) noexcept(noexcept(__o_.async_construct(std::declval<storage&>(), (_Args&&) __args...))) {
        return this->__o_.async_construct(stg, (_Args&&) __args...);
      },
      __an_);
  }

  auto async_destruct(storage& stg) noexcept { 
    return __o_.async_destruct(stg);
  }
};

template<class _O, class... _An>
packaged_async_object(_O&&, _An&&...) -> packaged_async_object<std::remove_cvref_t<_O>, std::remove_cvref_t<_An>...>;

struct pack_async_object_t {
  template<typename T, typename... Tn>
  auto operator()(T&& t, Tn&&... tn) const 
    -> packaged_async_object<std::remove_cvref_t<T>, std::remove_cvref_t<Tn>...> {
    return packaged_async_object{std::forward<T>(t), std::forward<Tn>(tn)...};
  }
};
constexpr inline static pack_async_object_t pack_async_object;

} // namespace exec
