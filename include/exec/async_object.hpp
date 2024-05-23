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

#include "../stdexec/execution.hpp"
#include "../stdexec/concepts.hpp"

#include "__detail/__manual_lifetime.hpp"

namespace exec {

struct async_construct_t {
  template<class _O, class _Stg, class... _An>
  auto operator()(_O&& __o, _Stg& __stg, _An&&... __an) const 
    noexcept(noexcept(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...)))
    -> std::enable_if_t<
      !stdexec::same_as<decltype(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...)), void>,
      decltype(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...))> {
    return ((_O&&)__o).async_construct(__stg, ((_An&&)__an)...);
  }
  template<class _O, class _Stg, class... _An>
  auto operator()(_O&& __o, _Stg& __stg, _An&&... __an) const 
    noexcept(noexcept(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...)))
    -> std::enable_if_t<
      stdexec::same_as<decltype(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...)), void>,
      decltype(((_O&&)__o).async_construct(__stg, ((_An&&)__an)...))> = delete;
};
constexpr inline static async_construct_t async_construct{};

template<class _O, class... _An>
using async_construct_result_t = stdexec::__call_result_t<async_construct_t, const std::remove_cvref_t<_O>&, typename std::remove_cvref_t<_O>::storage&, _An...>;

struct async_destruct_t {
  template<class _O, class _Stg>
  auto operator()(_O&& __o, _Stg& __stg) const 
    noexcept
    -> std::enable_if_t<
      !stdexec::same_as<decltype(((_O&&)__o).async_destruct(__stg)), void>,
      decltype(((_O&&)__o).async_destruct(__stg))> {
    static_assert(noexcept(((_O&&)__o).async_destruct(__stg)));
    return ((_O&&)__o).async_destruct(__stg);
  }
  template<class _O, class _Stg>
  auto operator()(_O&& __o, _Stg& __stg) const 
    noexcept
    -> std::enable_if_t<
      stdexec::same_as<decltype(((_O&&)__o).async_destruct(__stg)), void>,
      decltype(((_O&&)__o).async_destruct(__stg))> = delete;
};
constexpr inline static async_destruct_t async_destruct{};

template<class _O>
using async_destruct_result_t = stdexec::__call_result_t<async_destruct_t, const std::remove_cvref_t<_O>&, typename std::remove_cvref_t<_O>::storage&>;

namespace __async_object {

template<class _T>
concept __immovable_object = 
  !std::is_move_constructible_v<_T> &&
  !std::is_copy_constructible_v<_T> &&
  !std::is_move_assignable_v<_T> &&
  !std::is_copy_assignable_v<_T>;

template<class _T>
concept __object = 
  !std::is_default_constructible_v<_T> &&
  __immovable_object<_T>;

template<class _T>
concept __storage = 
  std::is_nothrow_default_constructible_v<_T> &&
  __immovable_object<_T>;

template<class _S>
concept __async_destruct_result_valid = 
  stdexec::__single_typed_sender<_S> &&
  stdexec::sender_of<_S, stdexec::set_value_t()>;

} // namespace __async_object

template<class _T>
concept async_object = 
  requires (){
    typename _T::object;
    typename _T::handle;
    typename _T::storage;
  } &&
  std::is_move_constructible_v<_T> &&
  std::is_nothrow_move_constructible_v<typename _T::handle> &&
  __async_object::__object<typename _T::object> &&
  __async_object::__storage<typename _T::storage> &&
  requires (const _T& __t_clv, typename _T::storage& __s_lv){
    { async_destruct_t{}(__t_clv, __s_lv) }
      -> stdexec::__nofail_sender;
  } && 
  __async_object::__async_destruct_result_valid<async_destruct_result_t<_T>>;

template<class _T, class... _An>
concept async_object_constructible_from = 
  async_object<_T> &&
  requires (const _T& __t_clv, typename _T::storage& __s_lv, _An... __an){
    { async_construct_t{}(__t_clv, __s_lv, __an...) } 
      -> stdexec::sender_of<stdexec::set_value_t(typename _T::handle)>;
  };

} // namespace exec
