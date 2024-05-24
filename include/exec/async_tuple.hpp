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

#include "stdexec/concepts.hpp"
#include "stdexec/functional.hpp"

#include "__detail/__tuple_reverse.hpp"

#include "async_object.hpp"

namespace exec {

//
// implementation of async_tuple
//

template <class... _FynId>
struct __async_tuple {

  struct __t {
    using __id = __async_tuple;

    using __fyn_t = stdexec::__decayed_tuple<stdexec::__t<_FynId>...>;
    using __stgn_t = stdexec::__decayed_tuple<typename stdexec::__t<_FynId>::storage...>;

    STDEXEC_ATTRIBUTE((no_unique_address)) __fyn_t __fyn_;

    
    explicit __t(__fyn_t __fyn) : __fyn_(std::move(__fyn)) {}
    explicit __t(stdexec::__t<_FynId>... __fyn) : __fyn_(std::move(__fyn)...) {}

    using object = std::tuple<typename stdexec::__t<_FynId>::handle...>;
    using handle = std::tuple<typename stdexec::__t<_FynId>::handle...>;
    struct storage : stdexec::__immovable { 
      STDEXEC_ATTRIBUTE((no_unique_address)) std::optional<__fyn_t> __fyn_;
      STDEXEC_ATTRIBUTE((no_unique_address)) __stgn_t __stgn_;
      std::optional<object> o;
    };

    auto async_construct(storage& stg) noexcept {
      stg.__fyn_.emplace(__fyn_);
      auto mc = stdexec::__apply(
        [&](typename stdexec::__t<_FynId>&... __fyn) noexcept {
          return stdexec::__apply(
            [&](typename stdexec::__t<_FynId>::storage&... __stgn) noexcept {
              return stdexec::when_all(
                stdexec::just(std::ref(stg)),
                exec::async_construct(__fyn, __stgn)...
              ); 
            }, stg.__stgn_);
        }, stg.__fyn_.value());
      auto oc = stdexec::then(mc, [](storage& stg, typename stdexec::__t<_FynId>::handle... hn) noexcept {
        auto construct = [&]() noexcept { return object{hn...}; };
        stg.o.emplace(stdexec::__conv{construct}); 
        return handle{stg.o.value()};
      });
      return oc;
    }
    auto async_destruct(storage& stg) noexcept { 
      auto md = stdexec::__apply(
        [&](typename stdexec::__t<_FynId>&&... __fyn) noexcept {
          return stdexec::__apply(
            [&](typename stdexec::__t<_FynId>::storage&... __stgn) noexcept {
              return stdexec::__apply(
                [&](auto&&... __d) noexcept {
                  return stdexec::when_all(
                    stdexec::just(std::ref(stg)),
                    __d...);
                }, exec::__tuple_reverse(std::make_tuple(exec::async_destruct(__fyn, __stgn)...))); 
            }, stg.__stgn_);
        }, std::move(stg.__fyn_.value()));
      auto od = stdexec::then(md, [](storage& stg) noexcept {
        stg.o.reset();
      });
      return od;
    }

  };
};

template<class... _Fyn>
using __async_tuple_t = stdexec::__t<__async_tuple<stdexec::__id<std::remove_cvref_t<_Fyn>>...>>;

// make_async_tuple is an algorithm that creates an async-object that 
// contains a tuple of the given async-objects.
// the async_tuple object will compose the async-constructors and 
// async-destructors of all the given async-objects
struct make_async_tuple_t {
  template<class... _Fyn>
  __async_tuple_t<_Fyn...> operator()(_Fyn&&... __fyn) const {
    using __fyn_t = typename __async_tuple_t<_Fyn...>::__fyn_t;
    return __async_tuple_t<_Fyn...>{__fyn_t{(_Fyn&&)__fyn...}};
  }
};
constexpr inline static make_async_tuple_t make_async_tuple{};

} // namespace exec
