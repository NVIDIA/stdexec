/*
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
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <test_common/type_helpers.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <concepts>
#  include <cstdint>
#  include <memory>
#endif

namespace ex = STDEXEC;

namespace
{

  TEST_CASE("read returns empty env", "[factories][read]")
  {
    auto sndr  = ex::read_env(ex::get_allocator);
    using Sndr = decltype(sndr);
    using Env  = ex::prop<ex::get_allocator_t, std::allocator<int>>;
    static_assert(ex::sender<Sndr>);
    static_assert(!ex::sender_in<Sndr>);
    static_assert(ex::sender_in<Sndr, Env>);
    static_assert(ex::__completes_inline<ex::set_value_t, ex::env_of_t<Sndr>, Env>);
  }

  struct answer_t
  {
    template <class Env>
      requires ex::__queryable_with<Env, answer_t>
    constexpr auto operator()(Env const & env) const noexcept -> int
    {
      return env.query(*this);
    }
  };

  struct answer_env
  {
    constexpr auto query(answer_t) const noexcept -> int
    {
      return 42;
    }
  };

  struct address_receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value(int&& value) noexcept
    {
      *address_ = std::addressof(value);
      *value_   = value;
    }

    [[nodiscard]]
    auto get_env() const noexcept -> answer_env
    {
      return {};
    }

    int const ** address_;
    int*         value_;
  };

  // A CUDA stream algorithm can pass the value to a kernel by reference, so it must not be
  // a temporary on the stack.
  TEST_CASE("read_env sends a value that lives in its operation state", "[factories][read]")
  {
    STATIC_REQUIRE(
      std::same_as<ex::completion_signatures_of_t<decltype(ex::read_env(answer_t())), answer_env>,
                   ex::completion_signatures<ex::set_value_t(int)>>);

    int const * address = nullptr;
    int         value   = 0;
    auto        op      = ex::connect(ex::read_env(answer_t()), address_receiver{&address, &value});
    ex::start(op);
    CHECK(value == 42);

    auto const begin = reinterpret_cast<std::uintptr_t>(std::addressof(op));
    auto const at    = reinterpret_cast<std::uintptr_t>(address);
    CHECK(begin <= at);
    CHECK(at < begin + sizeof(op));
  }
}  // namespace
