/*
 * Copyright (c) 2025 Ian Petersen
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/scope_tokens.hpp>

#include <array>
#include <cstdint>
#include <memory_resource>
#include <stdexcept>

namespace ex = STDEXEC;

namespace {
  // a sender adaptor that prepends the provided environment onto the provided sender's environment
  struct with_attrs_t {
    template <ex::sender Sender, class Env>
    auto operator()(Sender&& sender, Env&& env) const noexcept(
      std::is_nothrow_constructible_v<std::remove_cvref_t<Sender>, Sender>
      && std::is_nothrow_constructible_v<std::remove_cvref_t<Env>, Env>) {
      return ex::__make_sexpr<with_attrs_t>(std::forward<Env>(env), std::forward<Sender>(sender));
    }
  };

  inline constexpr with_attrs_t with_attrs{};

  struct with_attrs_impl : ex::__sexpr_defaults {
    static constexpr auto get_attrs =
      []<class Env>(auto, const Env& env, const ex::sender auto& child) noexcept {
        return ex::__env::__join(env, ex::get_env(child));
      };

    template <class Sender, class... Env>
    static consteval auto get_completion_signatures() //
      -> ex::__completion_signatures_of_t<ex::__child_of<std::remove_cvref_t<Sender>>, Env...> {
      return {};
    };
  };

  struct counting_resource : std::pmr::memory_resource {
    counting_resource() noexcept
      : counting_resource(*std::pmr::new_delete_resource()) {
    }

    explicit counting_resource(std::pmr::memory_resource& upstream) noexcept
      : upstream_(upstream) {
    }

    counting_resource(counting_resource&&) = delete;

    ~counting_resource() = default;

    std::intmax_t allocated() const noexcept {
      return allocated_;
    }

   private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
      auto ret = upstream_.allocate(bytes, alignment);
      allocated_ += bytes;
      return ret;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
      allocated_ -= bytes;
      upstream_.deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      auto* downCast = dynamic_cast<const counting_resource*>(&other);
      return downCast != nullptr && (upstream_ == downCast->upstream_);
    }

    std::pmr::memory_resource& upstream_;
    std::intmax_t allocated_{};
  };

  struct scope_with_alloc {
    std::pmr::polymorphic_allocator<> alloc;

    struct token : null_token {
      const scope_with_alloc* scope_;

      template <ex::sender Sender>
      auto wrap(Sender&& sender) const
        noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<Sender>, Sender>) {
        return with_attrs(std::forward<Sender>(sender), ex::prop(ex::get_allocator, scope_->alloc));
      }
    };

    token get_token() const noexcept {
      return token{{}, this};
    }
  };
} // namespace

template <>
struct ex::__sexpr_impl<with_attrs_t> : with_attrs_impl { };

namespace {
  TEST_CASE("Trivial spawns compile", "[consumers][spawn]") {
    ex::spawn(ex::just(), null_token{});
    ex::spawn(ex::just_stopped(), null_token{});
  }

  TEST_CASE("spawn doesn't leak", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto&& envAlloc) noexcept {
        // check that the allocator provided to spawn is in our environment
        REQUIRE(alloc == envAlloc);
        // check that we actually allocated something to run this op
        REQUIRE(rsc.allocated() > 0);
      }),
      null_token{},
      ex::prop(ex::get_allocator, alloc));

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE("spawn reads an allocator from the sender's environment", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<> alloc(&rsc);

    scope_with_alloc scope{alloc};

    REQUIRE(rsc.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto&& envAlloc) noexcept {
        // we should've pulled the scope's allocator into our environment
        REQUIRE(alloc == envAlloc);

        // we should've allocated some memory for this operation
        REQUIRE(rsc.allocated() > 0);
      }),
      scope.get_token());

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE(
    "The allocator provided directly to spawn overrides the allocator in the sender's environment",
    "[consumers][spawn]") {

    counting_resource rsc1;

    std::array<std::byte, 256> buffer{};
    std::pmr::monotonic_buffer_resource bumpAlloc(buffer.data(), buffer.size());

    counting_resource rsc2(bumpAlloc);

    std::pmr::polymorphic_allocator<> alloc1(&rsc1);
    std::pmr::polymorphic_allocator<> alloc2(&rsc2);

    REQUIRE(alloc1 != alloc2);

    scope_with_alloc scope{alloc1};

    REQUIRE(rsc1.allocated() == 0);
    REQUIRE(rsc2.allocated() == 0);

    ex::spawn(
      ex::read_env(ex::get_allocator) | ex::then([&](auto& envAlloc) noexcept {
        // the allocator in the environment should be the one provided to spawn
        // as an explicit argument and not the one provided by the scope
        REQUIRE(alloc1 != envAlloc);
        REQUIRE(alloc2 == envAlloc);

        // we should have allocated some memory for the op from rsc2 but not from rsc
        REQUIRE(rsc1.allocated() == 0);
        REQUIRE(rsc2.allocated() > 0);
      }),
      scope.get_token(),
      ex::prop(ex::get_allocator, alloc2));

    REQUIRE(rsc1.allocated() == 0);
    REQUIRE(rsc2.allocated() == 0);
  }

  TEST_CASE("spawn tolerates throwing scope tokens", "[consumers][spawn]") {
    counting_resource rsc;
    std::pmr::polymorphic_allocator<std::byte> alloc(&rsc);

    struct throwing_token : null_token {
      const counting_resource* rsc;

      assoc try_associate() const {
        REQUIRE(rsc->allocated() > 0);
        throw std::runtime_error("nope");
      }
    };

    REQUIRE(rsc.allocated() == 0);

    bool threw = false;
    try {
      ex::spawn(ex::just(), throwing_token{{}, &rsc}, ex::prop(ex::get_allocator, alloc));
    } catch (const std::runtime_error& e) {
      threw = true;
      REQUIRE(std::string{"nope"} == e.what());
    }

    REQUIRE(threw);

    REQUIRE(rsc.allocated() == 0);
  }

  TEST_CASE("spawn tolerates expired scope tokens", "[consumers][spawn]") {
    struct expired_token : null_token { // inherit the wrap method template
      const counting_resource* rsc;
      bool* tried;

      struct assoc {
        constexpr explicit operator bool() const noexcept {
          return false;
        }

        constexpr assoc try_associate() const noexcept {
          return {};
        }
      };

      assoc try_associate() const {
        REQUIRE(rsc->allocated() > 0);
        *tried = true;
        return {};
      }
    };

    counting_resource rsc;
    std::pmr::polymorphic_allocator<std::byte> alloc(&rsc);

    REQUIRE(rsc.allocated() == 0);

    bool triedToAssociate = false;

    ex::spawn(
      ex::just(), expired_token{{}, &rsc, &triedToAssociate}, ex::prop(ex::get_allocator, alloc));

    REQUIRE(rsc.allocated() == 0);
    REQUIRE(triedToAssociate);
  }
} // namespace
