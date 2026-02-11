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

#include <stdexec/execution.hpp>

#include "test_common/scope_tokens.hpp"

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <type_traits>
#include <utility>

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
      -> ex::completion_signatures_of_t<ex::__child_of<std::remove_cvref_t<Sender>>, Env...> {
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

    [[nodiscard]]
    std::intmax_t allocated() const noexcept {
      return allocated_;
    }

   private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
      auto ret = upstream_.allocate(bytes, alignment);
      allocated_ += static_cast<std::intmax_t>(bytes);
      return ret;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
      allocated_ -= static_cast<std::intmax_t>(bytes);
      upstream_.deallocate(p, bytes, alignment);
    }

    [[nodiscard]]
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

    [[nodiscard]]
    token get_token() const noexcept {
      return token{{}, this};
    }
  };
} // namespace

template <>
struct ex::__sexpr_impl<with_attrs_t> : with_attrs_impl { };
