/*
 * Copyright (c) NVIDIA
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

#include <test_helpers.hpp>
#include <schedulers/detail/variant.hpp>
#include <schedulers/detail/tuple.hpp>

namespace cuda = example::cuda;


template <class ActionT>
void match_std(ActionT action)
{
  tracer_t::state_t std_state{};
  tracer_t::state_t cuda_state{};

  {
    tracer_t tracer{};
    tracer_t::accessor_t accessor = tracer.get(42);
    std::variant<std::tuple<tracer_t::accessor_t>> storage(std::move(accessor));

    action(tracer, storage);

    std_state = tracer.state();
  }

  {
    tracer_t tracer{};
    tracer_t::accessor_t accessor = tracer.get(42);
    cuda::variant<cuda::tuple<tracer_t::accessor_t>> storage(std::move(accessor));

    action(tracer, storage);

    cuda_state = tracer.state();
  }

  REQUIRE(std_state.n_copy_assignments == cuda_state.n_copy_assignments);
  REQUIRE(std_state.n_copy_constructors == cuda_state.n_copy_constructors);
  REQUIRE(std_state.n_move_assignments == cuda_state.n_move_assignments);
  REQUIRE(std_state.n_move_constructors == cuda_state.n_move_constructors);
  REQUIRE(std_state.n_destroyed == cuda_state.n_destroyed);
}

template <class ActionT, class... Ts>
void invoke(ActionT action, std::variant<Ts...> &storage)
{
  std::visit([=](auto& tpl) {
    std::apply(action, tpl);
  }, storage);
}

template <class ActionT, class... Ts>
void invoke(ActionT action, cuda::variant<Ts...> &storage)
{
  cuda::invoke(action, storage);
}

TEST_CASE("cuda storage type is move constructible", "[cuda][utility]")
{
  match_std([](tracer_t &tracer, auto &storage) {
    // Copy constructor is called to initialize the value
    REQUIRE(tracer.get_n_copy_constructions() == 0);
    REQUIRE(tracer.get_n_copy_assignments() == 0);
    REQUIRE(tracer.get_n_move_constructions() == 1);
    REQUIRE(tracer.get_n_move_assignments() == 0);

    bool was_invoked = false;
    invoke([&](tracer_t::accessor_t &accessor) {
      REQUIRE(accessor.value_ == 42);
      REQUIRE(tracer.get_n_copy_constructions() == 0);
      REQUIRE(tracer.get_n_copy_assignments() == 0);
      REQUIRE(tracer.get_n_move_constructions() == 1);
      REQUIRE(tracer.get_n_move_assignments() == 0);
      was_invoked  = true;
    }, storage);
    REQUIRE(was_invoked);
  });
}

TEST_CASE("cuda storage type is move assignable", "[cuda][utility]")
{
  match_std([](tracer_t &tracer, auto &storage) {
    // Copy constructor is called to initialize the value
    REQUIRE(tracer.get_n_copy_constructions() == 0);
    REQUIRE(tracer.get_n_copy_assignments() == 0);
    REQUIRE(tracer.get_n_move_constructions() == 1);
    REQUIRE(tracer.get_n_move_assignments() == 0);

    // Copy assignment is used
    storage = tracer.get(24);
    REQUIRE(tracer.get_n_copy_constructions() == 0);
    REQUIRE(tracer.get_n_copy_assignments() == 0);
    REQUIRE(tracer.get_n_move_constructions() == 2);
    REQUIRE(tracer.get_n_move_assignments() == 1);

    bool was_invoked = false;
    invoke([&](tracer_t::accessor_t &accessor) {
      REQUIRE(accessor.value_ == 24);
      REQUIRE(tracer.get_n_copy_constructions() == 0);
      REQUIRE(tracer.get_n_copy_assignments() == 0);
      REQUIRE(tracer.get_n_move_constructions() == 2);
      REQUIRE(tracer.get_n_move_assignments() == 1);
      was_invoked  = true;
    }, storage);
    REQUIRE(was_invoked);
  });
}

TEST_CASE("cuda storage type is copy assignable", "[cuda][utility]")
{
  match_std([](tracer_t &tracer, auto &storage) {
    // Copy constructor is called to initialize the value
    REQUIRE(tracer.get_n_copy_constructions() == 0);
    REQUIRE(tracer.get_n_copy_assignments() == 0);
    REQUIRE(tracer.get_n_move_constructions() == 1);
    REQUIRE(tracer.get_n_move_assignments() == 0);

    // Copy assignment is used
    tracer_t::accessor_t new_accessor = tracer.get(24);
    storage = new_accessor;

    REQUIRE(tracer.get_n_copy_constructions() == 1);
    REQUIRE(tracer.get_n_copy_assignments() == 0);
    REQUIRE(tracer.get_n_move_constructions() == 1);
    REQUIRE(tracer.get_n_move_assignments() == 1);

    bool was_invoked = false;
    invoke([&](tracer_t::accessor_t &accessor) {
      REQUIRE(accessor.value_ == 24);
      REQUIRE(tracer.get_n_copy_constructions() == 1);
      REQUIRE(tracer.get_n_copy_assignments() == 0);
      REQUIRE(tracer.get_n_move_constructions() == 1);
      REQUIRE(tracer.get_n_move_assignments() == 1);
      was_invoked  = true;
    }, storage);
    REQUIRE(was_invoked);
  });
}
