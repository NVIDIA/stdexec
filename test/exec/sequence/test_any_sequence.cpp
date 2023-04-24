#include "exec/sequence/any_sequence_sender_of.hpp"

#include "exec/sequence/once.hpp"
#include "exec/sequence/first_value.hpp"

#include "catch2/catch.hpp"

template <class... Ts>
using any_receiver = typename exec::any_sequence_receiver_ref<
  stdexec::completion_signatures<stdexec::set_stopped_t(), stdexec::set_value_t(Ts)...>>;

template <class... Ts>
using any_sequence = typename any_receiver<Ts...>::template any_sender<>;

TEST_CASE("construct any sequence", "[sequence][any_sequence]") {
  any_sequence<int> s = exec::once(stdexec::just(42));
  auto [i] = stdexec::sync_wait(exec::first_value(std::move(s))).value();
  CHECK(i == 42);
}