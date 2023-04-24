#include "exec/sequence/any_sequence_sender_of.hpp"

#include "exec/sequence/once.hpp"
#include "exec/sequence/first_value.hpp"
#include "exec/sequence/empty_sequence.hpp"

#include "catch2/catch.hpp"

template <class... Ts>
using any_receiver = typename exec::any_sequence_receiver_ref<
  stdexec::completion_signatures<stdexec::set_stopped_t(), stdexec::set_value_t(Ts)...>>;

template <class... Ts>
using any_sequence = typename any_receiver<Ts...>::template any_sender<>;

TEST_CASE("construct any sequence from once", "[sequence_senders][any_sequence]") {
  any_sequence<int> s = exec::once(stdexec::just(42));
  auto [i] = stdexec::sync_wait(exec::first_value(std::move(s))).value();
  CHECK(i == 42);
}

TEST_CASE("empty sequence is any sequence", "[sequence_senders][any_sequence]") {
  any_sequence<> s1 = exec::empty_sequence();
  any_sequence<int> s2 = exec::empty_sequence();
  any_sequence<double> s3 = exec::empty_sequence();
  any_sequence<char, float> s4 = exec::empty_sequence();
}