#include <exec/materialize.hpp>

#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <catch2/catch.hpp>

using namespace STDEXEC;
using namespace exec;

namespace {

  template <class _Tag, class... _Args>
    requires __completion_tag<std::decay_t<_Tag>>
  using __dematerialize_value = completion_signatures<std::decay_t<_Tag>(_Args...)>;

  TEST_CASE("materialize completion signatures", "[adaptors][materialize]") {
    auto just_ = materialize(just());
    static_assert(sender<decltype(just_)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_)>,
                  completion_signatures<set_value_t(set_value_t)>
    >);
    auto demat_just = dematerialize(just_);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(demat_just)>,
                  completion_signatures<set_value_t()>
    >);

    auto just_string = materialize(just(std::string("foo")));
    static_assert(sender<decltype(just_string)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_string)>,
                  completion_signatures<set_value_t(set_value_t, std::string)>
    >);

    auto demat_just_string = dematerialize(just_string);
    static_assert(std::same_as<
                  completion_signatures_of_t<decltype(demat_just_string)>,
                  completion_signatures<set_value_t(std::string)>
    >);

    auto just_stopped_ = materialize(just_stopped());
    static_assert(sender<decltype(just_stopped_)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_stopped_)>,
                  completion_signatures<set_value_t(set_stopped_t)>
    >);
    auto demat_just_stopped = dematerialize(just_stopped_);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(demat_just_stopped)>,
                  completion_signatures<set_stopped_t()>
    >);
    // wait_for_value(std::move(snd), movable(42));
  }

  TEST_CASE("materialize value", "[adaptors][materialize]") {
    auto just_42 = materialize(just(42));
    auto [tag, i] = *sync_wait(just_42);
    static_assert(std::same_as<decltype(tag), set_value_t>);
    static_assert(std::same_as<decltype(i), int>);
    CHECK(i == 42);
    auto [tag2, i2] = *sync_wait(std::move(just_42));
    CHECK(i2 == 42);
  }

  TEST_CASE("materialize stop", "[adaptors][materialize]") {
    auto just_stop = materialize(just_stopped());
    auto [tag] = *sync_wait(just_stop);
    static_assert(std::same_as<decltype(tag), set_stopped_t>);
  }

  TEST_CASE("dematerialize value", "[adaptors][materialize]") {
    auto just_42 = dematerialize(materialize(just(42)));
    auto [i] = *sync_wait(just_42);
    CHECK(i == 42);
  }
} // namespace
