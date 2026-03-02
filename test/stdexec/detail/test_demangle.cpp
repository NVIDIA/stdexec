#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>

namespace
{
  TEST_CASE("demangling a type", "[detail][demangle]")
  {
    struct Dummy
    {
      void operator()(int const) const {}
    };
    auto sndr = STDEXEC::just(42) | STDEXEC::then(Dummy{});

    static_assert(
      std::same_as<STDEXEC::__demangle_t<decltype(sndr)>,
                   STDEXEC::__basic_sender<
                     STDEXEC::then_t,
                     Dummy,
                     STDEXEC::__basic_sender<STDEXEC::just_t, STDEXEC::__tuple<int>>::type>::type>);
  }
}  // namespace
