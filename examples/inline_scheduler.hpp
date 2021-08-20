#pragma once

#include <execution.hpp>
#include <type_traits>
#include <exception>

// A simple scheduler that executes its continuation inline, on the
// thread of the caller of start().
struct inline_scheduler {
  template <class R_>
    struct __op {
      using R = std::__t<R_>;
      [[no_unique_address]] R rec_;
      friend void tag_invoke(std::execution::start_t, __op& op) noexcept try {
        std::execution::set_value((R&&) op.rec_);
      } catch(...) {
        std::execution::set_error((R&&) op.rec_, std::current_exception());
      }
    };

  struct __sender {
    template <template <class...> class Tuple,
              template <class...> class Variant>
      using value_types = Variant<Tuple<>>;
    template <template <class...> class Variant>
      using error_types = Variant<std::exception_ptr>;
    static constexpr bool sends_done = false;

    template <std::execution::receiver_of R>
      friend auto tag_invoke(std::execution::connect_t, __sender, R&& rec)
        noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> __op<std::__id_t<std::remove_cvref_t<R>>> {
        return {(R&&) rec};
      }
  };

  friend __sender tag_invoke(std::execution::schedule_t, const inline_scheduler&) noexcept {
    return {};
  }

  bool operator==(const inline_scheduler&) const noexcept = default;
};
