/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include <exec/sequence_senders.hpp>
#include <exec/sequence/notification.hpp>
#include <exec/sequence/marbles.hpp>
#include "exec/sequence/iterate.hpp"
#include "exec/sequence/transform_each.hpp"
#include "exec/variant_sender.hpp"
#include "stdexec/__detail/__then.hpp"

#include <catch2/catch.hpp>

namespace std {
  inline std::string to_string(const std::error_code __error) noexcept {
    return __error.message();
  }
  inline std::string to_string(const std::exception_ptr __ex) noexcept {
    try {
      std::rethrow_exception(__ex);
    } catch (const std::exception& __ex) {
      return __ex.what();
    }
  }
} // namespace std

namespace stdexec::__rcvrs {
  inline std::string to_string(set_value_t) noexcept {
    return {"set_value"};
  }
  inline std::string to_string(set_error_t) noexcept {
    return {"set_error"};
  }
  inline std::string to_string(set_stopped_t) noexcept {
    return {"set_stopped"};
  }
} // namespace stdexec::__rcvrs

namespace exec::__sequence_sender {
  inline std::string to_string(set_next_t) noexcept {
    return {"set_next"};
  }
} // namespace exec::__sequence_sender

namespace exec::__marbles {
  inline std::string to_string(sequence_start_t) noexcept {
    return {"sequence_start"};
  }
  inline std::string to_string(sequence_connect_t) noexcept {
    return {"sequence_connect"};
  }
  inline std::string to_string(sequence_end_t) noexcept {
    return {"sequence_end"};
  }
  inline std::string to_string(sequence_error_t) noexcept {
    return {"sequence_error"};
  }
  inline std::string to_string(sequence_stopped_t) noexcept {
    return {"sequence_stopped"};
  }
  inline std::string to_string(request_stop_t) noexcept {
    return {"request_stop"};
  }
} // namespace exec::__marbles

namespace Catch {
  template <class _Clock, class _Duration>
  struct StringMaker<std::chrono::time_point<_Clock, _Duration>> {
    static std::string convert(const std::chrono::time_point<_Clock, _Duration>& __at) {
      return std::to_string(
               std::chrono::duration_cast<std::chrono::milliseconds>(__at.time_since_epoch())
                 .count())
           + "ms";
    }
  };

  template <class _Rep, class _Period>
  struct StringMaker<std::chrono::duration<_Rep, _Period>> {
    static std::string convert(const std::chrono::duration<_Rep, _Period>& __duration) {
      return std::to_string(
               std::chrono::duration_cast<std::chrono::milliseconds>(__duration).count())
           + "ms";
    }
  };

  template <class _Clock>
  struct StringMaker<exec::marble_t<_Clock>> {
    static std::string convert(const exec::marble_t<_Clock>& __value) {
      return to_string(__value);
    }
  };

  template <class _CompletionSignatures>
  struct StringMaker<exec::notification_t<_CompletionSignatures>> {
    static std::string convert(const exec::notification_t<_CompletionSignatures>& __value) {
      return to_string(__value);
    }
  };

  template <stdexec::__one_of<
    exec::sequence_end_t,
    exec::sequence_error_t,
    exec::sequence_stopped_t,
    stdexec::set_value_t,
    stdexec::set_error_t,
    stdexec::set_stopped_t,
    exec::set_next_t
  > _Tag>
  struct StringMaker<_Tag> {
    static std::string convert(const _Tag& __tag) {
      return to_string(__tag);
    }
  };
} // namespace Catch

namespace {
  using namespace exec;
  namespace ex = stdexec;
  using ex::operator""_mstr;
  using namespace std::chrono_literals;

  template <ex::sender Sender>
  struct as_sequence_t : Sender {
    using sender_concept = sequence_sender_t;
    template <stdexec::__decays_to<as_sequence_t> _Self, class... _Env>
    static auto get_item_types(_Self&&, _Env&&...) noexcept -> exec::__item_types_of_t<Sender> {
      return {};
    }
    auto subscribe(auto receiver) {
      return connect(set_next(receiver, *static_cast<Sender*>(this)), receiver);
    }
  };

  // a sequence adaptor that applies a function to each item
  [[maybe_unused]]
  static constexpr auto then_each = [](auto f) {
    return exec::transform_each(stdexec::then(f));
  };
  // a sequence adaptor that applies a function to each item
  // the function must produce a sequence
  // all the sequences returned from the function are merged
  [[maybe_unused]]
  static constexpr auto flat_map = [](auto&& f) {
    auto map_merge = [](auto&& sequence, auto&& f) noexcept {
      return merge_each(
        exec::transform_each(
          static_cast<decltype(sequence)&&>(sequence), ex::then(static_cast<decltype(f)&&>(f))));
    };
    return stdexec::__binder_back<decltype(map_merge), decltype(f)>{
      {static_cast<decltype(f)&&>(f)}, {}, {}};
  };
  // when_all requires a successful completion
  // however stop_after_on has no successful completion
  // this uses variant_sender to add a successful completion
  // (the successful completion will never occur)
  [[maybe_unused]]
  static constexpr auto with_void = [](auto&& sender) noexcept
    -> variant_sender<stdexec::__call_result_t<ex::just_t>, decltype(sender)> {
    return {static_cast<decltype(sender)&&>(sender)};
  };
  // with_stop_token_from adds get_stop_token query, that returns the
  // token for the provided stop_source, to the receiver env
  [[maybe_unused]]
  static constexpr auto with_stop_token_from = [](auto& stop_source) noexcept {
    return ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()});
  };
  // log_start completes with the provided sequence after printing provided string
  [[maybe_unused]]
  static constexpr auto log_start = [](auto sequence, auto message) {
    return exec::sequence(
      ex::read_env(ex::get_stop_token) | stdexec::then([message](auto&& token) noexcept {
        UNSCOPED_INFO(
          message << (token.stop_requested() ? ", stop was requested" : ", stop not requested")
                  << ", on thread id: " << std::this_thread::get_id());
      }),
      ex::just(sequence));
  };
  // log_sequence prints the message when each value in the sequence is emitted
  [[maybe_unused]]
  static constexpr auto log_sequence = [](auto sequence, auto message) {
    return sequence | then_each([message](auto&& value) mutable noexcept {
             UNSCOPED_INFO(message << ", on thread id: " << std::this_thread::get_id());
             return value;
           });
  };
  // emits_stopped completes with set_stopped after printing info
  [[maybe_unused]]
  static constexpr auto emits_stopped = []() {
    return ex::just() | stdexec::let_value([]() noexcept {
             UNSCOPED_INFO("emitting stopped, on thread id: " << std::this_thread::get_id());
             return ex::just_stopped();
           });
  };
  // emits_error completes with set_error(error) after printing info
  [[maybe_unused]]
  static constexpr auto emits_error = [](auto error) {
    return ex::just() | stdexec::let_value([error]() noexcept {
             UNSCOPED_INFO(error.what() << ", on thread id: " << std::this_thread::get_id());
             return ex::just_error(error);
           });
  };

#if STDEXEC_HAS_STD_RANGES()

  // a sequence of numbers from itoa()
  [[maybe_unused]]
  static constexpr auto range = [](auto from, auto to) {
    return exec::iterate(std::views::iota(from, to));
  };

#endif // STDEXEC_HAS_STD_RANGES()

} // namespace
