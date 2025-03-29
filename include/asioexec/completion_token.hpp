#pragma once

#include <cassert>
#include <concepts>
#include <exception>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <boost/asio/associated_allocator.hpp>
#include <boost/asio/associated_executor.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/cancellation_signal.hpp>
#include <boost/asio/cancellation_type.hpp>
#include <boost/asio/prefer.hpp>
#include <boost/asio/query.hpp>
#include <boost/asio/require.hpp>
#include <stdexec/execution.hpp>

namespace asioexec {

namespace detail::completion_token {

template<typename Signature>
struct signature;
template<typename... Args>
struct signature<void(Args...)> : std::type_identity<
  ::stdexec::set_value_t(Args...)> {};

template<typename... Signatures>
using completion_signatures = ::stdexec::completion_signatures<
  typename signature<Signatures>::type...,
  ::stdexec::set_error_t(std::exception_ptr),
  ::stdexec::set_stopped_t()>;

struct stop_callback {
  constexpr explicit stop_callback(::boost::asio::cancellation_signal& signal)
    noexcept : signal_(signal) {}
  void operator()() && noexcept {
    signal_.emit(::boost::asio::cancellation_type::partial);
  }
private:
  ::boost::asio::cancellation_signal& signal_;
};

template<typename>
class completion_handler;

template<typename Receiver>
struct operation_state_base {
  template<typename T>
    requires std::constructible_from<Receiver, T>
  constexpr explicit operation_state_base(T&& t) noexcept(
    std::is_nothrow_constructible_v<Receiver, T> &&
    std::is_nothrow_default_constructible_v<::boost::asio::cancellation_signal>)
    : r_(std::forward<T>(t))
  {}
  Receiver r_;
  ::boost::asio::cancellation_signal signal_;
  completion_handler<Receiver>* h_{nullptr};
  bool* completed_{nullptr};
  template<typename F>
  constexpr void run_(F&& f) noexcept {
    bool completed = false;
    const auto prev = std::exchange(completed_, &completed);
    if (prev) {
      assert(!*prev);
    }
    try {
      std::invoke(std::forward<F>(f));
    } catch (...) {
      completed = true;
      assert(h_);
      h_->release();
      h_ = nullptr;
      ::stdexec::set_error(std::move(r_), std::current_exception());
    }
    if (prev) {
      *prev = completed;
    }
    if (!completed) {
      completed_ = prev;
    }
  }
protected:
  std::optional<
    ::stdexec::stop_callback_for_t<
      ::stdexec::stop_token_of_t<
        ::stdexec::env_of_t<Receiver>>,
    stop_callback>> callback_;
};

template<typename Receiver>
class completion_handler {
  operation_state_base<Receiver>* self_;
  template<typename T, typename... Args>
  void complete_(const T& channel, Args&&... args) noexcept {
    assert(self_);
    auto&& r = self_->r_;
    self_->h_ = nullptr;
    if (self_->completed_) {
      assert(!*self_->completed_);
      *self_->completed_ = true;
      self_->completed_ = nullptr;
    }
    self_ = nullptr;
    std::invoke(channel, std::move(r), std::forward<Args>(args)...);
  }
public:
  constexpr explicit completion_handler(operation_state_base<Receiver>& self)
    noexcept : self_(&self)
  {
    assert(!self_->h_);
    self_->h_ = this;
  }
  constexpr completion_handler(completion_handler&& other) noexcept
    : self_(std::exchange(other.self_, nullptr))
  {
    if (self_) {
      self_->h_ = this;
    }
  }
  completion_handler& operator=(const completion_handler&) = delete;
  constexpr ~completion_handler() noexcept {
    if (self_ && !self_->completed_) {
      ::stdexec::set_stopped(std::move(self_->r_));
    }
  }
  template<typename... Args>
  void operator()(Args&&... args) && noexcept {
    complete_(::stdexec::set_value, std::forward<Args>(args)...);
  }
  using cancellation_slot_type = ::boost::asio::cancellation_slot;
  auto get_cancellation_slot() const noexcept {
    assert(self_);
    return self_->signal_.slot();
  }
  constexpr operation_state_base<Receiver>& state() const noexcept {
    assert(self_);
    return *self_;
  }
  constexpr void release() noexcept {
    self_ = nullptr;
  }
};

template<typename Receiver, typename Initiation, typename Args>
class operation_state : operation_state_base<Receiver> {
  using base_ = operation_state_base<Receiver>;
  Initiation init_;
  Args args_;
public:
  template<typename T, typename U, typename V>
    requires
      std::constructible_from<base_, T> &&
      std::constructible_from<Initiation, U> &&
      std::constructible_from<Args, V>
  constexpr explicit operation_state(T&& t, U&& u, V&& v) noexcept(
    std::is_nothrow_constructible_v<base_, T> &&
    std::is_nothrow_constructible_v<Initiation, U> &&
    std::is_nothrow_constructible_v<Args, V>)
    : base_(std::forward<T>(t)),
      init_(std::forward<U>(u)),
      args_(std::forward<Args>(v))
  {}
  constexpr void start() & noexcept {
    try {
      std::apply(
        [&](auto&&... args) {
          std::invoke(
            std::move(init_),
            completion_handler<Receiver>(*this),
            std::forward<decltype(args)>(args)...);
        },
        std::move(args_));
    } catch (...) {
      ::stdexec::set_error(std::move(base_::r_), std::current_exception());
      return;
    }
    base_::callback_.emplace(
      ::stdexec::get_stop_token(::stdexec::get_env(base_::r_)),
      stop_callback(base_::signal_));
  }
};

template<typename Signatures, typename Initiation, typename... Args>
class sender {
  using args_type_ = std::tuple<std::decay_t<Args>...>;
public:
  using sender_concept = ::stdexec::sender_t;
  using completion_signatures = Signatures;
  template<typename T, typename... Us>
    requires
      std::constructible_from<Initiation, T> &&
      std::constructible_from<args_type_, Us...>
  explicit constexpr sender(
    T&& t,
    Us&&... us) noexcept(
      std::is_nothrow_constructible_v<Initiation, T> &&
      std::is_nothrow_constructible_v<args_type_, Us...>)
      : init_(std::forward<T>(t)),
        args_(std::forward<Us>(us)...)
  {}
  template<typename Sender, typename Receiver>
    requires ::stdexec::receiver_of<
      std::remove_cvref_t<Receiver>,
      completion_signatures>
  constexpr auto connect(this Sender&& sender, Receiver&& receiver) noexcept(
    std::is_nothrow_constructible_v<
      operation_state<
        std::remove_cvref_t<Receiver>,
        Initiation,
        args_type_>,
      Receiver,
      decltype(std::declval<Sender>().init_),
      decltype(std::declval<Sender>().args_)>)
  {
    return operation_state<
      std::remove_cvref_t<Receiver>,
      Initiation,
      args_type_>(
        std::forward<Receiver>(receiver),
        std::forward<Sender>(sender).init_,
        std::forward<Sender>(sender).args_);
  }
private:
  Initiation init_;
  args_type_ args_;
};

template<typename Receiver, typename Executor>
class executor {
  operation_state_base<Receiver>& self_;
  Executor ex_;
  template<typename F>
  constexpr auto wrap_(F f) const noexcept(
    std::is_nothrow_move_constructible_v<F>)
  {
    return [&self = self_, f = std::move(f)]() mutable noexcept {
      self.run_(std::move(f));
    };
  }
public:
  explicit constexpr executor(
    operation_state_base<Receiver>& self,
    const Executor& ex) noexcept
    : self_(self),
      ex_(ex)
  {}
  template<typename Query>
    requires requires {
      boost::asio::query(
        std::declval<const Executor&>(),
        std::declval<const Query&>());
    }
  constexpr decltype(auto) query(const Query& q) const noexcept {
    return boost::asio::query(ex_, q);
  }
  template<typename... Args>
    requires requires {
      boost::asio::prefer(
        std::declval<const Executor&>(),
        std::declval<Args>()...);
    }
  constexpr decltype(auto) prefer(Args&&... args) const noexcept {
    const auto ex = boost::asio::prefer(ex_, std::forward<Args>(args)...);
    return executor<
      Receiver,
      std::remove_cvref_t<decltype(ex)>>(
        self_,
        ex);
  }
  template<typename... Args>
    requires requires {
      boost::asio::require(
        std::declval<const Executor&>(),
        std::declval<Args>()...);
    }
  constexpr decltype(auto) require(Args&&... args) const noexcept {
    const auto ex = boost::asio::require(ex_, std::forward<Args>(args)...);
    return executor<
      Receiver,
      std::remove_cvref_t<decltype(ex)>>(
        self_,
        ex);
  }
  template<typename T>
  void execute(T&& t) const noexcept {
    self_.run_([&]() {
      ex_.execute(wrap_(std::forward<T>(t)));
    });
  }
  constexpr void on_work_started() const noexcept requires requires {
    std::declval<const Executor&>().on_work_started();
  }
  {
    ex_.on_work_started();
  }
  constexpr void on_work_finished() const noexcept requires requires {
    std::declval<const Executor&>().on_work_finished();
  }
  {
    ex_.on_work_finished();
  }
  template<typename F, typename A>
    requires requires {
      std::declval<const Executor&>().dispatch(
        std::declval<F>(),
        std::declval<const A&>());
    }
  constexpr void dispatch(F&& f, const A& a) const noexcept {
    self_.run_([&]() {
      ex_.dispatch(wrap_(std::forward<F>(f)), a);
    });
  }
  template<typename F, typename A>
    requires requires {
      std::declval<const Executor&>().post(
        std::declval<F>(),
        std::declval<const A&>());
    }
  constexpr void post(F&& f, const A& a) const noexcept {
    self_.run_([&]() {
      ex_.post(wrap_(std::forward<F>(f)), a);
    });
  }
  template<typename F, typename A>
    requires requires {
      std::declval<const Executor&>().defer(
        std::declval<F>(),
        std::declval<const A&>());
    }
  constexpr void defer(F&& f, const A& a) const noexcept {
    self_.run_([&]() {
      ex_.defer(wrap_(std::forward<F>(f)), a);
    });
  }
  constexpr bool operator==(const executor& rhs) const noexcept {
    return (&self_ == &rhs.self_) && (ex_ == rhs.ex_);
  }
  bool operator!=(const executor& rhs) const = default;
};

}

struct completion_token_t {};
inline const completion_token_t completion_token{};

}

namespace boost::asio {

template<typename... Signatures>
struct async_result<::asioexec::completion_token_t, Signatures...> {
  template<typename Initiation, typename... Args>
  static constexpr auto initiate(
    Initiation&& i,
    const ::asioexec::completion_token_t&,
    Args&&... args)
  {
    return ::asioexec::detail::completion_token::sender<
      ::asioexec::detail::completion_token::completion_signatures<
        Signatures...>,
      std::remove_cvref_t<Initiation>,
      Args...>(
        std::forward<Initiation>(i),
        std::forward<Args>(args)...);
  }
};

template<typename Receiver, typename Executor>
struct associated_executor<
  ::asioexec::detail::completion_token::completion_handler<Receiver>,
  Executor>
{
  using type = ::asioexec::detail::completion_token::executor<
    Receiver,
    Executor>;
  static type get(
    const ::asioexec::detail::completion_token::completion_handler<Receiver>& h,
    const Executor& ex) noexcept
  {
    return type(h.state(), ex);
  }
};

template<typename Receiver, typename Allocator>
  requires requires(const Receiver& r) {
    ::stdexec::get_allocator(::stdexec::get_env(r));
  }
struct associated_allocator<
  ::asioexec::detail::completion_token::completion_handler<Receiver>,
  Allocator>
{
  using type = std::remove_cvref_t<
    decltype(
      ::stdexec::get_allocator(
        ::stdexec::get_env(
          std::declval<const Receiver&>())))>;
  static type get(
    const ::asioexec::detail::completion_token::completion_handler<Receiver>& h,
    const Allocator&) noexcept
  {
    return ::stdexec::get_allocator(
      ::stdexec::get_env(
        h.state().r_));
  }
};

}
