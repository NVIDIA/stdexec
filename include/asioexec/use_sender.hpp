#pragma once

#include <concepts>
#include <exception>
#include <type_traits>
#include <utility>
#include <boost/asio/error.hpp>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <stdexec/execution.hpp>
#include "completion_token.hpp"
//#include "inlinable_receiver.hpp"

namespace asioexec {

namespace detail::use_sender {

template<typename T>
concept is_error_code = std::same_as<
  std::remove_cvref_t<T>,
  ::boost::system::error_code>;

inline std::exception_ptr to_exception_ptr(::boost::system::error_code ec)
  noexcept
{
  try {
    return std::make_exception_ptr(
      ::boost::system::system_error(std::move(ec)));
  } catch (...) {
    return std::current_exception();
  }
}

template<typename Receiver>
struct receiver {
  template<typename T>
    requires std::constructible_from<Receiver, T>
  constexpr explicit receiver(T&& t) noexcept(
    std::is_nothrow_constructible_v<Receiver, T>)
    : r_(std::forward<T>(t))
  {}
  using receiver_concept = ::stdexec::receiver_t;
  constexpr void set_stopped() && noexcept {
    ::stdexec::set_stopped(std::move(r_));
  }
  constexpr void set_error(std::exception_ptr ex) && noexcept {
    ::stdexec::set_error(std::move(r_), std::move(ex));
  }
  template<typename T, typename... Args>
    requires is_error_code<T>
  constexpr void set_value(T&& t, Args&&... args) && noexcept {
    if (!t) {
      ::stdexec::set_value(std::move(r_), std::forward<Args>(args)...);
      return;
    }
    if (t == ::boost::asio::error::operation_aborted) {
      ::stdexec::set_stopped(std::move(r_));
      return;
    }
    ::stdexec::set_error(
      std::move(r_),
      use_sender::to_exception_ptr(std::forward<T>(t)));
  }
  constexpr decltype(auto) get_env() const noexcept {
    return ::stdexec::get_env(r_);
  }
  //template<typename ChildOp>
  //  requires
  //    execution::inlinable_receiver<Receiver, ChildOp> &&
  //    std::is_nothrow_move_constructible_v<Receiver>
  //constexpr static receiver make_receiver_for(ChildOp* op) noexcept {
  //  return receiver(Receiver::make_receiver_for(op));
  //}
private:
  Receiver r_;
};

template<typename Signature>
struct transform_completion_signature : std::type_identity<Signature> {};
template<typename T, typename... Args>
  requires is_error_code<T>
struct transform_completion_signature<::stdexec::set_value_t(T, Args...)>
  : std::type_identity<::stdexec::set_value_t(Args...)> {};

template<typename>
struct completion_signatures;
template<typename... Signatures>
struct completion_signatures<
  ::stdexec::completion_signatures<Signatures...>> : std::type_identity<
    ::stdexec::completion_signatures<
      typename transform_completion_signature<Signatures>::type...>> {};

template<typename Sender>
struct sender {
  template<typename T>
    requires std::constructible_from<Sender, T>
  constexpr explicit sender(T&& t) noexcept(
    std::is_nothrow_constructible_v<Sender, T>)
    : s_(std::forward<T>(t))
  {}
  using sender_concept = ::stdexec::sender_t;
  template<typename Self, typename Env>
  constexpr auto get_completion_signatures(this Self&&, const Env&) noexcept ->
    typename completion_signatures<
      ::stdexec::completion_signatures_of_t<
        decltype(std::declval<Self>().s_),
        Env>>::type;
  template<typename Self, ::stdexec::receiver Receiver>
    requires ::stdexec::sender_to<
      decltype(std::forward_like<Self>(std::declval<Sender&>())),
      receiver<Receiver>>
  constexpr auto connect(this Self&& self, Receiver r) noexcept(
    std::is_nothrow_constructible_v<receiver<Receiver>, Receiver> &&
    noexcept(::stdexec::connect(
      std::declval<Self>().s_,
      std::declval<receiver<Receiver>>())))
  {
    return ::stdexec::connect(
      std::forward<Self>(self).s_,
      receiver<Receiver>(std::move(r)));
  }
private:
  Sender s_;
};

template<typename Sender>
explicit sender(Sender) -> sender<Sender>;

}

struct use_sender_t {};
inline const use_sender_t use_sender{};

}

namespace boost::asio {

template<typename... Signatures>
struct async_result<::asioexec::use_sender_t, Signatures...> {
  template<typename Initiation, typename... Args>
  static constexpr auto initiate(
    Initiation&& i,
    const ::asioexec::use_sender_t&,
    Args&&... args)
  {
    return ::asioexec::detail::use_sender::sender(
      async_result<
        ::asioexec::completion_token_t,
        Signatures...>::initiate(
          std::forward<Initiation>(i),
          ::asioexec::completion_token,
          std::forward<Args>(args)...));
  }
};

}
