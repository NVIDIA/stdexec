/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <concepts>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <asioexec/asio_config.hpp>
#include <stdexec/execution.hpp>

namespace asioexec {

  namespace detail::completion_token {

    //  The machinery from here down through completion_token::set_value is to
    //  work around the fact that Asio allows operations to declare signatures which
    //  aren't reflective of the actual cv- & ref-qualifications with which the
    //  values will actually be sent. This means that one or more conversions may be
    //  required to actually send the values correctly which may throw and which
    //  therefore must occur before the call to ::stdexec::set_value.

    //  The technique used to achieve this is to utilize an unevaluated call against
    //  overload_set to determine the matching Asio signature. Then
    //  completion_token::convert is used to transform the values actually sent by
    //  the Asio operation so that they match the selected completion signature
    //  (this may throw). The converted values may then be sent to the receiver with
    //  no possibility of throwing.
    template <typename>
    struct has_function_call_operator {
      struct type { };

      std::tuple<> operator()(type) const;
    };

    template <typename... Args>
    struct has_function_call_operator<::stdexec::set_value_t(Args...)> {
      std::tuple<Args...> operator()(Args...) const;
    };

    template <typename>
    struct overload_set;

    template <typename... Signatures>
    struct overload_set<::stdexec::completion_signatures<Signatures...>>
      : has_function_call_operator<Signatures>... {
      using has_function_call_operator<Signatures>::operator()...;
    };

    template <typename T, typename U>
      requires std::is_same_v<T&&, U&&> || std::is_convertible_v<U&&, T&&>
    constexpr T&& convert(U&& u) noexcept {
      return static_cast<U&&>(u);
    }

    template <typename T, typename U>
    constexpr std::remove_cvref_t<T> convert(U&& u) {
      return static_cast<U&&>(u);
    }

    template <typename Tuple, typename Receiver, std::size_t... Ns, typename... Args>
    constexpr void set_value_impl(Receiver&& r, std::index_sequence<Ns...>, Args&&... args) {
      ::stdexec::set_value(
        static_cast<Receiver&&>(r),
        completion_token::convert<std::tuple_element_t<Ns, Tuple>>(static_cast<Args&&>(args))...);
    }

    template <typename Signatures, typename Receiver, typename... Args>
    constexpr void set_value(Receiver&& r, Args&&... args) {
      using tuple = decltype(std::declval<overload_set<Signatures>>()(std::declval<Args>()...));
      completion_token::set_value_impl<tuple>(
        static_cast<Receiver&&>(r),
        std::make_index_sequence<std::tuple_size_v<tuple>>{},
        static_cast<Args&&>(args)...);
    }

    template <typename Signature>
    struct signature;

    template <typename... Args>
    struct signature<void(Args...)> {
      using type = ::stdexec::set_value_t(Args...);
    };

    template <typename... Signatures>
    using completion_signatures = ::stdexec::completion_signatures<
      typename signature<Signatures>::type...,
      ::stdexec::set_error_t(std::exception_ptr),
      ::stdexec::set_stopped_t()
    >;

    struct stop_callback {
      constexpr explicit stop_callback(asio_impl::cancellation_signal& signal) noexcept
        : signal_(signal) {
      }

      void operator()() && noexcept {
        signal_.emit(asio_impl::cancellation_type::partial);
      }
     private:
      asio_impl::cancellation_signal& signal_;
    };

    template <typename, typename>
    class completion_handler;

    template <typename Signatures, typename Receiver>
    struct operation_state_base {
      class frame_;

      template <typename T>
        requires std::constructible_from<Receiver, T>
      explicit operation_state_base(T&& t) noexcept(
        std::is_nothrow_constructible_v<Receiver, T>
        && std::is_nothrow_default_constructible_v<asio_impl::cancellation_signal>)
        : r_(static_cast<T&&>(t)) {
      }

      Receiver r_;
      asio_impl::cancellation_signal signal_;
      std::recursive_mutex m_;
      frame_* frames_{nullptr};
      std::exception_ptr ex_;
      completion_handler<Signatures, Receiver>* h_{nullptr};

      class frame_ {
        operation_state_base& self_;
        std::unique_lock<std::recursive_mutex> l_;
        frame_* prev_;
       public:
        explicit frame_(operation_state_base& self) noexcept
          : self_(self)
          , l_(self.m_)
          , prev_(self.frames_) {
          self.frames_ = this;
        }

        frame_(const frame_&) = delete;

        ~frame_() noexcept {
          if (l_) {
            STDEXEC_ASSERT(self_.frames_ == this);
            self_.frames_ = prev_;
            if (!self_.frames_ && !self_.h_) {
              //  We are the last frame and the handler is gone so it's up to us to
              //  finalize the operation
              l_.unlock();
              if (self_.ex_) {
                ::stdexec::set_error(static_cast<Receiver&&>(self_.r_), std::move(self_.ex_));
              } else {
                ::stdexec::set_stopped(static_cast<Receiver&&>(self_.r_));
              }
            }
          }
        }

        explicit operator bool() const noexcept {
          return bool(l_);
        }

        void release() noexcept {
          auto ptr = this;
          do {
            STDEXEC_ASSERT(ptr->l_);
            STDEXEC_ASSERT(self_.frames_ == ptr);
            ptr = ptr->prev_;
            self_.frames_->l_.unlock();
            self_.frames_->prev_ = nullptr;
            self_.frames_ = ptr;
          } while (ptr);
        }
      };

      template <typename F>
      void run_(F&& f) noexcept {
        const frame_ frame(*this);
        STDEXEC_TRY {
          static_cast<F&&>(f)();
        }
        STDEXEC_CATCH_ALL {
          STDEXEC_ASSERT(frame);
          //  Do not overwrite the first exception encountered
          if (!ex_) {
            ex_ = std::current_exception();
          }
        }
      }
     protected:
      std::optional<::stdexec::stop_callback_for_t<
        ::stdexec::stop_token_of_t<::stdexec::env_of_t<Receiver>>,
        stop_callback
      >>
        callback_;
    };

    template <typename Signatures, typename Receiver>
    class completion_handler {
      operation_state_base<Signatures, Receiver>* self_;
     public:
      explicit completion_handler(operation_state_base<Signatures, Receiver>& self) noexcept
        : self_(&self) {
        STDEXEC_ASSERT(!self_->h_);
        self_->h_ = this;
      }

      completion_handler(completion_handler&& other) noexcept
        : self_(std::exchange(other.self_, nullptr)) {
        if (self_) {
          const std::lock_guard l(self_->m_);
          self_->h_ = this;
        }
      }

      completion_handler& operator=(const completion_handler&) = delete;

      ~completion_handler() noexcept {
        if (self_) {
          //  When this goes out of scope it might send set stopped or set error, or
          //  it might defer that to the executor frames above us on the call stack
          //  (if any)
          const typename operation_state_base<Signatures, Receiver>::frame_ frame(*self_);
          self_->h_ = nullptr;
        }
      }

      template <typename... Args>
      void operator()(Args&&... args) noexcept {
        STDEXEC_ASSERT(self_);
        {
          const std::lock_guard l(self_->m_);
          if (self_->frames_) {
            self_->frames_->release();
          }
          STDEXEC_ASSERT(!self_->frames_);
        }
        if (self_->ex_) {
          ::stdexec::set_error(static_cast<Receiver&&>(self_->r_), std::move(self_->ex_));
        } else {
          STDEXEC_TRY {
            completion_token::set_value<Signatures>(
              static_cast<Receiver&&>(self_->r_), static_cast<Args&&>(args)...);
          }
          STDEXEC_CATCH_ALL {
            ::stdexec::set_error(static_cast<Receiver&&>(self_->r_), std::current_exception());
          }
        }
        //  Makes destructor a no op, the operation is complete so there's nothing
        //  more to do
        self_ = nullptr;
      }

      using cancellation_slot_type = asio_impl::cancellation_slot;

      auto get_cancellation_slot() const noexcept {
        STDEXEC_ASSERT(self_);
        return self_->signal_.slot();
      }

      operation_state_base<Signatures, Receiver>& state() const noexcept {
        STDEXEC_ASSERT(self_);
        return *self_;
      }
    };

    template <typename Signatures, typename Receiver, typename Initiation, typename Args>
    class operation_state : operation_state_base<Signatures, Receiver> {
      using base_ = operation_state_base<Signatures, Receiver>;
      Initiation init_;
      Args args_;
     public:
      template <typename T, typename U, typename V>
        requires std::constructible_from<base_, T> && std::constructible_from<Initiation, U>
                && std::constructible_from<Args, V>
      explicit operation_state(T&& t, U&& u, V&& v) noexcept(
        std::is_nothrow_constructible_v<base_, T> && std::is_nothrow_constructible_v<Initiation, U>
        && std::is_nothrow_constructible_v<Args, V>)
        : base_(static_cast<T&&>(t))
        , init_(static_cast<U&&>(u))
        , args_(static_cast<V&&>(v)) {
      }

      void start() & noexcept {
        const typename base_::frame_ frame(*this);
        STDEXEC_TRY {
          std::apply(
            [&](auto&&... args) {
              std::invoke(
                static_cast<Initiation&&>(init_),
                completion_handler<Signatures, Receiver>(*this),
                static_cast<decltype(args)&&>(args)...);
            },
            std::move(args_));
        }
        STDEXEC_CATCH_ALL {
          if (!base_::ex_) {
            base_::ex_ = std::current_exception();
          }
          return;
        }
        //  In the case of an immediate completion *this may already be outside its
        //  lifetime so we can't proceed into the branch
        if (frame) {
          base_::callback_.emplace(
            ::stdexec::get_stop_token(::stdexec::get_env(base_::r_)),
            stop_callback(base_::signal_));
        }
      }
    };

    template <typename Signatures, typename Initiation, typename... Args>
    class sender {
      using args_type_ = std::tuple<std::decay_t<Args>...>;
     public:
      using sender_concept = ::stdexec::sender_t;

      template <typename T, typename... Us>
        requires std::constructible_from<Initiation, T>
                && std::constructible_from<args_type_, Us...>
      explicit constexpr sender(T&& t, Us&&... us) noexcept(
        std::is_nothrow_constructible_v<Initiation, T>
        && std::is_nothrow_constructible_v<args_type_, Us...>)
        : init_(static_cast<T&&>(t))
        , args_(static_cast<Us&&>(us)...) {
      }

      template <typename Env>
        requires std::is_copy_constructible_v<Initiation>
              && std::is_copy_constructible_v<args_type_>
      constexpr Signatures get_completion_signatures(const Env&) const & noexcept {
        return {};
      }

      template <typename Env>
        requires std::is_move_constructible_v<Initiation>
              && std::is_move_constructible_v<args_type_>
      constexpr Signatures get_completion_signatures(const Env&) && noexcept {
        return {};
      }

      template <typename Receiver>
        requires ::stdexec::receiver_of<
          std::remove_cvref_t<Receiver>,
          ::stdexec::completion_signatures_of_t<const sender&, ::stdexec::env_of_t<Receiver>>
        >
      constexpr auto connect(Receiver&& receiver) const & noexcept(
        std::is_nothrow_constructible_v<
          operation_state<Signatures, std::remove_cvref_t<Receiver>, Initiation, args_type_>,
          Receiver,
          const Initiation&,
          const args_type_&
        >) {
        return operation_state<Signatures, std::remove_cvref_t<Receiver>, Initiation, args_type_>(
          static_cast<Receiver&&>(receiver), init_, args_);
      }

      template <typename Receiver>
        requires ::stdexec::receiver_of<
          std::remove_cvref_t<Receiver>,
          ::stdexec::completion_signatures_of_t<sender, ::stdexec::env_of_t<Receiver>>
        >
      constexpr auto connect(Receiver&& receiver) && noexcept(
        std::is_nothrow_constructible_v<
          operation_state<Signatures, std::remove_cvref_t<Receiver>, Initiation, args_type_>,
          Receiver,
          Initiation,
          args_type_
        >) {
        return operation_state<Signatures, std::remove_cvref_t<Receiver>, Initiation, args_type_>(
          static_cast<Receiver&&>(receiver),
          static_cast<Initiation&&>(init_),
          static_cast<args_type_&&>(args_));
      }
     private:
      Initiation init_;
      args_type_ args_;
    };

    template <typename Signatures, typename Receiver, typename Executor>
    class executor {
      operation_state_base<Signatures, Receiver>& self_;
      Executor ex_;

      template <typename F>
      constexpr auto wrap_(F f) const noexcept(std::is_nothrow_move_constructible_v<F>) {
        return [&self = self_, f = std::move(f)]() mutable noexcept {
          self.run_(std::move(f));
        };
      }
     public:
      explicit constexpr executor(
        operation_state_base<Signatures, Receiver>& self,
        const Executor& ex) noexcept
        : self_(self)
        , ex_(ex) {
      }

      template <typename Query>
        requires requires {
          asio_impl::query(std::declval<const Executor&>(), std::declval<const Query&>());
        }
      constexpr decltype(auto) query(const Query& q) const noexcept {
        return asio_impl::query(ex_, q);
      }

      template <typename... Args>
        requires requires {
          asio_impl::prefer(std::declval<const Executor&>(), std::declval<Args>()...);
        }
      constexpr decltype(auto) prefer(Args&&... args) const noexcept {
        const auto ex = asio_impl::prefer(ex_, static_cast<Args&&>(args)...);
        return executor<Signatures, Receiver, std::remove_cvref_t<decltype(ex)>>(self_, ex);
      }

      template <typename... Args>
        requires requires {
          asio_impl::require(std::declval<const Executor&>(), std::declval<Args>()...);
        }
      constexpr decltype(auto) require(Args&&... args) const noexcept {
        const auto ex = asio_impl::require(ex_, static_cast<Args&&>(args)...);
        return executor<Signatures, Receiver, std::remove_cvref_t<decltype(ex)>>(self_, ex);
      }

      template <typename T>
      void execute(T&& t) const noexcept {
        self_.run_([&]() { ex_.execute(wrap_(static_cast<T&&>(t))); });
      }

      constexpr void on_work_started() const noexcept
        requires requires { std::declval<const Executor&>().on_work_started(); }
      {
        ex_.on_work_started();
      }

      constexpr void on_work_finished() const noexcept
        requires requires { std::declval<const Executor&>().on_work_finished(); }
      {
        ex_.on_work_finished();
      }

      template <typename F, typename A>
        requires requires {
          std::declval<const Executor&>().dispatch(std::declval<F>(), std::declval<const A&>());
        }
      constexpr void dispatch(F&& f, const A& a) const noexcept {
        self_.run_([&]() { ex_.dispatch(wrap_(static_cast<F&&>(f)), a); });
      }

      template <typename F, typename A>
        requires requires {
          std::declval<const Executor&>().post(std::declval<F>(), std::declval<const A&>());
        }
      constexpr void post(F&& f, const A& a) const noexcept {
        self_.run_([&]() { ex_.post(wrap_(static_cast<F&&>(f)), a); });
      }

      template <typename F, typename A>
        requires requires {
          std::declval<const Executor&>().defer(std::declval<F>(), std::declval<const A&>());
        }
      constexpr void defer(F&& f, const A& a) const noexcept {
        self_.run_([&]() { ex_.defer(wrap_(static_cast<F&&>(f)), a); });
      }

      constexpr bool operator==(const executor& rhs) const noexcept {
        return (&self_ == &rhs.self_) && (ex_ == rhs.ex_);
      }

      bool operator!=(const executor& rhs) const = default;
    };

  } // namespace detail::completion_token

  struct completion_token_t { };

  inline const completion_token_t completion_token{};

} // namespace asioexec

namespace ASIOEXEC_ASIO_NAMESPACE {

  template <typename... Signatures>
  struct async_result<::asioexec::completion_token_t, Signatures...> {
    template <typename Initiation, typename... Args>
    static constexpr auto
      initiate(Initiation&& i, const ::asioexec::completion_token_t&, Args&&... args) {
      return ::asioexec::detail::completion_token::sender<
        ::asioexec::detail::completion_token::completion_signatures<Signatures...>,
        std::remove_cvref_t<Initiation>,
        Args...
      >(static_cast<Initiation&&>(i), static_cast<Args&&>(args)...);
    }
  };

  template <typename Signatures, typename Receiver, typename Executor>
  struct associated_executor<
    ::asioexec::detail::completion_token::completion_handler<Signatures, Receiver>,
    Executor
  > {
    using type = ::asioexec::detail::completion_token::executor<Signatures, Receiver, Executor>;

    static type get(
      const ::asioexec::detail::completion_token::completion_handler<Signatures, Receiver>& h,
      const Executor& ex) noexcept {
      return type(h.state(), ex);
    }
  };

  template <typename Signatures, typename Receiver, typename Allocator>
    requires requires(const Receiver& r) { ::stdexec::get_allocator(::stdexec::get_env(r)); }
  struct associated_allocator<
    ::asioexec::detail::completion_token::completion_handler<Signatures, Receiver>,
    Allocator
  > {
    using type = std::remove_cvref_t<decltype(::stdexec::get_allocator(
      ::stdexec::get_env(std::declval<const Receiver&>())))>;

    static type get(
      const ::asioexec::detail::completion_token::completion_handler<Signatures, Receiver>& h,
      const Allocator&) noexcept {
      return ::stdexec::get_allocator(::stdexec::get_env(h.state().r_));
    }
  };

} // namespace ASIOEXEC_ASIO_NAMESPACE
