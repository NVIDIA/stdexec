#pragma once

#include <stdexec/execution.hpp>
#include <boost/asio.hpp>

namespace asioexec {

  template <class... Ts>
  class sender {
   public:
    using sender_concept = stdexec::sender_t;
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(Ts...),
      stdexec::set_error_t(std::exception_ptr),
      stdexec::set_stopped_t()>;

    sender(auto &&init_start_func, auto &&...init_start_args)
      : initiation(new asio_initiation(
          std::forward<decltype(init_start_func)>(init_start_func),
          std::tuple(std::forward<decltype(init_start_args)>(init_start_args)...))) {
    }

    sender(const sender &other)
      : initiation(&other.initiation->copy()) {
    }

    sender(sender &&) = default;

    stdexec::operation_state auto connect(stdexec::receiver auto &&);

    template <stdexec::receiver R>
    struct operation_state {
      enum class state_type {
        constructed,
        emplaced,
        initiated,
        stopped
      };

      struct stopper {
        operation_state &op;

        void operator()() noexcept {
          state_type expected = op.state.load(std::memory_order_relaxed);
          while (not op.state.compare_exchange_weak(
            expected, state_type::stopped, std::memory_order_acq_rel)) {
          }
          if (expected == state_type::initiated)
            op.cancel_sig.emit(boost::asio::cancellation_type::total);
        }
      };

      using stop_callback =
        typename stdexec::stop_token_of_t<stdexec::env_of_t<R> &>::template callback_type<stopper>;

      sender snd;
      R recv;
      std::atomic<state_type> state = state_type::constructed;
      boost::asio::cancellation_signal cancel_sig;
      std::optional<stop_callback> stop_cb;

      void start() noexcept {
        if (stdexec::get_stop_token(stdexec::get_env(recv)).stop_requested()) {
          stdexec::set_stopped(recv);
          return;
        }

        stop_cb.emplace(stdexec::get_stop_token(stdexec::get_env(recv)), stopper(*this));

        auto expected = state_type::constructed;
        if (not state.compare_exchange_strong(
              expected, state_type::emplaced, std::memory_order_acq_rel)) {
          stop_cb.reset();
          stdexec::set_stopped(recv);
          return;
        }

        try {
          snd.initiation->initiate(boost::asio::bind_cancellation_slot<
                                   boost::asio::cancellation_slot,
                                   std::function<void(Ts...)>>(cancel_sig.slot(), [&](Ts... args) {
            stdexec::set_value(std::move(recv), std::forward<decltype(args)>(args)...);
          })); // Value channel.
        } catch (...) {
          stop_cb.reset();
          stdexec::set_error(std::move(recv), std::current_exception()); // Error channel.
          return;
        }

        expected = state_type::emplaced;
        if (not state.compare_exchange_strong(
              expected, state_type::initiated, std::memory_order_acq_rel)) {
          stop_cb.reset();
          cancel_sig.emit(boost::asio::cancellation_type::total);
          return;
        }
      }
    };

    template <stdexec::receiver R>
    struct operation_state_unstoppable {
      sender snd;
      R recv;

      void start() noexcept {
        try {
          snd.initiation->initiate([&](Ts... args) {
            stdexec::set_value(std::move(recv), std::forward<decltype(args)>(args)...);
          }); // Value channel.
        } catch (...) {
          stdexec::set_error(std::move(recv), std::current_exception()); // Error channel.
        }
      }
    };

   private:
    struct asio_initiation_base // Type erase some additional type-info of asio_initiation
    {
      virtual ~asio_initiation_base() = default;
      virtual asio_initiation_base &copy() const = 0;

      virtual void initiate(std::function<void(Ts...)>) = 0;
      virtual void initiate(
        boost::asio::
          cancellation_slot_binder<std::function<void(Ts...)>, boost::asio::cancellation_slot>) = 0;
    };

    template <class Func, class... Args> // Refers to boost::asio::async_initiate.
    struct asio_initiation : asio_initiation_base {
      Func func;
      std::tuple<Args...> args;

      asio_initiation(Func init_func, std::tuple<Args...> init_args)
        : func(std::move(init_func))
        , args(std::move(init_args)) {

        };

      asio_initiation &copy() const override {
        return *new asio_initiation(func, args);
      }

      void initiate(std::function<void(Ts...)> token) override {
        std::apply(func, std::tuple_cat(std::make_tuple(std::move(token)), std::move(args)));
      }

      void initiate(boost::asio::cancellation_slot_binder<
                    std::function<void(Ts...)>,
                    boost::asio::cancellation_slot> token) override {
        std::apply(func, std::tuple_cat(std::make_tuple(std::move(token)), std::move(args)));
      }
    };

    std::unique_ptr<asio_initiation_base>
      initiation; // In most cases initiation is larger than 16 bits.
  };

  template <class... Ts>
  stdexec::operation_state auto sender<Ts...>::connect(stdexec::receiver auto &&recv) {
    if constexpr (
      stdexec::stoppable_token<stdexec::stop_token_of_t<stdexec::env_of_t<decltype(recv)>>>)
      return operation_state<std::decay_t<decltype(recv)>>(
        std::move(*this), std::forward<decltype(recv)>(recv));
    else
      return operation_state_unstoppable<std::decay_t<decltype(recv)>>(
        std::move(*this), std::forward<decltype(recv)>(recv));
  }

  struct use_sender_t {
    constexpr use_sender_t() = default;

    template <class Executor>
    struct executor_with_default : Executor {
      using default_completion_token_type = use_sender_t;

      executor_with_default(const Executor &ex) noexcept
        : Executor(ex) {
      }

      executor_with_default(const auto &ex) noexcept
        requires(
          not std::same_as<std::decay_t<decltype(ex)>, executor_with_default>
          and std::convertible_to<std::decay_t<decltype(ex)>, Executor>)
        /* this requires-clause should first requires "not std::same_as<ex,*this>", then requires "std::convertible_to<ex,Executor>",
         * because "std::convertible_to" depends on itself.
         */
        : Executor(ex) {
      }
    };

    template <class Context>
    using as_default_on_t = Context::template rebind_executor<
      executor_with_default<typename Context::executor_type>>::other;

    static auto as_default_on(auto &&obj) {
      return typename std::decay_t<decltype(obj)>::template rebind_executor<
        executor_with_default<typename std::decay_t<decltype(obj)>::executor_type>>::
        other(std::forward<decltype(obj)>(obj));
    }
  };

  struct use_nothrow_sender_t {
    constexpr use_nothrow_sender_t() = default;

    template <class Executor>
    struct executor_with_default : Executor {
      using default_completion_token_type = use_nothrow_sender_t;

      executor_with_default(const Executor &ex) noexcept
        : Executor(ex) {
      }

      executor_with_default(const auto &ex) noexcept
        requires(
          not std::same_as<std::decay_t<decltype(ex)>, executor_with_default>
          and std::convertible_to<std::decay_t<decltype(ex)>, Executor>)
        /* this requires-clause should first requires "not std::same_as<ex,*this>", then requires "std::convertible_to<ex,Executor>",
                 * because "std::convertible_to" depends on itself.
                 */
        : Executor(ex) {
      }
    };

    template <class Context>
    using as_default_on_t = Context::template rebind_executor<
      executor_with_default<typename Context::executor_type>>::other;

    static auto as_default_on(auto &&obj) {
      return typename std::decay_t<decltype(obj)>::template rebind_executor<
        executor_with_default<typename std::decay_t<decltype(obj)>::executor_type>>::
        other(std::forward<decltype(obj)>(obj));
    }
  };

  inline constexpr use_sender_t use_sender = use_sender_t();
  inline constexpr use_nothrow_sender_t use_nothrow_sender = use_nothrow_sender_t();

} // namespace asioexec

namespace boost::asio {
  template <class... Ts>
  struct async_result<asioexec::use_sender_t, void(Ts...)> {
    static auto initiate(auto &&asio_initiation, asioexec::use_sender_t, auto &&...asio_args) {
      auto sched_sender = stdexec::read_env(stdexec::get_scheduler);
      auto value_sender = stdexec::just(
        std::forward<decltype(asio_initiation)>(asio_initiation),
        std::forward<decltype(asio_args)>(asio_args)...);

      // clang-format off
        return stdexec::when_all(std::move(sched_sender), std::move(value_sender)) 
             | stdexec::let_value([](auto &&sched, auto &&...args) 
                 {
                     if constexpr (std::same_as<std::decay_t<std::tuple_element_t<0,std::tuple<Ts...>>>, boost::system::error_code>)
                       return asioexec::sender<Ts...>(std::forward<decltype(args)>(args)...) 
                            | stdexec::let_value([] (boost::system::error_code ec, auto&&... args)
                                {
                                  if (ec)
                                    throw boost::system::system_error(ec);
                                  return stdexec::just(std::forward<decltype(args)>(args)...);
                                })
                            | stdexec::continues_on(std::forward<decltype(sched)>(sched));
                     else
                       return asioexec::sender<Ts...>(std::forward<decltype(args)>(args)...) 
                            | stdexec::continues_on(std::forward<decltype(sched)>(sched));
                    });
      // clang-format on
    }

    using return_type =
      decltype(initiate([](auto &&...) { }, std::declval<asioexec::use_sender_t>()));
  };

  template <class... Ts>
  struct async_result<asioexec::use_nothrow_sender_t, void(Ts...)> {
    static auto
      initiate(auto &&asio_initiation, asioexec::use_nothrow_sender_t, auto &&...asio_args) {
      auto sched_sender = stdexec::read_env(stdexec::get_scheduler);
      auto value_sender = stdexec::just(
        std::forward<decltype(asio_initiation)>(asio_initiation),
        std::forward<decltype(asio_args)>(asio_args)...);

      // clang-format off
        return stdexec::when_all(std::move(sched_sender), std::move(value_sender)) 
             | stdexec::let_value([](auto &&sched, auto &&...args) 
                 {
                   return asioexec::sender<Ts...>(std::forward<decltype(args)>(args)...) 
                        | stdexec::continues_on(std::forward<decltype(sched)>(sched)); 
                 });
      // clang-format on
    }

    using return_type =
      decltype(initiate([](auto &&...) { }, std::declval<asioexec::use_nothrow_sender_t>()));
  };
} // namespace boost::asio
