/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#pragma once

#include <stdexec/execution.hpp>
#include <test_common/type_helpers.hpp>

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace ex = STDEXEC;

// Put all the test utilities in an anonymous namespace to avoid ODR violations
namespace {
  template <class Scheduler, ex::__completion_tag... Tags>
  struct sched_attrs {
    sched_attrs(Scheduler sched, Tags...)
      : scheduler_(std::move(sched)) {
    }

    template <ex::__one_of<Tags...> Tag>
    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<Tag>) const noexcept {
      return scheduler_;
    }

    Scheduler scheduler_;
  };

  //! Scheduler that will send impulses on user's request.
  //! One can obtain senders from this, connect them to receivers and start the operation states.
  //! Until the scheduler is told to start the next operation, the actions in the operation states are
  //! not executed. This is similar to a task scheduler, but it's single threaded. It has basic
  //! thread-safety to allow it to be run with `sync_wait` (which makes us not control when the
  //! operation_state object is created and started).
  struct impulse_scheduler {
    using scheduler_concept = ex::scheduler_t;

    impulse_scheduler()
      : shared_data_(std::make_shared<data>(0)) {
    }

    explicit impulse_scheduler(int id)
      : shared_data_(std::make_shared<data>(id)) {
    }

    ~impulse_scheduler() = default;

    //! Actually start the command from the last started operation_state
    //! Returns immediately if no command registered (i.e., no operation state started)
    auto try_start_next() -> bool {
      // Wait for a command that we can execute
      std::unique_lock lock{shared_data_->mutex_};

      // If there are no commands in the queue, return false
      if (shared_data_->all_commands_.empty()) {
        return false;
      }

      // Pop one command from the queue
      auto cmd = std::move(shared_data_->all_commands_.front());
      shared_data_->all_commands_.erase(shared_data_->all_commands_.begin());
      // Exit the lock before executing the command
      lock.unlock();
      // Execute the command, i.e., send an impulse to the connected sender
      cmd();
      // Return true to signal that we started a command
      return true;
    }

    //! Actually start the command from the last started operation_state
    //! Blocks if no command registered (i.e., no operation state started)
    void start_next() {
      // Wait for a command that we can execute
      std::unique_lock lock{shared_data_->mutex_};
      while (shared_data_->all_commands_.empty()) {
        shared_data_->cv_.wait(lock);
      }

      // Pop one command from the queue
      auto cmd = std::move(shared_data_->all_commands_.front());
      shared_data_->all_commands_.erase(shared_data_->all_commands_.begin());
      // Exit the lock before executing the command
      lock.unlock();
      // Execute the command, i.e., send an impulse to the connected sender
      cmd();
    }

    [[nodiscard]]
    auto schedule() const noexcept {
      return sender{shared_data_};
    }

    auto operator==(const impulse_scheduler&) const noexcept -> bool = default;

   private:
    //! Command type that can store the action of firing up a sender
    using oper_command_t = std::function<void()>;
    using cmd_vec_t = std::vector<oper_command_t>;

    struct data : std::enable_shared_from_this<data> {
      explicit data(int id)
        : id_(id) {
      }

      int id_;
      cmd_vec_t all_commands_;
      std::mutex mutex_;
      std::condition_variable cv_;
    };

    template <class Receiver>
    struct opstate {
      opstate(std::shared_ptr<data> shared_data, Receiver&& recv)
        : data_(std::move(shared_data))
        , receiver_(static_cast<Receiver&&>(recv)) {
      }

      opstate(opstate&&) = delete;

      void start() & noexcept {
        // Enqueue another command to the list of all commands
        // The scheduler will start this, whenever start_next() is called
        std::unique_lock lock{data_->mutex_};
        data_->all_commands_.emplace_back([this]() {
          if (ex::get_stop_token(ex::get_env(receiver_)).stop_requested()) {
            ex::set_stopped(static_cast<Receiver&&>(receiver_));
          } else {
            ex::set_value(static_cast<Receiver&&>(receiver_));
          }
        });
        data_->cv_.notify_all();
      }

      std::shared_ptr<data> data_;
      Receiver receiver_;
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;

      template <class Receiver>
      auto connect(Receiver rcvr) const -> opstate<Receiver> {
        return {data_, static_cast<Receiver&&>(rcvr)};
      }

      [[nodiscard]]
      auto get_env() const noexcept {
        return sched_attrs(impulse_scheduler(data_), ex::set_value, ex::set_stopped);
      }

      std::shared_ptr<data> data_;
    };

    explicit impulse_scheduler(std::shared_ptr<data> shared_data) noexcept
      : shared_data_(std::move(shared_data)) {
    }

    //! That data shared between the operation state and the actual scheduler
    //! Shared pointer to allow the scheduler to be copied (not the best semantics, but it will do)
    std::shared_ptr<data> shared_data_{};
  };

  //! Scheduler that executes everything inline, i.e., on the same thread
  template <class Domain = void>
  struct basic_inline_scheduler {
    using scheduler_concept = ex::scheduler_t;

    auto schedule() const noexcept {
      return sender{};
    }

    auto operator==(const basic_inline_scheduler&) const noexcept -> bool = default;

    auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept -> Domain
      requires(!std::same_as<Domain, void>)
    {
      return {};
    }

   private:
    template <class Receiver>
    struct opstate : immovable {
      void start() & noexcept {
        ex::set_value(static_cast<Receiver&&>(rcvr_));
      }

      Receiver rcvr_;
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      template <class Receiver>
      auto connect(Receiver rcvr) const -> opstate<Receiver> {
        return {{}, static_cast<Receiver&&>(rcvr)};
      }

      auto get_env() const noexcept {
        return sched_attrs(basic_inline_scheduler(), ex::set_value);
      }
    };
  };

  using STDEXEC::inline_scheduler;

  template <class Type>
  struct nothrow_copyable_box {
    nothrow_copyable_box() noexcept = default;

    explicit nothrow_copyable_box(Type value)
      : value_(std::make_shared<Type>(static_cast<Type&&>(value))) {
    }

    [[nodiscard]]
    auto value() const noexcept -> const Type& {
      return *value_;
    }

    [[nodiscard]]
    auto operator==(const nothrow_copyable_box& other) const
      noexcept(noexcept(*value_ == *(other.value_))) -> bool {
      if (value_ && other.value_) {
        return *value_ == *(other.value_);
      }
      return !value_ && !other.value_;
    }

   private:
    std::shared_ptr<const Type> value_{};
  };

  template <class Type>
    requires ex::__nothrow_copy_constructible<Type>
  struct nothrow_copyable_box<Type> {
    nothrow_copyable_box() noexcept = default;

    explicit nothrow_copyable_box(Type value) noexcept(ex::__nothrow_copy_constructible<Type>)
      : value_(static_cast<Type&&>(value)) {
    }

    [[nodiscard]]
    auto value() const noexcept -> const Type& {
      return value_;
    }

    [[nodiscard]]
    auto operator==(const nothrow_copyable_box&) const -> bool = default;

   private:
    Type value_{};
  };

  //! Scheduler that returns a sender that always completes with error.
  template <class Error = std::exception_ptr>
  struct error_scheduler {
    using scheduler_concept = ex::scheduler_t;

    error_scheduler() = default;

    explicit error_scheduler(Error err)
      : err_(static_cast<Error&&>(err)) {
    }

    [[nodiscard]]
    auto schedule() const noexcept {
      return sender{err_};
    }

    auto operator==(const error_scheduler&) const noexcept -> bool = default;

    template <ex::__completion_tag Tag>
    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<Tag>) const noexcept {
      return *this;
    }

   private:
    template <class Receiver>
    struct opstate : immovable {
      void start() & noexcept {
        ex::set_error(static_cast<Receiver&&>(rcvr_), Error(err_.value()));
      }

      Receiver rcvr_;
      nothrow_copyable_box<Error> err_;
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(Error), ex::set_stopped_t()>;

      template <class Receiver>
      auto connect(Receiver rcvr) && -> opstate<Receiver> {
        return {{}, static_cast<Receiver&&>(rcvr), std::move(err_)};
      }

      auto get_env() const noexcept {
        return sched_attrs{error_scheduler(err_), ex::set_value, ex::set_error, ex::set_stopped};
      }

      nothrow_copyable_box<Error> err_;
    };

    error_scheduler(nothrow_copyable_box<Error> err) noexcept
      : err_(static_cast<nothrow_copyable_box<Error>&&>(err)) {
    }

    nothrow_copyable_box<Error> err_{};
  };

  //! Scheduler that returns a sender that always completes with cancellation.
  struct stopped_scheduler {
   private:
    struct sender;

   public:
    using scheduler_concept = ex::scheduler_t;

    auto operator==(const stopped_scheduler&) const noexcept -> bool = default;

    [[nodiscard]]
    auto schedule() const noexcept {
      return sender{};
    }

    template <ex::__one_of<ex::set_value_t, ex::set_stopped_t> Tag>
    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<Tag>) const noexcept {
      return stopped_scheduler{};
    }

   private:
    template <class Receiver>
    struct opstate : immovable {
      void start() & noexcept {
        ex::set_stopped(static_cast<Receiver&&>(rcvr_));
      }

      Receiver rcvr_;
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;

      template <class Receiver>
      auto connect(Receiver rcvr) const -> opstate<Receiver> {
        return {{}, static_cast<Receiver&&>(rcvr)};
      }

      [[nodiscard]]
      auto get_env() const noexcept {
        return sched_attrs(stopped_scheduler(), ex::set_value, ex::set_stopped);
      }
    };
  };

  namespace _dummy {
    template <class Domain>
    struct _attrs_t {
      constexpr auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept;

      constexpr auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept {
        return Domain{};
      }
    };

    template <class Rcvr>
    struct _opstate_t : ex::__immovable {
      using operation_state_concept = ex::operation_state_t;

      constexpr _opstate_t(Rcvr rcvr) noexcept
        : _rcvr(static_cast<Rcvr&&>(rcvr)) {
      }

      constexpr void start() noexcept {
        ex::set_value(static_cast<Rcvr&&>(_rcvr));
      }

      Rcvr _rcvr;
    };

    template <class Domain>
    struct _sndr_t {
      using sender_concept = ex::sender_t;

      template <class Self>
      static consteval auto get_completion_signatures() noexcept {
        return ex::completion_signatures<ex::set_value_t()>();
      }

      template <class Rcvr>
      constexpr auto connect(Rcvr rcvr) const noexcept -> _opstate_t<Rcvr> {
        return _opstate_t<Rcvr>(static_cast<Rcvr&&>(rcvr));
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept {
        return _attrs_t<Domain>{};
      }
    };
  } // namespace _dummy

  //! Scheduler that returns a sender that always completes inline (successfully).
  template <class Domain = ex::default_domain>
  struct dummy_scheduler : _dummy::_attrs_t<Domain> {
    using scheduler_concept = ex::scheduler_t;

    static constexpr auto schedule() noexcept -> _dummy::_sndr_t<Domain> {
      return {};
    }

    friend constexpr bool operator==(dummy_scheduler, dummy_scheduler) noexcept {
      return true;
    }

    friend constexpr bool operator!=(dummy_scheduler, dummy_scheduler) noexcept {
      return false;
    }
  };

  namespace _dummy {
    template <class Domain>
    constexpr auto
      _attrs_t<Domain>::query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept {
      return dummy_scheduler<Domain>{};
    }
  } // namespace _dummy
} // anonymous namespace
