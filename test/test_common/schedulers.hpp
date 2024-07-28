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

namespace ex = stdexec;

// Put all the test utilities in an anonymous namespace to avoid ODR violations
namespace {
  template <class S>
  struct scheduler_env {
    template <stdexec::__completion_tag Tag>
    S query(ex::get_completion_scheduler_t<Tag>) const noexcept {
      return {};
    }
  };

  //! Scheduler that will send impulses on user's request.
  //! One can obtain senders from this, connect them to receivers and start the operation states.
  //! Until the scheduler is told to start the next operation, the actions in the operation states are
  //! not executed. This is similar to a task scheduler, but it's single threaded. It has basic
  //! thread-safety to allow it to be run with `sync_wait` (which makes us not control when the
  //! operation_state object is created and started).
  struct impulse_scheduler {
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

    //! That data shared between the operation state and the actual scheduler
    //! Shared pointer to allow the scheduler to be copied (not the best semantics, but it will do)
    std::shared_ptr<data> shared_data_{};

    template <class R>
    struct oper {
      data* data_;
      R receiver_;

      oper(data* shared_data, R&& recv)
        : data_(shared_data)
        , receiver_(static_cast<R&&>(recv)) {
      }

      oper(oper&&) = delete;

      friend void tag_invoke(ex::start_t, oper& self) noexcept {
        // Enqueue another command to the list of all commands
        // The scheduler will start this, whenever start_next() is called
        std::unique_lock lock{self.data_->mutex_};
        self.data_->all_commands_.emplace_back([&self]() {
          if (ex::get_stop_token(ex::get_env(self.receiver_)).stop_requested()) {
            ex::set_stopped(static_cast<R&&>(self.receiver_));
          } else {
            ex::set_value(static_cast<R&&>(self.receiver_));
          }
        });
        self.data_->cv_.notify_all();
      }
    };

    struct env {
      data* data_;

      template <stdexec::__one_of<ex::set_value_t, ex::set_stopped_t> Tag>
      auto query(ex::get_completion_scheduler_t<Tag>) const noexcept {
        return impulse_scheduler{data_};
      }
    };

    struct my_sender {
      using __id = my_sender;
      using __t = my_sender;

      using sender_concept = stdexec::sender_t;
      using completion_signatures = ex::completion_signatures< //
        ex::set_value_t(),                                     //
        ex::set_stopped_t()>;
      data* data_;

      template <class R>
      friend oper<std::decay_t<R>> tag_invoke(ex::connect_t, my_sender self, R&& r) {
        return {self.data_, static_cast<R&&>(r)};
      }

      auto get_env() const noexcept -> env {
        return env{data_};
      }
    };

    explicit impulse_scheduler(data* shared_data)
      : shared_data_(shared_data->shared_from_this()) {
    }

   public:
    using __id = impulse_scheduler;
    using __t = impulse_scheduler;

    impulse_scheduler()
      : shared_data_(std::make_shared<data>(0)) {
    }

    explicit impulse_scheduler(int id)
      : shared_data_(std::make_shared<data>(id)) {
    }

    ~impulse_scheduler() = default;

    //! Actually start the command from the last started operation_state
    //! Returns immediately if no command registered (i.e., no operation state started)
    bool try_start_next() {
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

    my_sender schedule() const {
      return my_sender{shared_data_.get()};
    }

    bool operator==(const impulse_scheduler&) const noexcept = default;
  };

  //! Scheduler that executes everything inline, i.e., on the same thread
  template <class Domain = void>
  struct basic_inline_scheduler {
    using __t = basic_inline_scheduler;
    using __id = basic_inline_scheduler;

    template <class R>
    struct oper : immovable {
      R recv_;

      friend void tag_invoke(ex::start_t, oper& self) noexcept {
        ex::set_value(static_cast<R&&>(self.recv_));
      }
    };

    struct my_sender {
      using __t = my_sender;
      using __id = my_sender;
      using sender_concept = stdexec::sender_t;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      template <class R>
      friend oper<R> tag_invoke(ex::connect_t, my_sender, R r) {
        return {{}, static_cast<R&&>(r)};
      }

      auto get_env() const noexcept -> scheduler_env<basic_inline_scheduler> {
        return {};
      }
    };

    my_sender schedule() const noexcept {
      return {};
    }

    bool operator==(const basic_inline_scheduler&) const noexcept = default;

    Domain query(ex::get_domain_t) const noexcept
      requires(!ex::same_as<Domain, void>)
    {
      return Domain();
    }
  };

  using inline_scheduler = basic_inline_scheduler<>;

  //! Scheduler that returns a sender that always completes with error.
  template <class E = std::exception_ptr>
  struct error_scheduler {
    using __id = error_scheduler;
    using __t = error_scheduler;

    error_scheduler() = default;

    error_scheduler(E err)
      : err_(static_cast<E&&>(err)) {
    }

    error_scheduler(error_scheduler&&) noexcept = default;

    error_scheduler(const error_scheduler&) noexcept = default;

   private:
    template <class R>
    struct oper : immovable {
      R recv_;
      E err_;

      friend void tag_invoke(ex::start_t, oper& self) noexcept {
        ex::set_error(static_cast<R&&>(self.recv_), static_cast<E&&>(self.err_));
      }
    };

    struct my_sender {
      using __id = my_sender;
      using __t = my_sender;

      using sender_concept = stdexec::sender_t;
      using completion_signatures = ex::completion_signatures< //
        ex::set_value_t(),                                     //
        ex::set_error_t(E),
        ex::set_stopped_t()>;

      E err_;

      template <class R>
      friend oper<R> tag_invoke(ex::connect_t, my_sender self, R&& r) {
        return {{}, static_cast<R&&>(r), static_cast<E&&>(self.err_)};
      }

      scheduler_env<error_scheduler> get_env() const noexcept {
        return {};
      }
    };

    E err_{};

   public:
    my_sender schedule() const {
      return {err_};
    }

    bool operator==(const error_scheduler&) const noexcept = default;
  };

  //! Scheduler that returns a sender that always completes with cancellation.
  struct stopped_scheduler {
    using __id = stopped_scheduler;
    using __t = stopped_scheduler;

    template <class R>
    struct oper : immovable {
      R recv_;

      void start() & noexcept {
        ex::set_stopped(static_cast<R&&>(recv_));
      }
    };

    struct my_sender {
      using __id = my_sender;
      using __t = my_sender;

      using sender_concept = stdexec::sender_t;
      using completion_signatures = ex::completion_signatures< //
        ex::set_value_t(),                                     //
        ex::set_stopped_t()>;

      template <class R>
      oper<R> connect(R r) const {
        return {{}, static_cast<R&&>(r)};
      }

      scheduler_env<stopped_scheduler> get_env() const noexcept {
        return {};
      }
    };

    my_sender schedule() const {
      return {};
    }

    bool operator==(const stopped_scheduler&) const noexcept = default;
  };
} // anonymous namespace
