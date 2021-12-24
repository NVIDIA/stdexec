/*
 * Copyright (c) Lucian Radu Teodorescu
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

#include <execution.hpp>

#include <functional>
#include <vector>

namespace ex = std::execution;

//! Scheduler that will send impulses on user's request.
//! One can obtain senders from this, connect them to receivers and start the operation states.
//! Until the scheduler is told to start the next operation, the actions in the operation states are
//! not executed. This is similar to a task scheduler, but it's single threaded. Doesn't have any
//! thread safety built in.
struct impulse_scheduler {
  private:
  //! Command type that can store the action of firing up a sender
  using oper_command_t = std::function<void()>;
  using cmd_vec_t = std::vector<oper_command_t>;
  //! All the commands that we need to somehow
  std::shared_ptr<cmd_vec_t> all_commands_{};

  template <typename R, typename StopToken>
  struct oper {
    cmd_vec_t* all_commands_;
    R receiver_;
    StopToken stoken_;

    oper(cmd_vec_t* all_commands, R&& recv, StopToken stoken)
        : all_commands_(all_commands)
        , receiver_((R &&) recv)
        , stoken_((StopToken&&) stoken) {}

    friend void tag_invoke(ex::start_t, oper& self) noexcept {
      // Enqueue another command to the list of all commands
      // The scheduler will start this, whenever start_next() is called
      self.all_commands_->emplace_back([&self]() {
        if (self.stoken_.stop_requested()) {
          ex::set_done((R &&) self.receiver_);
        } else {
          try {
            ex::set_value((R &&) self.receiver_);
          } catch (...) {
            ex::set_error((R &&) self.receiver_, std::current_exception());
          }
        }
      });
    }
  };

  struct my_sender : ex::completion_signatures<               //
                         ex::set_value_t(),                   //
                         ex::set_error_t(std::exception_ptr), //
                         ex::set_done_t()> {
    cmd_vec_t* all_commands_;

    template <typename R, typename C>
    friend oper<R, ex::stop_token_of_t<C>>
    tag_invoke(ex::connect_t, my_sender self, R&& r, C&& c) {
      return {self.all_commands_, (R &&) r, ex::get_stop_token(c)};
    }

    friend impulse_scheduler tag_invoke(
        ex::get_completion_scheduler_t<ex::set_value_t>, my_sender) {
      return {};
    }
  };

  public:
  impulse_scheduler()
      : all_commands_(std::make_shared<cmd_vec_t>(cmd_vec_t{})) {}
  ~impulse_scheduler() = default;

  //! Actually start the command from the last started operation_state
  void start_next() {
    if (!all_commands_->empty()) {
      // Pop one command from the queue
      auto cmd = std::move(all_commands_->front());
      all_commands_->erase(all_commands_->begin());
      // Execute the command, i.e., send an impulse to the connected sender
      cmd();
    }
  }

  friend my_sender tag_invoke(ex::schedule_t, const impulse_scheduler& self) {
    return my_sender{{}, self.all_commands_.get()};
  }

  friend bool operator==(impulse_scheduler, impulse_scheduler) noexcept { return true; }
  friend bool operator!=(impulse_scheduler, impulse_scheduler) noexcept { return false; }
};

//! Scheduler that executes everything inline, i.e., on the same thread
struct inline_scheduler {
  template <typename R>
  struct oper {
    R recv_;
    friend void tag_invoke(ex::start_t, oper& self) noexcept {
      try {
        ex::set_value((R &&) self.recv_);
      } catch (...) {
        ex::set_error((R &&) self.recv_, std::current_exception());
      }
    }
  };

  struct my_sender : ex::completion_signatures< //
                         ex::set_value_t(),     //
                         ex::set_error_t(std::exception_ptr)> {
    template <typename R>
    friend oper<R> tag_invoke(ex::connect_t, my_sender self, R&& r, auto) {
      return {(R &&) r};
    }

    friend inline_scheduler tag_invoke(ex::get_completion_scheduler_t<ex::set_value_t>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, inline_scheduler) { return {}; }

  friend bool operator==(inline_scheduler, inline_scheduler) noexcept { return true; }
  friend bool operator!=(inline_scheduler, inline_scheduler) noexcept { return false; }
};

//! Scheduler that returns a sender that always completes with error.
template <typename E = std::exception_ptr>
struct error_scheduler {
  template <typename R>
  struct oper {
    R recv_;
    E err_;

    friend void tag_invoke(ex::start_t, oper& self) noexcept {
      ex::set_error((R &&) self.recv_, (E &&) self.err_);
    }
  };

  struct my_sender : ex::completion_signatures< //
                         ex::set_value_t(),     //
                         ex::set_error_t(E)> {
    E err_;

    template <typename R>
    friend oper<R> tag_invoke(ex::connect_t, my_sender self, R&& r, auto) {
      return {(R &&) r, (E &&) self.err_};
    }

    friend error_scheduler tag_invoke(ex::get_completion_scheduler_t<ex::set_value_t>, my_sender) {
      return {};
    }
  };

  E err_;

  friend my_sender tag_invoke(ex::schedule_t, error_scheduler self) {
    return {{}, (E &&) self.err_};
  }

  friend bool operator==(error_scheduler, error_scheduler) noexcept { return true; }
  friend bool operator!=(error_scheduler, error_scheduler) noexcept { return false; }
};

//! Scheduler that returns a sender that always completes with cancellation.
struct done_scheduler {
  template <typename R>
  struct oper {
    R recv_;
    friend void tag_invoke(ex::start_t, oper& self) noexcept { ex::set_done((R &&) self.recv_); }
  };

  struct my_sender : ex::completion_signatures< //
                         ex::set_value_t(),     //
                         ex::set_done_t()> {
    template <typename R>
    friend oper<R> tag_invoke(ex::connect_t, my_sender self, R&& r, auto) {
      return {(R &&) r};
    }

    template <typename CPO>
    friend done_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, done_scheduler) { return {}; }

  friend bool operator==(done_scheduler, done_scheduler) noexcept { return true; }
  friend bool operator!=(done_scheduler, done_scheduler) noexcept { return false; }
};
