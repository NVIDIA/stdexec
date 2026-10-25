/*
 * Copyright (c) 2026 stdexec contributors
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

#include <stdexec/__detail/__config.hpp>

#if STDEXEC_USE_MODULES()
import std;
import stdexec;
#else
#  include <stdexec/execution.hpp>

#  include <atomic>
#  include <chrono>
#  include <condition_variable>
#  include <mutex>
#  include <thread>
#  include <utility>
#endif

namespace
{
  //! A stop source for deterministically testing a stop request that races
  //! with the completion of an operation.
  //!
  //! request_stop() takes the registered callback and then holds it, the way
  //! a thread can be preempted inside inplace_stop_source::request_stop(),
  //! until another thread is waiting in that callback's destructor. Only then
  //! does it invoke the callback. As for any stop callback, the destructor
  //! blocks while the callback runs on another thread and does not block when
  //! the callback destroys itself. At most one callback can be registered.
  class racing_stop_source
  {
    struct callback_base
    {
      void (*invoke_)(callback_base*) noexcept;
    };

   public:
    class token;

    racing_stop_source()                     = default;
    racing_stop_source(racing_stop_source&&) = delete;

    [[nodiscard]]
    auto get_token() noexcept -> token;

    void request_stop()
    {
      std::unique_lock lock{mutex_};
      if (stop_requested_.exchange(true))
      {
        return;
      }
      callback_base* callback = std::exchange(registered_, nullptr);
      running_                = callback;
      running_on_             = std::this_thread::get_id();
      cv_.notify_all();
      if (callback == nullptr)
      {
        return;
      }
      // The timeout only keeps a test from hanging if the destructor never
      // runs while the callback is held.
      (void) cv_.wait_for(lock, std::chrono::seconds(10), [this] { return destructor_waiting_; });
      lock.unlock();
      callback->invoke_(callback);
      lock.lock();
      running_ = nullptr;
      cv_.notify_all();
    }

    //! Blocks until request_stop() has taken the registered callback, if any.
    void wait_for_request()
    {
      std::unique_lock lock{mutex_};
      cv_.wait(lock, [this] { return stop_requested_.load(); });
    }

    //! Whether the callback's destructor ran on another thread while
    //! request_stop() was holding the callback.
    [[nodiscard]]
    auto destructor_waited() -> bool
    {
      std::lock_guard lock{mutex_};
      return destructor_waiting_;
    }

   private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::atomic<bool>       stop_requested_{false};
    callback_base*          registered_{nullptr};
    callback_base*          running_{nullptr};
    std::thread::id         running_on_{};
    bool                    destructor_waiting_{false};
  };

  class racing_stop_source::token
  {
   public:
    template <class Fn>
    class callback_type : callback_base
    {
     public:
      template <class Init>
      explicit callback_type(token tok, Init&& init)
        : callback_base{&invoke}
        , source_(tok.source_)
        , fn_(static_cast<Init&&>(init))
      {
        std::unique_lock lock{source_->mutex_};
        if (source_->stop_requested_.load())
        {
          lock.unlock();
          fn_();
        }
        else
        {
          source_->registered_ = this;
        }
      }

      callback_type(callback_type&&) = delete;

      ~callback_type()
      {
        std::unique_lock lock{source_->mutex_};
        if (source_->registered_ == this)
        {
          source_->registered_ = nullptr;
        }
        else if (source_->running_ == this && source_->running_on_ != std::this_thread::get_id())
        {
          source_->destructor_waiting_ = true;
          source_->cv_.notify_all();
          source_->cv_.wait(lock, [this] { return source_->running_ != this; });
        }
      }

     private:
      static void invoke(callback_base* self) noexcept
      {
        static_cast<callback_type*>(self)->fn_();
      }

      racing_stop_source* source_;
      Fn                  fn_;
    };

    [[nodiscard]]
    auto stop_requested() const noexcept -> bool
    {
      return source_->stop_requested_.load();
    }

    [[nodiscard]]
    auto stop_possible() const noexcept -> bool
    {
      return true;
    }

    auto operator==(token const &) const noexcept -> bool = default;

   private:
    friend racing_stop_source;

    explicit token(racing_stop_source* source) noexcept
      : source_(source)
    {}

    racing_stop_source* source_;
  };

  inline auto racing_stop_source::get_token() noexcept -> token
  {
    return token{this};
  }

  //! Counts its value and stopped completions. Its environment carries a
  //! racing_stop_source's token.
  struct racing_stop_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    template <class... Values>
    void set_value(Values&&...) && noexcept
    {
      ++*values_;
    }

    void set_stopped() && noexcept
    {
      ++*stopped_;
    }

    [[nodiscard]]
    auto get_env() const noexcept
    {
      return STDEXEC::prop{STDEXEC::get_stop_token, token_};
    }

    std::atomic<int>*         values_;
    std::atomic<int>*         stopped_;
    racing_stop_source::token token_;
  };
}  // namespace
