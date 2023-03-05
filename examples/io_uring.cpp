#include "stdexec/execution.hpp"

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <linux/io_uring.h>

#include <chrono>
#include <system_error>
#include <utility>

namespace exec {
  namespace __io_uring {
    class safe_file_descriptor {
      int __fd_{-1};
     public:
      safe_file_descriptor() = default;

      explicit safe_file_descriptor(int __fd) noexcept
        : __fd_(__fd) {
      }

      safe_file_descriptor(const safe_file_descriptor&) = delete;
      safe_file_descriptor& operator=(const safe_file_descriptor&) = delete;

      safe_file_descriptor(safe_file_descriptor&& __other) noexcept
        : __fd_(std::exchange(__other.__fd_, -1)) {
      }

      safe_file_descriptor& operator=(safe_file_descriptor&& __other) noexcept {
        if (this != &__other) {
          if (__fd_ != -1) {
            ::close(__fd_);
          }
          __fd_ = std::exchange(__other.__fd_, -1);
        }
        return *this;
      }

      ~safe_file_descriptor() {
        if (__fd_ != -1) {
          ::close(__fd_);
        }
      }

      explicit operator bool() const noexcept {
        return __fd_ != -1;
      }

      operator int() const noexcept {
        return __fd_;
      }

      int native_handle() const noexcept {
        return __fd_;
      }
    };

    class memory_mapped_region {
      void* __ptr_{nullptr};
      size_t __size_{0};
     public:
      memory_mapped_region() = default;

      memory_mapped_region(void* __ptr, size_t __size) noexcept
        : __ptr_(__ptr)
        , __size_(__size) {
        if (__ptr_ == MAP_FAILED) {
          __ptr_ = nullptr;
        }
      }

      ~memory_mapped_region() {
        if (__ptr_) {
          ::munmap(__ptr_, __size_);
        }
      }

      memory_mapped_region(const memory_mapped_region&) = delete;
      memory_mapped_region& operator=(const memory_mapped_region&) = delete;

      memory_mapped_region(memory_mapped_region&& __other) noexcept
        : __ptr_(std::exchange(__other.__ptr_, nullptr))
        , __size_(std::exchange(__other.__size_, 0)) {
      }

      memory_mapped_region& operator=(memory_mapped_region&& __other) noexcept {
        if (this != &__other) {
          if (__ptr_) {
            ::munmap(__ptr_, __size_);
          }
          __ptr_ = std::exchange(__other.__ptr_, nullptr);
          __size_ = std::exchange(__other.__size_, 0);
        }
        return *this;
      }

      explicit operator bool() const noexcept {
        return __ptr_ != nullptr;
      }

      void* data() const noexcept {
        return __ptr_;
      }

      size_t size() const noexcept {
        return __size_;
      }
    };

    void __throw_on_error(int __ec) {
      if (__ec) {
        throw std::system_error(__ec, std::system_category());
      }
    }

    void __throw_on_error(bool __cond, int __ec) {
      if (__cond) {
        throw std::system_error(__ec, std::system_category());
      }
    }

    safe_file_descriptor __io_uring_setup(unsigned __entries, ::io_uring_params& __params) {
      int rc = (int) ::syscall(__NR_io_uring_setup, __entries, &__params);
      __throw_on_error(-rc);
      return safe_file_descriptor{rc};
    }

    int __io_uring_enter(
      int __ring_fd,
      unsigned int __to_submit,
      unsigned int __min_complete,
      unsigned int __flags) {
      return (int) ::syscall(
        __NR_io_uring_enter, __ring_fd, __to_submit, __min_complete, __flags, nullptr, 0);
    }

    struct schedule_after_t { };

    class __context {
     public:
      explicit __context(unsigned __entries, unsigned __flags = 0)
        : __params_{.flags = __flags}
        , __ring_fd_{__io_uring_setup(__entries, __params_)} {
        auto __sring_sz = __params_.sq_off.array + __params_.sq_entries * sizeof(unsigned);
        auto __cring_sz = __params_.cq_off.cqes + __params_.cq_entries * sizeof(::io_uring_cqe);
        auto __sqes_sz = __params_.sq_entries * sizeof(::io_uring_sqe);
        if (__params_.features & IORING_FEAT_SINGLE_MMAP) {
          __sring_sz = std::max(__sring_sz, __cring_sz);
          __cring_sz = __sring_sz;
        }
        __submission_queue_ = __map_region(__ring_fd_, IORING_OFF_SQ_RING, __sring_sz);
        __submission_queue_entries_ = __map_region(__ring_fd_, IORING_OFF_SQES, __sqes_sz);
        if (!(__params_.features & IORING_FEAT_SINGLE_MMAP)) {
          __completion_queue_ = __map_region(__ring_fd_, IORING_OFF_CQ_RING, __cring_sz);
        }
      }

      __context(const __context&) = delete;

      __context& operator=(const __context&) = delete;

      const ::io_uring_params& params() const noexcept {
        return __params_;
      }

     private:
      static memory_mapped_region __map_region(int __fd, off_t __offset, size_t __size) {
        void* __ptr = ::mmap(
          nullptr, __size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, __fd, __offset);
        __throw_on_error(__ptr == MAP_FAILED, errno);
        return memory_mapped_region{__ptr, __size};
      }

      class __schedule_after_sender;

      class __scheduler {
        __context* __context_{};

        friend __schedule_after_sender
          tag_invoke(schedule_after_t, __context& __self, std::chrono::nanoseconds __timeout);
      };

      template <class _Receiver>
      void __submit_timeout(std::chrono::nanoseconds __timeout, _Receiver&& __receiver);

      struct __completion_task {
        void (*__execute_)(void*, const ::io_uring_cqe*);
      };

      template <class _Receiver>
      struct __schedule_after_operation {
        __context& __context_;
        _Receiver __receiver_;
        std::chrono::nanoseconds __timeout_;

        // friend void tag_invoke(start_t, __schedule_after_operation& __self) noexcept;
      };

      class __schedule_after_sender {
        struct __env {
          __scheduler __scheduler_{};
        } __env_{};

        std::chrono::nanoseconds __timeout_{};

        friend __env
          tag_invoke(stdexec::get_env_t, const __schedule_after_sender& __self) noexcept {
          return __self.__env_;
        }

        template <class _Self, class _Receiver>
        friend __schedule_after_operation<_Receiver>
          tag_invoke(stdexec::connect_t, _Self&& __self, _Receiver&& __receiver) {
          return {
            *__self.__env_.__scheduler_.__context_, (_Receiver&&) __receiver, __self.__timeout_};
        }
      };

      ::io_uring_params __params_{};
      safe_file_descriptor __ring_fd_{};
      memory_mapped_region __submission_queue_{};
      memory_mapped_region __submission_queue_entries_{};
      memory_mapped_region __completion_queue_{};
    };
  }

  using io_uring_context = __io_uring::__context;
}

int main() {
  exec::io_uring_context context(1);
  stdexec::scheduler auto scheduler = context.get_scheduler();
  stdexec::sender auto hello = exec::schedule_after(scheduler, std::chrono::seconds(1))
                             | stdexec::then([] { std::cout << "Hello, world!" << std::endl; });
  stdexec::sync_wait(hello);
}