#include "stdexec/execution.hpp"

#include "exec/linux/io_uring_context.hpp"

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <linux/io_uring.h>

#include <cstring>
#include <chrono>
#include <system_error>
#include <utility>

int main() {
  exec::io_uring_context context(1);
  context.run();
}