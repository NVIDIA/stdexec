#pragma once

#include <execpools/tbb/tbb_thread_pool.hpp>

#warning Deprecated header file, please include the <execpools/tbb/tbb_thread_pool.hpp> header file instead and use the execpools::tbb_thread_pool class that is identical as tbbexec::thread_pool class.

namespace tbbexec {
  using tbb_thread_pool = execpools::tbb_thread_pool;
} // namespace tbbexec
