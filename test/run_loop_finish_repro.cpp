#include <stdexec/execution.hpp>

#include <thread>

namespace ex = STDEXEC_NAMESPACE;

void run_once()
{
  ex::run_loop loop;

  std::thread worker([&loop] { loop.run(); });

  // Ensure run() is actively servicing the loop before initiating
  // shutdown from this thread.
  if (!ex::sync_wait(ex::schedule(loop.get_scheduler()) | ex::then([]() {})))
  {
    loop.finish();
    worker.join();
  }

  loop.finish();
  worker.join();
}

int main()
{
  for (int i = 0; i < 1000; ++i)
  {
    run_once();
  }
}
