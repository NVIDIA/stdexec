#include <schedulers/stream.cuh>
#include <execution.hpp>

#include <cstdio>

namespace ex = std::execution;
namespace stream = example::cuda::stream;

int main() {
  using example::cuda::is_on_gpu;

  stream::context_t stream_context{};
  ex::scheduler auto sch = stream_context.get_scheduler();

  auto bulk_fn = [](int lbl) {
    return [=](int i) { 
      std::printf("B%d: i = %d\n", lbl, i); 
    };
  };

  auto then_fn = [](int lbl) {
    return [=] {
      std::printf("T%d\n", lbl);
    };
  };

  auto snd = ex::transfer_when_all(
               sch,
               ex::schedule(sch) | ex::bulk(4, bulk_fn(1)),
               ex::schedule(sch) | ex::then(then_fn(1)),
               ex::schedule(sch) | ex::bulk(4, bulk_fn(2)))
           | ex::then(then_fn(2));

  std::this_thread::sync_wait(std::move(snd));
}

