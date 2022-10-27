#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <cstdio>

namespace ex = stdexec;

int main() {
  using nvexec::is_on_gpu;

  nvexec::stream_context stream_ctx{};
  ex::scheduler auto sch = stream_ctx.get_scheduler();

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

  auto fork = ex::schedule(sch)
            | ex::bulk(4, bulk_fn(0))
            | ex::split();

  auto snd = ex::transfer_when_all(
               sch,
               fork | ex::bulk(4, bulk_fn(1)),
               fork | ex::then(then_fn(1)),
               fork | ex::bulk(4, bulk_fn(2)))
           | ex::then(then_fn(2));

  stdexec::sync_wait(std::move(snd));
}

