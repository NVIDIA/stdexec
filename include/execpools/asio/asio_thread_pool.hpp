/*
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <execpools/thread_pool_base.hpp>
#include <execpools/asio/asio_config.hpp>

namespace execpools {
  class asio_thread_pool : public execpools::thread_pool_base<asio_thread_pool> {
   public:
    asio_thread_pool()
      : pool_()
      , executor_(pool_.executor()) {
    }

    explicit asio_thread_pool(uint32_t num_threads)
      : pool_(num_threads)
      , executor_(pool_.executor()) {
    }

    ~asio_thread_pool() = default;

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t {
      return asio_impl::query(executor_, asio_impl::execution::occupancy);
    }
   private:
    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() -> stdexec::forward_progress_guarantee {
      return stdexec::forward_progress_guarantee::parallel;
    }

    friend execpools::thread_pool_base<asio_thread_pool>;

    template <class PoolType, class ReceiverId>
    friend struct execpools::operation;

    void enqueue(execpools::task_base* task, std::uint32_t tid = 0) noexcept {
      asio_impl::post(pool_, [task, tid] { task->__execute(task, /*tid=*/tid); });
    }

    asio_impl::thread_pool pool_;
    // Need to store implicitly the executor, thread_pool::executor() is not const
    asio_impl::thread_pool::executor_type executor_;
  };
} // namespace execpools
