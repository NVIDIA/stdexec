/*
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "boost/asio/execution/occupancy.hpp"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <execpools/thread_pool_base.hpp>

namespace execpools {

  class boost_thread_pool : public execpools::thread_pool_base<boost_thread_pool> {
   public:
    boost_thread_pool()
     : pool_()
     , executor_(pool_.executor()) {

     }

    explicit boost_thread_pool(uint32_t num_threads)
     : pool_(num_threads)
     , executor_(pool_.executor()) {

    }

    ~boost_thread_pool() = default;

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t {
      return boost::asio::query(executor_, boost::asio::execution::occupancy);
    }
   private:
    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() -> stdexec::forward_progress_guarantee {
      return stdexec::forward_progress_guarantee::parallel;
    }

    friend execpools::thread_pool_base<boost_thread_pool>;

    template <class PoolType, class ReceiverId>
    friend struct execpools::operation;

    void enqueue(execpools::task_base* task, std::uint32_t tid = 0) noexcept {
      boost::asio::post(pool_, [task, tid] { task->__execute(task, /*tid=*/tid); });
    }

    boost::asio::thread_pool pool_;
    // Need to store implicitly the executor, thread_pool::executor() is not const
    boost::asio::thread_pool::executor_type executor_;
  };
} // namespace tbbexec
