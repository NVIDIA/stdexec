/*
 * Copyright (c) NVIDIA
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

#include <schedulers/detail/distributed/concepts.hpp>
#include <schedulers/detail/distributed/environment.hpp>
#include <schedulers/detail/distributed/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <execution.hpp>
#include <span>

#include <mpi.h>

namespace example::cuda::distributed
{

namespace detail::rotate_exchange
{

template <class Receiver, class... Ts>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<Receiver, Ts...>,
                                       Receiver>
{
  using tuple_t = std::tuple<Ts...>;
  using super_t =
    std::execution::receiver_adaptor<receiver_t<Receiver, Ts...>,
                                     Receiver>;
  friend super_t;

  int elements_{};
  static constexpr int n_fields = sizeof...(Ts);
  static constexpr int prev_tag = 1;
  static constexpr int next_tag = 2;

  tuple_t prev_recv_;
  tuple_t prev_send_;
  tuple_t next_recv_;
  tuple_t next_send_;

  template <std::size_t I>
  using value_t =
    typename std::iterator_traits<std::tuple_element_t<I, tuple_t>>::value_type;

  template <std::size_t... I>
  void register_recv(int prev_rank,
                     int next_rank,
                     MPI_Request *requests,
                     std::index_sequence<I...>)
  {
    ((MPI_Irecv(std::get<I>(prev_recv_), elements_ * sizeof(value_t<I>), MPI_BYTE, prev_rank, next_tag, MPI_COMM_WORLD, requests + I)),...);
    ((MPI_Irecv(std::get<I>(next_recv_), elements_ * sizeof(value_t<I>), MPI_BYTE, next_rank, prev_tag, MPI_COMM_WORLD, requests + n_fields + I)),...);
  }

  template <std::size_t... I>
  void register_send(int prev_rank,
                     int next_rank,
                     MPI_Request *requests,
                     std::index_sequence<I...>)
  {
    ((MPI_Isend(std::get<I>(next_send_), elements_ * sizeof(value_t<I>), MPI_BYTE, next_rank, next_tag, MPI_COMM_WORLD, requests + n_fields + I)),...);
    ((MPI_Isend(std::get<I>(prev_send_), elements_ * sizeof(value_t<I>), MPI_BYTE, prev_rank, prev_tag, MPI_COMM_WORLD, requests + I)),...);
  }

  template <std::size_t... I>
  void shortcut(cudaStream_t stream, std::index_sequence<I...>)
  {
    ((std::copy_n(std::get<I>(prev_send_), elements_, std::get<I>(next_recv_))),...);
    ((std::copy_n(std::get<I>(next_send_), elements_, std::get<I>(prev_recv_))),...);
  }

  void set_value() &&noexcept
  try
  {
    const context_t &ctx = std::execution::get_env(this->base()).context();

    const int size = ctx.size();

    if (size == 0)
    {
      shortcut(ctx.stream(), std::make_index_sequence<n_fields>{});
    }
    else
    {
      const int rank = ctx.rank();

      const int prev_rank = rank == 0 ? size - 1 : rank - 1;
      const int next_rank = rank == (size - 1) ? 0 : rank + 1;

      cudaStreamSynchronize(ctx.stream());

      MPI_Request recv_requests[2 * n_fields];
      register_recv(prev_rank,
                    next_rank,
                    recv_requests,
                    std::make_index_sequence<n_fields>{});

      MPI_Request send_requests[2 * n_fields];
      register_send(prev_rank,
                    next_rank,
                    send_requests,
                    std::make_index_sequence<n_fields>{});

      MPI_Waitall(2 * n_fields, send_requests, MPI_STATUSES_IGNORE);
      MPI_Waitall(2 * n_fields, recv_requests, MPI_STATUSES_IGNORE);
    }

    std::execution::set_value(std::move(this->base()));
  }
  catch (...)
  {
    std::execution::set_error(std::move(this->base()),
                              std::current_exception());
  }

  void set_error(std::exception_ptr ex_ptr) &&noexcept
  {
    std::execution::set_error(std::move(this->base()), ex_ptr);
  }

  void set_stopped() &&noexcept
  {
    std::execution::set_stopped(std::move(this->base()));
  }

public:
  explicit receiver_t(Receiver receiver,
                      int elements,
                      std::tuple<Ts...> prev_recv,
                      std::tuple<Ts...> prev_send,
                      std::tuple<Ts...> next_recv,
                      std::tuple<Ts...> next_send)
      : super_t(std::move(receiver))
      , elements_(elements)
      , prev_recv_{prev_recv}
      , prev_send_{prev_send}
      , next_recv_{next_recv}
      , next_send_{next_send}
  {}

  static constexpr bool is_cuda_distributed_api = true;
};

template <distributed_sender S, class... Ts>
struct sender_t : sender_base_t<sender_t<S, Ts...>, S>
{
  using value_t = value_of_t<S>;
  using super_t = sender_base_t<sender_t<S, Ts...>, S>;
  friend super_t;

  int elements_{};
  std::tuple<Ts...> prev_recv_;
  std::tuple<Ts...> prev_send_;
  std::tuple<Ts...> next_recv_;
  std::tuple<Ts...> next_send_;

  template <distributed_receiver Receiver>
  auto connect(Receiver &&receiver) &&noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<Receiver, Ts...>{std::forward<Receiver>(receiver),
                                  elements_,
                                  prev_recv_,
                                  prev_send_,
                                  next_recv_,
                                  next_send_});
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  explicit sender_t(S sender,
                    int elements,
                    std::tuple<Ts...> prev_recv,
                    std::tuple<Ts...> prev_send,
                    std::tuple<Ts...> next_recv,
                    std::tuple<Ts...> next_send)
      : super_t{std::forward<S>(sender)}
      , elements_{elements}
      , prev_recv_{prev_recv}
      , prev_send_{prev_send}
      , next_recv_{next_recv}
      , next_send_{next_send}
  {}
};

struct rotate_exchange_t
{
  template <distributed_sender Sender, class... Ts>
  auto operator()(Sender &&sndr,
                  int elements,
                  std::tuple<Ts...> prev_recv,
                  std::tuple<Ts...> prev_send,
                  std::tuple<Ts...> next_recv,
                  std::tuple<Ts...> next_send) const noexcept
  {
    return sender_t<Sender, Ts...>{std::forward<Sender>(sndr),
                                   elements,
                                   prev_recv,
                                   prev_send,
                                   next_recv,
                                   next_send};
  }

  template <class... Ts>
  std::execution::__binder_back<rotate_exchange_t, int, std::tuple<Ts...>, std::tuple<Ts...>, std::tuple<Ts...>, std::tuple<Ts...>>
  operator()(int elements,
             std::tuple<Ts...> prev_recv,
             std::tuple<Ts...> prev_send,
             std::tuple<Ts...> next_recv,
             std::tuple<Ts...> next_send) const
  {
    return {{}, {}, {elements, prev_recv, prev_send, next_recv, next_send}};
  }
};

}

inline constexpr detail::rotate_exchange::rotate_exchange_t rotate_exchange{};

} // namespace example::cuda::distributed::detail::then

