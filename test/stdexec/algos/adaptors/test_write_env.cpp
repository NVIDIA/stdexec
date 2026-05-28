/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexec/__detail/__basic_sender.hpp>
#include <stdexec/__detail/__completion_signatures.hpp>
#include <stdexec/__detail/__optional.hpp>
#include <stdexec/__detail/__queries.hpp>

namespace
{
  struct query_t
  {
    template <class Env>
    decltype(auto) operator()(Env const & env) const noexcept
    {
      return env.query(*this);
    }
  };

  inline constexpr query_t query;
}  // namespace

namespace STDEXEC
{
  namespace __read_env
  {
    template <class _Receiver, class _Query>
    struct __opstate
    {
      constexpr void start() noexcept
      {
        // make sure our simplification stays valid
        static_assert(std::is_reference_v<decltype(_Query()(__rcvr_.get_env()))>);

        // The query returns a reference type; pass it straight through to the receiver.
        auto&& result = _Query()(__rcvr_.get_env());
        std::printf("completing with %p\n", (void*) &result);
        static_cast<_Receiver&&>(__rcvr_).set_value(static_cast<decltype(result)&&>(result));
      }

      _Receiver __rcvr_;
    };

    struct __read_env_impl : __sexpr_defaults
    {
      template <class _Self, class _Env>
      static consteval auto __get_completion_signatures()
      {
        using __query_t = __data_of<_Self>;
        {
          using __result_t = __call_result_t<__query_t, _Env>;
          return completion_signatures<set_value_t(__result_t)>();
        }
      };

      static constexpr auto __connect =
        []<class _Self, class _Receiver>(_Self const &, _Receiver&& __rcvr) noexcept
      {
        using __query_t = __data_of<_Self>;
        return __opstate<_Receiver, __query_t>{static_cast<_Receiver&&>(__rcvr)};
      };
    };
  }  // namespace __read_env

  struct __read_env_t
  {
    template <class _Query>
    constexpr auto operator()(_Query) const noexcept
    {
      return __make_sexpr<__read_env_t>(_Query());
    }
  };

  inline constexpr __read_env_t read_env{};

  template <>
  struct __sexpr_impl<__read_env_t> : __read_env::__read_env_impl
  {};

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write
  {
    struct __write_env_impl : __sexpr_defaults
    {
      static constexpr auto __get_env = []<class _State>(__ignore, _State const & __state) noexcept
      {
        auto&& logger = query(__state.__data_);
        std::printf("write_env::get_env %p\n", (void const *) &logger);
        return __state.__data_;
      };

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        return completion_signatures<set_value_t(int)>();
      }
    };
  }  // namespace __write

  struct __write_env_t
  {
    template <sender _Sender, class _Env>
    constexpr auto operator()(_Sender&& __sndr, _Env __env) const
    {
      return __make_sexpr<__write_env_t>(static_cast<_Env&&>(__env),
                                         static_cast<_Sender&&>(__sndr));
    }
  };

  inline constexpr __write_env_t write_env{};

  template <>
  struct __sexpr_impl<__write_env_t> : __write::__write_env_impl
  {};
}  // namespace STDEXEC

#include <memory>

namespace
{
  struct logger
  {
    logger() noexcept
    {
      std::printf("default %p\n", (void*) this);
    }

    logger(logger const & other) noexcept
    {
      std::printf("copy from %p to %p\n", (void const *) &other, (void*) this);
    }

    logger(logger&& other) noexcept
    {
      std::printf("move from %p to %p\n", (void*) &other, (void*) this);
    }

    ~logger()
    {
      std::printf("destroy %p\n", (void*) this);
    }

    logger& operator=(logger const & rhs) noexcept
    {
      std::printf("copy= from %p to %p\n", (void const *) &rhs, (void*) this);
      return *this;
    }

    logger& operator=(logger&& rhs) noexcept
    {
      std::printf("move= from %p to %p\n", (void*) &rhs, (void*) this);
      return *this;
    }
  };
  using namespace STDEXEC;

  template <class IncompleteType, class Env = env_of_t<IncompleteType>>
  struct ReceiverIncomplete
  {
    using receiver_concept = receiver_tag;

    IncompleteType* m_ptr;

    template <class V>
    void set_value(V&&) && noexcept
    {
      using rcvr_t = decltype(m_ptr->rcvr);
      static_cast<rcvr_t&&>(m_ptr->rcvr).set_value();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> Env
    {
      auto&& logger = query(m_ptr->rcvr.get_env());
      std::printf("ReceiverIncomplete::get_env %p\n", (void const *) &logger);
      return m_ptr->rcvr.get_env();
    }
  };

  template <sender Sndr, receiver Rcvr>
  struct OpStateIncomplete
  {
    using operation_state_concept = operation_state_tag;

    using rcvr_t          = ReceiverIncomplete<OpStateIncomplete, env_of_t<Rcvr>>;
    using inner_opstate_t = connect_result_t<Sndr, rcvr_t>;

    Rcvr            rcvr;
    inner_opstate_t inner_opstate;

    OpStateIncomplete(Sndr&& sndr, Rcvr rcvr_)
      : rcvr(static_cast<Rcvr&&>(rcvr_))
      , inner_opstate(connect(static_cast<Sndr&&>(sndr), rcvr_t{this}))
    {}

    void start() & noexcept
    {
      inner_opstate.start();
    }
  };

  template <sender Sndr>
  struct SenderIncomplete
  {
    using sender_concept = sender_tag;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures() -> completion_signatures<set_value_t()>
    {
      return {};
    }

    template <receiver Rcvr>
    auto connect(Rcvr rcvr) && -> OpStateIncomplete<Sndr, Rcvr>
    {
      return {static_cast<Sndr&&>(sndr), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr;
  };

  using result_t = std::optional<int>;
}  // namespace

int main()
{
  //alignas(64)
  char     state[246]{};
  result_t result{};

  struct receiver_t
  {
    using receiver_concept = receiver_tag;

    void set_value() && noexcept
    {
      result->emplace(0);
    }

    void*     state;
    result_t* result;
  };

  logger const alloc;
  auto         op = write_env(SenderIncomplete(read_env(query)), prop{query, alloc})
              .connect(receiver_t{&state, &result});

  op.start();

  return 0;
}
