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

#pragma once

#include "../stdexec/__detail/__receiver_ref.hpp"
#include "../stdexec/execution.hpp"

#include "completion_signatures.hpp"
#include "sender_for.hpp"

namespace experimental::execution
{
  struct unless_stop_requested_t;

  namespace _unless_stop_requested
  {
    using namespace STDEXEC;

    template <typename Env>
    concept _unstoppable_env = unstoppable_token<stop_token_of_t<Env>>;

    template <typename Receiver>
    concept _unstoppable_receiver = _unstoppable_env<env_of_t<Receiver>>;

    template <class CvChild, class Receiver>
    struct _opstate
    {
      using receiver_concept = receiver_tag;
      using child_opstate_t  = connect_result_t<CvChild, __receiver_ref<Receiver>>;

      constexpr explicit _opstate(CvChild&& child, Receiver rcvr)
        noexcept(__nothrow_connectable<CvChild, __receiver_ref<Receiver>>)
        : rcvr_(static_cast<Receiver&&>(rcvr))
        , child_op_(STDEXEC::connect(static_cast<CvChild&&>(child), STDEXEC::__receiver_ref(rcvr_)))
      {}

      constexpr void start() noexcept
      {
        if (STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)).stop_requested())
        {
          STDEXEC::set_stopped(static_cast<Receiver&&>(rcvr_));
        }
        else
        {
          STDEXEC::start(child_op_);
        }
      }

      Receiver        rcvr_;
      child_opstate_t child_op_;
    };

    struct _impls : __sexpr_defaults
    {
      template <class Self, class Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(sender_for<Self, unless_stop_requested_t>);
        if constexpr (_unstoppable_env<Env>)
        {
          return STDEXEC::get_completion_signatures<__child_of<Self>, Env>();
        }
        else
        {
          return exec::concat_completion_signatures(
            STDEXEC::get_completion_signatures<__child_of<Self>, Env>(),
            completion_signatures<set_stopped_t()>());
        }
      }

      template <class Receiver>
      using _receiver_t =
        __if_c<_unstoppable_receiver<Receiver>, Receiver, __receiver_ref<Receiver>>;

      static constexpr auto __connect =  //
        []<class Sender, class Receiver>(Sender&& sndr, Receiver rcvr) noexcept(
          __nothrow_connectable<__child_of<Sender>, _receiver_t<Receiver>>)
      {
        auto& [tag, ign, child] = sndr;

        if constexpr (_unstoppable_receiver<Receiver>)
        {
          return STDEXEC::connect(STDEXEC::__forward_like<Sender>(child),
                                  static_cast<Receiver&&>(rcvr));
        }
        else
        {
          using opstate_t = _opstate<__child_of<Sender>, Receiver>;
          return opstate_t(STDEXEC::__forward_like<Sender>(child), static_cast<Receiver&&>(rcvr));
        }
      };
    };
  }  // namespace _unless_stop_requested

  struct unless_stop_requested_t : STDEXEC::sender_adaptor_closure<unless_stop_requested_t>
  {
    [[nodiscard]]
    constexpr auto operator()() const noexcept
    {
      return *this;
    }

    template <STDEXEC::sender Sender>
    [[nodiscard]]
    constexpr auto operator()(Sender&& sndr) const -> STDEXEC::__well_formed_sender auto
    {
      return STDEXEC::__sexpr{unless_stop_requested_t(), {}, static_cast<Sender&&>(sndr)};
    }
  };

  inline constexpr unless_stop_requested_t unless_stop_requested{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC
{
  template <>
  struct __sexpr_impl<::exec::unless_stop_requested_t> : ::exec::_unless_stop_requested::_impls
  {};
}  // namespace STDEXEC
