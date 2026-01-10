/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "../stdexec/__detail/__tuple.hpp"
#include "../stdexec/__detail/__variant.hpp"
#include "../stdexec/execution.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace exec {
  namespace _seq {
    template <class... Senders>
    struct _sndr;

    struct sequence_t {
      template <class Sender>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto operator()(Sender sndr) const -> Sender;

      template <class... Senders>
        requires(sizeof...(Senders) > 1)
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto operator()(Senders... sndrs) const -> _sndr<Senders...>;
    };

    template <class Rcvr, class OpStateId, class Index>
    struct _rcvr {
      using receiver_concept = stdexec::receiver_t;
      using _opstate_t = stdexec::__t<OpStateId>;
      _opstate_t* _opstate;

      template <class... Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_value(Args&&... args) && noexcept {
        _opstate->_set_value(Index(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(Error&& err) && noexcept {
        stdexec::set_error(static_cast<Rcvr&&>(_opstate->_rcvr), static_cast<Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(host, device) void set_stopped() && noexcept {
        stdexec::set_stopped(static_cast<Rcvr&&>(_opstate->_rcvr));
      }

      // TODO: use the predecessor's completion scheduler as the current scheduler here.
      STDEXEC_ATTRIBUTE(host, device) auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(_opstate->_rcvr);
      }
    };

    template <class _Tuple>
    struct __convert_tuple_fn {
      template <class... _Ts>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto
        operator()(_Ts&&... __ts) const STDEXEC_AUTO_RETURN(_Tuple{static_cast<_Ts&&>(__ts)...});
    };

    template <class Rcvr, class... Senders>
    struct _opstate;

    template <class Rcvr, class Sender0, class... Senders>
    struct _opstate<Rcvr, Sender0, Senders...> {
      using operation_state_concept = stdexec::operation_state_t;

      // We will be connecting the first sender in the opstate constructor, so we don't need to
      // store it in the opstate. The use of `stdexec::__ignore` causes the first sender to not
      // be stored.
      using _senders_tuple_t = stdexec::__tuple<stdexec::__ignore, Senders...>;

      template <size_t Idx>
      using _rcvr_t = _seq::_rcvr<Rcvr, stdexec::__id<_opstate>, stdexec::__msize_t<Idx>>;

      template <class Sender, class Idx>
      using _child_opstate_t = stdexec::connect_result_t<Sender, _rcvr_t<stdexec::__v<Idx>>>;

      using _mk_child_ops_variant_fn = stdexec::__mzip_with2<
        stdexec::__q2<_child_opstate_t>,
        stdexec::__qq<stdexec::__variant_for>
      >;

      using _ops_variant_t = stdexec::__minvoke<
        _mk_child_ops_variant_fn,
        stdexec::__tuple<Sender0, Senders...>,
        stdexec::__make_indices<sizeof...(Senders) + 1>
      >;

      template <class CvrefSndrs>
      STDEXEC_ATTRIBUTE(host, device)
      explicit _opstate(Rcvr&& rcvr, CvrefSndrs&& sndrs)
        : _rcvr{static_cast<Rcvr&&>(rcvr)}
        , _sndrs{stdexec::__apply(
            __convert_tuple_fn<_senders_tuple_t>{},
            static_cast<CvrefSndrs&&>(sndrs))} // move all but the first sender into the opstate.
      {
        // Below, it looks like we are using `sndrs` after it has been moved from. This is not the
        // case. `sndrs` is moved into a tuple type that has `__ignore` for the first element. The
        // result is that the first sender in `sndrs` is not moved from, but the rest are.
        _ops.template emplace_from_at<0>(
          stdexec::connect, stdexec::__get<0>(static_cast<CvrefSndrs&&>(sndrs)), _rcvr_t<0>{this});
      }

      template <class Index, class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      void _set_value(Index, [[maybe_unused]] Args&&... args) noexcept {
        STDEXEC_TRY {
          constexpr size_t Idx = stdexec::__v<Index> + 1;
          if constexpr (Idx == sizeof...(Senders) + 1) {
            stdexec::set_value(static_cast<Rcvr&&>(_rcvr), static_cast<Args&&>(args)...);
          } else {
            auto& sndr = stdexec::__get<Idx>(_sndrs);
            auto& op = _ops.template emplace_from_at<Idx>(
              stdexec::connect, std::move(sndr), _rcvr_t<Idx>{this});
            stdexec::start(op);
          }
        }
        STDEXEC_CATCH_ALL {
          stdexec::set_error(static_cast<Rcvr&&>(_rcvr), std::current_exception());
        }
      }

      STDEXEC_ATTRIBUTE(host, device) void start() & noexcept {
        stdexec::start(_ops.template get<0>());
      }

      Rcvr _rcvr;
      _senders_tuple_t _sndrs;
      _ops_variant_t _ops{};
    };


    // The completions of the sequence sender are the error and stopped completions of all the
    // child senders plus the value completions of the last child sender.
    template <class... Env>
    struct _completions {
      // When folding left, the first sender folded will be the last sender in the list. That is
      // also when the "state" of the fold is void. For this case we want to include the value
      // completions; otherwise, we want to exclude them.
      template <class State, class... Args>
      struct _fold_left;

      template <class State, class Head, class... Tail>
      struct _fold_left<State, Head, Tail...> {
        using __t = stdexec::__gather_completion_signatures<
          stdexec::__completion_signatures_of_t<Head, Env...>,
          stdexec::set_value_t,
          stdexec::__mconst<stdexec::completion_signatures<>>::__f,
          stdexec::__cmplsigs::__default_completion,
          stdexec::__mtry_q<stdexec::__concat_completion_signatures>::__f,
          stdexec::__t<_fold_left<State, Tail...>>
        >;
      };

      template <class Head>
      struct _fold_left<void, Head> {
        using __t = stdexec::__mtry_q<stdexec::__concat_completion_signatures>::__f<
          stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
          stdexec::__completion_signatures_of_t<Head, Env...>
        >;
      };

      template <class... Sender>
      using __f = stdexec::__t<_fold_left<void, Sender...>>;
    };

    template <class Sender0, class... Senders>
    struct _sndr<Sender0, Senders...> {
      using sender_concept = stdexec::sender_t;

      template <class Self, class... Env>
      using _completions_t =
        stdexec::__minvoke<_completions<Env...>, stdexec::__copy_cvref_t<Self, Sender0>, Senders...>;

      template <stdexec::__decay_copyable Self, class... Env>
      STDEXEC_ATTRIBUTE(host, device)
      STDEXEC_EXPLICIT_THIS_BEGIN(auto get_completion_signatures)(this Self&&, Env&&...) {
        return _completions_t<Self, Env...>{};
      }
      STDEXEC_EXPLICIT_THIS_END(get_completion_signatures)

      template <class Self, class Rcvr>
        requires stdexec::__decay_copyable<Self>
      STDEXEC_ATTRIBUTE(host, device)
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Rcvr rcvr) {
        return _opstate<Rcvr, stdexec::__copy_cvref_t<Self, Sender0>, Senders...>{
          static_cast<Rcvr&&>(rcvr), static_cast<Self&&>(self)._sndrs};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) sequence_t _tag;
      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) stdexec::__ignore _ignore;
      stdexec::__tuple<Sender0, Senders...> _sndrs;
    };

    template <class Sender>
    STDEXEC_ATTRIBUTE(host, device)
    auto sequence_t::operator()(Sender sndr) const -> Sender {
      return sndr;
    }

    template <class... Senders>
      requires(sizeof...(Senders) > 1)
    STDEXEC_ATTRIBUTE(host, device)
    auto sequence_t::operator()(Senders... sndrs) const -> _sndr<Senders...> {
      return _sndr<Senders...>{{}, {}, {static_cast<Senders&&>(sndrs)...}};
    }
  } // namespace _seq

  using _seq::sequence_t;
  inline constexpr sequence_t sequence{};
} // namespace exec

namespace std {
  template <class... Senders>
  struct tuple_size<exec::_seq::_sndr<Senders...>>
    : std::integral_constant<std::size_t, sizeof...(Senders) + 2> { };

  template <size_t I, class... Senders>
  struct tuple_element<I, exec::_seq::_sndr<Senders...>> {
    using type = stdexec::__m_at_c<I, exec::sequence_t, stdexec::__, Senders...>;
  };
} // namespace std

STDEXEC_PRAGMA_POP()
