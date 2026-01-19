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
      using receiver_concept = STDEXEC::receiver_t;
      using _opstate_t = STDEXEC::__t<OpStateId>;
      _opstate_t* _opstate;

      template <class... Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_value(Args&&... args) && noexcept {
        _opstate->_set_value(Index(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(Error&& err) && noexcept {
        STDEXEC::set_error(static_cast<Rcvr&&>(_opstate->_rcvr), static_cast<Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(host, device) void set_stopped() && noexcept {
        STDEXEC::set_stopped(static_cast<Rcvr&&>(_opstate->_rcvr));
      }

      // TODO: use the predecessor's completion scheduler as the current scheduler here.
      STDEXEC_ATTRIBUTE(host, device) auto get_env() const noexcept -> STDEXEC::env_of_t<Rcvr> {
        return STDEXEC::get_env(_opstate->_rcvr);
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
      using operation_state_concept = STDEXEC::operation_state_t;

      // We will be connecting the first sender in the opstate constructor, so we don't need to
      // store it in the opstate. The use of `STDEXEC::__ignore` causes the first sender to not
      // be stored.
      using _senders_tuple_t = STDEXEC::__tuple<STDEXEC::__ignore, Senders...>;

      template <size_t Idx>
      using _rcvr_t = _seq::_rcvr<Rcvr, STDEXEC::__id<_opstate>, STDEXEC::__msize_t<Idx>>;

      template <class Sender, class Idx>
      using _child_opstate_t = STDEXEC::connect_result_t<Sender, _rcvr_t<Idx::value>>;

      using _mk_child_ops_variant_fn = STDEXEC::__mzip_with2<
        STDEXEC::__q2<_child_opstate_t>,
        STDEXEC::__qq<STDEXEC::__variant_for>
      >;

      using _ops_variant_t = STDEXEC::__minvoke<
        _mk_child_ops_variant_fn,
        STDEXEC::__tuple<Sender0, Senders...>,
        STDEXEC::__make_indices<sizeof...(Senders) + 1>
      >;

      template <class CvrefSndrs>
      STDEXEC_ATTRIBUTE(host, device)
      explicit _opstate(Rcvr&& rcvr, CvrefSndrs&& sndrs)
        : _rcvr{static_cast<Rcvr&&>(rcvr)}
        , _sndrs{STDEXEC::__apply(
            __convert_tuple_fn<_senders_tuple_t>{},
            static_cast<CvrefSndrs&&>(sndrs))} // move all but the first sender into the opstate.
      {
        // Below, it looks like we are using `sndrs` after it has been moved from. This is not the
        // case. `sndrs` is moved into a tuple type that has `__ignore` for the first element. The
        // result is that the first sender in `sndrs` is not moved from, but the rest are.
        _ops.template emplace_from_at<0>(
          STDEXEC::connect, STDEXEC::__get<0>(static_cast<CvrefSndrs&&>(sndrs)), _rcvr_t<0>{this});
      }

      template <class Index, class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      void _set_value(Index, [[maybe_unused]] Args&&... args) noexcept {
        STDEXEC_TRY {
          constexpr size_t Idx = Index::value + 1;
          if constexpr (Idx == sizeof...(Senders) + 1) {
            STDEXEC::set_value(static_cast<Rcvr&&>(_rcvr), static_cast<Args&&>(args)...);
          } else {
            auto& sndr = STDEXEC::__get<Idx>(_sndrs);
            auto& op = _ops.template emplace_from_at<Idx>(
              STDEXEC::connect, std::move(sndr), _rcvr_t<Idx>{this});
            STDEXEC::start(op);
          }
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(static_cast<Rcvr&&>(_rcvr), std::current_exception());
        }
      }

      STDEXEC_ATTRIBUTE(host, device) void start() & noexcept {
        STDEXEC::start(_ops.template get<0>());
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
        using __t = STDEXEC::__gather_completion_signatures_t<
          STDEXEC::__completion_signatures_of_t<Head, Env...>,
          STDEXEC::set_value_t,
          STDEXEC::__mconst<STDEXEC::completion_signatures<>>::__f,
          STDEXEC::__cmplsigs::__default_completion,
          STDEXEC::__mtry_q<STDEXEC::__concat_completion_signatures_t>::__f,
          STDEXEC::__t<_fold_left<State, Tail...>>
        >;
      };

      template <class Head>
      struct _fold_left<void, Head> {
        using __t = STDEXEC::__mtry_q<STDEXEC::__concat_completion_signatures_t>::__f<
          STDEXEC::completion_signatures<STDEXEC::set_error_t(std::exception_ptr)>,
          STDEXEC::__completion_signatures_of_t<Head, Env...>
        >;
      };

      template <class... Sender>
      using __f = STDEXEC::__t<_fold_left<void, Sender...>>;
    };

    template <class Sender0, class... Senders>
    struct _sndr<Sender0, Senders...> {
      using sender_concept = STDEXEC::sender_t;

      template <class Self, class... Env>
      using _completions_t =
        STDEXEC::__minvoke<_completions<Env...>, STDEXEC::__copy_cvref_t<Self, Sender0>, Senders...>;

      template <STDEXEC::__decay_copyable Self, class... Env>
      STDEXEC_ATTRIBUTE(host, device)
      static consteval auto get_completion_signatures() {
        return _completions_t<Self, Env...>{};
      }

      template <class Self, class Rcvr>
        requires STDEXEC::__decay_copyable<Self>
      STDEXEC_ATTRIBUTE(host, device)
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Rcvr rcvr) {
        return _opstate<Rcvr, STDEXEC::__copy_cvref_t<Self, Sender0>, Senders...>{
          static_cast<Rcvr&&>(rcvr), static_cast<Self&&>(self)._sndrs};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) sequence_t _tag;
      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) STDEXEC::__ignore _ignore;
      STDEXEC::__tuple<Sender0, Senders...> _sndrs;
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
    using type = STDEXEC::__m_at_c<I, exec::sequence_t, STDEXEC::__, Senders...>;
  };
} // namespace std

STDEXEC_PRAGMA_POP()
