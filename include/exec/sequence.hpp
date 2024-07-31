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

#include <stdexec/execution.hpp>
#include <stdexec/__detail/__manual_lifetime.hpp>

namespace exec {
  namespace _seq {
    template <class... Sndrs>
    struct _sndr;

    struct sequence_t {
      template <class Sndr>
      Sndr operator()(Sndr sndr) const;

      template <class... Sndrs>
        requires(sizeof...(Sndrs) > 1) && stdexec::__domain::__has_common_domain<Sndrs...>
      _sndr<Sndrs...> operator()(Sndrs... sndrs) const;
    };

    template <class... Args>
    struct _ops_tuple;

    template <class Sndr, class... Rest>
    struct _ops_tuple<Sndr, Rest...> : _ops_tuple<Rest...> {
      explicit _ops_tuple(Sndr&& sndr, Rest&&... rest)
        : _ops_tuple<Rest...>{static_cast<Rest&&>(rest)...}
        , _head{static_cast<Sndr&&>(sndr)} {
      }

      Sndr _head;

      _ops_tuple<Rest...>& _tail() noexcept {
        return *this;
      }
    };

    template <class Rcvr>
    struct _ops_tuple<Rcvr> {
      using _rcvr_t = Rcvr;
      Rcvr _rcvr;
    };

    template <class... Args>
    union _ops_variant { };

    template <class Sndr, class... Rest>
    struct _rcvr {
      using receiver_concept = stdexec::receiver_t;
      using _rcvr_t = typename _ops_tuple<Rest...>::_rcvr_t;
      _ops_variant<Sndr, Rest...>* _self;

      template <class... Args>
      void set_value(Args&&... args) && noexcept {
        auto& sndrs = *_self->_head.__get()._sndrs;
        try {
          if constexpr (sizeof...(Rest) == 1) {
            // destroy _head after completing the operation in case the arguments are references
            // to objects owned by _head.
            stdexec::set_value(static_cast<_rcvr_t&&>(sndrs._rcvr), static_cast<Args&&>(args)...);
            _self->_head.__destroy();
          } else {
            _self->_head.__destroy();
            _self->_tail.__construct(sndrs._head, sndrs._tail()); // potentially throwing
            stdexec::start(_self->_tail.__get()._head.__get()._op);
          }
        } catch (...) {
          stdexec::set_error(static_cast<_rcvr_t&&>(sndrs._rcvr), std::current_exception());
        }
      }

      template <class Error>
      void set_error(Error&& err) && noexcept {
        stdexec::set_error(
          static_cast<_rcvr_t&&>(_self->_head.__get()._sndrs->_rcvr), static_cast<Error&&>(err));
        _self->_head.__destroy();
      }

      void set_stopped() && noexcept {
        stdexec::set_stopped(static_cast<_rcvr_t&&>(_self->_head.__get()._sndrs->_rcvr));
        _self->_head.__destroy();
      }

      stdexec::env_of_t<_rcvr_t> get_env() const noexcept {
        return stdexec::get_env(_self->_head.__get()._sndrs->_rcvr);
      }
    };

    template <class Sndr, class... Rest>
      requires(sizeof...(Rest) > 0)
    union _ops_variant<Sndr, Rest...> {
      explicit _ops_variant(Sndr& sndr, _ops_tuple<Rest...>& sndrs) {
        auto connect_fn = [&] {
          return stdexec::connect(static_cast<Sndr&&>(sndr), _rcvr<Sndr, Rest...>{this});
        };
        _head.__construct(&sndrs, stdexec::__emplace_from{connect_fn});
      }

      ~_ops_variant() {
      }

      struct _head_t {
        _ops_tuple<Rest...>* _sndrs;
        stdexec::connect_result_t<Sndr, _rcvr<Sndr, Rest...>> _op;
      };

      stdexec::__manual_lifetime<_head_t> _head;
      stdexec::__manual_lifetime<_ops_variant<Rest...>> _tail;
    };

    template <class Rcvr, class... Sndrs>
    struct _opstate;

    template <class Rcvr, class Sndr, class... Rest>
    struct _opstate<Rcvr, Sndr, Rest...> {
      using operation_state_concept = stdexec::operation_state_t;

      _ops_tuple<Rest..., Rcvr> _tupl;
      _ops_variant<Sndr, Rest..., Rcvr> _var;

      explicit _opstate(Rcvr&& rcvr, Sndr sndr, Rest&&... rest)
        : _tupl{static_cast<Rest&&>(rest)..., static_cast<Rcvr&&>(rcvr)}
        , _var{sndr, _tupl} {
      }

      void start() & noexcept {
        stdexec::start(_var._head.__get()._op);
      }
    };

    // The completions of the sequence sender are the error and stopped completions of all the
    // child senders plus the value completions of the last child sender.
    template <class... Env>
    struct _completions {
      // When folding left, the first sender folded will be the last sender in the list. That is
      // also when the "state" of the fold is void. For this case we want to include the value
      // completions; otherwise, we want to exclude them.
      template <class Completions, class Sndr>
      using _fold_last_fn = //
        stdexec::__mtry_q<stdexec::__concat_completion_signatures>::__f<
          stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
          stdexec::__completion_signatures_of_t<Sndr, Env...>>;

      // For the rest of the senders (besides the last), the value completions are discarded. That
      // is achieved by the third template argument below, which transforms all value completions to
      // completion_signatures<>.
      template <class Completions, class Sndr>
      using _fold_rest_fn = //
        stdexec::__gather_completion_signatures<
          stdexec::__completion_signatures_of_t<Sndr, Env...>,
          stdexec::set_value_t,
          stdexec::__mconst<stdexec::completion_signatures<>>::__f,
          stdexec::__sigs::__default_completion,
          stdexec::__mtry_q<stdexec::__concat_completion_signatures>::__f,
          Completions>;

      template <class Completions, class Sndr>
      using _fold_fn = //
        stdexec::__minvoke_if_c<
          stdexec::__same_as<Completions, void>,
          stdexec::__q2<_fold_last_fn>,
          stdexec::__q2<_fold_rest_fn>,
          Completions,
          Sndr>;

      template <class... Sndrs>
      using __f = //
        stdexec::__minvoke<stdexec::__mfold_left<void, stdexec::__q2<_fold_fn>>, Sndrs...>;
    };

    template <class... Sndrs>
    struct _sndr : stdexec::__tuple_for<sequence_t, stdexec::__, Sndrs...> {
      using sender_concept = stdexec::sender_t;

      template <class... Env>
      using _completions_t = stdexec::__minvoke<_completions<Env...>, Sndrs...>;

      template <class Self, class... Env>
        requires(stdexec::__decay_copyable<stdexec::__copy_cvref_t<Self, Sndrs>> && ...)
      static auto get_completion_signatures(Self&&, Env&&...) -> _completions_t<Env...> {
        return {};
      }

      template <class Self, class Rcvr>
      static auto connect(Self&& self, Rcvr rcvr) {
        return self.apply(
          [](Rcvr&& rcvr, auto, auto, Sndrs... sndrs) {
            return _opstate<Rcvr, Sndrs...>{
              static_cast<Rcvr&&>(rcvr), static_cast<Sndrs&&>(sndrs)...};
          },
          static_cast<typename _sndr::__tuple&&>(self),
          static_cast<Rcvr&&>(rcvr));
      }
    };

    template <class Sndr>
    Sndr sequence_t::operator()(Sndr sndr) const {
      return sndr;
    }

    template <class... Sndrs>
      requires(sizeof...(Sndrs) > 1) && stdexec::__domain::__has_common_domain<Sndrs...>
    _sndr<Sndrs...> sequence_t::operator()(Sndrs... sndrs) const {
      return _sndr<Sndrs...>{
        {{}, {}, {static_cast<Sndrs&&>(sndrs)}...}
      };
    }
  } // namespace _seq

  using _seq::sequence_t;
  inline constexpr sequence_t sequence{};
} // namespace exec

namespace std {
  template <class... Sndrs>
  struct tuple_size<exec::_seq::_sndr<Sndrs...>>
    : std::integral_constant<std::size_t, sizeof...(Sndrs) + 2> { };

  template <size_t I, class... Sndrs>
  struct tuple_element<I, exec::_seq::_sndr<Sndrs...>> {
    using type = stdexec::__m_at_c<I, exec::sequence_t, stdexec::__, Sndrs...>;
  };
} // namespace std
