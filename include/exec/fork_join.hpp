/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"
#include "../stdexec/__detail/__receiver_ref.hpp"

#include <exception>

namespace exec {
  struct PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE { };

  struct fork_join_t {
    template <class Sndr, class... Closures>
    struct _sndr_t;

    struct _dematerialize_fn {
      struct _impl_fn {
        template <class Rcvr, class Tag, class... Args>
        STDEXEC_ATTRIBUTE(always_inline, host, device)
        void operator()(Rcvr& rcvr, Tag, const Args&... args) const noexcept {
          Tag{}(static_cast<Rcvr&&>(rcvr), args...);
        }
      };

      template <class Rcvr, class Tuple>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void operator()(Rcvr& rcvr, const Tuple& tupl) const noexcept {
        tupl.apply(_impl_fn{}, tupl, rcvr);
      }
    };

    struct _mk_when_all_fn {
      template <class CacheSndr, class... Closures>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      auto operator()(CacheSndr sndr, Closures&&... closures) const {
        return stdexec::when_all(static_cast<Closures&&>(closures)(sndr)...);
      }
    };

    template <class Completions>
    using _maybe_eptr_completion_t = stdexec::__if_c<
      stdexec::__nothrow_decay_copyable_results_t<Completions>::value,
      stdexec::__mset_nil,
      stdexec::__tuple_for<stdexec::set_error_t, ::std::exception_ptr>
    >;

    template <class Completions>
    using _variant_t = typename stdexec::__mset_insert<
      stdexec::__for_each_completion_signature<
        Completions,
        stdexec::__decayed_tuple,
        stdexec::__mset
      >,
      _maybe_eptr_completion_t<Completions>
    >::template rebind<stdexec::__variant_for>;

    template <class Domain>
    struct _env_t {
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto query(stdexec::get_domain_t) noexcept -> Domain {
        return {};
      }
    };

    template <class Tag, class... Args>
    using _cref_sig_t = Tag(const Args&...);

    // Given a set of async results, each of the form `tuple<Tag, Args...>`, compute
    // the corresponding completion signatures, where each signature is of the form
    // `Tag(const Args&...)`.
    template <class... AsyncResults>
    using _cache_sndr_completions_t =
      stdexec::completion_signatures<stdexec::__mapply<stdexec::__q<_cref_sig_t>, AsyncResults>...>;

    template <class Variant, class Domain>
    struct _cache_sndr_t {
      using sender_concept = stdexec::sender_t;

      template <class Rcvr>
      struct _opstate_t {
        using operation_state_concept = stdexec::operation_state_t;

        STDEXEC_ATTRIBUTE(host, device) void start() noexcept {
          Variant::visit(_dematerialize_fn{}, *_results_, _rcvr_);
        }

        Rcvr _rcvr_;
        const Variant* _results_;
      };

      template <class _Self, class... _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static auto get_completion_signatures(_Self&&, _Env&&...) noexcept {
        return stdexec::__mapply<stdexec::__qq<_cache_sndr_completions_t>, Variant>{};
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      auto connect(Rcvr rcvr) const -> _opstate_t<Rcvr> {
        return _opstate_t<Rcvr>{static_cast<Rcvr&&>(rcvr), _results_};
      }

      STDEXEC_ATTRIBUTE(host, device) static auto get_env() noexcept -> _env_t<Domain> {
        return {};
      }

      const Variant* _results_;
    };

    template <class Completions, class Closures, class Domain>
    using _when_all_sndr_t = stdexec::__tup::__apply_result_t<
      _mk_when_all_fn,
      Closures,
      _cache_sndr_t<_variant_t<Completions>, Domain>
    >;

    template <class Sndr, class Closures, class Rcvr>
    struct _opstate_t {
      using operation_state_concept = stdexec::operation_state_t;
      using _env_t = stdexec::__call_result_t<stdexec::__env::__fwd_fn, stdexec::env_of_t<Rcvr>>;
      using _child_completions_t = stdexec::__completion_signatures_of_t<Sndr, _env_t>;
      using _domain_t = stdexec::__early_domain_of_t<Sndr, stdexec::__none_such>;
      using _when_all_sndr_t =
        fork_join_t::_when_all_sndr_t<_child_completions_t, Closures, _domain_t>;
      using _child_opstate_t =
        stdexec::connect_result_t<Sndr, stdexec::__rcvr_ref_t<_opstate_t, _env_t>>;
      using _fork_opstate_t =
        stdexec::connect_result_t<_when_all_sndr_t, stdexec::__rcvr_ref_t<Rcvr>>;
      using _cache_sndr_t = fork_join_t::_cache_sndr_t<_variant_t<_child_completions_t>, _domain_t>;

      STDEXEC_ATTRIBUTE(host, device)
      explicit _opstate_t(Sndr&& sndr, Closures&& closures, Rcvr rcvr) noexcept
        : _rcvr_(static_cast<Rcvr&&>(rcvr))
        , _fork_opstate_(
            stdexec::connect(
              closures.apply(
                _mk_when_all_fn{},
                static_cast<Closures&&>(closures),
                _cache_sndr_t{&_cache_}),
              stdexec::__ref_rcvr(_rcvr_))) {
        _child_opstate_.__construct_from(
          stdexec::connect, static_cast<Sndr&&>(sndr), stdexec::__ref_rcvr(*this));
      }

      STDEXEC_IMMOVABLE(_opstate_t);

      STDEXEC_ATTRIBUTE(host, device) ~_opstate_t() {
        // If this opstate was never started, we must explicitly destroy the _child_opstate_.
        if (_cache_.is_valueless()) {
          _child_opstate_.__destroy();
        }
      }

      STDEXEC_ATTRIBUTE(host, device) void start() noexcept {
        stdexec::start(_child_opstate_.__get());
      }

      template <class Tag, class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      void _complete(Tag, Args&&... args) noexcept {
        STDEXEC_TRY {
          using _tuple_t = stdexec::__decayed_tuple<Tag, Args...>;
          _cache_.template emplace<_tuple_t>(Tag{}, static_cast<Args&&>(args)...);
        }
        STDEXEC_CATCH_ALL {
          if constexpr (!stdexec::__nothrow_decay_copyable<Args...>) {
            using _tuple_t = stdexec::__tuple_for<stdexec::set_error_t, ::std::exception_ptr>;
            _cache_._results_
              .template emplace<_tuple_t>(stdexec::set_error, ::std::current_exception());
          }
        }
        _child_opstate_.__destroy();
        stdexec::start(_fork_opstate_);
      }

      template <class... Values>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_value(Values&&... values) noexcept {
        this->_complete(stdexec::set_value, static_cast<Values&&>(values)...);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void set_error(Error&& err) noexcept {
        this->_complete(stdexec::set_error, static_cast<Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(always_inline, host, device) void set_stopped() noexcept {
        this->_complete(stdexec::set_stopped);
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>> {
        return stdexec::__env::__fwd_fn{}(stdexec::get_env(_rcvr_));
      }

      Rcvr _rcvr_;
      _variant_t<_child_completions_t> _cache_{};
      stdexec::__manual_lifetime<_child_opstate_t> _child_opstate_{};
      _fork_opstate_t _fork_opstate_;
    };

    template <class... Closures>
    struct _closure_t {
      using _closures_t = stdexec::__tuple_for<Closures...>;

      template <class Sndr>
      STDEXEC_ATTRIBUTE(host, device)
      friend constexpr auto
        operator|(Sndr sndr, _closure_t self) noexcept -> _sndr_t<Sndr, Closures...> {
        return _sndr_t<Sndr, Closures...>{
          {}, static_cast<_closures_t&&>(self._closures_), static_cast<Sndr&&>(sndr)};
      }

      _closures_t _closures_;
    };

    template <class Sndr, class... Closures>
      requires stdexec::sender<Sndr>
    STDEXEC_ATTRIBUTE(host, device)
    auto operator()(Sndr sndr, Closures... closures) const -> _sndr_t<Sndr, Closures...> {
      return {{}, {static_cast<Closures&&>(closures)...}, static_cast<Sndr&&>(sndr)};
    }

    template <class... Closures>
      requires((!stdexec::sender<Closures>) && ...)
    STDEXEC_ATTRIBUTE(host, device)
    auto operator()(Closures... closures) const -> _closure_t<Closures...> {
      return {{static_cast<Closures&&>(closures)...}};
    }
  };

  template <>
  struct fork_join_t::_env_t<stdexec::__none_such> { };

  template <class Sndr, class... Closures>
  struct fork_join_t::_sndr_t {
    using sender_concept = stdexec::sender_t;
    using _closures_t = stdexec::__tuple_for<Closures...>;

    template <class Self, class... Env>
    STDEXEC_ATTRIBUTE(host, device)
    static auto get_completion_signatures(Self&&, Env&&...) noexcept {
      using namespace stdexec;
      using _domain_t = __early_domain_of_t<Sndr, __none_such>;
      using _child_t = __copy_cvref_t<Self, Sndr>;
      using _child_completions_t = __completion_signatures_of_t<_child_t, __fwd_env_t<Env>...>;
      using __decay_copyable_results_t = stdexec::__decay_copyable_results_t<_child_completions_t>;

      if constexpr (!stdexec::__valid_completion_signatures<_child_completions_t>) {
        return _child_completions_t{};
      } else if constexpr (!__decay_copyable_results_t::value) {
        return _ERROR_<
          _WHAT_<>(PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE),
          _IN_ALGORITHM_(exec::fork_join_t)
        >();
      } else {
        using _sndr_t = _when_all_sndr_t<_child_completions_t, _closures_t, _domain_t>;
        return completion_signatures_of_t<_sndr_t, __fwd_env_t<Env>...>{};
      }
    }

    template <class Rcvr>
    STDEXEC_ATTRIBUTE(host, device)
    auto connect(Rcvr rcvr) && -> _opstate_t<Sndr, _closures_t, Rcvr> {
      return _opstate_t<Sndr, _closures_t, Rcvr>{
        static_cast<Sndr&&>(sndr_),
        static_cast<_closures_t&&>(_closures_),
        static_cast<Rcvr&&>(rcvr)};
    }

    template <class Rcvr>
    STDEXEC_ATTRIBUTE(host, device)
    auto connect(Rcvr rcvr) const & -> _opstate_t<Sndr const &, _closures_t const &, Rcvr> {
      return _opstate_t<Sndr const &, _closures_t const &, Rcvr>{
        sndr_, _closures_, static_cast<Rcvr&&>(rcvr)};
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sndr>> {
      return stdexec::__env::__fwd_fn{}(stdexec::get_env(sndr_));
    }

    STDEXEC_ATTRIBUTE(no_unique_address) fork_join_t _tag_;
    stdexec::__tuple_for<Closures...> _closures_;
    Sndr sndr_;
  };

  inline constexpr fork_join_t fork_join{};
} // namespace exec
