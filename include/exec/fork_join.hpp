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

#include "../stdexec/__detail/__receiver_ref.hpp"
#include "../stdexec/execution.hpp"

#include <exception>

namespace exec {
  struct PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE { };

  struct fork_join_impl_t {
    struct _dematerialize_fn {
      struct _impl_fn {
        template <class Rcvr, class Tag, class... Args>
        STDEXEC_ATTRIBUTE(always_inline, host, device)
        constexpr void operator()(Rcvr& rcvr, Tag, const Args&... args) const noexcept {
          Tag{}(static_cast<Rcvr&&>(rcvr), args...);
        }
      };

      template <class Rcvr, class Tuple>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void operator()(Rcvr& rcvr, const Tuple& tupl) const noexcept {
        STDEXEC::__apply(_impl_fn{}, tupl, rcvr);
      }
    };

    struct _mk_when_all_fn {
      template <class CacheSndr, class... Closures>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto operator()(CacheSndr sndr, Closures&&... closures) const {
        return STDEXEC::when_all(static_cast<Closures&&>(closures)(sndr)...);
      }
    };

    template <class Completions>
    using _maybe_eptr_completion_t = STDEXEC::__if_c<
      STDEXEC::__nothrow_decay_copyable_results_t<Completions>::value,
      STDEXEC::__mset_nil,
      STDEXEC::__tuple<STDEXEC::set_error_t, ::std::exception_ptr>
    >;

    template <class Completions>
    using _variant_t = STDEXEC::__mset_insert<
      STDEXEC::__for_each_completion_signature_t<
        Completions,
        STDEXEC::__decayed_tuple,
        STDEXEC::__mset
      >,
      _maybe_eptr_completion_t<Completions>
    >::template rebind<STDEXEC::__variant>;

    template <class Domain>
    struct _env_t {
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto query(STDEXEC::get_domain_t) noexcept -> Domain {
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
      STDEXEC::completion_signatures<STDEXEC::__mapply<STDEXEC::__q<_cref_sig_t>, AsyncResults>...>;

    template <class Variant, class Domain>
    struct _cache_sndr_t {
      using sender_concept = STDEXEC::sender_t;

      template <class Rcvr>
      struct _opstate_t {
        using operation_state_concept = STDEXEC::operation_state_t;

        STDEXEC_ATTRIBUTE(host, device) void start() noexcept {
          STDEXEC::__visit(_dematerialize_fn{}, *_results_, _rcvr_);
        }

        Rcvr _rcvr_;
        const Variant* _results_;
      };

      template <class _Self, class... _Env>
      STDEXEC_ATTRIBUTE(host, device)
      static consteval auto get_completion_signatures() {
        return STDEXEC::__mapply<STDEXEC::__qq<_cache_sndr_completions_t>, Variant>{};
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto connect(Rcvr rcvr) const -> _opstate_t<Rcvr> {
        return _opstate_t<Rcvr>{static_cast<Rcvr&&>(rcvr), _results_};
      }

      STDEXEC_ATTRIBUTE(host, device) static constexpr auto get_env() noexcept -> _env_t<Domain> {
        return {};
      }

      const Variant* _results_;
    };

    template <class Completions, class Closures, class Domain>
    using _when_all_sndr_t = STDEXEC::__apply_result_t<
      _mk_when_all_fn,
      Closures,
      _cache_sndr_t<_variant_t<Completions>, Domain>
    >;

    template <class Sndr, class Closures, class Rcvr>
    struct _opstate_t {
      using operation_state_concept = STDEXEC::operation_state_t;
      using _env_t = STDEXEC::__fwd_env_t<STDEXEC::env_of_t<Rcvr>>;
      using _child_completions_t = STDEXEC::__completion_signatures_of_t<Sndr, _env_t>;
      using _domain_t = STDEXEC::__completion_domain_of_t<STDEXEC::set_value_t, Sndr, _env_t>;
      using _when_all_sndr_t =
        fork_join_impl_t::_when_all_sndr_t<_child_completions_t, Closures, _domain_t>;
      using _child_opstate_t =
        STDEXEC::connect_result_t<Sndr, STDEXEC::__rcvr_ref_t<_opstate_t, _env_t>>;
      using _fork_opstate_t =
        STDEXEC::connect_result_t<_when_all_sndr_t, STDEXEC::__rcvr_ref_t<Rcvr>>;
      using _cache_sndr_t =
        fork_join_impl_t::_cache_sndr_t<_variant_t<_child_completions_t>, _domain_t>;

      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit _opstate_t(Sndr&& sndr, Closures&& closures, Rcvr rcvr) noexcept
        : _rcvr_(static_cast<Rcvr&&>(rcvr))
        , _fork_opstate_(
            STDEXEC::connect(
              STDEXEC::__apply(
                _mk_when_all_fn{},
                static_cast<Closures&&>(closures),
                _cache_sndr_t{&_cache_}),
              STDEXEC::__ref_rcvr(_rcvr_))) {
        _child_opstate_.__construct_from(
          STDEXEC::connect, static_cast<Sndr&&>(sndr), STDEXEC::__ref_rcvr(*this));
      }

      STDEXEC_IMMOVABLE(_opstate_t);

      STDEXEC_ATTRIBUTE(host, device) constexpr ~_opstate_t() {
        // If this opstate was never started, we must explicitly destroy the _child_opstate_.
        if (_cache_.__is_valueless()) {
          _child_opstate_.__destroy();
        }
      }

      STDEXEC_ATTRIBUTE(host, device) constexpr void start() noexcept {
        STDEXEC::start(_child_opstate_.__get());
      }

      template <class Tag, class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr void _complete(Tag, Args&&... args) noexcept {
        STDEXEC_TRY {
          using _tuple_t = STDEXEC::__decayed_tuple<Tag, Args...>;
          _cache_.template emplace<_tuple_t>(Tag{}, static_cast<Args&&>(args)...);
        }
        STDEXEC_CATCH_ALL {
          if constexpr (!STDEXEC::__nothrow_decay_copyable<Args...>) {
            using _tuple_t = STDEXEC::__tuple<STDEXEC::set_error_t, ::std::exception_ptr>;
            _cache_._results_
              .template emplace<_tuple_t>(STDEXEC::set_error, ::std::current_exception());
          }
        }
        _child_opstate_.__destroy();
        STDEXEC::start(_fork_opstate_);
      }

      template <class... Values>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_value(Values&&... values) noexcept {
        this->_complete(STDEXEC::set_value, static_cast<Values&&>(values)...);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_error(Error&& err) noexcept {
        this->_complete(STDEXEC::set_error, static_cast<Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(always_inline, host, device) void set_stopped() noexcept {
        this->_complete(STDEXEC::set_stopped);
      }

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto get_env() const noexcept -> _env_t {
        return STDEXEC::__fwd_env(STDEXEC::get_env(_rcvr_));
      }

      Rcvr _rcvr_;
      _variant_t<_child_completions_t> _cache_{STDEXEC::__no_init};
      STDEXEC::__manual_lifetime<_child_opstate_t> _child_opstate_{};
      _fork_opstate_t _fork_opstate_;
    };
  };

  struct fork_join_t {
    template <class Sndr, class... Closures>
      requires STDEXEC::sender<Sndr>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto
      operator()(Sndr&& sndr, Closures&&... closures) const -> STDEXEC::__well_formed_sender auto {
      return STDEXEC::__make_sexpr<fork_join_t>(
        STDEXEC::__tuple{std::forward<Closures>(closures)...}, std::forward<Sndr>(sndr));
    }

    template <class... Closures>
      requires((!STDEXEC::sender<Closures>) && ...)
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto operator()(Closures&&... closures) const {
      return STDEXEC::__closure{*this, std::forward<Closures>(closures)...};
    }
  };

  template <>
  struct fork_join_impl_t::_env_t<STDEXEC::indeterminate_domain<>> { };

  inline constexpr fork_join_t fork_join{};

} // namespace exec

namespace exec::__fork_join {
  struct __impls : STDEXEC::__sexpr_defaults {
    template <class Self, class... Env>
    STDEXEC_ATTRIBUTE(host, device)
    static consteval auto get_completion_signatures() {
      using namespace STDEXEC;

      using _closures_t = STDEXEC::__data_of<Self>;
      using _child_sndr_t = STDEXEC::__child_of<Self>;

      using _domain_t = __completion_domain_of_t<set_value_t, _child_sndr_t, Env...>;
      using _child_t = __copy_cvref_t<Self, _child_sndr_t>;
      using _child_completions_t = __completion_signatures_of_t<_child_t, __fwd_env_t<Env>...>;
      using __decay_copyable_results_t = STDEXEC::__decay_copyable_results_t<_child_completions_t>;

      if constexpr (!STDEXEC::__valid_completion_signatures<_child_completions_t>) {
        return _child_completions_t{};
      } else if constexpr (!__decay_copyable_results_t::value) {
        return _ERROR_<
          _WHAT_(PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE),
          _IN_ALGORITHM_(exec::fork_join_t)
        >();
      } else {
        using _sndr_t =
          fork_join_impl_t::_when_all_sndr_t<_child_completions_t, _closures_t, _domain_t>;
        return __completion_signatures_of_t<_sndr_t, __fwd_env_t<Env>...>{};
      }
    }

    static constexpr auto connect =
      []<class _Receiver, class _Sender>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept {
        using _closures_t = STDEXEC::__data_of<_Sender>;
        using _sndr_t = STDEXEC::__child_of<_Sender>;

        return fork_join_impl_t::_opstate_t<_sndr_t, _closures_t, _Receiver>{
          STDEXEC::__get<2>(static_cast<_Sender&&>(__sndr)),
          STDEXEC::__get<1>(static_cast<_Sender&&>(__sndr)),
          static_cast<_Receiver&&>(__rcvr)};
      };
  };
} // namespace exec::__fork_join

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::fork_join_t> : exec::__fork_join::__impls { };
} // namespace STDEXEC
