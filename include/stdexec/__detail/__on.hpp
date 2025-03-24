/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__continues_on.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__inline_scheduler.hpp"
#include "__meta.hpp"
#include "__schedulers.hpp"
#include "__senders_core.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__sender_introspection.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"
#include "__write_env.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on_v2 {
    inline constexpr __mstring __on_context = "In stdexec::on(Scheduler, Sender)..."_mstr;
    inline constexpr __mstring __no_scheduler_diag =
      "stdexec::on() requires a scheduler to transition back to."_mstr;
    inline constexpr __mstring __no_scheduler_details =
      "The provided environment lacks a value for the get_scheduler() query."_mstr;

    template <
      __mstring _Context = __on_context,
      __mstring _Diagnostic = __no_scheduler_diag,
      __mstring _Details = __no_scheduler_details>
    struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_ { };

    struct on_t;

    template <class _Sender, class _Env>
    struct __no_scheduler_in_environment {
      using sender_concept = sender_t;

      static auto
        get_completion_signatures(const __no_scheduler_in_environment&, const auto&) noexcept {
        return __mexception<
          _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_<>,
          _WITH_SENDER_<_Sender>,
          _WITH_ENVIRONMENT_<_Env>>{};
      }
    };

    template <class _Scheduler, class _Closure>
    struct __on_data {
      _Scheduler __sched_;
      _Closure __clsur_;
    };
    template <class _Scheduler, class _Closure>
    __on_data(_Scheduler, _Closure) -> __on_data<_Scheduler, _Closure>;

    template <class _Scheduler>
    struct __with_sched {
      using __t = __with_sched;
      using __id = __with_sched;

      _Scheduler __sched_;

      auto query(get_scheduler_t) const noexcept -> _Scheduler {
        return __sched_;
      }

      auto query(get_domain_t) const noexcept {
        return query_or(get_domain, __sched_, default_domain());
      }
    };

    template <class _Scheduler>
    __with_sched(_Scheduler) -> __with_sched<_Scheduler>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<on_t>(static_cast<_Scheduler&&>(__sched), static_cast<_Sender&&>(__sndr)));
      }

      template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const
        -> __well_formed_sender auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<on_t>(
            __on_data{static_cast<_Scheduler&&>(__sched), static_cast<_Closure&&>(__clsur)},
            static_cast<_Sender&&>(__sndr)));
      }

      template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
      STDEXEC_ATTRIBUTE((always_inline)) auto operator()(_Scheduler&& __sched, _Closure&& __clsur) const {
        return __binder_back<on_t, __decay_t<_Scheduler>, __decay_t<_Closure>>{
          {{static_cast<_Scheduler&&>(__sched)}, {static_cast<_Closure&&>(__clsur)}},
          {},
          {}
        };
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&]<class _Data>(__ignore, _Data&& __data, __ignore) noexcept -> decltype(auto) {
          if constexpr (scheduler<_Data>) {
            return __env::__join(
              __sched_env{static_cast<_Data&&>(__data)}, static_cast<_Env&&>(__env));
          } else {
            return static_cast<_Env>(static_cast<_Env&&>(__env));
          }
        };
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) static auto __transform_sender_fn(const _Env& __env) noexcept {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          if constexpr (scheduler<_Data>) {
            // This branch handles the case where `on` was called like `on(sch, snd)`
            auto __old = query_or(get_scheduler, __env, __none_such{});
            if constexpr (__same_as<decltype(__old), __none_such>) {
              if constexpr (__is_root_env<_Env>) {
                return continues_on(
                  starts_on(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)),
                  __inln::__scheduler{});
              } else {
                return __none_such{};
              }
            } else {
              return continues_on(
                starts_on(static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)),
                static_cast<decltype(__old)&&>(__old));
            }
          } else {
            // This branch handles the case where `on` was called like `on(snd, sch, clsur)`
            auto __old = query_or(
              get_completion_scheduler<set_value_t>,
              get_env(__child),
              query_or(get_scheduler, __env, __none_such{}));
            if constexpr (__same_as<decltype(__old), __none_such>) {
              return __none_such{};
            } else {
              auto&& [__sched, __clsur] = static_cast<_Data&&>(__data);
              return __write_env(                                                       //
                continues_on(                                                           //
                  __forward_like<_Data>(__clsur)(                                       //
                    continues_on(                                                       //
                      __write_env(static_cast<_Child&&>(__child), __with_sched{__old}), //
                      __sched)),                                                        //
                  __old),
                __with_sched{__sched});
            }
          }
        };
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) static auto transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn(static_cast<_Env&&>(__env)));
      }

      template <class _Sender, class _Env>
      STDEXEC_ATTRIBUTE((always_inline)) static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        auto __tfx_sndr_fn = __transform_sender_fn(__env);
        using _TfxSndrFn = decltype(__tfx_sndr_fn);
        using _NewSndr = __sexpr_apply_result_t<_Sender, _TfxSndrFn>;
        if constexpr (__same_as<_NewSndr, __none_such>) {
          return __no_scheduler_in_environment<_Sender, _Env>{};
        } else {
          return __sexpr_apply(
            static_cast<_Sender&&>(__sndr), static_cast<_TfxSndrFn&&>(__tfx_sndr_fn));
        }
      }
    };
  } // namespace __on_v2

  namespace v2 {
    using __on_v2::on_t;
    inline constexpr on_t on{};

    using continue_on_t = v2::on_t;
    inline constexpr continue_on_t continue_on{}; // for back-compat
  } // namespace v2

  template <>
  struct __sexpr_impl<v2::on_t> : __sexpr_defaults {
    static constexpr auto get_completion_signatures = //
      []<class _Sender>(_Sender&&) noexcept           //
      -> __merror_or_t<                               //
        __completion_signatures_of_t<                 //
          transform_sender_result_t<default_domain, _Sender, env<>>>,
        dependent_completions> {
      return {};
    };
  };
} // namespace stdexec
