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
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __compl_sigs {
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;
  } // namespace __compl_sigs

  template <__completion_signature... _Sigs>
  struct completion_signatures { };

  namespace __compl_sigs {
    template <class _TaggedTuple, class _Tag, class... _Ts>
    auto __as_tagged_tuple_(_Tag (*)(_Ts...), _TaggedTuple*)
      -> __mconst<__minvoke<_TaggedTuple, _Tag, _Ts...>>;

    template <class _Sig, class _TaggedTuple>
    using __as_tagged_tuple = decltype(__compl_sigs::__as_tagged_tuple_(
      static_cast<_Sig*>(nullptr),
      static_cast<_TaggedTuple*>(nullptr)));

    template <class _TaggedTuple, class _Variant, class... _Sigs>
    auto __for_all_sigs_(completion_signatures<_Sigs...>*, _TaggedTuple*, _Variant*)
      -> __mconst<__minvoke<_Variant, __minvoke<__as_tagged_tuple<_Sigs, _TaggedTuple>>...>>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __for_all_sigs =                      //
      __minvoke<                                //
        decltype(__compl_sigs::__for_all_sigs_( //
          static_cast<_Completions*>(nullptr),
          static_cast<_TaggedTuple*>(nullptr),
          static_cast<_Variant*>(nullptr)))>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __maybe_for_all_sigs = __meval<__for_all_sigs, _Completions, _TaggedTuple, _Variant>;

    template <class...>
    struct __normalize_completions {
      void operator()() const;
    };

    template <class _Ry, class... _As, class... _Rest>
    struct __normalize_completions<_Ry(_As...), _Rest...> : __normalize_completions<_Rest...> {
      auto operator()(_Ry (*)(_As&&...)) const -> _Ry (*)(_As...);
      using __normalize_completions<_Rest...>::operator();
    };

    // MSVCBUG: https://developercommunity.visualstudio.com/t/ICE-in-stdexec-metaprogramming/10642778
    // Previously a lambda was used here, along with __result_of in __normalize_completions.
    // As a workaround for this compiler bug, the lambda was replaced by a function
    // and the use of __result_of was expanded inline.
    template <class... _As, class _Ry, class... _Bs>
    auto __merge_sigs(_Ry (*)(_Bs...)) -> _Ry (*)(__if_c<__same_as<_Bs&&, _Bs>, _As, _Bs>...);

    template <class _Ry, class... _As, class... _Rest>
      requires __callable<__normalize_completions<_Rest...>, _Ry (*)(_As&&...)>
    struct __normalize_completions<_Ry(_As...), _Rest...> {
      auto operator()(_Ry (*)(_As&&...)) const -> decltype(__merge_sigs<_As...>(
        __declval<__call_result_t<__normalize_completions<_Rest...>, _Ry (*)(_As&&...)>>()));

      template <class _Sig>
      auto operator()(_Sig*) const -> __call_result_t<__normalize_completions<_Rest...>, _Sig*>;
    };

    template <class _Fn, class... _As>
    using __norm_sig_t = _Fn (*)(_As&&...);

    template <class T>
    extern __undefined<T> __norm;

    template <class _Ry, class... _As>
    extern __norm_sig_t<_Ry, _As...> __norm<_Ry(_As...)>;

    template <class _Sig>
    using __norm_t = decltype(+__norm<_Sig>);

    inline constexpr auto __convert_to_completion_signatures =
      []<class... Sigs>(__types<Sigs*...>*) -> completion_signatures<Sigs...> {
      return {};
    };

    template <class... _Sigs>
    using __unique_completion_signatures = __result_of<
      __convert_to_completion_signatures,
      __minvoke<
        __transform<
          __mbind_front_q<__call_result_t, __normalize_completions<_Sigs...>>,
          __munique<__q<__types>>>,
        __norm_t<_Sigs>...>*>;
  } // namespace __compl_sigs

  template <class _Completions>
  concept __valid_completion_signatures = //
    __same_as<__ok_t<_Completions>, __msuccess>
    && __is_instance_of<_Completions, completion_signatures>;

  template <class _Completions>
  using __invalid_completion_signatures_t = //
    __mbool<!__valid_completion_signatures<_Completions>>;

  template <__mstring _Msg = "Expected an instance of template completion_signatures<>"_mstr>
  struct _INVALID_COMPLETION_SIGNATURES_TYPE_ {
    template <class... _Completions>
    using __f = //
      __mexception<
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>,
        __minvoke<
          __mfind_if<
            __q<__invalid_completion_signatures_t>,
            __mcompose<__q<_WITH_TYPE_>, __q<__mfront>>>,
          _Completions...>>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_impl_t = //
    __minvoke<
      __if_c<
        (__valid_completion_signatures<_Completions> && ...),
        __mconcat<__q<__compl_sigs::__unique_completion_signatures>>,
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>>,
      _Completions...>;

  template <class... _Completions>
  struct __concat_completion_signatures_ {
    using __t = __meval<__concat_completion_signatures_impl_t, _Completions...>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_t = __t<__concat_completion_signatures_<_Completions...>>;

  template <class _Sender, class _Env>
  using __unrecognized_sender_error = //
    __mexception<_UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>>;
} // namespace stdexec
