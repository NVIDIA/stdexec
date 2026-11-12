/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__concepts.hpp"
#  include "__meta.hpp"
#  include "__type_traits.hpp"
#  include "__utility.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  namespace __cmplsigs
  {
    // __partitions is a cache of completion signatures for fast access. The
    // completion_signatures<Sigs...>::__partitioned nested struct contains an
    // alias to a __partitions specialization. If the cache is never accessed,
    // it is never instantiated.
    template <class _ValueTuplesList = __mlist<>,
              class _ErrorsList      = __mlist<>,
              class _StoppedList     = __mlist<>>
    struct __partitions;

    template <class... _ValueTuples, class... _Errors, class... _Stopped>
    struct __partitions<__mlist<_ValueTuples...>, __mlist<_Errors...>, __mlist<_Stopped...>>
    {
      template <class _Tuple, class _Variant>
      using __value_types = _Variant::template __f<__mapply<_Tuple, _ValueTuples>...>;

      template <class _Variant, class _Transform = __q1<__midentity>>
      using __error_types = _Variant::template __f<typename _Transform::template __f<_Errors>...>;

      template <class _Variant, class _Type = set_stopped_t()>
      using __stopped_types = _Variant::template __f<__msecond<_Stopped, _Type>...>;

      using __count_values  = __msize_t<sizeof...(_ValueTuples)>;
      using __count_errors  = __msize_t<sizeof...(_Errors)>;
      using __count_stopped = __msize_t<sizeof...(_Stopped)>;

      struct __decay_copyable
      {
        // These types are placed in a separate struct to avoid computing them
        // if they are not needed.
        struct __values : __mand_t<__mapply_q<__decay_copyable_t, _ValueTuples>...>
        {};

        struct __errors : __decay_copyable_t<_Errors...>
        {};

        struct __all : __mand_t<__values, __errors>
        {};
      };

      struct __nothrow_decay_copyable
      {
        // These types are placed in a separate struct to avoid computing them
        // if they are not needed.
        struct __values : __mand_t<__mapply_q<__nothrow_decay_copyable_t, _ValueTuples>...>
        {};

        struct __errors : __nothrow_decay_copyable_t<_Errors...>
        {};

        struct __all : __mand_t<__values, __errors>
        {};
      };
    };

    template <class _Sig, class _PartitionsDeclFn>
    extern __undefined<_Sig, _PartitionsDeclFn> __partition_declfn;

    template <class... _Args, class... _Values, class _Errors, class _Stopped>
    extern __declfn_t<__partitions<__mlist<_Values..., __mlist<_Args...>>, _Errors, _Stopped>>
      __partition_declfn<set_value_t(_Args...),
                         __declfn_t<__partitions<__mlist<_Values...>, _Errors, _Stopped>>>;

    template <class _Error, class _Values, class... _Errors, class _Stopped>
    extern __declfn_t<__partitions<_Values, __mlist<_Errors..., _Error>, _Stopped>>
      __partition_declfn<set_error_t(_Error),
                         __declfn_t<__partitions<_Values, __mlist<_Errors...>, _Stopped>>>;

    template <class _Values, class _Errors, class _Stopped>
    extern __declfn_t<__partitions<_Values, _Errors, __mlist<__mlist<>>>>
      __partition_declfn<set_stopped_t(), __declfn_t<__partitions<_Values, _Errors, _Stopped>>>;

    template <class _PartitionsDeclFn, class _Sig>
    using __partitions_declfn_t = decltype(__partition_declfn<_Sig, _PartitionsDeclFn>);

    template <class... _Sigs>
    using __partition_completion_signatures_t = __call_result_t<
      __mcall<__mfold_left<__declfn_t<__partitions<>>, __q2<__partitions_declfn_t>>, _Sigs...>>;

    template <class _Completions>
    using __partitions_of_t = _Completions::__partitioned::__t;
  }  // namespace __cmplsigs

  //////////////////////////////////////////////////////////////////////////////
  // completion signatures type traits
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sigs,
            class _Tuple   = __qq<__decayed_std_tuple>,
            class _Variant = __qq<__std_variant>>
  using __value_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __value_types<_Tuple, _Variant>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sigs, class _Variant = __qq<__std_variant>, class _Transform = __q1<__midentity>>
  using __error_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __error_types<_Variant, _Transform>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sigs, class _Variant, class _Type = __fn_t<set_stopped_t>>
  using __stopped_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __stopped_types<_Variant, _Type>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <__valid_completion_signatures _Sigs>
  inline constexpr bool __sends_stopped =
    __cmplsigs::__partitions_of_t<_Sigs>::__count_stopped::value != 0;

  // set_value_t() when _Type is void; otherwise, set_value_t(_Type)
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Type>
  using __single_value_sig_t = __mcall1<__mremove<void, __qf<set_value_t>>, _Type>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
