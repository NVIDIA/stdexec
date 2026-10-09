/*
 * Copyright (c) 2021-2026 NVIDIA Corporation
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

#  include "../functional.hpp"
#  include "__meta.hpp"
#  include "__sender_concepts.hpp"
#  include "__structured_apply.hpp"
#  include "__structured_binding_size.hpp"
#  include "__tuple.hpp"  // IWYU pragma: keep for __is_tuple and __tuple_size_v
#  include "__type_traits.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <cstddef>
#    include <exception>  // IWYU pragma: keep for std::terminate
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Tag, class _Data, class... _Child>
  struct __sexpr;

  namespace __detail
  {
    template <class _Tag, class _Data, class... _Child>
    STDEXEC_ATTRIBUTE(host, device)
    auto __get_tag_of(__tuple<_Tag, _Data, _Child...> const &) -> _Tag;

#  if !STDEXEC_MSVC() && (!STDEXEC_CLANG() || STDEXEC_CLANG_VERSION >= 2100)
    template <class _Sender>
      requires(!__is_tuple<_Sender>) && (__structured_binding_size_v<_Sender> >= 2)
    STDEXEC_ATTRIBUTE(host, device)
    auto __get_tag_of(_Sender const &) -> __call_result_t<__structured_apply_t, __front, _Sender>;
#  else
    // MSVC and clang < 21 have a bug where the return type of the function is
    // computed even if the constraints are not satisfied. Use return type
    // deduction instead to defer the evaluation until the function is actually
    // instantiated.
    template <class _Sender>
      requires(!__is_tuple<_Sender>) && (__structured_binding_size_v<_Sender> >= 2)
    STDEXEC_ATTRIBUTE(host, device)
    auto __get_tag_of(_Sender const &)
    {
      return __call_result_t<__structured_apply_t, __front, _Sender>();
    }
#  endif

    template <class _Sender>
    using __tag_of_t = decltype(__detail::__get_tag_of(__declval<_Sender const &>()));

    template <class _CvSender>
      requires __mcallable1_q<__tag_of_t, _CvSender>
    inline constexpr auto __tag_of_v = __declfn<__tag_of_t<_CvSender>>();
  }  // namespace __detail

  // NOT TO SPEC: in the specification, the tparam of tag_of_t is constrained with the
  // sender concept
  STDEXEC_MODULE_EXPORT
  template <class _CvSender>
    requires enable_sender<__decay_t<_CvSender>>
  using tag_of_t = decltype(__detail::__tag_of_v<_CvSender>());

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _CvSender>
  using __data_of = __copy_cvref_t<_CvSender, typename __decay_t<_CvSender>::__data_t>;

  template <class _CvSender, class _Continuation = __qq<__mlist>>
  using __children_of = __mapply<__mtransform<__copy_cvref_fn<_CvSender>, _Continuation>,
                                 typename __decay_t<_CvSender>::__children_t>;

  template <class _Ny, class _CvSender>
  using __nth_child_of = __children_of<_CvSender, __mbind_front_q<__m_at, _Ny>>;

  template <std::size_t _Ny, class _CvSender>
  using __nth_child_of_c = __nth_child_of<__msize_t<_Ny>, _CvSender>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _CvSender>
  using __child_of = __children_of<_CvSender, __qq<__mfront>>;

  template <class _CvSender>
  inline constexpr std::size_t __nbr_children_of = __children_of<_CvSender, __msize>::value;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _CvSender, class... _Tag>
  concept __sender_for = sender<_CvSender> && __minvocable_q<tag_of_t, _CvSender>
                      && (__std::same_as<tag_of_t<_CvSender>, _Tag> && ...);

  template <class _CvSender>
  concept sender_expr
    STDEXEC_DEPRECATE_CONCEPT("Please use exec::sender_for from "
                              "<exec/sender_for.hpp> instead") = __sender_for<_CvSender>;

  template <class _CvSender, class _Tag>
  concept sender_expr_for
    STDEXEC_DEPRECATE_CONCEPT("Please use exec::sender_for from "
                              "<exec/sender_for.hpp> instead") = __sender_for<_CvSender, _Tag>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
