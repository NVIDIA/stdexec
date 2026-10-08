/* Copyright (c) 2026 Ian Petersen
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

#include "../stdexec/__detail/__completion_signatures.hpp"
#include "../stdexec/__detail/__concepts.hpp"
#include "../stdexec/__detail/__domain.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__read_env.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__sender_concepts.hpp"
#include "../stdexec/__detail/__tuple.hpp"
#include "../stdexec/__detail/__utility.hpp"
#include "../stdexec/functional.hpp"

#include "__frame_allocator.hpp"
#include "__memory_resource_adaptor.hpp"
// TODO: split this header into pieces
#include "any_sender_of.hpp"
#include "get_frame_allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

// This file defines function<Signature, ...>, a type-erased sender intended
// for ABI-stable asynchronous API boundaries, including virtual member
// functions. Like a task coroutine, it represents an asynchronous function from
// arguments to results; unlike a coroutine, it allocates nothing when it is
// constructed.
//
// A function stores its arguments and a sender factory (a pointer to function,
// a pointer to member, or an empty, trivially copyable callable). connect
// invokes the factory with the stored arguments and connects the resulting
// sender to a type-erased receiver; that is why the template parameter is a
// function type rather than just a return type. The type-erased operation
// state is created in connect, so its storage can come from a frame allocator
// found in the receiver's environment (see get_frame_allocator) without relying
// on thread-local state.
//
// The completions, required environment queries and required completion
// domains are part of the type; see the documentation of the function alias
// template at the end of this file for the accepted forms.
namespace experimental::execution
{
  // for specifying required sender attributes in exec::function
  template <_query::_query_signature... Sigs>
  struct attrs
  {};

  namespace __func
  {
    using namespace STDEXEC;

    //! given the concrete receiver's environment, choose the frame allocator;
    //! first choice is the result of get_frame_allocator(env), second choice is
    //! get_allocator(env), and the default is std::allocator
    inline constexpr auto __choose_frame_allocator =
      __first_callable{get_frame_allocator, get_allocator, __always{std::allocator<std::byte>()}};

    //! Satisfied when choosing the frame allocator from an environment of type
    //! _Env and allocating from it can't throw. The check is made on the
    //! __frame_allocator_t adaptation of the chosen allocator, which is what
    //! actually allocates.
    template <class _Env>
    concept __has_nothrow_frame_allocator =
      __nothrow_callable<decltype(__choose_frame_allocator) const &, _Env const &>
      && requires(__frame_allocator_t<__result_of<__choose_frame_allocator, _Env const &>> &__alloc,
                  std::size_t                                                               __n) {
           { __alloc.allocate(__n) } noexcept;
         };

    //! Wrap _Receiver, which is a type-erased receiver, in a type that can
    //! extract the concrete, to-be-erased receiver from the operation state
    //! that contains it.
    //!
    //! This wrapper exists primarily as a hook for injecting a defaulted frame
    //! allocator when _Receiver *doesn't* have get_frame_allocator in its
    //! environment. That injection happens in the partial specialization,
    //! below.
    template <class _Receiver>
    struct __receiver_wrapper : public _Receiver
    {
      template <class _Opstate>
      constexpr explicit __receiver_wrapper(_Opstate *__opstate)
        : _Receiver(__opstate->__rcvr_)
      {}
    };

    //! Wrap _Receiver, which is a type-erased receiver, in a type that can
    //! extract the concrete, to-be-erased receiver from the operation state
    //! that contains it, and inject an environment that contains a defaulted
    //! frame allocator.
    //!
    //! This partial specialization handles the case that _Receiver doesn't have
    //! a frame allocator in its environment, in which case we need to provide a
    //! type-erasing frame allocator in the injected environment because we
    //! won't know the concrete type of the allocator that's actually used as
    //! our frame allocator until we're connected to a concrete receiver. We
    //! could provide either std::pmr::memory_resource*, or
    //! std::pmr::polymorphic_allocator<> with basically the same tradeoffs so
    //! we provide an allocator rather than a memory resource to better match
    //! the name of the injected query.
    template <class _Receiver>
      requires(!__queryable_with<env_of_t<_Receiver>, get_frame_allocator_t>)
    struct __receiver_wrapper<_Receiver> : public _Receiver
    {
      //! the injected query response for the frame allocator
      using __prop_t = prop<get_frame_allocator_t, std::pmr::polymorphic_allocator<std::byte>>;

      template <class _Opstate>
      constexpr explicit __receiver_wrapper(_Opstate *__opstate)
        : _Receiver(__opstate->__rcvr_)
        , __env_(&__opstate->__env_)
      {}

      constexpr auto get_env() const noexcept  //
        -> __join_env_t<__prop_t const &, env_of_t<_Receiver>>
      {
        return __env::__join(*__env_, STDEXEC::get_env(*static_cast<_Receiver const *>(this)));
      }

     private:
      __prop_t const *__env_;
    };

    template <class _Sigs, class _Queries>
    using __any_receiver_ref = ::exec::_any::_any_receiver_ref<_Sigs, _Queries>;

    template <class _Receiver, class _Sigs, class _Queries>
    struct __opstate_base
    {
      using __receiver_t   = __receiver_wrapper<__any_receiver_ref<_Sigs, _Queries>>;
      using __stop_token_t = stop_token_of_t<env_of_t<__receiver_t>>;

      //! The declared get_frame_allocator query's result is what allocates, so
      //! check it as the type-erased receiver's environment reports it.
      static constexpr bool __nothrow_frame_allocation =
        __has_nothrow_frame_allocator<env_of_t<__receiver_t>>;

      _any::_state<_Receiver, __stop_token_t> __rcvr_;
    };

    template <class _Receiver, class _Sigs, class _Queries>
      requires(
        !__queryable_with<env_of_t<__any_receiver_ref<_Sigs, _Queries>>, get_frame_allocator_t>)
    struct __opstate_base<_Receiver, _Sigs, _Queries>
    {
      using __receiver_t   = __receiver_wrapper<__any_receiver_ref<_Sigs, _Queries>>;
      using __prop_t       = __receiver_t::__prop_t;
      using __stop_token_t = stop_token_of_t<env_of_t<__receiver_t>>;
      using __adaptee_t    = __result_of<__choose_frame_allocator, env_of_t<_Receiver>>;

      //! The injected allocator forwards to the one chosen from the concrete
      //! receiver's environment, so check that one.
      static constexpr bool __nothrow_frame_allocation =
        __has_nothrow_frame_allocator<env_of_t<_Receiver>>;

      __memory_resource_adaptor_t<__adaptee_t> __resource_;
      __prop_t                                 __env_;
      _any::_state<_Receiver, __stop_token_t>  __rcvr_;

      explicit __opstate_base(_Receiver __rcvr)
        : __resource_(__choose_frame_allocator(STDEXEC::get_env(__rcvr)))
        , __env_(__make_env())
        , __rcvr_(static_cast<_Receiver &&>(__rcvr))
      {}

     private:
      //! the indirection through __make_env and __make_alloc is to work around
      //! what appears to be miscompilation with Clang 16; initializing __env_
      //! inline rather than delegating to these helpers results in passing an
      //! invalid address to the polymorphic_allocator constructor instead of
      //! the address of __resource_, leading to segfaults
      __prop_t __make_env()
      {
        return __prop_t(get_frame_allocator, __make_alloc());
      }

      std::pmr::polymorphic_allocator<> __make_alloc()
      {
        return std::pmr::polymorphic_allocator<>(&__resource_);
      }
    };

    //! The concrete operation state resulting from connecting a function<...>
    //! to a concrete receiver of type Receiver. This type manages an
    //! _any::_any_opstate_base instance, which is the type-erased operation
    //! state resulting from connecting the type-erased sender to an
    //! _any::_any_receiver_ref with the given completion signatures and
    //! queries.
    template <class _Receiver, class _Sigs, class _Queries>
    class __opstate : public __opstate_base<_Receiver, _Sigs, _Queries>
    {
      using __base = __opstate_base<_Receiver, _Sigs, _Queries>;
      using typename __base::__receiver_t;

      _any::_any_opstate_base __op_;

     public:
      using operation_state_concept = operation_state_tag;

      // The type-erased receiver holds a pointer to this object, and
      // __opstate_base may hold a polymorphic_allocator pointing at its own
      // __resource_, so moving an __opstate would leave dangling pointers. Say
      // so explicitly rather than relying on _any_opstate_base being immovable.
      STDEXEC_IMMOVABLE(__opstate);

      template <class _Factory>
      explicit constexpr __opstate(_Receiver __rcvr, _Factory __factory)
        : __base(static_cast<_Receiver &&>(__rcvr))
        , __op_(__factory(__receiver_t(this)))
      {}

      constexpr void start() & noexcept
      {
        __op_.start();
      }
    };

    template <class _Tag, class _Query>
    struct __make_domain_impl
    {};

    template <class _Domain, class _Tag>
    struct __make_domain_impl<_Tag, _Domain(get_completion_domain_t<_Tag>) noexcept>
    {
      constexpr _Domain operator()() const noexcept
      {
        return _Domain();
      }
    };

    //! get_completion_domain<> is a special case; its type parameter is void
    //! and it's equivalent to get_completion_domain<set_value_t>.
    template <class _Domain>
    struct __make_domain_impl<void, _Domain(get_completion_domain_t<set_value_t>) noexcept>
      : __make_domain_impl<set_value_t, _Domain(get_completion_domain_t<set_value_t>) noexcept>
    {};

    //! get_completion_domain ought to be no-throw, so make it optional to
    //! specify noexcept on the signature provided with attrs<...>
    template <class _Domain, class _Tag1, class _Tag2>
    struct __make_domain_impl<_Tag1, _Domain(get_completion_domain_t<_Tag2>)>
      : __make_domain_impl<_Tag1, _Domain(get_completion_domain_t<_Tag2>) noexcept>
    {};

    template <class _Tag, class... _Attrs>
    inline constexpr auto __make_domain = __first_callable<__make_domain_impl<_Tag, _Attrs>...>();

    template <class _Attrs>
    struct __attrs;

    template <class... _Attrs>
    struct __attrs<attrs<_Attrs...>>
    {
      template <class _Tag, class... _Env>
      constexpr auto query(get_completion_domain_t<_Tag>, _Env &&...) const noexcept
        -> decltype(__make_domain<_Tag, _Attrs...>())
      {
        return __make_domain<_Tag, _Attrs...>();
      }
    };

    template <class _Tag, class _Attrs, class... _Env>
    using __completion_domain_t = __call_result_or_t<
      get_completion_domain_t<_Tag>,
      __call_result_or_t<get_completion_domain_t<_Tag>, indeterminate_domain<>, _Attrs>,
      _Attrs,
      _Env const &...>;

    template <class _ActualDomain, class _ExpectedDomain>
    concept __completion_domain_matches_impl =
      __same_as<_ExpectedDomain, __common_domain_t<_ActualDomain, _ExpectedDomain>>;

    template <class _Tag, class _ActualAttrs, class _ExpectedAttrs, class... _Env>
    concept __completion_domain_matches =
      __completion_domain_matches_impl<__completion_domain_t<_Tag, _ActualAttrs, _Env...>,
                                       __completion_domain_t<_Tag, _ExpectedAttrs, _Env...>>;

    template <class _ActualAttrs, class _ExpectedAttrs, class... _Env>
    concept __completion_domains_match_impl =
      __completion_domain_matches<set_value_t, _ActualAttrs, _ExpectedAttrs, _Env...>
      && __completion_domain_matches<set_error_t, _ActualAttrs, _ExpectedAttrs, _Env...>
      && __completion_domain_matches<set_stopped_t, _ActualAttrs, _ExpectedAttrs, _Env...>;

    template <class _Actual, class _Expected, class... _Env>
    concept __completion_domains_match =
      __completion_domains_match_impl<env_of_t<_Actual>, env_of_t<_Expected>, _Env...>;

    template <class _Queries, class... _Env>
    struct __check_queries;

    template <class... _Queries, class... _Env>
    struct __check_queries<queries<_Queries...>, _Env...>
    {
      using type = __mfind_error<_any::_check_query_t<_Queries, _Env...>...>;
    };

    template <class _Queries, class... _Env>
    using __check_queries_t = __check_queries<_Queries, _Env...>::type;

    //! get_completion_domain_t<> (i.e. get_completion_domain_t<void>) asks for
    //! the set_value domain, so attrs<D(get_completion_domain_t<>)> means the
    //! same as attrs<D(get_completion_domain_t<set_value_t>)>. Normalize to the
    //! latter before canonicalizing so the rest of the implementation only
    //! sees explicit completion tags.
    template <class _Attr>
    struct __normalize_attr
    {
      using type = _Attr;
    };

    template <class _Domain>
    struct __normalize_attr<_Domain(get_completion_domain_t<>)>
    {
      using type = _Domain(get_completion_domain_t<set_value_t>);
    };

    template <class _Domain>
    struct __normalize_attr<_Domain(get_completion_domain_t<>) noexcept>
    {
      using type = _Domain(get_completion_domain_t<set_value_t>) noexcept;
    };

    template <class _Attrs>
    struct __normalize_attrs;

    template <class... _Attrs>
    struct __normalize_attrs<attrs<_Attrs...>>
    {
      using type = attrs<typename __normalize_attr<_Attrs>::type...>;
    };

    template <class _Attrs>
    using __normalized_attrs_t = __normalize_attrs<_Attrs>::type;

    template <class _Attr>
    struct __get_completion_domain_tag;

    template <class _Tag, class _Domain>
    struct __get_completion_domain_tag<_Domain(get_completion_domain_t<_Tag>)>
    {
      using type = _Tag;
    };

    template <class _Domain>
    struct __get_completion_domain_tag<_Domain(get_completion_domain_t<>)>
    {
      using type = set_value_t;
    };

    template <class _Tag, class _Domain>
    struct __get_completion_domain_tag<_Domain(get_completion_domain_t<_Tag>) noexcept>
      : __get_completion_domain_tag<_Domain(get_completion_domain_t<_Tag>)>
    {};

    template <class _Attr>
    using __get_completion_domain_tag_t = __get_completion_domain_tag<_Attr>::type;

    template <class _Attrs, class _Tag>
    inline constexpr bool __has_completion_domain = false;

    template <class... _Attrs, class _Tag>
      requires __one_of<_Tag, __get_completion_domain_tag_t<_Attrs>...>
    inline constexpr bool __has_completion_domain<attrs<_Attrs...>, _Tag> = true;

    //! it is undefined behaviour for a sender to advertise a completion domain
    //! for a completion channel that it never completes on so make sure there
    //! are no completion domains required by _Attrs that correspond to
    //! completion channels not advertised as possible by _Sigs
    template <class _Sigs, class _Attrs>
    concept __completion_signatures_and_domains_are_compatible =
      ((!__has_completion_domain<_Attrs, set_value_t>) || _Sigs::__count(set_value) > 0)     //
      && ((!__has_completion_domain<_Attrs, set_error_t>) || _Sigs::__count(set_error) > 0)  //
      && ((!__has_completion_domain<_Attrs, set_stopped_t>) || _Sigs::__count(set_stopped) > 0);

    template <class _Self>
    struct __self_box
    {
      using __void_pointer =
        __if_c<STDEXEC_IS_CONST(STDEXEC_REMOVE_REFERENCE(_Self)), void const *, void *>;

      using __self_tag = _Self;
    };

    struct __self_tag
    {};

    //! Satisfied when an object of type _Ty, passed as `self`, can bind to the
    //! implicit object parameter whose qualifiers _SelfBox records for
    //! __self_tag (which come from the function type's qualifiers, e.g.
    //! `int() const &`), as it could for a synchronous member function with
    //! those qualifiers, except that rvalues are never accepted: the function
    //! stores a pointer to the object, so accepting an rvalue would leave the
    //! function borrowing an object that is about to expire. Concretely:
    //!  - `&` accepts a non-const lvalue, and
    //!  - `const &` accepts a const or non-const lvalue (a qualification
    //!    conversion; the factory still receives a const reference).
    //! volatile objects are rejected, since no supported function type can
    //! bind to them.
    template <class _Ty, class _SelfBox>
    concept __binds_to_self =
      std::is_lvalue_reference_v<_Ty> && (!std::is_volatile_v<std::remove_reference_t<_Ty>>)
      && (__same_as<__copy_cvref_t<_Ty, __self_tag>, typename _SelfBox::__self_tag>
          || __same_as<typename _SelfBox::__self_tag, __self_tag const &>);

    //! The type with which the implicit object parameter is passed to the
    //! factory: _Self's unqualified type with the qualifiers the function type
    //! declares, so a non-const lvalue passed to a `const &` function reaches
    //! the factory as a const reference.
    template <class _Self, class _SelfBox>
    using __declared_self_t =
      __copy_cvref_t<typename _SelfBox::__self_tag, std::remove_cvref_t<_Self>>;

    //! The sender factory passed to function must be one of:
    //!  1. a pointer-to-function,
    //!  2. a pointer-to-member (function or object), or
    //!  3. an empty, trivially-copyable, callable object (like a captureless lambda or a
    //!     CPO like STDEXEC::just).
    //!
    //! Concept __is_callable_pointer matches options 1 and 2, and concept
    //! __is_empty_callable matches option 3.

    //! Never defined. A pointer to a member of an incomplete class gets the
    //! ABI's most general member-pointer representation, so these types bound
    //! the size and alignment of any pointer factory a user can provide.
    struct __incomplete;
    using __general_pmf_t = void (__incomplete::*)();
    using __general_pmd_t = int __incomplete::*;
    using __fn_ptr_t      = void (*)();

    //! The size required to store any factory __function accepts: an empty
    //! callable, a pointer to function, or a pointer to member. Under the
    //! Itanium C++ ABI a pointer to member function is two pointers, but under
    //! the Microsoft ABI its size depends on the class's inheritance model, and
    //! the general representation is larger.
    inline constexpr std::size_t __factory_storage_size = std::max(
      {sizeof(__fn_ptr_t), sizeof(__general_pmf_t), sizeof(__general_pmd_t)});

    //! The alignment required to store any factory __function accepts: an
    //! empty callable, a pointer to function, or a pointer to member.
    inline constexpr std::size_t __factory_storage_align = std::max(
      {alignof(__fn_ptr_t), alignof(__general_pmf_t), alignof(__general_pmd_t)});

    //! Satisfied when _Ty is a pointer-to-function or pointer-to-member
    template <class _Ty>
    concept __is_callable_pointer = (std::is_pointer_v<_Ty>
                                     && std::is_function_v<std::remove_pointer_t<_Ty>>)
                                 || std::is_member_pointer_v<_Ty>;

    //! Satisfied when _Ty is an empty, trivially-copyable object; callability is not
    //! actually checked here because __is_suitable_factory checks __invocable before
    //! checking is_empty_callable
    template <class _Ty>
    concept __is_empty_callable = std::is_empty_v<_Ty> && (STDEXEC_IS_TRIVIALLY_COPYABLE(_Ty));

    //! Defines the constraints on a function's sender factory argument; satisfied when
    //! _Factory:
    //!  - is invocable as a const lvalue with the given argument types (function
    //!    stores the factory and always invokes it that way; a stateless factory
    //!    has nothing to mutate or consume),
    //!  - returns a sender that is connectable to the given receiver type, and
    //!  - returns a sender whose completion domains match the declared completion
    //!    domains of the function that will use it.
    //!
    //! \tparam _Factory the ostensible factory to check
    //! \tparam _Func the function specialization that will use _Factory
    //! \tparam _Receiver the type of the receiver to which the sender returned from the
    //!                   factory will be connected
    //! \tparam _Args... the factory arguments
    template <class _Factory, class _Func, class _Receiver, class... _Args>
    concept __is_factory_of_suitable_sender =
      __invocable<_Factory const &, _Args...>
      && sender_to<__invoke_result_t<_Factory const &, _Args...>, _Receiver>
      && __completion_domains_match<__invoke_result_t<_Factory const &, _Args...>,
                                    _Func,
                                    env_of_t<_Receiver>>;

    //! Defines the constraints on a function's sender factory; satisfied when _Factory:
    //!  - produces a suitable sender as defined by __is_factory_of_suitable_sender, and
    //!  - is either of the above-defined kinds of callable.
    template <class _Factory, class _Func, class _Receiver, class... _Args>
    concept __is_suitable_factory =
      __is_factory_of_suitable_sender<_Factory, _Func, _Receiver, _Args...>
      && (__is_callable_pointer<_Factory> || __is_empty_callable<_Factory>);

    //! A frame allocator whose allocate is declared noexcept. It exists only to
    //! be named in unevaluated operands, so its members are never defined.
    template <class _Ty>
    struct __nothrow_frame_allocator_stub
    {
      using value_type = _Ty;

      __nothrow_frame_allocator_stub() = default;

      template <class _Uy>
      constexpr __nothrow_frame_allocator_stub(__nothrow_frame_allocator_stub<_Uy> const &) noexcept
      {}

      auto allocate(std::size_t) noexcept -> _Ty *;
      void deallocate(_Ty *, std::size_t) noexcept;

      friend constexpr auto operator==(__nothrow_frame_allocator_stub,
                                       __nothrow_frame_allocator_stub) noexcept -> bool = default;
    };

    //! A receiver used only to check, at construction, whether connecting the
    //! factory's sender can throw. It is the type-erased receiver _Receiver
    //! except that its environment's frame allocator is declared not to throw.
    //! Frame-allocation failure is the caller's concern (it appears in the
    //! conditional noexcept of function's connect), so a noexcept function type
    //! promises only that the type-erased path doesn't throw *other than* by
    //! failing to allocate a frame. In particular, a noexcept function whose
    //! factory returns another noexcept function is accepted, even though the
    //! inner function's connect allocates its frame through the outer one's
    //! type-erased frame allocator.
    //!
    //! This is sound as long as a sender's connect depends on the frame
    //! allocator's exception behaviour only through calls to allocate: the check
    //! sees this stub's type, while at run time the sender sees the type-erased
    //! receiver's frame allocator.
    template <class _Receiver>
    struct __nothrow_frame_allocation_receiver : _Receiver
    {
      using __prop_t = prop<get_frame_allocator_t, __nothrow_frame_allocator_stub<std::byte>>;

      auto get_env() const noexcept -> __join_env_t<__prop_t, env_of_t<_Receiver>>;
    };

    //! Satisfied when _Factory can be invoked as a const lvalue with arguments
    //! of types _Args without throwing.
    template <class _Factory, class... _Args>
    concept __factory_is_nothrow_invocable = __nothrow_invocable<_Factory const &, _Args...>;

    //! Satisfied when transform_sender, given a _Sender and an _Env, returns a
    //! sender of a different type, i.e. a domain has transformed it.
    template <class _Sender, class _Env>
    concept __is_transformed_for =
      !__same_as<__decay_t<transform_sender_result_t<_Sender, _Env>>, __decay_t<_Sender>>;

    //! Satisfied when connecting the sender _Factory returns to the type-erased
    //! receiver _Receiver can't throw, short of failing to allocate a frame (see
    //! __nothrow_frame_allocation_receiver). If a domain transforms that sender,
    //! the transformation and the transformed sender's connect are trusted
    //! rather than checked. The domains that can transform it depend only on
    //! the sender and on the queries the function type declares: _Receiver's
    //! environment provides nothing else but a frame allocator, so the
    //! caller's domain can't reach the sender. Those domains are chosen only
    //! implicitly (declaring a get_scheduler_t query, for example, brings that
    //! scheduler's domain into play) and nothing checks the choice, so
    //! declaring the function type noexcept vouches for whatever they do.
    template <class _Factory, class _Receiver, class... _Args>
    concept __factory_sender_connects_without_throwing =
      __is_transformed_for<__invoke_result_t<_Factory const &, _Args...>,
                           env_of_t<__nothrow_frame_allocation_receiver<_Receiver>>>
      || __nothrow_connectable<__invoke_result_t<_Factory const &, _Args...>,
                               __nothrow_frame_allocation_receiver<_Receiver>>;

    //! Defines the additional constraints on the sender factory of a noexcept
    //! function type, whose noexcept promises that the type-erased path
    //! (invoking the factory and connecting the sender it returns) doesn't
    //! throw. Satisfied when the function type isn't noexcept, or when _Factory:
    //!  - is nothrow-invocable with the curried arguments as connect() && passes
    //!    them (rvalues, unless the parameter type is a reference), and
    //!  - returns a sender that connects to _Receiver without throwing as
    //!    defined by __factory_sender_connects_without_throwing.
    //! These are the only checks: each is exact, and when one fails, the
    //! function type's author can fix it. Anything a domain transformation
    //! does is trusted instead.
    template <bool _Nothrow, class _Factory, class _Receiver, class... _Args>
    concept __meets_noexcept_contract =
      (!_Nothrow)
      || (__factory_is_nothrow_invocable<_Factory, _Args...>
          && __factory_sender_connects_without_throwing<_Factory, _Receiver, _Args...>);

    //! A class template that adapts a user-provided callable expecting a "self" reference
    //! in the first argument to a factory that accepts a pointer to (const) void, which
    //! is what's actually invoked by the underlying __function
    template <class _Factory, class _Self>
    struct __self_adapting_factory
    {
      using __value   = STDEXEC_REMOVE_REFERENCE(_Self);
      using __pointer = __value *;

      _Factory __factory;

      template <class _Void, class... _Args>
        requires std::is_void_v<_Void>
      constexpr decltype(auto) operator()(_Void *__self, _Args &&...__args) const
        noexcept(__nothrow_invocable<_Factory const &, _Self, _Args...>)
      {
        static_assert(STDEXEC_IS_CONST(__value) == STDEXEC_IS_CONST(_Void));

        return __invoke(__factory,
                        static_cast<_Self>(*static_cast<__pointer>(__self)),
                        static_cast<_Args &&>(__args)...);
      }
    };

    //! The type with which function's constructor takes a curried argument
    //! declared as _Ty: `_Ty const &` when _Ty is a non-reference type that is
    //! both trivially copy-constructible and trivially move-constructible,
    //! otherwise `_Ty &&`. An lvalue is therefore accepted (and copied)
    //! exactly when writing std::move would change nothing; otherwise the
    //! caller must move or copy explicitly (e.g. with auto(x)), so a costly
    //! copy is never made silently. Both traits are required: with a
    //! `const &` parameter even std::move(x) copies, so a type whose move is
    //! non-trivial must not take this path. Reference types are unchanged.
    template <class _Ty>
    using __curried_param_t =
      __if_c<!std::is_reference_v<_Ty> && std::is_trivially_copy_constructible_v<_Ty>
               && std::is_trivially_move_constructible_v<_Ty>,
             _Ty const &,
             _Ty &&>;

    //! the main implementation of the type-erasing sender function<...>
    //
    //! @tparam _Nothrow Whether the function type is declared noexcept, which
    //! promises that the type-erased path (invoking the factory and connecting
    //! the sender it returns) doesn't throw; the completion signatures describe
    //! only the asynchronous contract
    //!
    //! @tparam _Sigs The supported completion signatures
    //!
    //! @tparam _Queries The list of environment queries that must be supported
    //! by the eventual receiver; it's a pack of function type like
    //! Return(Query, Args...) or Return(Query, Args...) noexcept. The named
    //! query, when given the specified arguments, must return a value
    //! convertible to Return, and it must be noexcept, or not, as appropriate
    //!
    //! @tparam _Attrs The list of completion domains the erased sender must
    //! report; a pack of function types like Domain(get_completion_domain_t<Tag>)
    //! noexcept, also reported by the function's own environment
    //!
    //! @tparam _Args The argument types used to construct the erased sender
    template <bool _Nothrow, class _Sigs, class _Queries, class _Attrs, class... _Args>
    class __function
    {
      // check these with asserts rather than requires because the only way to
      // violate them is to circumvent the exec::function alias template so any
      // violation is a user hitting themselves
      static_assert(__is_instance_of<_Sigs, completion_signatures>);
      static_assert(__is_instance_of<_Queries, queries>);
      static_assert(__is_instance_of<_Attrs, attrs>);
      static_assert(__completion_signatures_and_domains_are_compatible<_Sigs, _Attrs>);

     protected:
      using __receiver_t = __receiver_wrapper<__any_receiver_ref<_Sigs, _Queries>>;

      template <class _Receiver>
      using __opstate_t = __opstate<_Receiver, _Sigs, _Queries>;

      template <class _Factory>
      static constexpr auto
      __mk_opstate(void *__storage, __receiver_t __rcvr, _Args &&...__args)  //
        -> _any::_any_opstate_base
      {
        auto const &__make_sender = *__std::start_lifetime_as<_Factory>(__storage);
        using __alloc_t           = decltype(__choose_frame_allocator(STDEXEC::get_env(__rcvr)));
        auto __alloc              = __frame_allocator_t<__alloc_t>(
          __choose_frame_allocator(STDEXEC::get_env(__rcvr)));
        // Ideally, function would allocate raw storage for the operation state
        // from __alloc and construct it with construct_at. Instead, an
        // operation state too big for the inline buffer is wrapped by the
        // __any machinery that function shares with any_sender_of, and the
        // wrapper is constructed and destroyed with
        // allocator_traits<...>::construct and destroy.
        // Uses-allocator construction never reaches the operation state: the
        // wrapper doesn't declare allocator_type, and the operation state is
        // initialized inside it directly from connect's result. What does
        // leak is that a frame allocator's own construct and destroy members,
        // if it has them, are called with the wrapper's type. That happens
        // only when the function declares the get_frame_allocator query;
        // otherwise __alloc is the injected polymorphic_allocator, which
        // forwards only allocate and deallocate to the caller's allocator.
        // Tolerated because frame allocators are unlikely to customize
        // construct or destroy. A fix would give function its own path through
        // __any (raw storage plus construct_at and destroy_at), leaving
        // any_sender_of unchanged.
        return _any::_any_opstate_base(__in_place_from,
                                       std::allocator_arg,
                                       __alloc,
                                       STDEXEC::connect,
                                       __invoke(__make_sender, static_cast<_Args &&>(__args)...),
                                       static_cast<__receiver_t &&>(__rcvr));
      }

      // The members are declared in order of decreasing alignment, with the
      // curried arguments, whose alignment varies, last. That puts any padding
      // at the end of the object, where it can be reused by a following
      // member of an enclosing class when a function is a [[no_unique_address]]
      // member or a base class.

      //! Storage for the sender factory passed to our constructor template;
      //! __make_opstate_ will reconstitute the actual factory from this
      //! bag-of-bytes with start_lifetime_as because it internally knows the
      //! concrete type of the user-provided sender factory. The storage is
      //! sized and aligned for the largest pointer to member the ABI can
      //! produce (see __factory_storage_size).
      alignas(__factory_storage_align) std::byte __make_sender_[__factory_storage_size]{};
      //! The type-erased operation state factory; it points to a function that
      //! knows the concrete type of the sender factory stored in __make_sender_
      //! so that it can construct the desired sender on demand and connect it
      //! to the given receiver. The expected arguments are the address of
      //! __make_sender_, the __any_receiver_ref to connect the sender to, and
      //! the arguments to pass to __make_sender_ to construct the sender.
      _any::_any_opstate_base (*__make_opstate_)(void *, __receiver_t, _Args &&...);
      //! The curried arguments that will be passed to __make_sender_ from
      //! inside __make_opstate_.
      STDEXEC_ATTRIBUTE(no_unique_address)
      __tuple<_Args...> __args_;

      struct __tag
      {};

      template <class _Factory>
      constexpr explicit __function(__curried_param_t<_Args>... __args, _Factory __factory, __tag)
        noexcept((__nothrow_constructible_from<_Args, __curried_param_t<_Args>> && ...))
        : __make_opstate_(&__mk_opstate<_Factory>)
        , __args_(static_cast<__curried_param_t<_Args>>(__args)...)
      {
        static_assert(sizeof(_Factory) <= sizeof(__make_sender_));
        static_assert(alignof(_Factory) <= __factory_storage_align);

        std::memcpy(__make_sender_, std::addressof(__factory), sizeof(_Factory));
      }

     public:
      using sender_concept = sender_tag;

      // check __not_decays_to first: the conjunction short-circuits, so
      // overload resolution for an ordinary copy or move never instantiates
      // the much more expensive __is_suitable_factory check
      template <class _Factory>
        requires __not_decays_to<_Factory, __function>
              && __is_suitable_factory<_Factory, __function, __receiver_t, _Args...>
              && __meets_noexcept_contract<_Nothrow, _Factory, __receiver_t, _Args...>
      constexpr explicit __function(__curried_param_t<_Args>... __args, _Factory __factory)
        noexcept((__nothrow_constructible_from<_Args, __curried_param_t<_Args>> && ...))
        : __function(static_cast<__curried_param_t<_Args>>(__args)..., __factory, __tag{})
      {}

      //! this implementation of get_completion_signatures is taken directly
      //! from the equivalent function on any_sender_of
      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures()
      {
        static_assert(__decays_to_derived_from<_Self, __function>);
        //! throw if _Env does not contain the queries needed to type-erase the
        //! receiver:
        if constexpr (__merror<__check_queries_t<_Queries, _Env...>>)
          return __throw_compile_time_error(__check_queries_t<_Queries, _Env...>());
        else
          return _Sigs();
      }

      constexpr __attrs<_Attrs> get_env() const noexcept
      {
        return {};
      }

      //! connect is noexcept exactly when nothing on the path to the operation
      //! state can throw: the function type is noexcept and the frame allocator
      //! chosen from the receiver's environment allocates without throwing. If
      //! a trusted domain transformation or a transformed sender's connect
      //! throws anyway, std::terminate is called.
      template <class _Receiver>
      static constexpr bool __nothrow_connect = _Nothrow
                                             && __opstate_t<_Receiver>::__nothrow_frame_allocation;

      template <receiver _Receiver>
      constexpr auto connect(_Receiver __rcvr) && noexcept(__nothrow_connect<_Receiver>)  //
        -> __opstate_t<_Receiver>
      {
        auto __factory = [this]<class _RcvrRef>(_RcvrRef __rcvr)
        {
          return __apply(__make_opstate_,
                         static_cast<__tuple<_Args...> &&>(__args_),
                         __make_sender_,
                         static_cast<_RcvrRef &&>(__rcvr));
        };
        return __opstate_t<_Receiver>{static_cast<_Receiver &&>(__rcvr), __factory};
      }

      //! as for connect() &&, plus copying the curried arguments mustn't throw
      template <receiver _Receiver>
        requires __std::copy_constructible<__function>
      constexpr auto connect(_Receiver __rcvr) const &  //
        noexcept(__nothrow_connect<_Receiver> && __nothrow_copy_constructible<_Args...>)
          -> __opstate_t<_Receiver>
      {
        return __function(*this).connect(static_cast<_Receiver &&>(__rcvr));
      }
    };

    //! This specialization of __function handles "member functions", which are created
    //! by specializing __make_function (below) with a cvref-qualified function type to
    //! indicate that the argument list includes an implicit self object (with the given
    //! cvref qualifiers). _SelfBox is effectively a tag type whose type parameter is a
    //! tag type conveying the cvref qualifiers of the implicit object parameter.
    //!
    //! Member functions are implemented in terms of a (possibly const) void pointer and
    //! a sender factory adaptor that captures the real type of the implicit object and
    //! casts the stored void pointer back to the correct type upon invocation.
    //!
    //! \tparam _Nothrow whether the function type is declared noexcept
    //! \tparam _Sigs the function's possible completion signatures
    //! \tparam _Queries the queries required to be supported by the environment of the
    //!                  receiver to which this function is ultimately connected
    //! \tparam _Attrs the queries supported by this function's attributes
    //! \tparam _SelfBox the tag type conveying the cvref qualifiers of the implicit
    //!                  object parameter
    //! \tparam _Args the pack of explicit arguments to the sender factory
    template <bool _Nothrow,
              class _Sigs,
              class _Queries,
              class _Attrs,
              __is_instance_of<__self_box> _SelfBox,
              class... _Args>
    class __function<_Nothrow, _Sigs, _Queries, _Attrs, _SelfBox, _Args...>
      : public __function<_Nothrow,
                          _Sigs,
                          _Queries,
                          _Attrs,
                          typename _SelfBox::__void_pointer,
                          _Args...>
    {
      using __void_pointer = _SelfBox::__void_pointer;
      using __base = __function<_Nothrow, _Sigs, _Queries, _Attrs, __void_pointer, _Args...>;

      using __receiver_t = __base::__receiver_t;
      using __tag        = __base::__tag;

     public:
      // check the cheap __binds_to_self first; the conjunction short-circuits
      template <class _Self, class _Factory>
        requires __binds_to_self<_Self &&, _SelfBox>
              && __is_suitable_factory<_Factory,
                                       __function,
                                       __receiver_t,
                                       __declared_self_t<_Self, _SelfBox>,
                                       _Args...>
              && __meets_noexcept_contract<_Nothrow,
                                           _Factory,
                                           __receiver_t,
                                           __declared_self_t<_Self, _SelfBox>,
                                           _Args...>
      constexpr explicit __function(_Self &&__self,
                                    __curried_param_t<_Args>... __args,
                                    _Factory __fact)
        // like the base's constructor: storing the self pointer and the
        // (pointer or empty) factory can't throw, so only initializing the
        // curried arguments matters
        noexcept((__nothrow_constructible_from<_Args, __curried_param_t<_Args>> && ...))
        : __base(static_cast<__void_pointer>(std::addressof(__self)),
                 static_cast<__curried_param_t<_Args>>(__args)...,
                 __self_adapting_factory<_Factory, __declared_self_t<_Self, _SelfBox>>{__fact},
                 __tag{})
      {}
    };

    template <class _Sigs>
    struct __canonical;

    template <template <class...> class _List, class... _Types>
    struct __canonical<_List<_Types...>>
    {
      using type = __minvoke<__munique<__msort<__q<_List>>>, _Types...>;
    };

    //! Map the type-list _Sigs to a canonical form, which sorts and uniques the
    //! contained elements to ensure user-specified type-lists are not
    //! order-dependent.
    //!
    //! @tparam _Sigs a type-list of types to be sorted and uniqued; expected to
    //! be a specialization of completion_signatures, queries, or attrs.
    template <class _Sigs>
    using __canonical_t = __canonical<_Sigs>::type;

    template <class _Signature>
    struct __function_meta;

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...)>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = false;

      template <class... _LeadingArgs>
      using __make_function = __function<__noexcept, _LeadingArgs..., _Args...>;
    };

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...) &>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = false;

      template <class... _LeadingArgs>
      using __make_function =
        __function<__noexcept, _LeadingArgs..., __self_box<__self_tag &>, _Args...>;
    };

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...) const &>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = false;

      template <class... _LeadingArgs>
      using __make_function =
        __function<__noexcept, _LeadingArgs..., __self_box<__self_tag const &>, _Args...>;
    };

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...) noexcept>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = true;

      template <class... _LeadingArgs>
      using __make_function = __function<__noexcept, _LeadingArgs..., _Args...>;
    };

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...) & noexcept>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = true;

      template <class... _LeadingArgs>
      using __make_function =
        __function<__noexcept, _LeadingArgs..., __self_box<__self_tag &>, _Args...>;
    };

    template <class _Return, class... _Args>
    struct __function_meta<_Return(_Args...) const & noexcept>
    {
      using __return_type = _Return;

      static constexpr bool __noexcept = true;

      template <class... _LeadingArgs>
      using __make_function =
        __function<__noexcept, _LeadingArgs..., __self_box<__self_tag const &>, _Args...>;
    };

    template <class _Ty>
    using __return_type_t = __function_meta<_Ty>::__return_type;

    template <class _Ty>
    inline constexpr bool __is_noexcept = __function_meta<_Ty>::__noexcept;

    template <class _Ty>
    concept __is_function_type = std::is_function_v<_Ty>;

    //! Satisfied when _Ty is one of the function types function accepts:
    //!
    //!   R(A...), R(A...) &, R(A...) const &, each optionally noexcept
    //!
    //! Rejected forms (__function_meta has no specialization for them):
    //!  - `&&` and `const &&`: the function would borrow an object that is
    //!    about to expire;
    //!  - `const` without a ref-qualifier: as for synchronous member
    //!    functions, it would bind rvalues too, reopening the same problem;
    //!  - anything `volatile`.
    template <class _Ty>
    concept __is_supported_function_type = __is_function_type<_Ty> && requires {
      typename __function_meta<_Ty>::__return_type;
    };

    template <class _Ty>
    concept __is_sender_tag_function = __is_function_type<_Ty>
                                    && __same_as<sender_tag, __return_type_t<_Ty>>;

    template <class _Ty>
    concept __is_not_sender_tag_function = __is_function_type<_Ty>
                                        && __not_same_as<sender_tag, __return_type_t<_Ty>>;

    //! Given a return type and a bool indicating whether the function is
    //! noexcept, compute the appropriate completion_signatures. The result is a
    //! set_value overload taking either Return&& or no args when Return is
    //! void, set_stopped, and, when the function type is not noexcept,
    //! set_error(std::exception_ptr)
    template <class _Ty>
    using __completion_sigs_from = __canonical_t<__concat_completion_signatures_t<
      completion_signatures<__single_value_sig_t<__return_type_t<_Ty>>, set_stopped_t()>,
      __eptr_completion_unless_t<__mbool<__is_noexcept<_Ty>>>>>;

    //! maps a completion signature to the default completion domain query
    struct __domain_query_from_sig
    {
      template <class _Tag, class... _Args>
      consteval auto operator()(_Tag (*)(_Args...)) const noexcept  //
        -> default_domain (*)(get_completion_domain_t<_Tag>) noexcept
      {
        return nullptr;
      }
    };

    //! maps a pack of domain queries produced by __domain_query_from_sig to the
    //! corresponding attrs<_Attrs...> type
    class __attrs_from_domain_queries
    {
      template <class _Tag>
      using __query_sig = default_domain (*)(get_completion_domain_t<_Tag>) noexcept;

     public:
      template <class... _Tag>
      consteval auto operator()(__query_sig<_Tag>...) const noexcept  //
        -> __canonical_t<attrs<default_domain(get_completion_domain_t<_Tag>) noexcept...>>
      {
        return {};
      }
    };

    //! computes the set of get_completion_domain queries that must be supported
    //! by any sender that might be erased by the corresponding function
    //!
    //! we should support get_completion_domain<Tag> only if _Sigs contains a
    //! completion of type Tag
    //!
    //! the query form should be
    //!
    //!   default_domain(get_completion_domain_t<Tag>)
    template <class _Sigs>
    using __default_attrs = decltype(_Sigs::__transform_reduce(__domain_query_from_sig(),
                                                               __attrs_from_domain_queries()));

    //! Map a variety of function<...> specifications into the canonical type-erased
    //! contract represented by the user-provided specification.
    //!
    //! The canonical specification looks like this:
    //!
    //!   function<
    //!       sender_tag(Args...),
    //!       completion_signatures<Sigs...>,
    //!       queries<Queries...>,
    //!       attrs<Attrs...>>
    //!
    //! where:
    //! - Args... is the type-erased sender factory's parameter list
    //! - Sigs... is the set of completion signatures that the erased sender is
    //!   allowed to advertise
    //! - Queries... is the set of queries that the eventual receiver's
    //!   environment must support
    //! - Attrs... is the set of attributes the type-erased sender must report;
    //!   only supports the specification of the sender's completion domains
    //!
    //! The order of Args... is obviously important, but Sigs..., Queries...,
    //! and Attrs... are all canonicalized into a sorted and uniqued list to
    //! ensure order is irrelevant.
    template <__is_function_type, class...>
    struct __make_function;

    //! Handle the cases where the given function signature matches
    //!
    //!  Return(Args...) noexcept(???)
    //!
    //! Note that none of these specializations accept a fully-specified completion
    //! signatures since they are derived from _Signature.

    template <__is_not_sender_tag_function _Signature>
    struct __make_function<_Signature>
      : __make_function<_Signature, queries<>, __default_attrs<__completion_sigs_from<_Signature>>>
    {};

    template <__is_not_sender_tag_function _Signature, __is_instance_of<queries> _Queries>
    struct __make_function<_Signature, _Queries>
      : __make_function<_Signature, _Queries, __default_attrs<__completion_sigs_from<_Signature>>>
    {};

    template <__is_not_sender_tag_function _Signature, __is_instance_of<attrs> _Attrs>
      requires __completion_signatures_and_domains_are_compatible<
        __completion_sigs_from<_Signature>,
        _Attrs>
    struct __make_function<_Signature, _Attrs> : __make_function<_Signature, queries<>, _Attrs>
    {};

    template <__is_not_sender_tag_function _Signature,
              __is_instance_of<queries>    _Queries,
              __is_instance_of<attrs>      _Attrs>
      requires __completion_signatures_and_domains_are_compatible<
        __completion_sigs_from<_Signature>,
        _Attrs>
    struct __make_function<_Signature, _Queries, _Attrs>
    {
      using __attrs_t = __canonical_t<__normalized_attrs_t<_Attrs>>;
      using type =
        __function_meta<_Signature>::template __make_function<__completion_sigs_from<_Signature>,
                                                              __canonical_t<_Queries>,
                                                              __attrs_t>;
    };

    //! Handle the cases where the given function signature matches
    //!
    //!  sender_tag(Args...)
    //!
    //! Note that all these specializations require fully-specified completion signatures
    //! since they can't be derived from _Signature.

    template <__is_sender_tag_function                _Signature,
              __is_instance_of<completion_signatures> _ComplSigs>
    struct __make_function<_Signature, _ComplSigs>
      : __make_function<_Signature, _ComplSigs, queries<>, __default_attrs<_ComplSigs>>
    {};

    template <__is_sender_tag_function                _Signature,
              __is_instance_of<completion_signatures> _ComplSigs,
              __is_instance_of<queries>               _Queries>
    struct __make_function<_Signature, _ComplSigs, _Queries>
      : __make_function<_Signature, _ComplSigs, _Queries, __default_attrs<_ComplSigs>>
    {};

    template <__is_sender_tag_function                _Signature,
              __is_instance_of<completion_signatures> _ComplSigs,
              __is_instance_of<attrs>                 _Attrs>
      requires __completion_signatures_and_domains_are_compatible<_ComplSigs, _Attrs>
    struct __make_function<_Signature, _ComplSigs, _Attrs>
      : __make_function<_Signature, _ComplSigs, queries<>, _Attrs>
    {};

    template <__is_sender_tag_function                _Signature,
              __is_instance_of<completion_signatures> _ComplSigs,
              __is_instance_of<queries>               _Queries,
              __is_instance_of<attrs>                 _Attrs>
      requires __completion_signatures_and_domains_are_compatible<_ComplSigs, _Attrs>
    struct __make_function<_Signature, _ComplSigs, _Queries, _Attrs>
    {
      using __attrs_t = __canonical_t<__normalized_attrs_t<_Attrs>>;
      using type = __function_meta<_Signature>::template __make_function<__canonical_t<_ComplSigs>,
                                                                         __canonical_t<_Queries>,
                                                                         __attrs_t>;
    };
  }  // namespace __func

  //! the user-facing interface to exec::function that supports several
  //! different declaration styles, including:
  //!
  //! - function<int(bar, baz)>: a fallible function from (bar, baz) to int
  //! - function<int(bar, baz) noexcept>: an infallible function from (bar, baz)
  //!   to int
  //! - function<sender_tag(bar, baz), completion_signatures<...>>: a function
  //!   from (bar, baz) that completes in the ways specified by the given
  //!   specialization of completion_signatures
  //! - function<int(bar, baz), queries<Return(Query, Args...), ...>: a function
  //!   from (bar, baz) to int that requires the final receiver to have an
  //!   environment that supports the Query query, taking arguments Args..., and
  //!   returning an object convertible to Return; queries may be required to be
  //!   no-throw by declaring the function type noexcept
  //! - function<sender_tag(bar, baz), completion_signatures<...>,
  //!   queries<Return(Query, Args...)>>: a fully-specified async function that
  //!   maps (bar, baz) to the specified completions, requiring the specified
  //!   queries in the ultimate receiver's environment
  //! - any of the above with a trailing attrs<Domain(get_completion_domain_t<Tag>),
  //!   ...>: additionally requires the erased sender to complete with Tag in
  //!   Domain, and reports that domain from the function's environment;
  //!   get_completion_domain_t<> means get_completion_domain_t<set_value_t>
  //! - a signature with an lvalue reference qualifier, like
  //!   function<int(bar) const &> or function<int(bar) &> (optionally
  //!   noexcept): an async member function; the constructor takes the object,
  //!   which must be an lvalue, as its first argument, the function holds a
  //!   pointer to it (so the object must outlive the function's operation),
  //!   and the factory receives it, with the declared cv- and ref-qualifiers,
  //!   before (bar); a `const &` function also accepts a non-const lvalue
  //!
  //! `&&`, `const &&`, `const` without a ref-qualifier and `volatile` function
  //! types are not supported.
  //!
  //! The constructor takes each curried argument declared as T as `T const &`
  //! when T is trivially copy- and move-constructible (so an lvalue `int` can
  //! be passed directly), and as `T &&` otherwise (so copying a std::string
  //! takes an explicit copy, e.g. auto(s)).
  //!
  //! When present, the completion signatures, queries and attrs must appear in
  //! that order.
  //!
  //! Declaring any form noexcept promises that the type-erased path (invoking
  //! the factory and connecting the sender it returns) doesn't throw, except
  //! by failing to allocate a frame, which connect's own conditional noexcept
  //! accounts for. For the forms with computed completions it also drops
  //! set_error(exception_ptr); for the sender_tag form the completion
  //! signatures are exactly as given, since they describe only the
  //! asynchronous contract. The constructor enforces the promise where it can
  //! be checked exactly: it requires the factory to be nothrow-invocable with
  //! the curried arguments and, unless a domain transforms the factory's
  //! sender, that sender's connect not to throw. A domain transformation and
  //! the transformed sender's connect are trusted; if either throws anyway,
  //! std::terminate is called.
  //!
  //! Future: support C-style ellipsis arguments in the function signature to
  //! permit type-erased arguments as well, like function<int(bar, baz, ...)> (a
  //! fallible function from (bar, baz) plus unspecified, erased additional
  //! arguments to int)
  template <__func::__is_supported_function_type _Signature, class... _Ts>
  using function = __func::__make_function<_Signature, _Ts...>::type;
}  // namespace experimental::execution

namespace exec = experimental::execution;
