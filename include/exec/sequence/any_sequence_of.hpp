/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../any_sender_of.hpp"
#include "../sequence_senders.hpp"

namespace experimental::execution
{
  template <class _Sigs, class _Queries = queries<>>
  struct any_sequence_receiver;

  template <class _Sigs, class _Queries = queries<>>
  struct any_sequence_receiver_ref;

  template <class _AnyReceiver, class _SenderQueries = queries<>>
  struct any_sequence_sender;

  namespace _any
  {
    //////////////////////////////////////////////////////////////////////////////////
    // _isequence_receiver
    template <class _Sigs, class _Queries>
    struct _isequence_receiver;

    template <class... _Sigs, class... _Queries>
    struct _isequence_receiver<completion_signatures<_Sigs...>, queries<_Queries...>>
    {
      using _return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
      using _void_sender = any_sender<any_receiver<_return_sigs>>;
      using _next_sigs   = completion_signatures<_Sigs...>;
      using _sigs        = __to_sequence_completions_t<_next_sigs>;
      using _item_sender = any_sender<any_receiver<_next_sigs>>;
      using _item_types  = item_types<_item_sender>;

      using _extends_t = STDEXEC::__any::__extends<
        _ireceiver<_sigs, queries<_Queries...>>::template _isemi_receiver>;

      template <class _Base>
      struct _interface_ : STDEXEC::__any::__interface_base<_interface_, _Base, _extends_t>
      {
       private:
        using _base_t = STDEXEC::__any::__interface_base<_interface_, _Base, _extends_t>;
       public:
        using receiver_concept = STDEXEC::receiver_tag;
        using _base_t::_base_t;

        virtual constexpr auto set_next(_item_sender _sndr) -> _void_sender
        {
          return execution::set_next(STDEXEC::__any::__value(*this),
                                     static_cast<_item_sender &&>(_sndr));
        }
      };
    };
  }  // namespace _any

  namespace _any
  {
    //////////////////////////////////////////////////////////////////////////////////
    // _isequence_sender
    template <class _AnySequenceReceiver, class _SenderQueries>
    struct _isequence_sender;

    template <class _Sigs, class _Queries, class... _SenderQueries>
    struct _isequence_sender<any_sequence_receiver<_Sigs, _Queries>, queries<_SenderQueries...>>
    {
      using _item_sender_t = any_sender<any_receiver<_Sigs, _Queries>>;
      using _extends_t =
        STDEXEC::__any::__extends<_iquery_memfn<_SenderQueries>::template _interface_...,
                                  STDEXEC::__any::__imovable>;

      template <class _Base>
      struct _interface_ : STDEXEC::__any::__interface_base<_interface_, _Base, _extends_t>
      {
       private:
        using _base_t = STDEXEC::__any::__interface_base<_interface_, _Base, _extends_t>;
       public:
        using completion_signatures = __to_sequence_completions_t<_Sigs>;
        using item_types            = execution::item_types<_item_sender_t>;
        using sender_concept        = sequence_sender_tag;
        using _any_receiver_ref_t   = any_sequence_receiver_ref<_Sigs, _Queries>;
        using _base_t::_base_t;

        virtual constexpr auto subscribe(_any_receiver_ref_t _rcvr) && -> _any_opstate_base
        {
          STDEXEC_ASSERT(_Base::__box_kind != STDEXEC::__any::__box_kind::__abstract);

          if constexpr (_Base::__box_kind == STDEXEC::__any::__box_kind::__abstract)
            __std::unreachable();
          else if constexpr (_Base::__box_kind == STDEXEC::__any::__box_kind::__proxy)
            // The result of the call to _value(*this) below is a reference to a
            // polymophic sender. If we pass that to STDEXEC::subscribe, it will attempt
            // to transform that sender, which will cause it to be sliced. Instead, we
            // call .subscribe(_rcvr) directly on the contained value. transform_sender
            // gets called when the next branch is taken, which will happen as a result of
            // the call to .subscribe(_rcvr) in this branch.
            return STDEXEC::__any::__value(std::move(*this)).subscribe(std::move(_rcvr));
          else
            return _any_opstate_base{__in_place_from,
                                     execution::subscribe,
                                     STDEXEC::__any::__value(std::move(*this)),
                                     std::move(_rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> _interface_ const &
        {
          return *this;
        }
      };
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // _any_seq_opstate
    template <class _Receiver, class _TargetStopToken>
    struct _any_seq_opstate
    {
      using operation_state_concept = STDEXEC::operation_state_tag;

      template <class _AnySender>
      constexpr explicit _any_seq_opstate(_AnySender &&_sndr, _Receiver _rcvr)
        : _rcvr_{static_cast<_Receiver &&>(_rcvr)}
        , _opstate_(static_cast<_AnySender &&>(_sndr).subscribe(
            typename _AnySender::_any_receiver_ref_t(_rcvr_)))
      {}

      constexpr void start() & noexcept
      {
        _rcvr_._register_callback();
        _opstate_.start();
      }

     private:
      _state<_Receiver, _TargetStopToken> _rcvr_;
      _any_opstate_base                   _opstate_;
    };
  }  // namespace _any

  ////////////////////////////////////////////////////////////////////////////////////
  // any_sequence_receiver
  template <class _Sigs, class _Queries>
  struct any_sequence_receiver final
    : STDEXEC::__any::__any<_any::_isequence_receiver<_Sigs, _Queries>::template _interface_>
  {
   private:
    using _base_t =
      STDEXEC::__any::__any<_any::_isequence_receiver<_Sigs, _Queries>::template _interface_>;
   public:
    using _base_t::_base_t;
  };

  ////////////////////////////////////////////////////////////////////////////////////
  // any_sequence_receiver_ref
  template <class _Sigs, class _Queries>
  struct any_sequence_receiver_ref
    : STDEXEC::__pointer_receiver<
        STDEXEC::__any::__any_ptr<_any::_isequence_receiver<_Sigs, _Queries>::template _interface_>>
  {
   private:
    using _item_sender_t = execution::any_sender<any_receiver<_Sigs, _Queries>>;
    using _item_types_t  = execution::item_types<_item_sender_t>;
    using _base_t        = STDEXEC::__pointer_receiver<
             STDEXEC::__any::__any_ptr<_any::_isequence_receiver<_Sigs, _Queries>::template _interface_>>;

   public:
    template <STDEXEC::__not_decays_to<any_sequence_receiver_ref> _Receiver>
    constexpr any_sequence_receiver_ref(_Receiver &_rcvr) noexcept
      : _base_t(std::addressof(_rcvr))
    {
      static_assert(sequence_receiver_of<_Receiver, _item_types_t>);
    }

    template <auto... _SenderQueries>
    using any_sender = any_sequence_sender<any_sequence_receiver<_Sigs, _Queries>,
                                           _any::_queries_t<_SenderQueries...>>;
  };

  //////////////////////////////////////////////////////////////////////////////////////
  // any_sequence_sender
  template <class _Sigs, class _Queries, class _SenderQueries>
  struct any_sequence_sender<any_sequence_receiver<_Sigs, _Queries>, _SenderQueries> final
    : STDEXEC::__any::__any<_any::_isequence_sender<any_sequence_receiver<_Sigs, _Queries>,
                                                    _SenderQueries>::template _interface_>
  {
   private:
    using _receiver_ref_t = any_sequence_receiver_ref<_Sigs, _Queries>;
    using _stop_token_t   = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<_receiver_ref_t>>;
    using _base_t =
      STDEXEC::__any::__any<_any::_isequence_sender<any_sequence_receiver<_Sigs, _Queries>,
                                                    _SenderQueries>::template _interface_>;
   public:
    using item_types            = _base_t::item_types;
    using completion_signatures = _base_t::completion_signatures;

    template <STDEXEC::__not_same_as<any_sequence_sender> _Sender>
      requires sequence_sender_to<_Sender, _receiver_ref_t>
    constexpr any_sequence_sender(_Sender _sndr)
      : _base_t(static_cast<_Sender &&>(_sndr))
    {}

    template <STDEXEC::receiver _Receiver>
    constexpr auto subscribe(_Receiver _rcvr) &&  //
      -> _any::_any_seq_opstate<_Receiver, _stop_token_t>
    {
      static_assert(STDEXEC::receiver_of<_Receiver, __to_sequence_completions_t<_Sigs>>);
      using _opstate_t = _any::_any_seq_opstate<_Receiver, _stop_token_t>;
      return _opstate_t{static_cast<_base_t &&>(*this), static_cast<_Receiver &&>(_rcvr)};
    }
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;
