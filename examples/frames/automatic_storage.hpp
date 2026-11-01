/*
 * Exploration: LIFO "automatic storage" for asynchronous operations.
 *
 * Three layers:
 *
 *   - A slab source (here, any std::pmr::memory_resource) hands out slabs.
 *     It's shared and must be thread-safe.
 *   - A slab_chain belongs to one strand: a sequence of logically
 *     sequential operations. It isn't thread-safe and is used LIFO. It keeps
 *     slabs it has handed out until it's destroyed, so a strand that pushes
 *     and pops across a slab boundary doesn't allocate on every crossing.
 *   - automatic_storage is a copyable cursor into a strand's storage. push()
 *     returns room for one object and the storage after it. A copy may be
 *     pushed from only while nothing allocated from a later copy is alive,
 *     exactly as with a stack pointer: the operation that pushed reuses its
 *     own copy once its child's operation state is destroyed.
 *
 * Prototype simplifications: fixed-size slabs (an oversized object fails
 * rather than getting an oversized slab), allocation failure always throws
 * std::bad_alloc (or terminates without exceptions), and the query is a
 * forwarding query (see demo.cpp for the consequences).
 */
#pragma once

#include <stdexec/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <memory_resource>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace frames
{
  namespace ex = stdexec;

  //////////////////////////////////////////////////////////////////////////////
  // Slabs and chains

  struct slab
  {
    slab*       next = nullptr;
    std::size_t size = 0;  // payload bytes, following the header

    auto begin() noexcept -> std::byte*
    {
      return reinterpret_cast<std::byte*>(this + 1);
    }

    auto end() noexcept -> std::byte*
    {
      return begin() + size;
    }
  };

  class slab_chain
  {
   public:
    slab_chain(std::pmr::memory_resource* source, std::size_t slab_size) noexcept
      : source_(source)
      , slab_size_(slab_size)
    {}

    slab_chain(slab_chain&&) = delete;

    ~slab_chain()
    {
      while (head_ != nullptr)
      {
        slab* next = head_->next;
        source_->deallocate(head_, sizeof(slab) + head_->size, alignof(std::max_align_t));
        head_ = next;
      }
    }

    // The slab that follows `prev` (the first slab if `prev` is null),
    // reusing one this chain already holds.
    auto after(slab* prev) -> slab*
    {
      slab*& link = prev == nullptr ? head_ : prev->next;
      if (link == nullptr)
      {
        void* p = source_->allocate(sizeof(slab) + slab_size_, alignof(std::max_align_t));
        link    = ::new (p) slab{nullptr, slab_size_};
      }
      return link;
    }

    [[nodiscard]]
    auto source() const noexcept -> std::pmr::memory_resource*
    {
      return source_;
    }

    [[nodiscard]]
    auto slab_size() const noexcept -> std::size_t
    {
      return slab_size_;
    }

   private:
    std::pmr::memory_resource* source_;
    std::size_t                slab_size_;
    slab*                      head_ = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////////
  // automatic_storage

  class automatic_storage
  {
   public:
    // No storage: every push fails.
    automatic_storage() = default;

    // The bottom of a strand's chain. Takes no slab until the first push.
    explicit automatic_storage(slab_chain& chain) noexcept
      : chain_(&chain)
    {}

    // A buffer offered by some operation state, falling back to `outer` when
    // it's full. `outer` must outlive every push from this storage.
    automatic_storage(std::span<std::byte> buffer, automatic_storage const & outer) noexcept
      : cur_(buffer.data())
      , end_(buffer.data() + buffer.size())
      , outer_(&outer)
    {}

    [[nodiscard]]
    auto empty() const noexcept -> bool
    {
      return chain_ == nullptr && outer_ == nullptr && cur_ == end_;
    }

    // Room for `size` bytes aligned to `align`, and the storage after it.
    [[nodiscard]]
    auto push(std::size_t size, std::size_t align) const -> std::pair<void*, automatic_storage>
    {
      if (void* p = fit(size, align))
      {
        automatic_storage after = *this;
        after.cur_              = static_cast<std::byte*>(p) + size;
        return {p, after};
      }
      if (chain_ != nullptr)
      {
        automatic_storage next = *this;
        next.slab_             = chain_->after(slab_);
        next.cur_              = next.slab_->begin();
        next.end_              = next.slab_->end();
        if (next.fit(size, align) == nullptr)
        {
          STDEXEC_THROW(std::bad_alloc());  // prototype: no oversized slabs yet
        }
        return next.push(size, align);
      }
      if (outer_ != nullptr)
      {
        return outer_->push(size, align);
      }
      STDEXEC_THROW(std::bad_alloc());
    }

    template <class T>
    [[nodiscard]]
    auto push() const -> std::pair<void*, automatic_storage>
    {
      return push(sizeof(T), alignof(T));
    }

    // The chain at the bottom of this storage, if any: what a fork draws its
    // new chain's slabs from.
    [[nodiscard]]
    auto chain() const noexcept -> slab_chain*
    {
      return chain_ != nullptr ? chain_ : outer_ != nullptr ? outer_->chain() : nullptr;
    }

   private:
    [[nodiscard]]
    auto fit(std::size_t size, std::size_t align) const noexcept -> void*
    {
      if (cur_ == nullptr)
      {
        return nullptr;
      }
      auto addr    = reinterpret_cast<std::uintptr_t>(cur_);
      auto aligned = (addr + align - 1) & ~(std::uintptr_t(align) - 1);
      if (aligned + size > reinterpret_cast<std::uintptr_t>(end_))
      {
        return nullptr;
      }
      return reinterpret_cast<void*>(aligned);
    }

    std::byte*                cur_   = nullptr;
    std::byte*                end_   = nullptr;
    slab*                     slab_  = nullptr;  // the slab cur_ points into, if chain-backed
    slab_chain*               chain_ = nullptr;
    automatic_storage const * outer_ = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////////
  // The query

  struct get_automatic_storage_t : ex::forwarding_query_t
  {
    template <class Env, class Self = get_automatic_storage_t>
      requires requires(Env const & env, Self const & self) {
        { env.query(self) } -> std::convertible_to<automatic_storage>;
      }
    auto operator()(Env const & env) const noexcept -> automatic_storage
    {
      return env.query(*this);
    }
  };

  inline constexpr get_automatic_storage_t get_automatic_storage{};

  template <class Env>
  auto storage_of(Env const & env) noexcept -> automatic_storage
  {
    if constexpr (std::invocable<get_automatic_storage_t, Env const &>)
    {
      return get_automatic_storage(env);
    }
    else
    {
      return {};
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // A receiver that forwards completions to its parent's receiver and offers
  // its child a given storage.

  // The environment a child sees: its parent receiver's environment, with
  // the storage the child may push from.
  template <class Env>
  struct storage_env
  {
    [[nodiscard]]
    auto query(get_automatic_storage_t) const noexcept -> automatic_storage
    {
      return storage;
    }

    template <class Query, class... As>
      requires(!std::same_as<Query, get_automatic_storage_t>)
           && requires(Env const & env, Query query, As&&... as) {
                env.query(query, static_cast<As&&>(as)...);
              }
    [[nodiscard]]
    auto query(Query query, As&&... as) const
      noexcept(noexcept(std::declval<Env const &>().query(query, static_cast<As&&>(as)...)))
        -> decltype(auto)
    {
      return env.query(query, static_cast<As&&>(as)...);
    }

    automatic_storage storage;
    Env               env;
  };

  template <class Rcvr>
  struct storage_receiver
  {
    using receiver_concept = ex::receiver_t;

    template <class... As>
    void set_value(As&&... as) && noexcept
    {
      ex::set_value(static_cast<Rcvr&&>(*rcvr), static_cast<As&&>(as)...);
    }

    template <class E>
    void set_error(E&& e) && noexcept
    {
      ex::set_error(static_cast<Rcvr&&>(*rcvr), static_cast<E&&>(e));
    }

    void set_stopped() && noexcept
    {
      ex::set_stopped(static_cast<Rcvr&&>(*rcvr));
    }

    [[nodiscard]]
    auto get_env() const noexcept -> storage_env<ex::env_of_t<Rcvr>>
    {
      return {storage, ex::get_env(*rcvr)};
    }

    Rcvr*             rcvr;
    automatic_storage storage;
  };

  template <class Self, class Child, class... Env>
  consteval auto child_completions()
  {
    using child_t =
      std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, Child const &, Child>;
    return ex::get_completion_signatures<child_t, storage_env<Env>...>();
  }

  // Prints where each frame went.
  inline void trace(char const * what, char const * name, void const * p, std::size_t size)
  {
    std::printf("    %-10s %-12s %p (%zu bytes)\n", what, name, p, size);
  }

  //////////////////////////////////////////////////////////////////////////////
  // with_automatic_storage(sndr, source, slab_size): the root of a strand.

  template <class Sndr>
  struct with_storage_sender
  {
    using sender_concept = ex::sender_t;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return child_completions<Self, Sndr, Env...>();
    }

    template <class Rcvr>
    struct opstate
    {
      using child_rcvr_t = storage_receiver<Rcvr>;

      opstate(Sndr&& sndr, Rcvr rcvr, std::pmr::memory_resource* source, std::size_t slab_size)
        : rcvr_(static_cast<Rcvr&&>(rcvr))
        , chain_(source, slab_size)
        , child_(
            ex::connect(static_cast<Sndr&&>(sndr), child_rcvr_t{&rcvr_, automatic_storage(chain_)}))
      {}

      void start() & noexcept
      {
        ex::start(child_);
      }

      Rcvr                                     rcvr_;
      slab_chain                               chain_;
      ex::connect_result_t<Sndr, child_rcvr_t> child_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Rcvr&&>(rcvr), source_, slab_size_};
    }

    Sndr                       sndr_;
    std::pmr::memory_resource* source_;
    std::size_t                slab_size_;
  };

  template <class Sndr>
  auto with_automatic_storage(Sndr&&                     sndr,
                              std::pmr::memory_resource* source = std::pmr::get_default_resource(),
                              std::size_t                slab_size = 4096)
  {
    return with_storage_sender<std::remove_cvref_t<Sndr>>{static_cast<Sndr&&>(sndr),
                                                          source,
                                                          slab_size};
  }

  //////////////////////////////////////////////////////////////////////////////
  // indirect(sndr, name): connects its child into storage it pushes, as type
  // erasure would. Falls back to the heap when it's given no storage.

  template <class Sndr>
  struct indirect_sender
  {
    using sender_concept = ex::sender_t;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return child_completions<Self, Sndr, Env...>();
    }

    template <class Rcvr>
    struct opstate
    {
      using child_rcvr_t = storage_receiver<Rcvr>;
      using child_op_t   = ex::connect_result_t<Sndr, child_rcvr_t>;

      opstate(Sndr&& sndr, Rcvr rcvr, char const * name)
        : rcvr_(static_cast<Rcvr&&>(rcvr))
      {
        automatic_storage storage = storage_of(ex::get_env(rcvr_));
        if (storage.empty())
        {
          on_heap_ = true;
          child_   = new child_op_t(
            ex::connect(static_cast<Sndr&&>(sndr), child_rcvr_t{&rcvr_, storage}));
          trace("heap", name, child_, sizeof(child_op_t));
        }
        else
        {
          auto [p, after] = storage.template push<child_op_t>();
          trace("push", name, p, sizeof(child_op_t));
          child_ = ::new (p)
            child_op_t(ex::connect(static_cast<Sndr&&>(sndr), child_rcvr_t{&rcvr_, after}));
        }
      }

      opstate(opstate&&) = delete;

      ~opstate()
      {
        if (on_heap_)
        {
          delete child_;
        }
        else
        {
          std::destroy_at(child_);
        }
      }

      void start() & noexcept
      {
        ex::start(*child_);
      }

      Rcvr        rcvr_;
      child_op_t* child_   = nullptr;
      bool        on_heap_ = false;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Rcvr&&>(rcvr), name_};
    }

    Sndr         sndr_;
    char const * name_;
  };

  template <class Sndr>
  auto indirect(Sndr&& sndr, char const * name = "indirect")
  {
    return indirect_sender<std::remove_cvref_t<Sndr>>{static_cast<Sndr&&>(sndr), name};
  }

  //////////////////////////////////////////////////////////////////////////////
  // strand(sndr): runs its child on a new chain drawn from the same slab
  // source as the storage it's given (a fork). The new chain takes no slab
  // until its first push, so forking work that never allocates is cheap.

  template <class Sndr>
  struct strand_sender
  {
    using sender_concept = ex::sender_t;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return child_completions<Self, Sndr, Env...>();
    }

    template <class Rcvr>
    struct opstate
    {
      using child_rcvr_t = storage_receiver<Rcvr>;

      opstate(Sndr&& sndr, Rcvr rcvr)
        : rcvr_(static_cast<Rcvr&&>(rcvr))
        , parent_chain_(storage_of(ex::get_env(rcvr_)).chain())
        , chain_(parent_chain_ ? parent_chain_->source() : nullptr,
                 parent_chain_ ? parent_chain_->slab_size() : 0)
        , child_(ex::connect(
            static_cast<Sndr&&>(sndr),
            child_rcvr_t{&rcvr_, parent_chain_ ? automatic_storage(chain_) : automatic_storage()}))
      {}

      void start() & noexcept
      {
        ex::start(child_);
      }

      Rcvr                                     rcvr_;
      slab_chain*                              parent_chain_;
      slab_chain                               chain_;  // unused without a parent chain
      ex::connect_result_t<Sndr, child_rcvr_t> child_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
  };

  template <class Sndr>
  auto strand(Sndr&& sndr)
  {
    return strand_sender<std::remove_cvref_t<Sndr>>{static_cast<Sndr&&>(sndr)};
  }

  //////////////////////////////////////////////////////////////////////////////
  // reserve_async_stack<N>(sndr): offers its child N bytes inside its own
  // operation state, falling back to the storage it was given.

  template <std::size_t N, class Sndr>
  struct reserve_sender
  {
    using sender_concept = ex::sender_t;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return child_completions<Self, Sndr, Env...>();
    }

    template <class Rcvr>
    struct opstate
    {
      using child_rcvr_t = storage_receiver<Rcvr>;

      opstate(Sndr&& sndr, Rcvr rcvr)
        : rcvr_(static_cast<Rcvr&&>(rcvr))
        , outer_(storage_of(ex::get_env(rcvr_)))
        , child_(ex::connect(static_cast<Sndr&&>(sndr), child_rcvr_t{&rcvr_, reserved()}))
      {}

      auto reserved() -> automatic_storage
      {
        std::printf("    reserve    %-12s %p (%zu bytes)\n",
                    "buffer",
                    static_cast<void*>(buffer_),
                    N);
        return automatic_storage(buffer_, outer_);
      }

      void start() & noexcept
      {
        ex::start(child_);
      }

      Rcvr              rcvr_;
      automatic_storage outer_;
      alignas(std::max_align_t) std::byte buffer_[N];
      ex::connect_result_t<Sndr, child_rcvr_t> child_;
    };

    template <class Rcvr>
    auto connect(Rcvr rcvr) && -> opstate<Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Rcvr&&>(rcvr)};
    }

    Sndr sndr_;
  };

  template <std::size_t N, class Sndr>
  auto reserve_async_stack(Sndr&& sndr)
  {
    return reserve_sender<N, std::remove_cvref_t<Sndr>>{static_cast<Sndr&&>(sndr)};
  }
}  // namespace frames
