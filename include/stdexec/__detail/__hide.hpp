#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include "__config.hpp"
#include "../concepts.hpp"

namespace h {

template<class T>
concept with_hide = requires{
    typename T::hide;
  };

template<class T>
concept with_unhide = requires{
    typename T::unhide;
  };

template<class T>
concept hidden = 
  stdexec::destructible<T> && 
  with_unhide<T> &&
  (!with_hide<T>);

template<class T>
concept no_hide = hidden<T> || (!with_hide<T>);
template<class T>
concept yes_hide = (!hidden<T>) && with_hide<T>;

template<class T>
concept no_unhide = (!hidden<T>) && (!with_hide<T>);
template<class T>
concept yes_unhide = hidden<T>;

// generate FNV-1a hash at compile-time
std::size_t constexpr string_length(const char* str) noexcept {
  std::size_t l = 0;
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  if (++l, str[l - 1] == 0) {
    return l - 1;
  }
  return (l - 1) + string_length(str + l);
}

namespace {
constexpr const std::uint64_t FnvOffset = 14695981039346656037ULL;
constexpr const std::uint64_t FnvPrime = 1099511628211ULL;
}  // namespace

std::uint64_t constexpr fnv_1a(const char* first,
                               std::uint64_t val = FnvOffset) noexcept {
  // FNV-1a 64bit hash function
  const size_t count = string_length(first);
  for (size_t next = 0; next < count; ++next) {
    val ^= (std::uint64_t)first[next];
    val *= FnvPrime;
  }
  return val;
}
template <class T> 
  requires std::is_integral_v<std::remove_cvref_t<T>>
std::uint64_t constexpr fnv_1a(T&& t, std::uint64_t val = FnvOffset) noexcept {
  // FNV-1a 64bit hash function
  for (size_t next = 0; next < sizeof(T); ++next) {
    val ^= ((std::uint64_t)(t >> (next * 8))) & 0xFF;
    val *= FnvPrime;
  }
  return val;
}

/// @brief get unique uint id value for T
template <class T>
constexpr std::uint64_t get_id() {
  return fnv_1a(__PRETTY_FUNCTION__);
}

template <class OSig, class ASig, class = void>
struct unique;
template <class... On, class Void>
struct unique<void(On...), void(), Void> {
  using type = void(On...);
};
template <class... On, class A0, class... An>
  requires (!std::same_as<A0, On> && ...)
struct unique<void(On...), void(A0, An...)> {
  using type = typename unique<void(On..., A0), void(An...)>::type;
};
template <class... On, class A0, class... An>
  requires (std::same_as<A0, On> || ...)
struct unique<void(On...), void(A0, An...)> {
  using type = typename unique<void(On...), void(An...)>::type;
};

template <std::uint64_t Id>
struct id {
  static constexpr std::uint64_t value = Id;
};

template <std::uint64_t Id, class T, class Tag, class... An>
struct htb : id<Id> {
  using unhide = T;
  using sig = Tag(An...);
  template <class... On>
  using append = typename unique<void(On...), void(An...)>::type;
};

template <class Tag, class H>
struct ht : H {
  using tag = Tag;
  using base = H;
};

template <std::uint64_t Id, class T, class Tag, class... An>
struct idd : id<Id> {
  using self_t = idd;
  using type = T;
  using tag = Tag;
  using sig = Tag(An...);
  template <class... On>
  using append = typename unique<void(On...), void(An...)>::type;
  static constexpr std::uint64_t value = Id;

private:
  template<class I>
  friend auto hf(void*) noexcept(noexcept(I::value == self_t::value)) requires(self_t::value == I::value) {
    class h : public htb<Id, T, Tag, An...> {};
    return ht<Tag, h>{};
  }
public:
  using hide = decltype(hf<id<Id>>(std::declval<self_t*>()));
};

template <std::uint64_t Id, class Derived, class DerivedTag, class Out,
          class In>
struct hide_filter;

template <std::uint64_t Id, class Derived, class DerivedTag, class... On>
struct hide_filter<Id, Derived, DerivedTag, void(On...), void()> {
  using type = idd<Id, std::remove_cvref_t<Derived>,
                   std::remove_cvref_t<DerivedTag>, On...>;
};

template <std::uint64_t Id, class Derived, class DerivedTag, class... On,
          with_unhide I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(On...), void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(I0::value, Id), Derived, DerivedTag,
                                    typename unique<void(On...), void(I0)>::type, void(In...)>::type;
};
template <std::uint64_t Id, class Derived, class DerivedTag, class... On,
          with_hide I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(On...), void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(I0::value, Id), Derived, DerivedTag,
                                    typename unique<void(On...), void(typename I0::hide)>::type, void(In...)>::type;
};
template <std::uint64_t Id, class Derived, class DerivedTag, class... On,
          class I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(On...), void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(get_id<I0>(), Id), Derived,
                                    DerivedTag, typename unique<void(On...), void(I0)>::type, void(In...)>::type;
};

template <class Derived, class DerivedTag, class... An>
using base_id_t =
    typename hide_filter<get_id<std::remove_cvref_t<Derived>>(), Derived,
                         DerivedTag, void(),
                         void(std::remove_cvref_t<An>...)>::type;

template <class Derived, class DerivedTag, class... An>
struct hide_this : base_id_t<Derived, DerivedTag, An...> {};

template <class T>
  requires no_hide<T> || yes_hide<T>
struct hide_impl;

template <no_hide T>
struct hide_impl<T> {
  using type = T;
};
template <yes_hide T>
struct hide_impl<T> {
  using type = typename T::hide;
};

template <class T>
  requires no_unhide<T> || yes_unhide<T>
struct unhide_impl;

template <no_unhide T>
struct unhide_impl<T> {
  using type = T;
};
template <yes_unhide T>
struct unhide_impl<T> {
  using type = typename T::unhide;
};

#if 1
/// @brief used to retrieve a type that can be
/// used to represent T in a memoized (unexpanded)
/// form. T is later retrieved with unhide
template <class T>
using hide = typename hide_impl<T>::type;

/// @brief used to retrieve T from the memoized
/// (unexpanded) form that is returned from hide.
template <class T>
using unhide = typename unhide_impl<T>::type;

static_assert(stdexec::same_as<hide<int>, int>, "");
static_assert(stdexec::same_as<hide<float>, float>, "");

struct tg {};
struct my_long_type : h::hide_this<my_long_type, tg> {};

static_assert(yes_hide<my_long_type>, "");

static_assert(!stdexec::same_as<hide<my_long_type>, my_long_type>, "");
static_assert(stdexec::same_as<unhide<hide<my_long_type>>, my_long_type>, "");

#else
/// @brief used to retrieve a type that can be
/// used to represent T in a memoized (unexpanded)
/// form. T is later retrieved with unhide
template <class T>
using hide = T;

/// @brief used to retrieve T from the memoized
/// (unexpanded) form that is returned from hide.
template <class T>
using unhide = T;
#endif

}  // namespace h
