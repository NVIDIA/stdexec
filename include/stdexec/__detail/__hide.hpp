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


struct htt {};

template <class T>
struct htb : htt {
  using unhide = T;
};

template <class Derived, class DerivedTag, class... An>
struct hide_this : DerivedTag {
  friend auto hf(DerivedTag*, An*...) {
    class ht : public htb<Derived> {};
    return ht{};
  }
  using hide =
      decltype(hf(std::declval<hide_this*>(), std::declval<An*>()...));
};

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

}  // namespace h
