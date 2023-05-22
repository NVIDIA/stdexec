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

/// @brief A constexpr function to count characters in the zero terminated 
/// string specified by @c str at compile-time
///
/// @param str A constexpr zero terminated string
/// @return std::size_t The count of characters in @c str
std::size_t constexpr string_length(const char* str) noexcept {
  std::size_t l = 0;
  // unroll the counting to reduce recursion depth. required 
  // to get the length of long strings
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

//
// generate FNV-1a hash at compile-time
//

constexpr const std::uint64_t FnvOffset = 14695981039346656037ULL;
constexpr const std::uint64_t FnvPrime = 1099511628211ULL;

/// @brief A function that computes a hash at compile-time by adding the 
/// characters in the specified constexpr zero terminated string @ first
///
/// @param first A constexpr zero terminated string
/// @param val An existing hash value to extend - defaults to the 
/// constant @c FnvOffset
/// @return std::uint64_t The hash after the contents of first are applied
std::uint64_t constexpr fnv_1a(const char* first,
                               std::uint64_t val = FnvOffset) noexcept {
  // FNV-1a 64bit hash function
  const size_t count = string_length(first);
  size_t next = 0;
  for (; next + 8 < count; next += 8) {
    std::uint64_t v = 
      (((std::uint64_t)first[next + 0]) << (0 * 8)) + 
      (((std::uint64_t)first[next + 1]) << (1 * 8)) + 
      (((std::uint64_t)first[next + 2]) << (2 * 8)) + 
      (((std::uint64_t)first[next + 3]) << (3 * 8)) + 
      (((std::uint64_t)first[next + 4]) << (4 * 8)) + 
      (((std::uint64_t)first[next + 5]) << (5 * 8)) + 
      (((std::uint64_t)first[next + 6]) << (6 * 8)) + 
      (((std::uint64_t)first[next + 7]) << (7 * 8));
    val ^= v;
    val *= FnvPrime;
  }
  for (; next < count; ++next) {
    val ^= (std::uint64_t)first[next];
    val *= FnvPrime;
  }
  return val;
}

/// @brief A function that computes a hash at compile-time by adding the 
/// characters in the specified constexpr zero terminated string @ first
///
/// @tparam T An integral type
/// @param t An integral value to add to the hash
/// @param val An existing hash value to extend - defaults to the 
/// constant @c FnvOffset
/// @return std::uint64_t The hash after the contents of first are applied
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

/// @brief get unique 64bit id value for @c T
///
/// @tparam T The type for which to generate an id value
/// @return std::uint64_t A unique value that represents @c T
template <class T>
constexpr std::uint64_t get_id() {
  return fnv_1a(__PRETTY_FUNCTION__);
}

/// @brief a type that stores a constexpr id
///
/// @tparam Id The 64bit id to store
template <std::uint64_t Id>
struct id {
  static constexpr std::uint64_t value = Id;
};

/// @brief a type that maps from @c Id to @c T
///
/// @tparam Id The 64bit id, used as a key
/// @tparam T The type that is associated with the @c Id value
/// @tparam Tag A type, with a user-friendly name, that indicates what 
/// kind of T is hidden
template <std::uint64_t Id, class T, class Tag>
struct htb : id<Id> {
  using unhide = T;
};

/// @brief a type that maps from @c T to @c Id and generates a type that 
/// maps from @c Id back to @c T
///
/// @tparam Id The 64bit id, used as a key
/// @tparam T The type that is associated with the @c Id value
/// @tparam Tag A type, with a user-friendly name, that indicates what 
/// kind of T is hidden
template <std::uint64_t Id, class T, class Tag>
struct idd : id<Id> {
  using self_t = idd;
  using type = T;
  using tag = Tag;
  static constexpr std::uint64_t value = Id;

private:
  /// @brief This definition of a private friend function defines a new type @c h that 
  /// only depends on the @c Tag and the @c Id
  ///
  /// The @c void* argument is used to consume the self_t* that allows this function 
  /// definition to be found by ADL
  ///
  /// The @c noexcept clause ensures a unique mangled name even when the Id is not perfectly unique
  /// The @c requires clause ensures a unique definition even when the Id is not perfectly unique
  /// 
  /// @tparam Tg The Tag dependency type for the function
  /// @tparam I The Id dependency type for the function
  /// @return h
  template<class Tg, class I>
  friend auto hf(void*) noexcept(noexcept(I::value == self_t::value)) requires(self_t::value == I::value) {
    class h : public htb<Id, T, Tag> {};
    return h{};
  }
public:
  /// @brief Invoke the function defined in this type and extract the @c h type that it defined
  using hide = decltype(hf<Tag, id<Id>>(std::declval<self_t*>()));
};

/// @brief Combine the Ids of all the related types to get a better unique Id
template <std::uint64_t Id, class Derived, class DerivedTag, class In>
struct hide_filter;

//
// produce filtered result
//

template <std::uint64_t Id, class Derived, class DerivedTag>
struct hide_filter<Id, Derived, DerivedTag, void()> {
  using type = idd<Id, std::remove_cvref_t<Derived>,
                   std::remove_cvref_t<DerivedTag>>;
};

//
// extend hash
//

/// @brief hidden case - @c I0 has a @c value member, use that to modify the @c Id
template <std::uint64_t Id, class Derived, class DerivedTag,
          with_unhide I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(I0::value, Id), Derived, DerivedTag,
                                    void(In...)>::type;
};
/// @brief hide_this case - @c I0 has a @c value member, use that to modify the @c Id
template <std::uint64_t Id, class Derived, class DerivedTag, 
          with_hide I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(I0::value, Id), Derived, DerivedTag,
                                    void(In...)>::type;
};
/// @brief create Id case - generate an id for @c I0 , use that to modify the @c Id
template <std::uint64_t Id, class Derived, class DerivedTag, 
          class I0, class... In>
struct hide_filter<Id, Derived, DerivedTag, void(I0, In...)> {
  using type = typename hide_filter<fnv_1a(get_id<I0>(), Id), Derived,
                                    DerivedTag, void(In...)>::type;
};

/// @brief Generate a type that can be used to hide @c Derived
template <class Derived, class DerivedTag, class... An>
using base_id_t =
    typename hide_filter<get_id<std::remove_cvref_t<Derived>>(), Derived,
                         DerivedTag, void(std::remove_cvref_t<An>...)>::type;

/// @brief Generate a type that can be used to hide @c Derived
template <class Derived, class DerivedTag, class... An>
struct hide_this : base_id_t<Derived, DerivedTag, An...> {};

/// @brief resolve from an arbitrary @c T to a hidden type
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

/// @brief resolve from an arbitrary @c T to an unhidden type
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
