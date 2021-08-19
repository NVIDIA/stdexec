#pragma once

#include <functional>

namespace std {
    // [func.tag_invoke], tag_invoke
    inline namespace __tag_invoke {
        template <class Tag, class... Args>
        concept __has_tag_invoke =
            requires (Tag tag, Args&&... args) {
                tag_invoke((Tag&&) tag, (Args&&) args...);
            };

        inline constexpr struct __fn {
            template <class... Args, __has_tag_invoke<Args...> Tag>
            decltype(auto) operator()(Tag tag, Args&&... args) const
                noexcept(noexcept(tag_invoke((Tag&&) tag, (Args&&) args...))) {
                return tag_invoke((Tag&&) tag, (Args&&) args...);
            }
        } tag_invoke {};
    }

    template<auto& Tag>
    using tag_t = decay_t<decltype(Tag)>;

    template<class Tag, class... Args>
    concept tag_invocable =
        invocable<decltype(tag_invoke), Tag, Args...>;

    template<class Tag, class... Args>
    concept nothrow_tag_invocable =
        tag_invocable<Tag, Args...> &&
        is_nothrow_invocable_v<decltype(tag_invoke), Tag, Args...>;

    template<class Tag, class... Args>
    using tag_invoke_result = invoke_result<decltype(tag_invoke), Tag, Args...>;

    template<class Tag, class... Args>
    using tag_invoke_result_t = invoke_result_t<decltype(tag_invoke), Tag, Args...>;
}
