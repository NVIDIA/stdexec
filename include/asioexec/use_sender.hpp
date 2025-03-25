#pragma once

#include <stdexec/execution.hpp>
#include <boost/asio.hpp>

namespace asioexec
{

    template <class... Ts>
    class sender
    {
    public:
        using sender_concept = stdexec::sender_t;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(Ts...), stdexec::set_error_t(std::exception_ptr), stdexec::set_stopped_t()>;

        sender(auto &&init_start_func, auto &&...init_start_args)
            : starter(new async_starter(std::forward<decltype(init_start_func)>(init_start_func), std::tuple(std::forward<decltype(init_start_args)>(init_start_args)...)))
        {
        }

        sender(const sender &init)
            : starter(&init.starter->copy())
        {
        }

        sender(sender &&) = default;

        stdexec::operation_state auto connect(stdexec::receiver auto &&);

        template <class R> // TODO: bug in clang++ when using template <stdexec::receiver R>. See respository=github/llvm/llvm-project, issue=#102320.
        struct operation_state
        {
            sender snd;
            R recv;

            void start() noexcept
            {
                if (stdexec::get_stop_token(stdexec::get_env(recv)).stop_requested())
                {
                    stdexec::set_stopped(std::move(recv)); // Stop channel.
                    return;
                }

                try
                {
                    snd.starter->start([&](Ts... args)
                                       { stdexec::set_value(std::move(recv), std::forward<decltype(args)>(args)...); }); // Value channel.
                }
                catch (...)
                {
                    stdexec::set_error(std::move(recv), std::current_exception()); // Error channel.
                }
            }
        };

    private:
        struct async_starter_base // Type erase some additional type-info of async_starter
        {
            virtual ~async_starter_base() = default;
            virtual async_starter_base &copy() const = 0;

            virtual void start(std::function<void(Ts...)>) = 0;
        };

        template <class Func, class... Args> // Refers to boost::asio::async_initiate.
        struct async_starter
            : async_starter_base
        {
            Func func;
            std::tuple<Args...> args;

            async_starter(Func init_func, std::tuple<Args...> init_args)
                : func(std::move(init_func)),
                  args(std::move(init_args)) {

                  };

            async_starter_base &copy() const override
            {
                return *new async_starter(func, args);
            }

            void start(std::function<void(Ts...)> completion_token) override
            {
                std::apply(func, std::tuple_cat(std::make_tuple(std::move(completion_token)), std::move(args)));
            }
        };

        std::unique_ptr<async_starter_base> starter;
    };

    template <class... Ts>
    stdexec::operation_state auto sender<Ts...>::connect(stdexec::receiver auto &&recv)
    {
        return operation_state<std::decay_t<decltype(recv)>>(std::move(*this), std::forward<decltype(recv)>(recv));
    }

    struct use_sender_t
    {
        constexpr use_sender_t() = default;

        template <class Executor>
        struct executor_with_default
            : Executor
        {
            using default_completion_token_type = use_sender_t;

            executor_with_default(const Executor &ex) noexcept
                : Executor(ex)
            {
            }

            executor_with_default(const auto &ex) noexcept
                requires(not std::same_as<std::decay_t<decltype(ex)>, executor_with_default> and std::convertible_to<std::decay_t<decltype(ex)>, Executor>)
                /* this requires-clause should first require "not std::same_as<ex,*this>", then require "std::convertible_to<lower_executor>",
                 * because "std::convertible_to" depends on itself.
                 */
                : Executor(ex)
            {
            }
        };

        template <class Context>
        using as_default_on_t = Context::template rebind_executor<executor_with_default<typename Context::executor_type>>::other;

        static auto as_default_on(auto &&obj)
        {
            return typename std::decay_t<decltype(obj)>::
                template rebind_executor<executor_with_default<typename std::decay_t<decltype(obj)>::executor_type>>::other(std::forward<decltype(obj)>(obj));
        }
    };

    constexpr use_sender_t use_sender = use_sender_t();

} // namespace asioexec

namespace boost::asio
{
    template <class... Ts>
    struct async_result<asioexec::use_sender_t, void(Ts...)>
    {
        static auto initiate(auto &&start_func, asioexec::use_sender_t, auto &&...start_args)
        {
            auto sched_sender = stdexec::read_env(stdexec::get_scheduler);
            auto value_sender = stdexec::just(std::forward<decltype(start_func)>(start_func), std::forward<decltype(start_args)>(start_args)...);
    
            // clang-format off
            return stdexec::when_all(std::move(sched_sender), std::move(value_sender)) 
                 | stdexec::let_value([](auto &&sched, auto &&...args) 
                     {
                         return asioexec::sender<Ts...>(std::forward<decltype(args)>(args)...) 
                              | stdexec::continues_on(std::forward<decltype(sched)>(sched)); 
                     });
            // clang-format on
        }

        using return_type = decltype(initiate([](auto &&...) {}, std::declval<asioexec::use_sender_t>()));
    };
} // namespace boost::asio
