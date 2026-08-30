/*
 * Test for PR #2223: any_sender completion signatures for sequence with 3+ senders
 * Fixes #2101
 */
#include <exec/any_sender_of.hpp>
#include <exec/sequence.hpp>
#include <stdexec/execution.hpp>

#include <catch2/catch_all.hpp>

namespace ex = STDEXEC;

TEST_CASE("sequence with 3 any_sender works - PR #2223", "[sequence][any_sender]")
{
    // Test case from issue #2101: sequence with 3+ any_sender
    // sequence requires all but the last sender to be void senders
    using SigsVoid = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
    using SigsInt = ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;
    
    using AnySenderVoid = exec::any_receiver_ref<SigsVoid>::any_sender<>;
    using AnySenderInt = exec::any_receiver_ref<SigsInt>::any_sender<>;

    // Create 2 void any_senders and 1 int any_sender
    AnySenderVoid s1 = ex::just();
    AnySenderVoid s2 = ex::just();
    AnySenderInt s3 = ex::just(42);

    // This should work now (was failing before the fix)
    auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3));

    // Run it
    auto [a] = *ex::sync_wait(std::move(seq));
    CHECK(a == 42);
}

TEST_CASE("sequence with 4 any_sender works", "[sequence][any_sender]")
{
    using SigsVoid = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
    using SigsInt = ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;
    
    using AnySenderVoid = exec::any_receiver_ref<SigsVoid>::any_sender<>;
    using AnySenderInt = exec::any_receiver_ref<SigsInt>::any_sender<>;

    AnySenderVoid s1 = ex::just();
    AnySenderVoid s2 = ex::just();
    AnySenderVoid s3 = ex::just();
    AnySenderInt s4 = ex::just(42);

    auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3), std::move(s4));

    auto [a] = *ex::sync_wait(std::move(seq));
    CHECK(a == 42);
}

TEST_CASE("sequence with mixed senders including any_sender works", "[sequence][any_sender]")
{
    using SigsVoid = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
    using SigsInt = ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;
    
    using AnySenderVoid = exec::any_receiver_ref<SigsVoid>::any_sender<>;
    using AnySenderInt = exec::any_receiver_ref<SigsInt>::any_sender<>;

    AnySenderVoid s1 = ex::just();
    auto s2 = ex::just();  // regular sender
    AnySenderInt s3 = ex::just(42);

    auto seq = exec::sequence(std::move(s1), std::move(s2), std::move(s3));

    auto [a] = *ex::sync_wait(std::move(seq));
    CHECK(a == 42);
}
