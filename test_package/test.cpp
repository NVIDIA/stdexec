#include <execution.hpp>

int main()
{
	auto x = std::execution::just(42);

	auto [a] = _P2300::this_thread::sync_wait(std::move(x)).value();
	return (a==42)?0:-1;
}
