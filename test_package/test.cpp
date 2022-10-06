#include <stdexec/execution.hpp>

int main()
{
	auto x = std::execution::just(42);

	auto [a] = std::this_thread::sync_wait(std::move(x)).value();
	return (a==42)?0:-1;
}
