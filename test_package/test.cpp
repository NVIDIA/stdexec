#include <stdexec/execution.hpp>

int main()
{
	auto x = stdexec::just(42);

	auto [a] = stdexec::sync_wait(std::move(x)).value();
	return (a==42)?0:-1;
}
