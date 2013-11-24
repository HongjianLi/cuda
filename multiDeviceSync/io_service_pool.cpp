#include "io_service_pool.hpp"

io_service_pool::io_service_pool(const unsigned concurrency) : w(unique_ptr<work>(new work(*this)))
{
	for (int i = 0; i < concurrency; ++i)
	{
		futures.emplace_back(async(launch::async, [&]()
		{
			run();
		}));
	}
}

void io_service_pool::wait()
{
	w.reset();
	for (auto& f : futures)
	{
		f.get();
	}
}
