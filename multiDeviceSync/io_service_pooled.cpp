#include "io_service_pooled.hpp"

io_service_pooled::io_service_pooled(const int concurrency) : w(new work(*this))
{
	for (int i = 0; i < concurrency; ++i)
	{
		futures.emplace_back(async(launch::async, [&]()
		{
			run();
		}));
	}
}

void io_service_pooled::init(const int num_tasks)
{
	this->num_tasks = num_tasks;
	this->num_completed_tasks = 0;
}

void io_service_pooled::done()
{
	lock_guard<mutex> guard(m);
	if (++num_completed_tasks == num_tasks) cv.notify_one();
}

void io_service_pooled::sync()
{
	unique_lock<mutex> lock(m);
	cv.wait(lock);
}

void io_service_pooled::destroy()
{
	w.reset();
	for (auto& f : futures)
	{
		f.get();
	}
}
