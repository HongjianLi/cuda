#pragma once
#ifndef IO_SERVICE_POOLED_HPP
#define IO_SERVICE_POOLED_HPP

#include <vector>
#include <future>
#include <boost/asio/io_service.hpp>
using namespace std;
using namespace boost::asio;

class io_service_pooled : public io_service
{
public:
	io_service_pooled(const int concurrency);	
	void init(const int num_tasks);	
	void done();
	void sync();
	void destroy();
private:
	vector<future<void>> futures;
	unique_ptr<work> w;
	mutex m;
	condition_variable cv;
	size_t num_tasks;
	size_t num_completed_tasks;
};

#endif
