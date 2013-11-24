#pragma once
#ifndef IO_SERVICE_POOL_HPP
#define IO_SERVICE_POOL_HPP

//#include <vector>
#include <future>
#include <boost/asio/io_service.hpp>
using namespace std;
using namespace boost::asio;

class io_service_pool : public io_service
{
public:
	explicit io_service_pool(const unsigned concurrency);
	void wait();
private:
	vector<future<void>> futures;
	unique_ptr<work> w;
};

#endif
