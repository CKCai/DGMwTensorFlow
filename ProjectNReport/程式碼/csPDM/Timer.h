#pragma once
#include <chrono>
#include <iostream>
#include <string>
// a class for evaluate the elapsed time of the interested-section of code.
class Timer
{
public:
	Timer(std::string CN)
	{
		 StartTimepoint = std::chrono::high_resolution_clock::now();
		 name = CN;
	}
	~Timer()
	{
		Stop();
	}
	void Stop()
	{
		auto EndTimepoint = std::chrono::high_resolution_clock::now();
		auto start = std::chrono::time_point_cast<std::chrono::microseconds> (StartTimepoint).time_since_epoch().count();
		auto end = std::chrono::time_point_cast<std::chrono::microseconds> (EndTimepoint).time_since_epoch().count();

		auto duration = end - start ; 
		double ms = duration * 0.001; 
		std::cout << name;
		std::cout << duration << " micro-seconds ( "<<ms<<" msec ) \n";
		
	}

private:
	std::chrono::time_point <std::chrono::high_resolution_clock> StartTimepoint;
	std::string name;
};