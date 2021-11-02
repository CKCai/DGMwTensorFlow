#include <iostream>
#include "ILU.h"
#include "D:\dev\csPDM_0820\csPDM\Timer.h"
#include <omp.h>
int main(int argc, char **argv)
{
	int num2omp = 100;				// number of rows in order to activate openMP (>)
	int num_threads = 12;		
	int N = 1200;
	int T = 10;
	if (argc == 5)
	{
		N = std::stoi(argv[1]);
		num2omp = std::stoi(argv[2]);
		num_threads = std::stoi(argv[3]);
		T = std::stoi(argv[4]);
	}

	/*
	{
		Timer time_01("Execution time of  algorithm =  ");
		ILU foo;
		//foo.test_omp();
		foo.para_ilu();
	}
	*/
	//num2omp = 100;						// number of rows in order to activate openMP (>)
	//num_threads = 5;
	
	for (int i = 0; i < T; i++)
	{
		ILU foo2(N, num2omp, num_threads);
		double time1 = omp_get_wtime();
		foo2.para_ilu();
		double time2 = omp_get_wtime();
		std::cout << "Elapsed time of Parallel LU Factorization Algorithm using omp_get_wtime() = " << (time2 - time1)*1000.0 << " msec\n";

		double time5 = omp_get_wtime();
		foo2.LUsolve();
		double time6 = omp_get_wtime();
		std::cout << "Elapsed time of Parallel LU  solve using omp_get_wtime() = " << (time6 - time5)*1000.0 << " msec\n";

		double time3 = omp_get_wtime();
		foo2.EigenSolve();
		double time4 = omp_get_wtime();
		std::cout << "Elapsed time of Eigen::SparseLU algorithm solve using omp_get_wtime() = " << (time4 - time3)*1000.0 << " msec\n";
		std::cout << "\n================================================= \n \n" ;
	}
	std::cout << "Done ! \n";
	std::cout << "Max number of threads = " << omp_get_max_threads() << std::endl;
	//std::cin.get();
	
	return 0;
}