#include "csPDM.h"
//#include "Parameters.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <fstream>
#include <iterator>
#include "Timer.h"
#include "plot/matplotlibcpp.h"		// Python plotting library under Release Mode - x64,  original version  -> only support std::vector
#include  "cvmarkersobj.h"
#include <omp.h>
//#include "matplotlib/matplotlibcpp.h"		// new version -> support eigen object
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double 
typedef Eigen::Triplet<double> T;
using namespace Concurrency::diagnostic;

namespace param {
	double Vlb = -100.0, Vc = -40.0, EL = -65.0, Vr = -65.0, VT = -50.0,  K = 2.0;
	double V_AMPA = 0.0, V_GABA_A = -80.0, V_GABA_B = Vlb;
	double dVs_AMPA = 1.0, dVs_GABA_A = -0.25, dVs_GABA_B = -0.25;
	double C = 1.0, C2=1.0 , gL =0.05; int tref = 0; int cs = 200;
	double tau_AMPA = 5.0, tau_GABA_A = 10.0, tau_GABA_B = 100.0;
	double Gs_AMPA =  -C2 * log(1 - (dVs_AMPA / (V_AMPA - EL))); //0.015504186535965199;
	double Gs_GABA_A = -C2 * log(1 - (dVs_GABA_A / (V_GABA_A - EL))), Gs_GABA_B = -C2 * log(1 - (dVs_GABA_B / (V_GABA_B - EL)));
	double I = 3.0; double intensity = pow(I/C, 2) ;
	//  這是csPDMEIF 的定義
	csPDMEIF::csPDMEIF() :den2(1200), A(1200, 1200), V(601), den_store(1200, 100), _den(601), FR(100001), b(1200), V_(600), MeanV_arr(10001), time(10001)
	{
		_dt = 0.01;
		_T = 1000.0;
		M = 600;
		L = 0.1;
		V = Eigen::VectorXd::LinSpaced(601, -100.0, -40.0);
		double dV = V(1) - V(0);
		V_ = Eigen::VectorXd::LinSpaced(600, -100.0+dV, -40.0-dV);
		for (int i = 0; i < V.size(); i++)
		{
			if (abs(V(i) + 65.0) < 1 * pow(10, -6))
			{
				index_reset = 2*i;
			}	
		}
		//den2(1200) ;						 //***** this line will lead to run-time error, Need to use initialization list   *****//
	} // default constructor

	csPDMEIF::csPDMEIF(const int& len, const int& Tf, const double& dt, const double& Vlb, const double& Vub, Eigen::VectorXd& ini_denV) :
		M(len - 1), den2(2 * M), A(2 * M, 2 * M), V(M + 1), den_store( M, int(Tf / dt) + 1), _den(len-1), ini_den(len-1), FR(int(Tf/dt)+1), b(2*M), V_(M), MeanV_arr(int(Tf/dt)+1), time(int(Tf/dt)+1)
	{
		den_store = Eigen::MatrixXd::Zero( M, int(Tf / dt) + 1);
		MeanV_arr = Eigen::VectorXd::Zero(int(Tf / dt) + 1);
		time = Eigen::VectorXd::LinSpaced(int(Tf / dt) + 1, 0, Tf);
		ini_den = ini_denV;
		_den = ini_denV;
		den_store.col(0) = ini_denV;
		den2 = Eigen::VectorXd::Zero(2 * M);
		b = den2;
		A = Eigen::MatrixXd::Zero(2 * M, 2 * M);
		V = Eigen::VectorXd::LinSpaced(len, Vlb, Vub);
		double dV = V(1) - V(0);
		V_ = Eigen::VectorXd::LinSpaced(len-1, Vlb+dV, Vub-dV);
		for (int i = 0; i < V.size();  i++)
		{
			if (abs(V(i) + 65.0) < 1 * pow(10, -6))
			{
				index_reset = 2 * i;
			}
		}
		FR = Eigen::VectorXd::Zero(int(1/dt*1000)+1);
		_T = Tf;
		_dt = dt;
		L = (-40.0 + 100.0) / M;
	} // constructor #1    



	void csPDMEIF::prob2x(const Eigen::MatrixXd& den, const int& len)
	{
		for (int i = 0; i < len; i++) {
			den2(2 * i) = den(i);
			den2(2 * i + 1) = den(i);

		}
		b = den2;
	}  // expand the PDF's vector

	/*Comment it (2018 paper version)*/
	//void csPDMEIF::timestep(const int& Tf_step, MeanConductance* syn)//, double& tau_s, double& C, double& EL, double& gL, double& L, double& K, double& V_AMPA, double& VT)
	//{		
	//	//int index_reset = 701-1;																								// done! (2020 / 10 / 29)，Need to change into customary location(index) of resetting voltage
	//	//Eigen::VectorXd  den_temp  = _den;  Eigen::VectorXd den2_temp = den2;					// PDF temporary object
	//	//Eigen::VectorXd Mat_norm = Eigen::VectorXd::Zero(Tf_step);								// Matrix Norm checker 
	//	//Eigen::VectorXd fire_rate = Eigen::VectorXd::Zero(Tf_step);									// firing rate vector
	//	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;		// E
	//	//SpMat A_sparse;
	//	std::vector<std::thread> threads;						// a thread vector, has some issue -> how to manage the threads
	//	std::vector<std::thread> threads_assign(4);		// assign 4 threads -> Need to constrain the number of thread in order to prevent the overhead of context switching.
	//	int num_thread = 4;
	//	int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
	//	std::vector<int> index_divide(2 * num_thread);
	//	std::vector<std::thread> threads_assign2(num_thread);
	//	for (int k = 0; k < num_thread; k++)
	//	{
	//		index_divide[2 * k] = 2 + k * div_num;
	//		index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
	//	}
	//	//std::vector<std::thread> threads_assign2(4);
	//	double Vkm = 0.0, Vkp = 0.0, V_mean = setMean();
	//	double flux = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0, D_flux = 0.0;
	//	Eigen::VectorXd Syn(3); Syn << syn[0].mu(0), 0.0, 0.0;												//  Mean value, only AMPA synapse at initial time
	//	Eigen::VectorXd Std_syn(3); Std_syn << syn[0].std_gs(0), 0.0, 0.0;									// Standard deviation
	//	flux = H0(V(M), Syn); D_flux = D(V(M ), Syn, Std_syn, V_mean);							// numerical flux term
	//	if (flux > 0)
	//	{
	//		FR(0) =flux*den2(2*M-1) - D_flux*(1/L)*(-den2(2*M-2)-den2(2*M-1));
	//	}
	//	else
	//	{
	//		FR(0) =  - D_flux * (1 / L)*(-den2(2 * M - 2) - den2(2 * M - 1));
	//	}

	//	for (int ti = 1; ti <= Tf_step; ti++)
	//	{
	//		double mu_gs = syn[0].mu(ti);			// only AMPA synapse
	//		double std_gsn = syn[0].std_gs(ti);
	//		/*
	//		{
	//			Timer  time1("Elapsed time for Case 1 = ");
	//			for (int row = 2; row < 2 * M - 2; row++)
	//			{
	//				threads.push_back(std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset));  // spawn threads

	//			}// Need some improvement because it is slower than sequential program below.
	//		}
	//		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));  // call join() method for each thread
	//		threads.clear();   // clear threads in <vector> because <vector> container has its size limit
	//		*/
	//		/*
	//		{
	//			Timer time3("Elapsed time for Case 2");
	//			for (int row = 2; row < 2 * M - 2; row+=4)
	//			{
	//				threads_assign[0] = std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
	//				threads_assign[1] = std::thread(&csPDMEIF::mat_assign, this, row+1, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
	//				threads_assign[2] = std::thread(&csPDMEIF::mat_assign, this, row+2, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
	//				threads_assign[3] = std::thread(&csPDMEIF::mat_assign, this, row+3, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
	//				std::for_each(threads_assign.begin(), threads_assign.end(), std::mem_fn(&std::thread::join));
	//			}
	//		}	// a little bit faster than above, but is still slower than Sequantial Programming
	//		*/
	//		/*
	//		int num_thread = 4;
	//		int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
	//		std::vector<int> index_divide(2 * num_thread);
	//		std::vector<std::thread> threads_assign2(num_thread);
	//		for (int k = 0; k < num_thread; k++)
	//		{
	//			index_divide[2 * k] = 2 + k * div_num;
	//			index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
	//		}
	//		*/
	//		//{
	//			//Timer time4("Elapsed time for Case 3 = ");
	//		for (int ii = 0; ii < num_thread; ii++)
	//		{
	//			threads_assign2[ii] = std::thread(&csPDMEIF::mat_assign_2, this, index_divide[2 * ii], index_divide[2 * ii + 1], std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
	//		}
	//		std::for_each(threads_assign2.begin(), threads_assign2.end(), std::mem_fn(&std::thread::join));
	//		//} // divide-data version, faster than sequential programming

	//		/*
	//		{
	//			Timer time2("Elapsed time for Serial programming");
	//			for (int row = 2; row < 2 * M - 2; row++)
	//			{
	//				mat_assign(row, mu_gs, std_gsn, ti, index_reset);
	//			}

	//		}	// Sequential Programming
	//		*/
	//		// 加入邊界條件
	//		// k = 1
	//		Vkm = V(0), Vkp = V(1); V_mean = setMean();
	//		//flux = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0, D_flux = 0.0;
	//		 Syn << mu_gs, 0.0, 0.0;												//  Mean value, only AMPA synapse
	//		 Std_syn << std_gsn, 0.0, 0.0;									// Standard deviation

	//		H0_int1 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (6.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkm, 3) * gL) / (3.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*mu_gs) / (2.0 * C*L) - (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) - (pow(K, 2) * Vkp*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
	//		H0_int2 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (3.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkm, 3) * gL) / (6.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) + (pow(K, 2) * Vkm*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
	//		D_int1 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 8.0 * V_AMPA*Vkm - 4.0 * V_AMPA*Vkp + 3.0 * pow(Vkm, 2) + 2.0 * Vkm*Vkp + pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
	//		D_int2 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 4.0 * V_AMPA*Vkm - 8.0 * V_AMPA*Vkp + pow(Vkm, 2) + 2.0 * Vkm*Vkp + 3.0 * pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
	//		flux = H0(V(1), Syn), D_flux = D(V(1), Syn, Std_syn, V_mean);   // numerical flux terms

	//		A(0, 0) = L / 2 + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1)));
	//		A(0, 1) = +_dt * (-((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2)));
	//		A(0, 2) = _dt * (-(pow(2 / L, 2)*D_int2));
	//		A(1, 0) = +_dt * (-((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*D_int1 + D_flux * (-1 / L)));
	//		if (flux > 0)
	//		{
	//			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + D_flux * (-1 / L)));
	//			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L)));
	//		}
	//		else
	//		{
	//			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + D_flux * (-1 / L)));
	//			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L) + (-flux)));
	//		}
	//		// k = [2*M-1, 2*M]
	//		Vkm = V(M - 1); Vkp = V(M); flux = H0(V(M - 1), Syn); D_flux = D(V(M - 1), Syn, Std_syn, V_mean);
	//		H0_int1 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (6.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkm, 3) * gL) / (3.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*mu_gs) / (2.0 * C*L) - (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) - (pow(K, 2) * Vkp*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
	//		H0_int2 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (3.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkm, 3) * gL) / (6.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) + (pow(K, 2) * Vkm*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
	//		D_int1 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 8.0 * V_AMPA*Vkm - 4.0 * V_AMPA*Vkp + 3.0 * pow(Vkm, 2) + 2.0 * Vkm*Vkp + pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
	//		D_int2 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 4.0 * V_AMPA*Vkm - 8.0 * V_AMPA*Vkp + pow(Vkm, 2) + 2.0 * Vkm*Vkp + 3.0 * pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
	//		if (flux > 0)
	//		{
	//			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
	//			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L) + flux));
	//			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + (2 / L)*(-D_flux)));
	//			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
	//		}
	//		else
	//		{
	//			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
	//			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L)));
	//			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + flux + (2 / L)*(-D_flux)));
	//			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
	//		}
	//		flux = H0(V(M), Syn); D_flux = D(V(M), Syn, Std_syn, V_mean);
	//		if (flux > 0)
	//		{
	//			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
	//			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
	//		}
	//		else
	//		{
	//			flux = 0;
	//			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
	//			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
	//		}
	//		// 開始作線代運算 -> 看是要使用sparse matrix or dense matrix linear algebra
	//		if (tref > 0 && (ti*_dt) >= tref)
	//		{
	//			b(index_reset) += _dt * FR(ti - int(tref / _dt))*(2 / L);
	//		}
	//		/*
	//		{
	//			Timer time11;
	//			//den2 = A.fullPivLu().solve((L / 2)*b);
	//			//den2 = A.householderQr().solve((L / 2)*b);	// Single threaded Inversion, using dense matrix algebra is pretty slow compare to Matlab
	//			Eigen::setNbThreads(8);
	//			den2 = A.partialPivLu().solve((L / 2)*b);			// Multithread Inversion (8 threads)
	//		}
	//		std::cout << "Absolute Error for dense LU  = " << (A*den2 - (L / 2)*b).norm() << "\n";
	//		*/
	//		//{
	//		//	Timer time12("Elapsed time for Solving LA (sparseLU) = ");
	//		
	//		SpMat A_sparse = A.sparseView();						// convert dense matrix into sparse matrix
	//		//Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
	//		// fill A and b;
	//		// Compute the ordering permutation vector from the structural pattern of A
	//		solver.analyzePattern(A_sparse);
	//		//std::cout<< (solver.info() == Eigen::Success) <<" : "<< ti<<"\n" ;
	//		// Compute the numerical factorization 
	//		solver.factorize(A_sparse);
	//		//Use the factors to solve the linear system 
	//		den2 = solver.solve((L / 2)*b);
	//		//}
	//		//std::cout << "Absolute Error for sparse LU = " << (A*den2 - (L / 2)*b).norm() << "\n";
	//		
	//		// 經過slope limiter, 並計算firing rate
	//		slope_limiter();
	//		
	//		flux = H0(V(M), Syn); D_flux = D(V(M), Syn, Std_syn, V_mean);
	//		if (flux > 0)
	//		{
	//			FR(ti) = flux * den2(2 * M-1) - D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
	//		}
	//		else
	//		{
	//			FR(ti) =  - D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
	//		}
	//		// downsampling the den2 vector -> _den
	//		for (int i = 0; i < M; i++)
	//		{
	//			_den(i) = (den2(2 * i) + den2(2 * i + 1)) / 2;
	//		}
	//		
	//		// 加入機率守恆式
	//		double sumD = _den.sum()*L;
	//		if ( sumD > 1.0)
	//		{
	//			den2 /=  sumD;
	//			_den /= sumD;
	//			std::cout << "den : "<<_den.sum()*L<<"\n" ;
	//			std::cout << "den2 : " << den2.sum()*L << "\n";
	//		}
	//		
	//		b = den2;					 // refresh (update)
	//	}		// Time Iteration
	//	//std::cout << A;
	//}// time-stepping algorithm using Backward Euler Method，only excitatory synapse (AMPA) -> modify into multiple synapse ，未完成待補, 2020/09/22
	//

/* csPDM -> New version (2020/11/08) with H(V, mu_gs)*/
//void csPDMEIF::timestep(const int& Tf_step, MeanConductance* syn)//, double& tau_s, double& C, double& EL, double& gL, double& L, double& K, double& V_AMPA, double& VT)
//{
//	//int index_reset = 701-1;																								// done! (2020 / 10 / 29)，Need to change into customary location(index) of resetting voltage
//	//Eigen::VectorXd  den_temp  = _den;  Eigen::VectorXd den2_temp = den2;					// PDF temporary object
//	//Eigen::VectorXd Mat_norm = Eigen::VectorXd::Zero(Tf_step);								// Matrix Norm checker 
//	//Eigen::VectorXd fire_rate = Eigen::VectorXd::Zero(Tf_step);									// firing rate vector
//	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;		// E
//	
//	std::vector<std::thread> threads;						// a thread vector, has some issue -> how to manage the threads
//	std::vector<std::thread> threads_assign(4);		// assign 4 threads -> Need to constrain the number of thread in order to prevent the overhead of context switching.
//	int num_thread = 4;
//	int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
//	std::vector<int> index_divide(2 * num_thread);
//	std::vector<std::thread> threads_assign2(num_thread);
//	for (int k = 0; k < num_thread; k++)
//	{
//		index_divide[2 * k] = 2 + k * div_num;
//		index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
//	}
//	
//	double Vkm = 0.0, Vkp = 0.0;
//	double flux = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0, D_flux = 0.0;
//	Eigen::VectorXd Syn(3); Syn << syn[0].mu(0), 0.0, 0.0;												//  Mean value, only AMPA synapse at initial time
//	Eigen::VectorXd Std_syn(3); Std_syn << syn[0].std_gs(0), 0.0, 0.0;									// Standard deviation
//	double t0 = 0.0;			// initial time point
//	flux = H0(t0, V(M), Syn, Std_syn); D_flux = D(t0, V(M), Syn, Std_syn);							// numerical flux term
//	if (flux > 0)
//	{
//		FR(0) = flux * den2(2 * M - 1) - D_flux * (1 / L)*(-den2(2 * M - 2) - den2(2 * M - 1));
//	}
//	else
//	{
//		FR(0) = -D_flux * (1 / L)*(-den2(2 * M - 2) - den2(2 * M - 1));
//	}
//	
//	for (int ti = 1; ti <= Tf_step; ti++)
//	{
//		double mu_gs = syn[0].mu(ti);			// only AMPA synapse
//		double std_gsn = syn[0].std_gs(ti);
//		double t = ti * _dt;
//		/*
//		{
//			Timer  time1("Elapsed time for Case 1 = ");
//			for (int row = 2; row < 2 * M - 2; row++)
//			{
//				threads.push_back(std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset));  // spawn threads
//
//			}// Need some improvement because it is slower than sequential program below.
//		}
//		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));  // call join() method for each thread
//		threads.clear();   // clear threads in <vector> because <vector> container has its size limit
//		*/
//		/*
//		{
//			Timer time3("Elapsed time for Case 2");
//			for (int row = 2; row < 2 * M - 2; row+=4)
//			{
//				threads_assign[0] = std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
//				threads_assign[1] = std::thread(&csPDMEIF::mat_assign, this, row+1, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
//				threads_assign[2] = std::thread(&csPDMEIF::mat_assign, this, row+2, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
//				threads_assign[3] = std::thread(&csPDMEIF::mat_assign, this, row+3, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
//				std::for_each(threads_assign.begin(), threads_assign.end(), std::mem_fn(&std::thread::join));
//			}
//		}	// a little bit faster than above, but is still slower than Sequantial Programming
//		*/
//		/*
//		int num_thread = 4;
//		int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
//		std::vector<int> index_divide(2 * num_thread);
//		std::vector<std::thread> threads_assign2(num_thread);
//		for (int k = 0; k < num_thread; k++)
//		{
//			index_divide[2 * k] = 2 + k * div_num;
//			index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
//		}
//		*/
//		//{
//			//Timer time4("Elapsed time for Case 3 = ");
//		for (int ii = 0; ii < num_thread; ii++)
//		{
//			threads_assign2[ii] = std::thread(&csPDMEIF::mat_assign_2, this, index_divide[2 * ii], index_divide[2 * ii + 1], std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
//		}
//		std::for_each(threads_assign2.begin(), threads_assign2.end(), std::mem_fn(&std::thread::join));
//		//} // divide-data version, faster than sequential programming
//
//		/*
//		{
//			Timer time2("Elapsed time for Serial programming");
//			for (int row = 2; row < 2 * M - 2; row++)
//			{
//				mat_assign(row, mu_gs, std_gsn, ti, index_reset);
//			}
//
//		}	// Sequential Programming
//		*/
//		// 加入邊界條件
//		// k = 1
//		Vkm = V(0), Vkp = V(1); 
//		Syn << mu_gs, 0.0, 0.0;												//  Mean value, only AMPA synapse
//		Std_syn << std_gsn, 0.0, 0.0;									// Standard deviation
//
//		H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
//		H0_int2 = (L / 2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
//		D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
//		D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
//		flux = H0(t, V(1), Syn, Std_syn), D_flux = D(t, V(1), Syn, Std_syn);   // numerical flux terms
//
//		A(0, 0) = L / 2 + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1)));
//		A(0, 1) = +_dt * (-((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2)));
//		A(0, 2) = _dt * (-(pow(2 / L, 2)*D_int2));
//		A(1, 0) = +_dt * (-((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*D_int1 + D_flux * (-1 / L)));
//		if (flux > 0)
//		{
//			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + D_flux * (-1 / L)));
//			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L)));
//		}
//		else
//		{
//			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + D_flux * (-1 / L)));
//			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L) + (-flux)));
//		}
//		// k = [2*M-1, 2*M]
//		Vkm = V(M - 1); Vkp = V(M); flux = H0(t, V(M - 1), Syn, Std_syn); D_flux = D(t, V(M - 1), Syn, Std_syn);
//		H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
//		H0_int2 = (L / 2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
//		D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
//		D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
//		if (flux > 0)
//		{
//			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
//			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L) + flux));
//			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + (2 / L)*(-D_flux)));
//			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
//		}
//		else
//		{
//			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
//			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L)));
//			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + flux + (2 / L)*(-D_flux)));
//			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
//		}
//		flux = H0(t, V(M), Syn, Std_syn); D_flux = D(t, V(M), Syn, Std_syn);
//		if (flux > 0)
//		{
//			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
//			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
//		}
//		else
//		{
//			flux = 0;
//			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
//			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
//		}
//		// 開始作線代運算 -> 看是要使用sparse matrix or dense matrix linear algebra
//		if (tref > 0 && (ti*_dt) >= tref)
//		{
//			b(index_reset) += _dt * FR(ti - int(tref / _dt))*(2 / L);
//		}
//		/*
//		{
//			Timer time11;
//			//den2 = A.fullPivLu().solve((L / 2)*b);
//			//den2 = A.householderQr().solve((L / 2)*b);	// Single threaded Inversion, using dense matrix algebra is pretty slow compare to Matlab
//			Eigen::setNbThreads(8);
//			den2 = A.partialPivLu().solve((L / 2)*b);			// Multithread Inversion (8 threads)
//		}
//		std::cout << "Absolute Error for dense LU  = " << (A*den2 - (L / 2)*b).norm() << "\n";
//		*/
//		//{
//		//	Timer time12("Elapsed time for Solving LA (sparseLU) = ");
//
//		SpMat A_sparse = A.sparseView();						// convert dense matrix into sparse matrix
//		//Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
//		// fill A and b;
//		// Compute the ordering permutation vector from the structural pattern of A
//		solver.analyzePattern(A_sparse);
//		//std::cout<< (solver.info() == Eigen::Success) <<" : "<< ti<<"\n" ;
//		// Compute the numerical factorization 
//		solver.factorize(A_sparse);
//		//Use the factors to solve the linear system 
//		den2 = solver.solve((L / 2)*b);
//		//}
//		//std::cout << "Absolute Error for sparse LU = " << (A*den2 - (L / 2)*b).norm() << "\n";
//
//		// 經過slope limiter, 並計算firing rate
//		slope_limiter();
//
//		flux = H0(t, V(M), Syn, Std_syn); D_flux = D(t, V(M), Syn, Std_syn);
//		if (flux > 0)
//		{
//			FR(ti) = flux * den2(2 * M - 1) - D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
//		}
//		else
//		{
//			FR(ti) = -D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
//		}
//		// downsampling the den2 vector -> _den
//		for (int i = 0; i < M; i++)
//		{
//			_den(i) = (den2(2 * i) + den2(2 * i + 1)) / 2;
//		}
//
//		// 加入機率守恆式
//		double sumD = _den.sum()*L;
//		if (sumD > 1.0)
//		{
//			den2 /= sumD;
//			_den /= sumD;
//			std::cout << "den : " << _den.sum()*L << "\n";
//			std::cout << "den2 : " << den2.sum()*L << "\n";
//		}
//		double V_mean = setMean();
//		b = den2;					 // refresh (update)
//
//	}		// Time Iteration
//	//std::cout << A;
//}// time-stepping algorithm using Backward Euler Method，only excitatory synapse (AMPA) -> modify into multiple synapse 
/* csPDM -> New version (2020/11/08) with H(<V>, mu_gs) -> (exponential neuron with constant approximation) or H(mu_gs) -> (LIF)*/
void csPDMEIF::timestep(const int& Tf_step, MeanConductance* syn)//, double& tau_s, double& C, double& EL, double& gL, double& L, double& K, double& V_AMPA, double& VT)
{
	//int index_reset = 701-1;																								// done! (2020 / 10 / 29)，Need to change into customary location(index) of resetting voltage
	//Eigen::VectorXd  den_temp  = _den;  Eigen::VectorXd den2_temp = den2;					// PDF temporary object
	//Eigen::VectorXd Mat_norm = Eigen::VectorXd::Zero(Tf_step);								// Matrix Norm checker 
	//Eigen::VectorXd fire_rate = Eigen::VectorXd::Zero(Tf_step);									// firing rate vector
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;		// E
	//marker_series series;																					// profiling object
	//marker_series msgSeries(_T("message series"));										// profiling object#2

	std::vector<std::thread> threads;						// a thread vector, has some issue -> how to manage the threads
	std::vector<std::thread> threads_assign(4);		// assign 4 threads -> Need to constrain the number of thread in order to prevent the overhead of context switching.
	int num_thread = 4;
	int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
	std::vector<int> index_divide(2 * num_thread);
	std::vector<std::thread> threads_assign2(num_thread);
	for (int k = 0; k < num_thread; k++)
	{
		index_divide[2 * k] = 2 + k * div_num;
		index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
	}

	double Vkm = 0.0, Vkp = 0.0;
	double flux = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0, D_flux = 0.0;
	Eigen::VectorXd Syn(3); Syn << syn[0].mu(0), syn[1].mu(0), syn[2].mu(0);												//  Mean value, only AMPA synapse at initial time
	Eigen::VectorXd Std_syn(3); Std_syn << syn[0].std_gs(0), syn[1].std_gs(0), syn[2].std_gs(0);									// Standard deviation
	double t0 = 0.0;			// initial time point
	flux = H0(t0, V(M), Syn, Std_syn); D_flux = D(t0, V(M), Syn, Std_syn);							// numerical flux term
	if (flux > 0)
	{
		FR(0) = flux * den2(2 * M - 1) - D_flux * (1 / L)*(-den2(2 * M - 2) - den2(2 * M - 1));
	}
	else
	{
		FR(0) = -D_flux * (1 / L)*(-den2(2 * M - 2) - den2(2 * M - 1));
	}
	MeanV_arr(0) = setMean();
	for (int ti = 1; ti <= Tf_step; ti++)
	{
		//span *flagSpan_2 = new span(msgSeries, 1, _T("span: Each time step"));
		//msgSeries.write_message(1, _T("Here is the message."));
		double mu_gs[3] = { syn[0].mu(ti), syn[1].mu(ti), syn[2].mu(ti) };			// only AMPA synapse
		double std_gsn[3] = { syn[0].std_gs(ti), syn[1].std_gs(ti), syn[2].std_gs(ti) };
		double t = ti * _dt;
		/*
		{
			Timer  time1("Elapsed time for Case 1 = ");
			for (int row = 2; row < 2 * M - 2; row++)
			{
				threads.push_back(std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset));  // spawn threads

			}// Need some improvement because it is slower than sequential program below.
		}
		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));  // call join() method for each thread
		threads.clear();   // clear threads in <vector> because <vector> container has its size limit
		*/
		/*
		{
			Timer time3("Elapsed time for Case 2");
			for (int row = 2; row < 2 * M - 2; row+=4)
			{
				threads_assign[0] = std::thread(&csPDMEIF::mat_assign, this, row, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
				threads_assign[1] = std::thread(&csPDMEIF::mat_assign, this, row+1, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
				threads_assign[2] = std::thread(&csPDMEIF::mat_assign, this, row+2, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
				threads_assign[3] = std::thread(&csPDMEIF::mat_assign, this, row+3, std::ref(mu_gs), std::ref(std_gsn), ti, index_reset);
				std::for_each(threads_assign.begin(), threads_assign.end(), std::mem_fn(&std::thread::join));
			}
		}	// a little bit faster than above, but is still slower than Sequantial Programming
		*/
		/*
		int num_thread = 4;
		int div_num = (2 * M - 4) / num_thread;				// *** 須注意是否整除 ***
		std::vector<int> index_divide(2 * num_thread);
		std::vector<std::thread> threads_assign2(num_thread);
		for (int k = 0; k < num_thread; k++)
		{
			index_divide[2 * k] = 2 + k * div_num;
			index_divide[2 * k + 1] = 2 + (k + 1)*div_num - 1;
		}
		*/
		//{
			//Timer time4("Elapsed time for Case 3 = ");
		//span *flagSpan = new span(series, 1, _T("span: Parallel Matrix Assignment"));
		//series.write_flag(_T("Here is the flag."));
		for (int ii = 0; ii < num_thread; ii++)
		{
			threads_assign2[ii] = std::thread(&csPDMEIF::mat_assign_3, this, index_divide[2 * ii], index_divide[2 * ii + 1], mu_gs, std_gsn, ti, index_reset);
		}
		std::for_each(threads_assign2.begin(), threads_assign2.end(), std::mem_fn(&std::thread::join));
	/*	delete flagSpan;*/
		//} // divide-data version, faster than sequential programming

		//double temm = syn[0].mu(ti);
		//double std_temm = syn[0].std_gs(ti);
		//span *flagSpan2 = new span(series, 1, _T("span: Sequential Matrix Assignment"));
		//series.write_flag(_T("Here is the flag."));
		////{
		//	//Timer time2("Elapsed time for Serial programming: ");
		//	for (int row = 2; row < 2 * M - 2; row++)
		//	{
		//		mat_assign(row, temm, std_temm, ti, index_reset);
		//	}
		//delete flagSpan2;
		//}	// Sequential Programming
		
		// 加入邊界條件
		// k = 1
		Vkm = V(0), Vkp = V(1);
		Syn << mu_gs[0], mu_gs[1], mu_gs[2];												//  Mean value, only AMPA synapse
		Std_syn << std_gsn[0], std_gsn[1], std_gsn[2];									// Standard deviation

		H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		H0_int2 = (L / 2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		flux = H0(t, V(1), Syn, Std_syn), D_flux = D(t, V(1), Syn, Std_syn);   // numerical flux terms

		A(0, 0) = L / 2 + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1)));
		A(0, 1) = +_dt * (-((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2)));
		A(0, 2) = _dt * (-(pow(2 / L, 2)*D_int2));
		A(1, 0) = +_dt * (-((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*D_int1 + D_flux * (-1 / L)));
		if (flux > 0)
		{
			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + D_flux * (-1 / L)));
			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L)));
		}
		else
		{
			A(1, 1) = L / 2 + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + D_flux * (-1 / L)));
			A(1, 2) = _dt * (-(pow(2 / L, 2)*(-D_int2) + D_flux * (2 / L) + (-flux)));
		}
		// k = [2*M-1, 2*M]
		Vkm = V(M - 1); Vkp = V(M); flux = H0(t, V(M - 1), Syn, Std_syn); D_flux = D(t, V(M - 1), Syn, Std_syn);
		H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		H0_int2 = (L / 2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		if (flux > 0)
		{
			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L) + flux));
			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + (2 / L)*(-D_flux)));
			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
		}
		else
		{
			A(2 * M - 2, 2 * M - 4) = _dt * (-(-D_flux * (-1 / L)));
			A(2 * M - 2, 2 * M - 3) = _dt * (-(-D_flux * (-1 / L)));
			A(2 * M - 2, 2 * M - 2) = (L / 2) + _dt * (-((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) + pow(2 / L, 2)*(-D_int1) + flux + (2 / L)*(-D_flux)));
			A(2 * M - 2, 2 * M - 1) = +_dt * (-(-H0_int2 * (1 / L) + pow(1 / L, 2)*(D_int1 - D_int2) + 0));
		}
		flux = H0(t, V(M), Syn, Std_syn); D_flux = D(t, V(M), Syn, Std_syn);
		if (flux > 0)
		{
			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
		}
		else
		{
			flux = 0;
			A(2 * M - 1, 2 * M - 2) = _dt * (-(H0_int1*(1 / L) + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)+(-1 / L)*D_flux));
			A(2 * M - 1, 2 * M - 1) = (L / 2) + _dt * (-((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) + (-flux) + (-1 / L)*D_flux));
		}
		// 開始作線代運算 -> 看是要使用sparse matrix or dense matrix linear algebra
		if (tref > 0 && (ti*_dt) >= tref)
		{
			b(index_reset) += _dt * FR(ti - int(tref / _dt))*(2 / L);
		}
		/*
		{
			Timer time11;
			//den2 = A.fullPivLu().solve((L / 2)*b);
			//den2 = A.householderQr().solve((L / 2)*b);	// Single threaded Inversion, using dense matrix algebra is pretty slow compare to Matlab
			Eigen::setNbThreads(8);
			den2 = A.partialPivLu().solve((L / 2)*b);			// Multithread Inversion (8 threads)
		}
		std::cout << "Absolute Error for dense LU  = " << (A*den2 - (L / 2)*b).norm() << "\n";
		*/
		//{
		//	Timer time12("Elapsed time for Solving LA (sparseLU) = ");
		//span *messageSpan = new span(series, 2, _T("span: Matrix conversion (dense to sparse) and SparseLU"));
		//series.write_flag(_T("Here is the message."));
		/*
		SpMat A_sparse = A.sparseView();						// convert dense matrix into sparse matrix
		//Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
		// fill A and b;
		// Compute the ordering permutation vector from the structural pattern of A
		solver.analyzePattern(A_sparse);
		//std::cout<< (solver.info() == Eigen::Success) <<" : "<< ti<<"\n" ;
		// Compute the numerical factorization 
		solver.factorize(A_sparse);
		//Use the factors to solve the linear system 
		den2 = solver.solve((L / 2)*b);
		*/
		/*Sequential LU*/
		Eigen::VectorXd  B = (L / 2.0)*b;
		EigenSolve(A, B);
		/*delete messageSpan;*/
		//}
		//std::cout << "Absolute Error for sparse LU = " << (A*den2 - (L / 2)*b).norm() << "\n";

		/*Parallel LU Factorization*/
		//int num2omp = 100;
		//Eigen::VectorXd  B = (L / 2.0)*b;
		//para_ilu(A, num2omp);
		//LU_solve(B);

		// 經過slope limiter, 並計算firing rate
		slope_limiter();

		flux = H0(t, V(M), Syn, Std_syn); D_flux = D(t, V(M), Syn, Std_syn);
		if (flux > 0)
		{
			FR(ti) = flux * den2(2 * M - 1) - D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
		}
		else
		{
			FR(ti) = -D_flux * (1 / L)*(-den2(2 * M - 1) - den2(2 * M - 2));
		}
		// downsampling the den2 vector -> _den
		for (int i = 0; i < M; i++)
		{
			_den(i) = (den2(2 * i) + den2(2 * i + 1)) / 2;
		}

		// 加入機率守恆式
		double sumD = _den.sum()*L;
		if (sumD > 1.0)
		{
			den2 /= sumD;
			_den /= sumD;
			std::cout << "den : " << _den.sum()*L << "\n";
			std::cout << "den2 : " << den2.sum()*L << "\n";
		}
		double V_mean = setMean();
		den_store.col(ti) = _den;
		MeanV_arr(ti) = V_mean;
		b = den2;					 // refresh (update)
		/*delete flagSpan_2;*/
	}		// Time Iteration
	//std::cout << A;
}// time-stepping algorithm using Backward Euler Method，only excitatory synapse (AMPA) -> modify into multiple synapse (done! 2020/11/11)

	void csPDMEIF::mat_assign( int row, double& mu_gs, double& std_gsn, const int ti, const int index_reset) 
	{
		int index_odd = 0, index_even=0;
		double t = ti * _dt;			// 下一刻時間點
		if (row % 2 == 1)		//  row index is 奇數，決定mesh index
		{
			 index_odd = (row - 1) / 2;
			 index_even = (row  + 1) / 2;
		}
		else								//  row index is 偶數
		{
			 index_odd = (row) / 2;
			 index_even = (row + 2) / 2;
		}
		double Vkm = 0.0, Vkp = 0.0, V_mean=0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0;

		Vkm = V(index_odd); Vkp = V(index_even);  
		double flux = 0.0;																								// Numerical flux term
		
		Eigen::VectorXd Syn(3); Syn << mu_gs, 0.0 , 0.0 ;												// only AMPA synapse
		Eigen::VectorXd Std_syn(3); Std_syn << std_gsn, 0.0, 0.0 ;
		//V_mean = setMean();
		//  Trapzoidal integration rule 
		H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		H0_int2 = (L / 2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
		D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
		if (row == index_reset)
		{
			flux = H0(t, Vkm, Syn, Std_syn);
			if (tref == 0)		// if no refractory
			{
				double H0_flux = H0(t, V(M), Syn, Std_syn), D_flux = D(t, V(M), Syn, Std_syn);
				if (H0(V(M), Syn) > 0)
				{
					A(index_reset, 2 * M - 1) = -_dt * (H0_flux + D_flux / L); 
					A(index_reset, 2 * M - 2) = -_dt* D_flux / L;
				}
				else
				{
					A(index_reset, 2 * M - 1) = -_dt * ( 0.0 + D_flux / L); 
					A(index_reset, 2 * M - 2) = -_dt * D_flux / L;
				}
			}

		}
		else
		{
			if (row % 2 == 0) 
			{
				flux = H0(t, V(index_odd), Syn, Std_syn);
			}	 // 奇數行
			else
			{
				flux = H0(t, V(index_even), Syn, Std_syn);
			}	// 偶數行

		}
		if (row % 2 == 0) 
		{
			double D_flux = D(t, V(index_odd), Syn, Std_syn);
			if (flux > 0)
			{
				A(row, row - 2) = _dt * -(D_flux / L);
				A(row, row - 1) = _dt * -(D_flux / L + flux);
				A(row, row) = L / 2 - _dt * ((1 / L)*(-H0_int1) + pow(1 / L , 2)*(D_int1 - D_int2) - pow(2 / L , 2)*(D_int1)-(2 / L)*D_flux);
				A(row, row + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L , 2)*(D_int1 - D_int2));
				A(row, row + 2) = -_dt * (pow(2 / L , 2)*D_int2);
			}
			else
			{
				A(row, row - 2) = _dt * -(D_flux / L);
				A(row, row - 1) = _dt * -(D_flux / L);
				A(row, row) = L / 2 - _dt * (flux + (1 / L)*(-H0_int1) + pow(1 / L , 2)*(D_int1 - D_int2) - pow(2 / L , 2)*(D_int1)-(2 / L)*D_flux);
				A(row, row + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L , 2)*(D_int1 - D_int2));
				A(row, row + 2) = -_dt * (pow(2 / L , 2)*D_int2);
			}
		}	// 奇數行
		else
		{
			double D_flux = D(t, V(index_even), Syn, Std_syn);
			if (flux > 0)
			{
				A(row, row - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L , 2)*(-D_int1 + D_int2) + pow(2 / L , 2)*(D_int1)-(1 / L)*D_flux);
				A(row, row) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L , 2)*(-D_int1 + D_int2) - flux - (1 / L)*D_flux);
				A(row, row + 1) = -_dt * (-pow(2 / L , 2)*(D_int2)+(2 / L)*D_flux);
			}
			else
			{
				A(row, row - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L , 2)*(-D_int1 + D_int2) + pow(2 / L , 2)*(D_int1)-(1 / L)*D_flux);
				A(row, row) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L , 2)*(-D_int1 + D_int2) - (1 / L)*D_flux);
				A(row, row + 1) = -_dt * (-flux - pow(2 / L , 2)*(D_int2)+(2 / L)*D_flux);
			}
		}	//偶數行

	}// Parallel matrix assignment w/o boundary mesh, version #1

	void csPDMEIF::mat_assign_2(int row, int row_end,double& mu_gs, double& std_gsn, const int ti, const int index_reset)
	{
		int index_odd = 0, index_even = 0;
		double Vkm = 0.0, Vkp = 0.0, V_mean = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0;
		Eigen::VectorXd Syn(3); Syn << mu_gs, 0.0, 0.0;  // only AMPA synapse
		Eigen::VectorXd Std_syn(3); Std_syn << std_gsn, 0.0, 0.0;
		for (int i=row; i <= row_end; i++)
		{
			if (i % 2 == 1)		//  row index is 奇數，決定mesh index
			{
				 index_odd = (i - 1) / 2;
				 index_even = (i + 1) / 2;
			}
			else								//  row index is 偶數
			{
				 index_odd = (i) / 2;
				 index_even = (i + 2) / 2;
			}
			

			Vkm = V(index_odd); Vkp = V(index_even);
			double flux = 0.0;																	// Numerical flux term

																					
			V_mean = setMean();
			H0_int1 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (6.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkm, 3) * gL) / (3.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (pow(Vkm, 2) * Vkp*mu_gs) / (2.0 * C*L) - (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) - (pow(K, 2) * Vkp*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
			H0_int2 = (EL*pow(Vkm, 2) * gL) / (2.0 * C*L) - (pow(Vkp, 3) * gL) / (3.0 * C*L) - (pow(Vkm, 3) * mu_gs) / (6.0 * C*L) - (pow(Vkp, 3) * mu_gs) / (3.0 * C*L) - (pow(Vkm, 3) * gL) / (6.0 * C*L) + (EL*pow(Vkp, 2) * gL) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * gL) / (2.0 * C*L) + (V_AMPA*pow(Vkm, 2) * mu_gs) / (2.0 * C*L) + (V_AMPA*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (Vkm*pow(Vkp, 2) * mu_gs) / (2.0 * C*L) + (K*gL*(K*exp(-(VT - Vkm) / K)*(K - Vkm) - K * exp(-(VT - Vkp) / K)*(K - Vkp))) / (C*L) - (EL*Vkm*Vkp*gL) / (C*L) - (V_AMPA*Vkm*Vkp*mu_gs) / (C*L) + (pow(K, 2) * Vkm*gL*(exp(-(VT - Vkm) / K) - exp(-(VT - Vkp) / K))) / (C*L);
			D_int1 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 8.0 * V_AMPA*Vkm - 4.0 * V_AMPA*Vkp + 3.0 * pow(Vkm, 2) + 2.0 * Vkm*Vkp + pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
			D_int2 = (pow(std_gsn, 2) * tau_AMPA*exp((VT - V_mean) / K)*pow((Vkm - Vkp), 2) * (6.0 * pow(V_AMPA, 2) - 4.0 * V_AMPA*Vkm - 8.0 * V_AMPA*Vkp + pow(Vkm, 2) + 2.0 * Vkm*Vkp + 3.0 * pow(Vkp, 2))) / (12.0 * C*L*(C*exp((VT - V_mean) / K) - gL * tau_AMPA + gL * tau_AMPA*exp((VT - V_mean) / K) + mu_gs * tau_AMPA*exp((VT - V_mean) / K)));
			if (i == index_reset)
			{
				flux = H0(V(index_odd), Syn);
				if (tref == 0)
				{
					double H0_flux = H0(V(M), Syn), D_flux = D(V(M), Syn, Std_syn, V_mean);
					if (H0(V(M), Syn) > 0)
					{
						A(index_reset, 2 * M - 1) = -_dt * (H0_flux + D_flux / L);
						A(index_reset, 2 * M - 2) = -_dt * D_flux / L;

					}
					else
					{
						A(index_reset, 2 * M - 1) = -_dt * (0.0 + D_flux / L);
						A(index_reset, 2 * M - 2) = -_dt * D_flux / L;
					}
				}

			}
			else
			{
				if (i % 2 == 0)
				{
					flux = H0(V(index_odd), Syn);
				}	 // 奇數行
				else
				{
					flux = H0(V(index_even), Syn);
				}	// 偶數行

			}
			if (i % 2 == 0)
			{
				double D_flux = D(V(index_odd), Syn, Std_syn, V_mean);
				if (flux > 0)
				{
					A(i, i - 2) = _dt * -(D_flux / L);
					A(i, i - 1) = _dt * -(D_flux / L + flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) - pow(2 / L, 2)*(D_int1)-(2 / L)*D_flux);
					A(i, i + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2));
					A(i, i + 2) = -_dt * (pow(2 / L, 2)*D_int2);
				}
				else
				{
					A(i, i - 2) = _dt * -(D_flux / L);
					A(i, i - 1) = _dt * -(D_flux / L);
					A(i, i) = L / 2 - _dt * (flux + (1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) - pow(2 / L, 2)*(D_int1)-(2 / L)*D_flux);
					A(i, i + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2));
					A(i, i + 2) = -_dt * (pow(2 / L, 2)*D_int2);
				}
			}	// 奇數行
			else
			{
				double D_flux = D(V(index_even), Syn, Std_syn, V_mean);
				if (flux > 0)
				{
					A(i, i - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)-(1 / L)*D_flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) - flux - (1 / L)*D_flux);
					A(i, i + 1) = -_dt * (-pow(2 / L, 2)*(D_int2)+(2 / L)*D_flux);
				}
				else
				{
					A(i, i - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)-(1 / L)*D_flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) - (1 / L)*D_flux);
					A(i, i + 1) = -_dt * (-flux - pow(2 / L, 2)*(D_int2)+(2 / L)*D_flux);
				}
			}	//偶數行
		}
	}// Parallel matrix assignment w/o boundary mesh, version #2

	void csPDMEIF::mat_assign_3(int row, int row_end, double* mu_gs, double* std_gsn, const int ti, const int index_reset)
	{
		int index_odd = 0, index_even = 0;
		double Vkm = 0.0, Vkp = 0.0, V_mean = 0.0, H0_int1 = 0.0, H0_int2 = 0.0, D_int1 = 0.0, D_int2 = 0.0;
		Eigen::VectorXd Syn(3); Syn << mu_gs[0], mu_gs[1], mu_gs[2];
		Eigen::VectorXd Std_syn(3); Std_syn << std_gsn[0], std_gsn[1], std_gsn[2];
		double t = ti * _dt;			// 下一刻時間點
		for (int i = row; i <= row_end; i++)
		{
			if (i % 2 == 1)		//  row index is 奇數，決定mesh index
			{
				index_odd = (i - 1) / 2;
				index_even = (i + 1) / 2;
			}
			else								//  row index is 偶數
			{
				index_odd = (i) / 2;
				index_even = (i + 2) / 2;
			}


			Vkm = V(index_odd); Vkp = V(index_even);
			double flux = 0.0;																	// Numerical flux term


			//  Trapzoidal integration rule 
			H0_int1 = (L / 2)*H0(t, Vkm, Syn, Std_syn)*phi1(Vkm,Vkp);
			H0_int2 = (L/2)*H0(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
			D_int1 = (L / 2)*D(t, Vkm, Syn, Std_syn)*phi1(Vkm, Vkp);
			D_int2 = (L / 2)*D(t, Vkp, Syn, Std_syn)*phi2(Vkp, Vkm);
			if (i == index_reset)
			{
				flux = H0(t, Vkm, Syn, Std_syn);
				if (tref == 0)
				{
					double H0_flux = H0(t, V(M), Syn, Std_syn), D_flux = D(t, V(M), Syn, Std_syn);
					if (H0(t, V(M), Syn, Std_syn) > 0)
					{
						A(index_reset, 2 * M - 1) = -_dt * (H0_flux + D_flux / L);
						A(index_reset, 2 * M - 2) = -_dt * D_flux / L;

					}
					else
					{
						A(index_reset, 2 * M - 1) = -_dt * (0.0 + D_flux / L);
						A(index_reset, 2 * M - 2) = -_dt * D_flux / L;
					}
				}

			}
			else
			{
				if (i % 2 == 0)
				{
					flux = H0(t, V(index_odd), Syn, Std_syn);
				}	 // 奇數行
				else
				{
					flux = H0(t, V(index_even), Syn, Std_syn);
				}	// 偶數行

			}
			if (i % 2 == 0)
			{
				double D_flux = D(t, V(index_odd), Syn, Std_syn);
				if (flux > 0)
				{
					A(i, i - 2) = _dt * -(D_flux / L);
					A(i, i - 1) = _dt * -(D_flux / L + flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) - pow(2 / L, 2)*(D_int1)-(2 / L)*D_flux);
					A(i, i + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2));
					A(i, i + 2) = -_dt * (pow(2 / L, 2)*D_int2);
				}
				else
				{
					A(i, i - 2) = _dt * -(D_flux / L);
					A(i, i - 1) = _dt * -(D_flux / L);
					A(i, i) = L / 2 - _dt * (flux + (1 / L)*(-H0_int1) + pow(1 / L, 2)*(D_int1 - D_int2) - pow(2 / L, 2)*(D_int1)-(2 / L)*D_flux);
					A(i, i + 1) = -_dt * ((1 / L)*(-H0_int2) + pow(1 / L, 2)*(D_int1 - D_int2));
					A(i, i + 2) = -_dt * (pow(2 / L, 2)*D_int2);
				}
			}	// 奇數行
			else
			{
				double D_flux = D(t, V(index_even), Syn, Std_syn);
				if (flux > 0)
				{
					A(i, i - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)-(1 / L)*D_flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) - flux - (1 / L)*D_flux);
					A(i, i + 1) = -_dt * (-pow(2 / L, 2)*(D_int2)+(2 / L)*D_flux);
				}
				else
				{
					A(i, i - 1) = -_dt * ((1 / L)*H0_int1 + pow(1 / L, 2)*(-D_int1 + D_int2) + pow(2 / L, 2)*(D_int1)-(1 / L)*D_flux);
					A(i, i) = L / 2 - _dt * ((1 / L)*H0_int2 + pow(1 / L, 2)*(-D_int1 + D_int2) - (1 / L)*D_flux);
					A(i, i + 1) = -_dt * (-flux - pow(2 / L, 2)*(D_int2)+(2 / L)*D_flux);
				}
			}	//偶數行
		}
	}// Parallel matrix assignment w/o boundary mesh, version #2 -> new version csPDM (2020/11/08)
	
	 /*2018 paper version*/
	double csPDMEIF::H0(double& V, const Eigen::VectorXd& mu_gs)
	{
		//const int num_syn = mu_gs.size();			// number of synaptic type
		//Eigen::VectorXd Es(num_syn);
		double y = (1 / C)*(-gL * (V - EL) + gL * K*exp((V - VT) / K) - (mu_gs(0)*(V - V_AMPA)+mu_gs(1)*(V-V_GABA_A)+mu_gs(2)*(V-V_GABA_B)));
		return y;
		
	}

	/*2018 paper version*/
	double csPDMEIF::D(double& V, Eigen::VectorXd& mu_gs, Eigen::VectorXd& std_gs, double& V_mean)
	{
		double tau_eff = C / (gL - gL * exp((V_mean - VT) / K) + mu_gs.sum());
		double y = pow((std_gs(0)*(V_AMPA-V)/C),2)*(tau_AMPA*tau_eff/(tau_AMPA+tau_eff)) +
						 pow((std_gs(1)*(V_GABA_A - V) / C), 2)*(tau_GABA_A*tau_eff / (tau_GABA_A + tau_eff)) + 
						 pow((std_gs(2)*(V_GABA_B - V) / C), 2)*(tau_GABA_B*tau_eff / (tau_GABA_B + tau_eff));
		return y;
	}
	
	/*new version*/
	double csPDMEIF::H0(double& t, double& V, Eigen::VectorXd& mu_gs, Eigen::VectorXd& std_gs)
	{
		//Eigen::VectorXd tau =  
		//double s_ampa = tau_AMPA / (tau_AMPA*H_(V, mu_gs) - 1);			// 會發散，所以comment it
		//double s_ampa = tau_AMPA / (tau_AMPA*H_(mu_gs) - 1);				// 上式簡化的版本
		//double t = 100000.0;			// 代入穩態值
		double s_ampa = tau_AMPA / (tau_AMPA*H_LIF(mu_gs) - 1);			// LIF版本_常數
		double f_mean = (1.0 / C)*(-gL * (V - EL) + gL * K*exp((V - VT) / K) - (mu_gs(0)*(V - V_AMPA) + mu_gs(1)*(V - V_GABA_A) + mu_gs(2)*(V - V_GABA_B)));
		double s_gabaa = tau_GABA_A / (tau_GABA_A*H_LIF(mu_gs) - 1);
		double s_gabab = tau_GABA_B / (tau_GABA_B*H_LIF(mu_gs) - 1);
		//double f_mean = (1.0 / C)*(-gL * (V - EL) - mu_gs(0)*(V - V_AMPA) - mu_gs(1)*(V - V_GABA_A) - mu_gs(2)*(V - V_GABA_B));		// LIF 版本
		//double a1 = (pow(std_gs(0),2)/C)*((2*(V-V_AMPA)*s_ampa+ pow(V_AMPA-V,2)*(gL/(C*K))*s_ampa*t-pow(s_ampa,2))*exp(t/s_ampa)+(-2*(V-V_AMPA)*s_ampa+pow(s_ampa,2)));
		//double a1 = pow(std_gs(0) / C, 2)*s_ampa * 2 * (V - V_AMPA)*(exp(t / s_ampa)-1);
		double a1 = pow(std_gs(0) / C, 2) * 2 * (V - V_AMPA)*s_ampa*(exp(t / s_ampa) - 1)+ pow(std_gs(1) / C, 2) * 2 * (V - V_GABA_A)*s_gabaa*(exp(t / s_gabaa) - 1)+ pow(std_gs(2) / C, 2) * 2 * (V - V_GABA_B)*s_gabab*(exp(t / s_gabab) - 1);		// LIF_AMPA synapse
		//return (f_mean - a1);
		return (f_mean + 0.0);
		//return f_mean;
		//return 0.01;							// constant convection -> use it for testing the accuracy of  numerical scheme 
	}	// csPDM -> new verrsiom' s Convection coefficient  -> only AMPA synapse
	
	/*new version*/
	double csPDMEIF::D(double& t, double& V, Eigen::VectorXd& mu_gs, Eigen::VectorXd& std_gs)
	{
		 //double s_ampa = tau_AMPA / (tau_AMPA*H_(V, mu_gs) - 1);		// 會發散，所以comment it
		//double s_ampa = tau_AMPA / (tau_AMPA*H_(mu_gs) - 1);				// 上式簡化版本
		//double t = 100000.0;			// 代入穩態值
		double s_ampa = tau_AMPA / (tau_AMPA*H_LIF(mu_gs) - 1);					// LIF版本_常數
		double s_gabaa = tau_GABA_A / (tau_GABA_A*H_LIF(mu_gs) - 1);			// LIF版本_常數
		double s_gabab = tau_GABA_B / (tau_GABA_B*H_LIF(mu_gs) - 1);			// LIF版本_常數
		double wnt = intensity / 2.0;				// white-noise input term 
		 //return (pow(std_gs(0)*(V_AMPA-V)/C,2)*s_ampa*(exp(t/s_ampa)-1));
		/*
			return (pow(std_gs(0)*(V_AMPA - V) / C, 2)*s_ampa*(exp(t / s_ampa) - 1)+ pow(std_gs(1)*(V_GABA_A - V) / C, 2)*s_gabaa*(exp(t / s_gabaa) - 1)
			+ pow(std_gs(2)*(V_GABA_B - V) / C, 2)*s_gabab*(exp(t / s_gabab) - 1) + wnt);
		*/
		return 2.79628;
		//return 0.0;				// only consider convection , use it for testing the accuracy of numerical scheme 
	}	// csPDM -> new version' s Diffusion coefficient -> only AMPA synapse


	double csPDMEIF::setMean()
	{
		double dV = V(1) - V(0);
		double sumD = _den.sum()*dV;
		if (sumD < 1.0)
		{
			return (_den.dot(V_))*dV + (1 - sumD)*-65.0;
		}
		else
		{
			return (_den.dot(V_))*dV;
		}
		
		//return (_den.dot(V))*dV;
	}

	void csPDMEIF::slope_limiter()
	{
		double x_mean = 0.0, x_meanb = 0.0, x_meana = 0.0, x1 = 0.0, x1_=0.0 ,  x2=0.0, x3 = 0.0;
		for (int i = 0; i < M; i++)
		{
			if (i == 0)
			{
				x_mean = (den2(0) + den2(1)) / 2;
				x_meanb = (den2(0) + den2(1)) / 2; 
				x_meana = (den2(2) + den2(3)) / 2;
			}
			else if (i == M-1)
			{
				x_mean = (den2(2 * M-2 ) + den2(2 * M - 1)) / 2;
				x_meanb = (den2(2 * M - 4) + den2(2 * M - 3)) / 2;
				x_meana = 0.0;
			}
			else
			{
				x_mean = (den2(2 * i) + den2(2 * i + 1)) / 2;
				x_meanb = (den2(2 * i - 2) + den2(2 * i - 1)) / 2;
				x_meana = (den2(2 * i + 2) + den2(2 * i + 3)) / 2;
			}
			x1 = x_mean - den2(2 * i) ;	x1_ = den2(2 * i + 1) - x_mean;
			x2 = x_mean - x_meanb; 
			x3 = x_meana - x_mean;
			den2(2*i) = x_mean - minimod(x1, x2, x3);
			den2(2 * i + 1) = x_mean + minimod(x1_, x2, x3);
		}
		
	}

	double csPDMEIF::minimod(double& x1, double& x2, double& x3)
	{
		//double s = 0.0;
		if (sgn(x1) == sgn(x2) && (sgn(x1) == sgn(x2)))
		{
			return sgn(x1)*fmin(fmin(abs(x1), abs(x2)), abs(x3));
		}
		else
		{
			return 0.0;
		}
	}
	
	void csPDMEIF::plot_data(std::string obj)
	{
		namespace plt = matplotlibcpp;
		if (obj == "_den")
		{
			//Eigen::VectorXd& mat = _den;
			//Eigen::VectorXd& V = V_;
			std::vector <double> vec(_den.data(), _den.data() + _den.rows()*_den.cols());			// copy Eigen Vector object to STL vector object
			std::vector <double> Voltage(V_.data(), V_.data() + V_.rows()*V_.cols());			// copy Eigen Vector object to STL vector object
			plt::figure(1);
			plt::plot(Voltage,vec);
			plt::xlabel("mV");
			plt::title("Final-time PDF");
			plt::show();
		}
		else if (obj == "ini_pdf")
		{
			std::vector <double> vec(ini_den.data(),ini_den.data() + ini_den.rows()*ini_den.cols());			// copy Eigen Vector object to STL vector object
			std::vector <double> Voltage(V_.data(), V_.data() + V_.rows()*V_.cols());			// copy Eigen Vector object to STL vector object
			plt::figure(1);
			plt::plot(Voltage, vec);
			plt::xlabel("mV");
			plt::title("Initial PDF");
			plt::show();
		}
		else if (obj == "FR")
		{
			//Eigen::VectorXd& mat = FR;
			std::vector <double> fr(FR.data(), FR.data() + FR.rows()*FR.cols());			// copy Eigen Vector object to STL vector object
			std::vector <double> t(time.data(), time.data() + time.rows()*time.cols());
			plt::figure(1);
			plt::plot(t,fr);
			plt::xlabel("time (ms)");
			plt::ylabel("kHz");
			plt::title("Firing Rates");
			plt::show();
		}
		else if (obj == "MV")
		{
			std::vector <double> mv(MeanV_arr.data(), MeanV_arr.data() + MeanV_arr.rows()*MeanV_arr.cols());			// copy Eigen Vector object to STL vector object
			std::vector <double> t(time.data(), time.data() + time.rows()*time.cols());
			plt::figure(1);
			plt::plot(t,mv);
			plt::xlabel("time (ms)");
			plt::ylabel("mV");
			plt::title("Mean Value");
			plt::show();

		}
		else if (obj == "b")
		{
			std::vector <double> b_(b.data(), b.data() + b.rows()*b.cols());
			plt::figure(1);
			plt::plot(b_);
			plt::show();
		}
		else if (obj == "error")
		{

		}


	}// 畫圖函式
	
	/*
		Parallel LU Factorization  + Eigen Sparse Solver

		a : Factorization matrix {a = LU} 
	*/
	void csPDMEIF::para_ilu(Eigen::MatrixXd& a, int& num2omp)
	{
		A_temp = a;
		/*Diagonal scaling for matix A*/
		//scale_matrix();
		/*Convert A_temp into sparse matrix*/
		A_sparse = A_temp.sparseView();
		/*Find non-zero element*/
		Eigen::MatrixXi non_0_index(A_sparse.outerSize()*A_sparse.innerSize(), 2);

		int iter = 0;
		for (int k = 0; k < A_sparse.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it)
			{

				non_0_index(iter, 0) = it.row();   // row index
				non_0_index(iter, 1) = it.col();   // col index (here it is equal to k)
				iter++;

			}
		}

		Lm = A_temp.triangularView<Eigen::UnitLower>();
		Um = A_temp.triangularView<Eigen::Upper>();
		/*
			Parallel ILU algorithm
			Ref & papers :
			[1] Fine-Grained Parallel Incomplete LU Factorization, 2015
		*/
		double time3 = omp_get_wtime();
		int Num_threads = omp_get_max_threads();
		double conv_val = 10.0;
		for (int sweep = 1; sweep<=3; sweep++)				// 此演算法的收斂速度不夠快，因此考量到實際用途，論文使用sweep<=3
		{
			//#pragma omp parallel for schedule(static) default(shared) num_threads(1)
			#pragma omp parallel if(a.rows()>num2omp) num_threads(Num_threads)
			{
				#pragma omp for schedule(static) 
				for (int ii = 0; ii < iter; ii++)
				{
					int i = non_0_index(ii, 0);
					int j = non_0_index(ii, 1);
					if (i > j)
					{
						if (j == 0)
						{
							Lm.coeffRef(i, j) = A_temp.coeffRef(i, j) / Um.coeffRef(0, 0);
						}
						else
						{
							Lm.coeffRef(i, j) = (A_temp.coeffRef(i, j) - (Lm.block(i, 0, 1, j)*Um.block(0, j, j, 1))(0, 0)) / Um.coeffRef(j, j);
						}
					}
					else if (i <= j && i > 0)
					{
						Um.coeffRef(i, j) = A_temp.coeffRef(i, j) - (Lm.block(i, 0, 1, i)*Um.block(0, j, i, 1))(0, 0);
					}
				}
			}
			//conv_val = (A_temp - Lm * Um).norm();
		} // sweep++

	}// parallel LU Factorization 

	void csPDMEIF::LU_solve(Eigen::VectorXd& b) 
	{
		L_sparse = Lm.sparseView();
		U_sparse = Um.sparseView();
		Eigen::VectorXd y = L_sparse.triangularView<Eigen::Lower>().solve(b);
		den2 = U_sparse.triangularView<Eigen::Upper>().solve(y);
	}

	void csPDMEIF::EigenSolve(Eigen::MatrixXd& a, Eigen::VectorXd& b)
	{

		Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
		SpMat A_sparse = a.sparseView();
		solver.analyzePattern(A_sparse);
		solver.factorize(A_sparse);
		den2 = solver.solve(b);
	}

	/*
		這是MeanConductance 的定義
	*/

	MeanConductance::MeanConductance() : _vs(100001), mu(100001), mu_sq(100001), std_gs(100001)
	{
		mu_0(0) = 0.0;  mu_0(1) = 0.0;
		_vs = Eigen::VectorXd::Zero(100001);
		mu = Eigen::VectorXd::Zero(100001);
		mu_sq = Eigen::VectorXd::Zero(100001);
		std_gs = Eigen::VectorXd::Zero(100001);
		_dt = 0.01;
		_T = 1000.0;
	}// default constructor

	MeanConductance::MeanConductance(const Eigen::VectorXd& vs, const Eigen::Vector2d& mu_ini, const Eigen::VectorXd& t) :
		_vs(vs), mu(mu_ini), mu_sq(vs.rows()), std_gs(vs.rows())
	{
		 const int len = vs.rows();
		_dt = t(1) - t(0);
		_T = t.maxCoeff();
		// Initialize the data  with zeros
		mu = Eigen::VectorXd::Zero(len);
		mu_sq = Eigen::VectorXd::Zero(len);
		std_gs = Eigen::VectorXd::Zero(len);

	} // constructor #1


	void MeanConductance::print()
	{
		std::cout << _vs << "\n";
	} // printing function for private data (user-defined)


	void MeanConductance::ode45(double& taus, int& cs, double& eta_s)
	{
		//	double w1 = mu_0(0);  double w2 = mu_0(1);
		//	mu(0) = w1; mu_sq(0) = w2; 
			//double K1, K1_, K2, K2_, K3, K3_, K4, K4_;
		double w[2] = { mu(0), mu_sq(0) };
		double temp[2] = { 0.0,0.0 };
		double* K1;
		double* K2;
		double* K3;
		double* K4;
		for (int k = 1; k < _vs.rows(); k++)
		{
			K1 = (vec_function(w, temp, taus, cs, eta_s, k - 1, 0));
			w[0] = mu(k - 1) + K1[0] / 2; w[1] = mu_sq(k - 1) + K1[1] / 2;
			K2 = (vec_function(w, temp, taus, cs, eta_s, k - 1, 1));
			w[0] = mu(k - 1) + K2[0] / 2; w[1] = mu_sq(k - 1) + K2[1] / 2;
			K3 = (vec_function(w, temp, taus, cs, eta_s, k - 1, 1));
			w[0] = mu(k - 1) + K3[0]; w[1] = mu_sq(k - 1) + K3[1];
			K4 = (vec_function(w, temp, taus, cs, eta_s, k - 1, 2));

			mu(k) = mu(k - 1) + _dt * (K1[0] + 2.0*K2[0] + 2.0*K3[0] + K4[0]) / 6.0;
			mu_sq(k) = mu_sq(k - 1) + _dt * (K1[1] + 2.0*K2[1] + 2.0*K3[1] + K4[1]) / 6.0;
			std_gs(k) = sqrt(mu_sq(k) - mu(k)*mu(k));

			w[0] = mu(k); w[1] = mu_sq(k);
		}

	}// Time-stepping using RK4

	double* MeanConductance::vec_function(double* w, double* temp, double& taus, int & cs, double& eta_s, const int i, const int flag)
	{
		switch (flag)
		{
		case 0:
			temp[0] = -1.0 / taus * w[0] + cs * _vs(i)*eta_s / taus;
			temp[1] = -2.0 / taus * w[1] + cs * _vs(i)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
			break;
		case 1:
			temp[0] = -1.0 / taus * w[0] + cs * ((_vs(i) + _vs(i + 1)) / 2)*eta_s / taus;				// linearly interpolate the  vs(t)
			temp[1] = -2.0 / taus * w[1] + cs * ((_vs(i) + _vs(i + 1)) / 2)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
			break;
		default:  // flag ==2
			temp[0] = -1.0 / taus * w[0] + cs * _vs(i + 1)*eta_s / taus;
			temp[1] = -2.0 / taus * w[1] + cs * _vs(i + 1)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
		}

		//if (flag == 0)
		//{
		//	temp[0] = -1.0 / taus * w[0] + cs * _vs(i)*eta_s / taus;
		//	temp[1] = -2.0 / taus * w[1] + cs * _vs(i)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
		//}
		//else if (flag == 1)
		//{
		//	temp[0] = -1.0 / taus * w[0] + cs * ((_vs(i)+_vs(i+1))/2)*eta_s / taus;				// linearly interpolate the  vs(t)
		//	temp[1] = -2.0 / taus * w[1] + cs * ((_vs(i) + _vs(i + 1)) / 2)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
		//}
		//else  // flag == 2
		//{
		//	temp[0] = -1.0 / taus * w[0] + cs * _vs(i+1)*eta_s / taus;
		//	temp[1] = -2.0 / taus * w[1] + cs * _vs(i+1)*(2 * eta_s / taus * w[0] + (eta_s / taus)*(eta_s / taus));
		//}
		return  temp;
	}		// user-defined vector function (state-space equation)

	FileWriter::FileWriter(std::string name):filename(name)
	{
			// void
	}

	void FileWriter::write_Data(csPDMEIF& A)
	{
		std::vector <double> mv(A.MeanV_arr.data(), A.MeanV_arr.data() + A.MeanV_arr.rows()*A.MeanV_arr.cols());
		std::vector <double> den(A._den.data(), A._den.data() + A._den.rows()*A._den.cols());
		std::vector <double> fr(A.FR.data(), A.FR.data() + A.FR.rows()*A.FR.cols());
		std::vector< std::vector<double> > den_mat;
		den_mat.resize(A.den_store.rows(), std::vector<double>(A.den_store.cols(), 0));
		for (int i = 0; i < den_mat.size(); i++)
			for (int j = 0; j < den_mat.front().size(); j++)
				den_mat[i][j] = A.den_store(i, j);

		std::ofstream file;
		file.open(filename);			// open the file (stream)
		std::ostream_iterator<double, char> output_iterator(file, ",");
		std::copy(mv.begin(), mv.end(), output_iterator);
		file << "\n";
		std::copy(den.begin(), den.end(), output_iterator);
		file  << "\n";
		std::copy(fr.begin(), fr.end(), output_iterator);
		file << "\n";
		for (int i = 0; i < den_mat.size(); i++)
		{
			for (int j = 0; j < den_mat.front().size(); j++)
			{
				file << den_mat[i][j] << ",";
			}
			file << "\n";
		}
		
		file.close();
	}

}