#include <iostream>
#include <thread>
//# include "Parameters.h"
#include <Eigen\Dense>					// using Eigen for Linear Algebra
//#include "plot/matplotlibcpp.h"		// Python plotting library under Release Mode - x64 , 盡砞﹚把σAnalysis0911, error -> 2020/10/16
#include <vector>								// STL library
#include <cmath>
//#include <string>
#include "Timer.h" 
#include "csPDM.h"
#define pi 3.14159265359 
// include standard liibrary <chrono> to estimate the program's elapsed time (干)
using namespace std;
using Eigen::Matrix;
typedef Matrix<double, Eigen::Dynamic, 1> VectorM1;
typedef Matrix<double, 1200, 1> Vector2M1;
//namespace plt = matplotlibcpp;

int main(int argc, char** argv) {
	// Parameters setting
	using namespace param;
	/* Vlb = -100.0; Vc = -40.0; EL = -65.0; Vr = -65.0; VT = -50.0;  K = 2.0; 
	V_AMPA = 0.0; V_GABA_A = -80.0; V_GABA_B = Vlb ;
	dVs_AMPA = 1.0; dVs_GABA_A = -0.25; dVs_GABA_B = -0.25;
	C = 1.0; gL = 0.05; tref = 3.0; cs = 200; 
	tau_AMPA = 5.0 ; tau_GABA_A = 10.0 ; tau_GABA_B = 100.0;
	Gs_AMPA = -C*log(1 - (dVs_AMPA / (V_AMPA - EL))); Gs_GABA_A = -C*log(1 - (dVs_GABA_A / (V_GABA_A - EL))); Gs_GABA_B = -C*log(1 - (dVs_GABA_B / (V_GABA_B - EL)));
	*/
	/*
	if (argc > 1)
	{
		double amp = 0.0;
		amp = double(*argv[0]);
		std::cout << amp<<"\n";
	}
	*/
	// Mesh setting
	const int M = 600; const int len = M + 1;
	double L = (Vc - Vlb) / M;
	// Intial-Valued Probability Density function
	VectorM1 ini_V(len-1);
	{
		Timer Timer("Elapsed time for I.V. Density assignment = ");
		double std = 2.0; double mu = -60.0; double V0 = -60;
		for (int k = 0; k < M; k++) {
			double x_ = (Vlb +   L/2 + k*L);
			ini_V(k) = (1. / (std*sqrt(2.*pi)))*exp(-0.5*pow((x_ - mu) / std, 2));			// initial condition (gaussian pdf)
			// delta initial pdf
			/*if (abs(x_ - V0 - L / 2) < 1 * pow(10, -6))
			{
				ini_V(k) = 1.0 / L;
			}
			else
			{
				ini_V(k) = 0.0;
			}*/
			
		}
		//double initial_ro[M+1];    // ┪砛эΘvector or matrix object 穦, 干(08/25)
	} // adding a scope({}) to auto-delete the class Timer (automatically call the destructor) , then we can evaluate performance in this section  of code.
	//cout << ini_V << "\n";
	//cout << ini_V.maxCoeff()<<"\n";
	//cout << ini_V.size() << "\n";
																						/*Now run the simulation */
	cout << "Running the simulation! \n";
	double dt = 0.1;												// time step size
	int t_len = int(1000.0 / dt)+1;							// length of  time array
	csPDMEIF  G;													// declare class csPDM with default constructor
	csPDMEIF  G2(len, 1000, dt,Vlb,Vc, ini_V);		// declare class csPDM with  constructor #1
	G2.prob2x(ini_V, M);										// expand the initial PDF into 2 x dimension
	Eigen::VectorXd  time = Eigen::VectorXd::LinSpaced(t_len, 0.0 , 1000.0);		// [0 1000] msec
	Eigen::VectorXd  vs(t_len);
	Eigen::VectorXd vs2(t_len);
	Eigen::VectorXd vs3(t_len);
	for (int k = 0 ; k<t_len;k++) 
	{
		vs(k) = (0.0*sin(2.0 * pi*(1.0 / 100.0)*time(k)) + 8.0)*0.001;			// synaptic input rate for AMPA synapse (excitatory), unit: Hz
		vs2(k) = (0.0*sin(2.0 * pi*(3.0 / 100.0)*time(k)) + 4.0)*0.001;			// synaptic input rate for GABA_A synapse (Es = -80.0, inhibitory synapse) , unit: Hz
		vs3(k) = (0.0*sin(2.0 * pi*(2.0 / 100.0)*time(k)) + 3.0)*0.001;			// synaptic input rate (Es = -100.0, inhibitory synapse) , unit: Hz
	}
	Eigen::Vector2d  gs_initial;  gs_initial << 0, 0;
	cout << "size of gs_initial = "<<gs_initial.rows()<<"\n";
	cout << "size of vs(t) = " << vs.rows() << "\n";
	MeanConductance Synapse;												// declare class MeanConductance with default constructor
	MeanConductance Synapse_AMPA(vs, gs_initial, time);			// declare AMPA synapse (class MeanConductance with constructor #1)
	MeanConductance Synapse_GABA_A(vs2, gs_initial, time);			// declare GABA_A synapse (class MeanConductance with constructor #1)
	MeanConductance Synapse_GABA_B(vs3, gs_initial, time);			// declare GABA_B synapse (class MeanConductance with constructor #1)
	MeanConductance mat_syn[3] = { Synapse_AMPA, Synapse_GABA_A, Synapse_GABA_B };
	double mat_tau[3] = { tau_AMPA , tau_GABA_A , tau_GABA_B };
	double mat_Gs[3] = { Gs_AMPA , Gs_GABA_A , Gs_GABA_B };
	
	std::vector<std::thread> threads;   // create "thread-type " container
	{
		Timer Timer2("Elapsed time for Synapse ODEs = ");
		//Synapse_AMPA.ode45(tau_AMPA, cs, Gs_AMPA);				// AMPA synapse, solving mean-field equation of conductance
		//Synapse_GABA_A.ode45(tau_GABA_A, cs, Gs_GABA_A);
		//Synapse_GABA_B.ode45(tau_GABA_B, cs, Gs_GABA_B);

		for (unsigned i=0;i<3;i++)
		{
			//Timer Timer3;
			threads.push_back(std::thread(&MeanConductance::ode45, &mat_syn[i], std::ref(mat_tau[i]), std::ref(cs), std::ref(mat_Gs[i])));
		} // spawn threads
		std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));  // call join() method for each thread
		
	}	
	//Synapse = Synapse_AMPA;
	// ︽csPDM家览祘Α絏*****ゼЧΘ Time stepping algorithm for class csPDMEIF, 2020/ 09/ 21 *****
	//cout << G2.setMean() << endl;
	cout << "Number of  synaptic type= "<<sizeof(mat_syn) / sizeof(mat_syn[0]) <<" ("<<typeid(sizeof(mat_syn)/sizeof(mat_syn[0])).name() << ")\n";
	// *** Show the computed data (Need to access the private data, hence need some modification on the csPDM code )***
	//Eigen::VectorXd& mat = G2.getVector("_den");
	//vector <double> vec(mat.data(), mat.data() + mat.rows()*mat.cols());			// copy Eigen Vector object to STL vector object
	//plt::figure(1);
	//plt::plot(vec);
	//plt::show();

	int Tf_step = 1000;//t_len-1;  // 10;
	{
		Timer time("Total running time = ");
		G2.timestep(Tf_step, mat_syn);
	}
	
	cout << G2.setMean()<<"\n";
	G2.plot_data("_den");
	//G2.plot_data("FR");
	G2.plot_data("MV");
	//G2.plot_data("b");
	G2.plot_data("ini_pdf");
	
	//FileWriter FILE("data.txt");
	//FILE.write_Data(G2);
	
	cout << "Done ! \n";
	
	cin.get();
	return 0;

}