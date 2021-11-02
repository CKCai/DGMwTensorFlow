#pragma once		// This line has already define the header file 
#define EIGEN_DONT_PARALLELIZE

#include <Eigen\Dense>
#include <Eigen\Sparse>
#include <string>

// 這是標頭檔
namespace param {
	extern double Vlb, Vc, EL, Vr, VT, K, C, C2,gL; extern int tref;								// single neuron parameters
	extern double V_AMPA, V_GABA_A, V_GABA_B;						// Synapse Parameters
	extern double dVs_AMPA, dVs_GABA_A, dVs_GABA_B;				// Voltage jump
	extern double Gs_AMPA, Gs_GABA_A, Gs_GABA_B;					// Conductance jump
	extern int cs;																							// number of pre-synapse
	extern double tau_AMPA, tau_GABA_A, tau_GABA_B;					// time constant for synapses
	extern double I, intensity;																		// std of external current input & noise intensity
	// Mean conductance class
	class MeanConductance {
	public:
		friend class csPDMEIF;						// 宣告csPDMEIF為夥伴類別，表示csPDMEIF可以存取MeanConductance的私有成員
		MeanConductance();							// default constructor 
		MeanConductance(const Eigen::VectorXd&, const Eigen::Vector2d&, const Eigen::VectorXd&);  // constructor #1, specify the initial state by users
		void print();										// member function #1
		void ode45(double&, int&, double&);																							// ode45 for solving state-space equation
		double* vec_function(double*, double*, double&, int&, double&, const int, const int);			// state-space equation
	private:
		Eigen::Vector2d mu_0;
		Eigen::VectorXd _vs;
		Eigen::VectorXd mu;
		Eigen::VectorXd mu_sq;
		Eigen::VectorXd std_gs;
		double _T;
		double _dt;
	}; // calculating the statistical property of conductance using ode45

	// csPDM for EIF neuron
	class csPDMEIF {

	public: // 未完待補 -> 加入多種synapse後的H0/D/H0_int/D_int的推導及程式碼，2020/10/31
		friend class FileWriter;																	//  acessing and storing (private) data member  
		void prob2x(const Eigen::MatrixXd&, const int&);			 // member function #1
		csPDMEIF(const int&, const int&, const double&, const double&, const double&, Eigen::VectorXd&);	 // constructor #1
		csPDMEIF();																			// defaullt constructor

		void timestep(const int&, MeanConductance*);//, double&, double&, double&, double&, double&, double&, double&, double&);						// time-stepping algorithm using Backward Euler Method
		double H0(double&, const Eigen::VectorXd&);																					 // H0 function for excitatory synapse
		double D(double&, Eigen::VectorXd&, Eigen::VectorXd&, double&);			// D function for excitatory synapse
		double setMean();																	// set mean membrane voltage, unit: mV
		double f_rate();																		// firing rate function，未完成待補(2020/9/22)，算完PDF再計算firing rate
		void mat_assign(int  , double&, double&, const int , const int );
		void mat_assign_2(int, int , double&, double&, const int, const int);
		void mat_assign_3(int, int, double*, double*, const int, const int);		// parallel version for new version csPDM in contrast with mat_assign_2
		void slope_limiter();																// Slope Limiter
		double minimod(double&,double&,double&);					// minimod function
		Eigen::VectorXd& getVector(std::string temp);					// getter for Eigen Vector object，未完待補(2020 / 09 / 24)，方案一: 使用if else 比對string
		Eigen::MatrixXd& getMatrix(std::string temp_mat);			// getter for Eigen Matrix object，未完待補(2020 / 09 / 24)，方案二: 使用STL, 參考資料: https://www.codeguru.com/cpp/cpp/cpp_mfc/article.php/c4067/Switch-on-Strings-in-C.htm
		void plot_data(std::string );												// plotting function for private data，未完待補(2020/ 11/ 04)
		inline int sgn(double& x1) { 
			if (x1 > pow(10,-8)) 
			{ 
				return 1; 
			} 
			else if (x1< -pow(10,-8)) 
			{
				return -1;
			}
			else
			{
				return 0;
			}
		}
		// New csPDM's fumctions
		inline double H_(double& V, Eigen::VectorXd& mu_gs)		// H(V, mean_gs) for exponential neuron (constant)
		{ 
			double V_mean = setMean();
			return (1.0 / C)*(-gL + gL * exp((V - VT) / K) - mu_gs.sum()); 
		}		

		inline double H_(Eigen::VectorXd& mu_gs)							// H(<V>, mean_gs) for exponential neuron (constant)
		{
			double V_mean = setMean();
			return (1.0 / C)*(-gL + gL * exp((V_mean - VT) / K) - mu_gs.sum());
		}

		inline double H_LIF(Eigen::VectorXd& mu_gs)					// H(mean_gs) for LIF neuron (constant)
		{
			return (1 / C)*(-gL - mu_gs.sum());
		}

		double H0(double&, double&, Eigen::VectorXd&, Eigen::VectorXd&);   // H0(t, V, mean_gs, std_gs)  ->  convection coefficient
		double D(double&, double&, Eigen::VectorXd&, Eigen::VectorXd&);		 // D(t, V, mean_gs, std_gs) -> diffusion coefficient
		inline double phi1(double& V, double& Vkp) { return (Vkp - V) / L; };	 // phi1(V,Vkp)
		inline double phi2(double& V, double& Vkm) { return (V - Vkm) / L; }; // phi2(V,Vkm)

		/* Parallel LU Factorization algorithm + Eigen Sparse solver */
		void para_ilu(Eigen::MatrixXd&, int&);
		void LU_solve(Eigen::VectorXd&);
		void EigenSolve(Eigen::MatrixXd& , Eigen::VectorXd&);										// Comparison (對照組)
	private:
		int M;
		int index_reset;
		double L;
		double _T;   // final time steps , final time = _T*_dt , unit: 1
		double _dt;  // time step size, unit: msec
		Eigen::VectorXd FR;				// firing rate vector
		Eigen::VectorXd _den;			//  Original PDF (1xM)
		Eigen::VectorXd den2;			// 2 x PDF (1x(2M))
		Eigen::VectorXd b; 
		Eigen::MatrixXd den_store;  // PDF storing matrix  (M x time steps )
		Eigen::MatrixXd A;			// Time-stepping matrix, 可能要改成sparse-Matrix, 2020 / 09 / 24
		Eigen::VectorXd V;			// Voltage vector for calculating mass matrix : 1x(M+1)
		Eigen::VectorXd V_;			// Voltage vector for evaluating mean voltage : 1xM
		Eigen::VectorXd MeanV_arr;// mean voltage array
		Eigen::VectorXd time;
		Eigen::VectorXd ini_den;		// initial pdf (1xM)

		/* The parameters in Parallel LU Factorization*/
		Eigen::SparseMatrix<double> A_sparse;							// Sparse matrix A
		Eigen::MatrixXd A_temp;													// temporary  matrix for storing elements of  matrix A
		Eigen::MatrixXd Lm;
		Eigen::SparseMatrix<double> L_sparse;
		Eigen::MatrixXd Um;
		Eigen::SparseMatrix<double> U_sparse;

	};

	/*
		函數實作(函數定義)在csPDM.cpp
	*/

	class FileWriter {
	public:
		FileWriter(std::string);
		void write_Data(csPDMEIF&);
	private:
		std::string  filename;				// 檔案名稱，包含.txt檔名
	};

	
}