#pragma once

#define EIGEN_DONT_PARALLELIZE

#include <D:\dev\csPDM_0820\dependencies\Eigen\Dense>
#include <D:\dev\csPDM_0820\dependencies\Eigen\Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double 
// class for Parallel Incomplete LU Factorization 
class ILU {
public:
	ILU();						// default constructor
	ILU(int, int&, int&);
	void test_omp();
	void seq_ilu();
	void para_ilu();
	void scale_matrix();			// 目前未使用
	void test_LU_acc();
	void LUsolve();
	void EigenSolve();
	void Diag_Inv(Eigen::MatrixXd &);
private:
	Eigen::MatrixXd A;
	SpMat A_sparse;
	Eigen::MatrixXd A_temp;			// temporary  matrix for storing elements of  matrix A
	Eigen::MatrixXd L;
	SpMat L_sparse;
	Eigen::MatrixXd U;
	SpMat U_sparse;
	Eigen::VectorXd b;
	Eigen::VectorXd X;
	Eigen::VectorXd X0;
	int rows;
	int cols;
	int num2omp;
	int Num_threads;
	double conv_val ;
};
