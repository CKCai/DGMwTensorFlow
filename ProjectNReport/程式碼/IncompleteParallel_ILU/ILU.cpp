#include "ILU.h"
#include <thread>
#include <iostream>
#include <omp.h>
#include <cmath>
#include "D:\dev\csPDM_0820\csPDM\Timer.h"
#include  "cvmarkersobj.h"
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double 
using namespace Concurrency::diagnostic;
ILU::ILU() :A(1200, 1200), L(1200,1200), U(1200,1200), A_temp(1200,1200), b(1200)
{
	A = Eigen::MatrixXd::Zero(1200, 1200);
	b = Eigen::VectorXd::Random(1200);
	X = Eigen::VectorXd::Zero(1200);
	X0 = Eigen::VectorXd::Random(1200);
	rows = A.rows();
	cols = A.cols();
	
	for (int i = 0; i < rows; i++)
	{
		
		for (int j = 0; j < cols; j++)
		{
			if (i == j || abs(i - j) <= 2)
				A(i, j) =10.0*Eigen::MatrixXd::Random(1,1)(0,0);
		}
	}
	A_temp = A;
	L = A;
	U = A;
}

ILU::ILU(int N, int& num2omp_, int& Num_threads_) :A(N, N), L(N, N), U(N, N), A_temp(N, N), b(N)
{
	A = Eigen::MatrixXd::Zero(N, N);
	b = Eigen::VectorXd::Random(N);
	X = Eigen::VectorXd::Zero(N);
	X0 = Eigen::VectorXd::Random(N);
	rows = A.rows();
	cols = A.cols();
	num2omp = num2omp_;
	Num_threads = Num_threads_;
	for (int i = 0; i < rows; i++)
	{

		for (int j = 0; j < cols; j++)
		{
			if (i == j || abs(i - j) <= 2)
				A(i, j) = 10.0*Eigen::MatrixXd::Random(1, 1)(0, 0);
		}
	}
	A_temp = A;
	L = A;
	U = A;
}

void ILU::test_omp()
{
	std::cout << "Before operation, A = \n" << A << std::endl;
#pragma omp parallel for schedule(static, 2)  default(shared) num_threads(2)
	for (int i = 0; i < 4; i++)
	{
		A(i, i) = i;
	}
	std::cout << "After operation, A = \n" << A << std::endl;
}

void ILU::seq_ilu()
{
	//std::cout << "Before operation, L = \n" << L << std::endl;
	
	for (int i = 1; i < rows ; i++)
	{
		for (int k = 0; k <= i - 1 && i >= k; k++)
		{
			if (i >= k)
			{
				L(i, k) /= L(k, k);
			}
			for (int j = k + 1; j < rows ; j++)
			{
				if (i >= j)
				{
					L(i, j) -= L(i, k)*L(k, j);
				}
			}
		}
	}

	A_temp = L; 
	L = A_temp.triangularView<Eigen::UnitLower>();
	/*
	std::cout << "After operation, A_temp = \n" << A_temp << std::endl;
	std::cout << "After operation, L = \n" << L << std::endl;
	std::cout << "Before operation, U = \n" << U << std::endl;
	*/
	for (int i = 1; i < rows; i++)
	{
		for (int k = 0; k <= i - 1; k++)
		{
			if (i <= k)
			{
				U(i, k) /= U(k, k);
			}
			for (int j = k + 1; j < rows ; j++)
			{
				if (i <= j)
				{
					U(i, j) -= U(i, k)*U(k, j);
				}
			}
		}
	}
	A_temp = U; 
	U = A_temp.triangularView<Eigen::Upper>();
	//std::cout << "A = \n" << A << std::endl;
	//std::cout << "L*U = \n" << L * U << std::endl;
	//std::cout << "A-L*U = \n" << A - L * U << std::endl;
	//std::cout << "After operation, L = \n" << L << std::endl;
	//std::cout << "After operation, U = \n" << U << std::endl;
	/*
	std::cout << "After operation, A_temp = \n" << A_temp << std::endl;
	std::cout << "After operation, U = \n" << U << std::endl;
	*/
} // Sequential ILU

  /*1200x1200 matrix 的執行時間約為 50 msec  in Release mode */
void ILU::para_ilu()
{
	marker_series series;
	/*Diagonal scaling for matix A*/
	//scale_matrix();
	//double time5 = omp_get_wtime();
	/*Convert A_temp into sparse matrix*/
	 A_sparse = A_temp.sparseView();
	/*Find non-zero element*/
	Eigen::MatrixXi non_0_index(A_sparse.outerSize()*A_sparse.innerSize(),2);
	//double time6 = omp_get_wtime();
	//std::cout << "Elapsed time of Initial Process using omp_get_wtime() = " << (time6 - time5)*1000.0 << " msec\n";
	int iter = 0; 
	double time1 = omp_get_wtime();
	for (int k = 0; k < A_sparse.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it)
		{

			non_0_index(iter, 0) = it.row();   // row index
			non_0_index(iter, 1) = it.col();   // col index (here it is equal to k)
			iter++;

		}
	}
	double time2 = omp_get_wtime();
	std::cout << "Elapsed time of finding nonzero using omp_get_wtime() = " << (time2 - time1)*1000.0 << " msec\n";
	//std::cout << "A_sparse (sparse matrix) = \n" << A_sparse << std::endl;
	/*Initial guess for the algorithm*/
	L = A_temp.triangularView<Eigen::UnitLower>();
	U = A_temp.triangularView<Eigen::Upper>();
	/*
	std::cout << "A (scaling) = \n" << A_temp << std::endl;
	std::cout << "L = \n" << L << std::endl;
	std::cout << "U = \n" << U << std::endl;
	*/

	//double conv_val = (A_temp - L * U).norm();
	//std::cout << "Initial error = "<<conv_val << std::endl;
	
	/*
	Parallel ILU algorithm
	Ref & papers : 
	[1] Fine-Grained Parallel Incomplete LU Factorization, 2015 
	*/
	double time3 = omp_get_wtime();
	//int Num_threads = omp_get_max_threads();
	span *flagSpan = new span(series, 1, _T("Span for Parallel LU"));
	series.write_flag(_T("Here is the flag."));
	conv_val = 10.0;
	for (int sweep = 1; sweep<=3; sweep++)				// 此演算法的收斂速度不夠快，因此考量到實際用途，論文使用sweep<=3
	{	
		//#pragma omp parallel for schedule(static) default(shared) num_threads(1)
		#pragma omp parallel if(rows>num2omp) num_threads(Num_threads)
		{
			//std::cout << "Number of threads = "<<omp_get_num_threads()<<std::endl;
			#pragma omp for schedule(static) 
			for (int ii = 0; ii < iter; ii++)
			{
				//	for (int jj = 0; jj < 2; jj++)
				//	{
				int i = non_0_index(ii, 0);
				int j = non_0_index(ii, 1);
				if (i > j )
				{
					if (j == 0)
					{
						//L(i, j) = A_temp(i, j) / U(0, 0);
						L.coeffRef(i, j) = A_temp.coeffRef(i, j) / U.coeffRef(0, 0);
					}
					else
					{
						//L(i, j) = (A_temp(i, j) - (L.block(i, 0, 1, j)*U.block(0, j, j, 1))(0, 0)) / U(j, j);
						L.coeffRef(i, j) = (A_temp.coeffRef(i, j) - (L.block(i, 0, 1, j)*U.block(0, j, j, 1))(0, 0)) / U.coeffRef(j, j);
					}
					//L(i, j) = (A_temp(i, j) - (L.block(i, 0, 1, j)*U.block(0, j, j, 1))(0, 0)) / U(j, j);
					//std::cout<<"U(j, j) = " << U(j, j) << std::endl;
				}
				else if (i <= j && i > 0)
				{

					//U(i, j) = A_temp(i, j) - (L.block(i, 0, 1, i)*U.block(0, j, i, 1))(0, 0);
					U.coeffRef(i, j) = A_temp.coeffRef(i, j) - (L.block(i, 0, 1, i)*U.block(0, j, i, 1))(0, 0);
					//std::cout << "U(j, j) = " << U(j, j) << std::endl;
				}
				//	}
			}
		}
		//conv_val = (A_temp - L * U).norm();
		//std::cout << "error norm = " << conv_val << std::endl;
		//std::cout << "L2 norm of  A-LU matrix"<<" at < "<<sweep <<" > th sweep =  " << conv_val << std::endl;
	}
	delete flagSpan;
	double time4 = omp_get_wtime();
	std::cout << "Elapsed time of LU factorization using omp_get_wtime() = " << (time4 - time3)*1000.0 << " msec\n";
	/*
	std::cout << "A = \n" << A << std::endl;
	std::cout << "L = \n" << L << std::endl;
	std::cout << "U = \n" << U << std::endl;
	std::cout << "L*U = \n" << L*U << std::endl;
	std::cout << "A-LU = \n" << A_temp - L * U << std::endl;
	*/
	//std::cout << "L.block = "<<L.block(0, 0, 1, 0) << std::endl;    // This is left for reminder that sparsity pattern need to be correctly identify

} // Parallel ILU algorithm

void ILU::scale_matrix()
{

	std::cout << omp_get_num_threads() << std::endl;
	A = A.cwiseAbs();									// 確保A矩陣的對角線元素是正值，此行作為範例使用，正式使用時須確保對角線元素為正值。
	Eigen::MatrixXd A_1 = A; 
	Eigen::MatrixXd A_2 = Eigen::MatrixXd::Zero(rows, rows);
	A_2 = A.triangularView<Eigen::UnitDiag>();
	//std::cout << "A_2  = \n" << A_2 << std::endl;
	#pragma omp parallel if(rows>100) num_threads(12)
	{
		std::cout << omp_get_num_threads() << std::endl;
		#pragma omp for schedule(static) 
		for (int i = 0; i < rows; i++)
		{
			
			A_2(i, i) = 1.0 / sqrt(A(i, i));
		}
	}
	A_temp = A_2 * A_1*A_2;					// diagonal scaling operation, this is the bottleneck of  this function (ILU::scale_matrix) -> 2021/01/04 修改
	int i = 0;
	/*
	std::cout << "A = \n" << A << std::endl;
	std::cout << "A (diagonal scaling) = \n" << A_temp << std::endl;
	std::cout << "A_2  = \n" << A_2 << std::endl;
	*/
} // Diagonal scaling of matrix A


void ILU::test_LU_acc()
{
	conv_val = (A_temp - L * U).norm();
	std::cout << "L2 norm of  A-LU matrix =  " << conv_val << std::endl;
}

/* 
Using (Sparse Matrix).coeffRef(i,j) to read and write the elements of sparse matrix
Ref:
[1] https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
[2] https://stackoverflow.com/questions/25100079/c-eigen-sparse-matrix-multiplication-much-slower-than-python-scipy-sparse
*/
void ILU::LUsolve()
{
	L_sparse = L.sparseView();
	U_sparse = U.sparseView();
	Eigen::VectorXd y = L_sparse.triangularView<Eigen::Lower>().solve(b);
	X = U_sparse.triangularView<Eigen::Upper>().solve(y);
	//std::cout <<"X = \n"<< X << std::endl;
	std::cout << "Norm for AX-b = \n" << (A * X - b).norm() << std::endl;
// Slower version of LU solve 
//	Eigen::MatrixXd Tri_DL = L.diagonal().asDiagonal();		// extract the diagonal elements from matrix L
//	Eigen::MatrixXd Tri_DU = U.diagonal().asDiagonal();		// extract the diagonal elements from matrix U
//	Diag_Inv(Tri_DL);																	// inverse of diagnal matrix DL
//	Diag_Inv(Tri_DU);																// inverse of diagnal matrix DU								
//	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(rows, rows);
//	Eigen::VectorXd temp = X0;													// Initial guess
//	Eigen::VectorXd c1 = Tri_DL * b;
//	Eigen::MatrixXd G1 = (I - Tri_DL*L);
//	Eigen::MatrixXd G2 = (I - Tri_DU * U);
//	double err_val = 10.0;
//	int Num_threads = omp_get_max_threads();
//	/*Solve Ax = LUx=Ly =b*/
//	for (int iter = 1; err_val > 1e-2; iter++)
//	{
//		#pragma omp parallel if(rows>100) num_threads(Num_threads)
//		{
//			#pragma omp for schedule(static) 
//			for (int i = 0; i < rows; i++)
//			{
//				double s = 0.0;
//				for (int j = 0; j <= i; j++)
//				{
//					s += G1(i, j)*temp(j);
//				}
//				X(i) = s + c1(i);
//			}
//			
//		}
//		err_val = ((temp - X).cwiseAbs().sum())/(temp.cwiseAbs().sum());
//		temp = X;									// update for next iteration
//		//std::cout << "Error norm = " << err_val<<std::endl;
//	}
//	/* Solve Ux = y */
//	Eigen::VectorXd c2 = Tri_DU * temp;					// In this case, temp = y
//	temp = X0;
//	err_val = 10.0;
//	for (int iter = 1; err_val > 1e-2; iter++)
//	{	
//		#pragma omp parallel if(rows>100) num_threads(Num_threads)
//		{
//			#pragma omp for schedule(static) 
//			for (int i = 0; i < rows; i++)
//			{
//				double s = 0.0;
//				int end = cols - 1;
//				for (int j = end; j >=i ; j--)
//				{
//					s += G2(i, j)*temp(j);
//				}
//				X(i) = s + c2(i);
//			}
//
//		}
//		err_val = ((temp - X).cwiseAbs().sum()) / (temp.cwiseAbs().sum());
//		temp = X;									// update for next iteration
//		//std::cout << "Error norm = " << err_val << std::endl;
//	}
////	std::cout << "D = \n" << Tri_DL << std::endl;


}

void ILU::Diag_Inv(Eigen::MatrixXd & a)
{
	int Num_threads = omp_get_max_threads();
	//Eigen::MatrixXd b = a;
	#pragma omp parallel if(rows>num2omp) num_threads(Num_threads)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < rows; i++)
		{
			double temp = a(i, i);
			a(i, i) = 1.0 / temp;
		}
	}
} // Calculate the inversion of matrix a


void ILU::EigenSolve()
{
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
	SpMat A_sparse = A.sparseView();
	double time3 = omp_get_wtime();
	solver.analyzePattern(A_sparse);
	solver.factorize(A_sparse);
	double time4 = omp_get_wtime();
	X = solver.solve(b);
	//double time6 = omp_get_wtime();
	std::cout << "Elapsed time of Eigen SparseLU factorization solve using omp_get_wtime() = " << (time4 - time3)*1000.0 << " msec\n";
	std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
	//std::cout << "X = \n" << X<<std::endl;
	std::cout << "Norm for AX-b = \n" << (A * X - b).norm() << std::endl;
}