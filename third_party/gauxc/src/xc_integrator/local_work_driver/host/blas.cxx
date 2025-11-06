/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include "host/blas.hpp"
#include <type_traits>
#include <gauxc/exceptions.hpp>

#if BLAS_IS_LP64
  #define blas_int int32_t
#else
  #define blas_int int64_t
#endif

extern "C" {

//void dlacpy_( const char* UPLO, const int* M, const int* N, const double* A, 
//              const int* LDA, double* B, const int* LDB );
//void slacpy_( const char* UPLO, const int* M, const int* N, const float* A, 
//              const int* LDA, float* B, const int* LDB );

void dgemm_( const char* TA, const char* TB, const blas_int* M, const blas_int* N, 
             const blas_int* K, const double* ALPHA, const double* A, 
             const blas_int* LDA, const double* B, const blas_int* LDB, 
             const double* BETA, double* C, const blas_int* LDC );
void sgemm_( const char* TA, const char* TB, const blas_int* M, const blas_int* N, 
             const blas_int* K, const float* ALPHA, const float* A, 
             const blas_int* LDA, const float* B, const blas_int* LDB, 
             const float* BETA, float* C, const blas_int* LDC );

void dsyr2k_( const char* UPLO, const char* TRANS, const blas_int* N, const blas_int* K, 
              const double* ALPHA, const double* A, const blas_int* LDA, const double* B, 
              const blas_int* LDB, const double* BETA, double* C, const blas_int* LDC ); 
void ssyr2k_( const char* UPLO, const char* TRANS, const blas_int* N, const blas_int* K, 
              const float* ALPHA, const float* A, const blas_int* LDA, const float* B, 
              const blas_int* LDB, const float* BETA, float* C, const blas_int* LDC ); 

double ddot_( const blas_int* N, const double* X, const blas_int* INCX, const double* Y, 
              const blas_int* INCY );
float sdot_( const blas_int* N, const float* X, const blas_int* INCX, const float* Y, 
              const blas_int* INCY );


void daxpy_( const blas_int* N, const double* ALPHA, const double* A, const blas_int* INCX, 
             double* Y, const blas_int* INCY );
void saxpy_( const blas_int* N, const float* ALPHA, const float* A, const blas_int* INCX, 
             float* Y, const blas_int* INCY );

void dscal_( const blas_int* N, const double* ALPHA, const double* X, const blas_int* INCX );
void sscal_( const blas_int* N, const float* ALPHA, const float* X, const blas_int* INCX ); 
}

namespace GauXC::blas {

template <typename T>
void lacpy( char UPLO, int M, int N, const T* A, int LDA, T* B,
            int LDB ) {

/*
  if constexpr ( std::is_same_v<T,float> )
    slacpy_( &UPLO, &M, &N, A, &LDA, B, &LDB );
  else if constexpr ( std::is_same_v<T,double> )
    dlacpy_( &UPLO, &M, &N, A, &LDA, B, &LDB );
  else GAUXC_GENERIC_EXCEPTION("LACPY NYI");
*/

  if( UPLO == 'L' ) {

    for( int j = 0; j < N; ++j )
    for( int i = j; i < M; ++i )
      B[i + j*LDB] = A[i + j*LDA];

  } else if( UPLO == 'U' ) {

    for( int j = 0; j <  N; ++j )
    for( int i = 0; i <= j; ++i )
      B[i + j*LDB] = A[i + j*LDA];

  } else {

    for( int j = 0; j < N; ++j )
    for( int i = 0; i < M; ++i )
      B[i + j*LDB] = A[i + j*LDA];

  }

}

template void lacpy( char UPLO, int M, int N, const float* A, int LDA, 
                     float* B, int LDB );
template void lacpy( char UPLO, int M, int N, const double* A, int LDA, 
                     double* B, int LDB );









template <typename T>
void gemm( char TA, char TB, int _M, int _N, int _K, T ALPHA, 
           const T* A, int _LDA, const T* B, int _LDB, T BETA,
           T* C, int _LDC ) {

  blas_int M   = _M;
  blas_int N   = _N;
  blas_int K   = _K;
  blas_int LDA = _LDA;
  blas_int LDB = _LDB;
  blas_int LDC = _LDC;

  if constexpr ( std::is_same_v<T,float> )
    sgemm_( &TA, &TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC );
  else if constexpr ( std::is_same_v<T,double> )
    dgemm_( &TA, &TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC );
  else GAUXC_GENERIC_EXCEPTION("GEMM NYI");


}
template
void gemm( char floatA, char floatB, int M, int N, int K, float ALPHA, 
           const float* A, int LDA, const float* B, int LDB, float BETA,
           float* C, int LDC );
template
void gemm( char doubleA, char doubleB, int M, int N, int K, double ALPHA, 
           const double* A, int LDA, const double* B, int LDB, double BETA,
           double* C, int LDC );







template <typename T>
void syr2k( char UPLO, char TRANS, int _N, int _K, T ALPHA,
            const T* A, int _LDA, const T* B, int _LDB, T BETA, 
            T* C, int _LDC ) {

  blas_int N   = _N;
  blas_int K   = _K;
  blas_int LDA = _LDA;
  blas_int LDB = _LDB;
  blas_int LDC = _LDC;

  if constexpr ( std::is_same_v<T,float> )
    ssyr2k_( &UPLO, &TRANS, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC );
  else if constexpr ( std::is_same_v<T,double> )
    dsyr2k_( &UPLO, &TRANS, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC );
  else GAUXC_GENERIC_EXCEPTION("SYR2K NYI");


}

template
void syr2k( char UPLO, char floatRANS, int N, int K, float ALPHA,
            const float* A, int LDA, const float* B, int LDB, float BETA, 
            float* C, int LDC );
template
void syr2k( char UPLO, char doubleRANS, int N, int K, double ALPHA,
            const double* A, int LDA, const double* B, int LDB, double BETA, 
            double* C, int LDC );
            






template <typename T>
T dot( int _N, const T* X, int _INCX, const T* Y, int _INCY ) {

  blas_int N    = _N;
  blas_int INCX = _INCX;
  blas_int INCY = _INCY;

  if constexpr ( std::is_same_v<T,float> )
    return sdot_(&N, X, &INCX, Y, &INCY);
  else if constexpr ( std::is_same_v<T,double> )
    return ddot_(&N, X, &INCX, Y, &INCY);
  else GAUXC_GENERIC_EXCEPTION("DOT NYI");

  return 0.;
}

template
float dot( int N, const float* X, int INCX, const float* Y, int INCY );
template
double dot( int N, const double* X, int INCX, const double* Y, int INCY );






template <typename T>
void axpy( int _N, T ALPHA, const T* X, int _INCX, T* Y, int _INCY ) {

  blas_int N    = _N;
  blas_int INCX = _INCX;
  blas_int INCY = _INCY;

  if constexpr ( std::is_same_v<T,float> )
    saxpy_(&N, &ALPHA, X, &INCX, Y, &INCY );
  else if constexpr ( std::is_same_v<T,double> )
    daxpy_(&N, &ALPHA, X, &INCX, Y, &INCY );
  else GAUXC_GENERIC_EXCEPTION("AXPY NYI");

}

template
void axpy( int N, float ALPHA, const float* A, int INCX, float* Y, 
           int INCY );
template
void axpy( int N, double ALPHA, const double* A, int INCX, double* Y, 
           int INCY );
            





template <typename T>
void scal( int _N, T ALPHA, T* X, int _INCX ) {

  blas_int N    = _N;
  blas_int INCX = _INCX;

  if constexpr ( std::is_same_v<T,float> )
    sscal_(&N, &ALPHA, X, &INCX );
  else if constexpr ( std::is_same_v<T,double> )
    dscal_(&N, &ALPHA, X, &INCX );
  else GAUXC_GENERIC_EXCEPTION("SCAL NYI");

}

template
void scal( int N, float ALPHA, float* X, int INCX ); 
template
void scal( int N, double ALPHA, double* X, int INCX );

}


