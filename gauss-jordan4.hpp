#ifndef __GAUSS_JORDAN4_HPP__ 
#define __GAUSS_JORDAN4_HPP__

#include"alg_if.hpp"

template<typename T>
__global__ void gauss_jordan4(int n, T * AA){

  extern __shared__ T A[];

  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int blk = blockIdx.x;
  
  A[(idy*n)+idx] = AA[(n*n*blk)+(idy*n)+idx];

  __syncthreads();

  for(int i=0; i < n; ++i){

    T i_row = alg_if( idx != n-1 , A[(i*n)+idx+1] , 1. ) / A[i*n];
    T y_row = alg_if( idx != n-1 , A[(idy*n)+idx+1] , .0 ) - A[idy*n]*i_row;
 
    __syncthreads();
    
    A[(idy*n)+idx] = alg_if( idy != i , y_row , i_row );

    __syncthreads();

  }
  
  AA[(n*n*blk)+(idy*n)+idx] = A[(idy*n)+idx];


}


#endif
 
