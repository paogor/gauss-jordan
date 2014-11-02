#ifndef __GAUSS_JORDAN3_HPP__ 
#define __GAUSS_JORDAN3_HPP__

template<typename T>
__global__ void gauss_jordan3(int n, T * AA){

  extern __shared__ T A[];

  int idy = threadIdx.y; // inverted to avoid warp divergence
  int idx = threadIdx.x;
  int blk = blockIdx.x;
  
  A[(idy*n)+idx] = AA[(n*n*blk)+(idy*n)+idx];

  __syncthreads();

  for(int i=0; i < n; ++i){

    T i_row = (( idx != n-1 ) ? A[(i*n)+idx+1] : 1)  / A[i*n];
    T y_row = (( idx != n-1 ) ? A[(idy*n)+idx+1] : 0 ) - A[idy*n]*i_row;

    __syncthreads();
    
    A[(idy*n)+idx] = ( idy != i ) ? y_row : i_row ;

    __syncthreads();

  }
  
  AA[(n*n*blk)+(idy*n)+idx] = A[(idy*n)+idx];


}
#endif
 
