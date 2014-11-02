#ifndef __GAUSS_JORDAN6_HPP__ 
#define __GAUSS_JORDAN6_HPP__

template<typename T>
__global__ void gauss_jordan6(int n, T * AA){

  extern __shared__ T A[];

  int idy = threadIdx.x; // inverted to avoid warp divergence
  int idx = threadIdx.y;
  int blk = blockIdx.x;
  
  A[(idy*n)+idy] = AA[(n*n*blk)+(idy*n)+idx];

  __syncthreads();

  for(int i=0; i < n; ++i){

    T i_row = (( idy != n-1 ) ? A[(i*n)+idy+1] : 1)  / A[i*n];
    T y_row = (( idy != n-1 ) ? A[(idx*n)+idy+1] : 0 ) - A[idx*n]*i_row;

    __syncthreads();
    
    A[(idx*n)+idy] = ( idx != i ) ? y_row : i_row ;

    __syncthreads();

  }
  
  AA[(n*n*blk)+(idx*n)+idy] = A[(idy*n)+idx];


}
#endif
 
