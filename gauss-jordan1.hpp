#ifndef __GAUSS_JORDAN1_HPP__ 
#define __GAUSS_JORDAN1_HPP__

template <typename T>
__global__ void gauss_jordan1(int n, T * AA)
{

  extern __shared__ T A[];

  int idx = threadIdx.x;
  int blk = blockIdx.x;
  
  for(int i=0; i<n; ++i)
    A[(i*n)+idx] = AA[(n*n*blk)+(i*n)+idx];

  __syncthreads();

  for(int i=0; i < n; ++i)
  { 
    T a = ((idx != (n-1)) ? A[(i*n)+idx+1] : 1) / A[i*n];

    __syncthreads();
    
    A[(i*n)+idx] = a;
    
    __syncthreads();

    for(int j=0; j<n; ++j)
      if(j != i)
      {
        a = ((idx != (n-1)) ? A[(j*n)+idx+1] : 0) - A[j*n]*A[(i*n)+idx]; 

        __syncthreads();
    
        A[(j*n)+idx] = a;
      }

    __syncthreads();
  }
  
  for(int i=0; i<n; ++i)
    AA[(n*n*blk)+(i*n)+idx] = A[(i*n)+idx];

}

#endif
 
