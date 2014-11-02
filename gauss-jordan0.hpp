#ifndef __GAUSS_JORDAN0_HPP__ 
#define __GAUSS_JORDAN0_HPP__

// ERROR dont use it 


template <typename T>
__global__ void gauss_jordan0(int n, T * AA){

  extern __shared__ T A[];

  int idx = threadIdx.x;
  int blk = blockIdx.x;
  
  for(int i=0; i<n; ++i)
    A[(i*n)+idx] = AA[(n*n*blk)+(i*n)+idx];

  __syncthreads();

  for(int i=0; i < n; ++i){

    T a;

    if( idx != (n-1) )
      a = A[(i*n)+idx+1]/A[i*n]; // ERROR ERROR acces OK
      else a = 1/A[i*n];

    __syncthreads();

    A[(i*n)+idx] = a;

    __syncthreads();

    for(int j=0; j<n; ++j)
    if(j != i){   
   
      if( idx != (n-1) )
        a = A[(j*n)+idx+1] - A[j*n]* A[(i*n)+idx]; 
        else a = - A[j*n]* A[(i*n)+idx];

      __syncthreads();
    
      A[(j*n)+idx] = a;

    } 

    __syncthreads();

  }
  
  for(int i=0; i<n; ++i)
    AA[(n*n*blk)+(i*n)+idx] = A[(i*n)+idx];

}

#endif

