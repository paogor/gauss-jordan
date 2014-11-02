#ifndef __GAUSS_JORDANCPU_HPP__
#define __GAUSS_JORDANCPU_HPP__

template<typename T>
void gauss_jordanCPU(int n, T * A ){
  
  for(int i = 0; i<n; ++i){

    T a = A[(i*n)];
    
    for(int j = 0; j<n-1; ++j)
      A[(i*n)+j] = A[(i*n)+j+1]/a;
    A[(i*n)+n-1] = 1/a;

    for(int k = 0; k<n; ++k)
      if( k != i ){

        T b = A[(k*n)];
        
        for(int j = 0; j<n-1; ++j)
          A[(k*n)+j] = A[(k*n)+j+1] - A[(i*n)+j]*b;
        A[(k*n)+n-1] = - A[(i*n)+n-1]*b;        
        
      }
   
  }

}

#endif