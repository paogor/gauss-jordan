#include<iostream>
#include<vector>
#include<algorithm>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include"gauss-jordan5.hpp"
#include"gauss-jordan3.hpp"
#include"gauss-jordan1.hpp"
#include"gauss-jordan2.hpp"
#include"gauss-jordan6.hpp"

using std::vector;
using thrust::device_vector;
using thrust::host_vector;
using thrust::raw_pointer_cast;


int main()
{

  const size_t n = 17;
  const size_t m = 1024;

  vector<double> A(n*n, 500);

  for(int i = 0; i<n; ++i)
     A[(i*n)+i] += i*111;


  A.resize( m * n*n);
  for(int i = 1; i<m; ++i)
  std::copy(A.begin(), A.begin()+ n*n, A.begin() + i*(n*n));

  vector<double> ACPU = A;
  host_vector<double> h_A3, h_A5, h_A1, h_A2, h_A6;


  {
    device_vector<double> d_A5 = A;
    gauss_jordan5<<<m,n,n*n*sizeof(double)>>>
      (n, raw_pointer_cast(d_A5.data()));
    h_A5 = d_A5;
  }

  {
    device_vector<double> d_A3 = A;
    gauss_jordan3<<<dim3(m,1,1),dim3(n,n,1),(n*n)*sizeof(double)>>>
      (n, raw_pointer_cast(d_A3.data()));
    h_A3 = d_A3;
  }

  {
    device_vector<double> d_A1 = A;
    gauss_jordan1<<<m,n,n*n*sizeof(double)>>>
      (n, raw_pointer_cast(d_A1.data()));
    h_A1 = d_A1;
  }

  {
    device_vector<double> d_A2 = A;
    gauss_jordan2<<<m,n,(n*n+1)*sizeof(double)>>>
      (n, raw_pointer_cast(d_A2.data()));
    h_A2 = d_A2;
  }

  {
    device_vector<double> d_A6 = A;
    gauss_jordan6<<<dim3(m,1,1),dim3(n,n,1),(n*n)*sizeof(double)>>>
      (n, raw_pointer_cast(d_A6.data()));
    h_A6 = d_A6;
  }



  return 0;
}
