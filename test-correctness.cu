#include<iostream>
#include<vector>
#include <algorithm>

#include <boost/math/special_functions/next.hpp>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include"gauss-jordanCPU.hpp"
#include"gauss-jordan0.hpp"
#include"gauss-jordan1.hpp"
#include"gauss-jordan2.hpp"
#include"gauss-jordan3.hpp"
#include"gauss-jordan4.hpp"
#include"gauss-jordan5.hpp"

using std::vector;
using thrust::device_vector;
using thrust::host_vector;
using thrust::raw_pointer_cast;

using boost::math::float_distance;

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> A){

  for(size_t i=0; i < A.size(); ++i)
      os<<A[i]<<" ";
  
  return os;

}

template <typename T>
std::ostream& operator<<(std::ostream& os, thrust::host_vector<T> A){

  for(size_t i=0; i < A.size(); ++i)
      os<<A[i]<<" ";
  
  return os;

}

template <typename T>
std::vector<T> gmmm(size_t n, T * A, T * B )
{

  std::vector<T> C(n*n,0);

  for(size_t i=0; i <n; ++i)
    for(size_t j=0; j <n; ++j)
      for(size_t k=0; k< n; ++k)
        C[(i*n)+j] += A[(i*n)+k]*B[(k*n)+j];

  return C;

}


int main(){

  const int n = 4;
  vector<double> A(n*n);

//  std::generate(A.begin(), A.end(), std::rand);
//  double max = *std::max_element(A.begin(), A.end());

  for(int i = 0; i<n; ++i)
   for(int j = 0; j<n; ++j)
     A[(i*n)+i] = 1 ;
 
  std::cout<<A<<std::endl;

  vector<double> ACPU = A;
  host_vector<double> h_A0, h_A1, h_A2, h_A3, h_A4, h_A5;

  device_vector<double> d_A0 = A;
  device_vector<double> d_A1 = A;
  device_vector<double> d_A2 = A;
  device_vector<double> d_A3 = A;
  device_vector<double> d_A4 = A;
  device_vector<double> d_A5 = A;

  gauss_jordanCPU(n, ACPU.data());

  gauss_jordan0<<<1,n,n*n*sizeof(double)>>>(n, raw_pointer_cast(d_A0.data()));
  h_A0 = d_A0;

  gauss_jordan1<<<1,n,n*n*sizeof(double)>>>(n, raw_pointer_cast(d_A1.data()));
  h_A1 = d_A1;

  gauss_jordan2<<<1,n,(n*n+1)*sizeof(double)>>>(n, raw_pointer_cast(d_A2.data()));
  h_A2 = d_A2;
 
  gauss_jordan3<<<dim3(1,1,1),dim3(n,n,1),(n*n)*sizeof(double)>>>(n, raw_pointer_cast(d_A3.data()));
  h_A3 = d_A3;

  gauss_jordan4<<<dim3(1,1,1),dim3(n,n,1),(n*n+1)*sizeof(double)>>>(n, raw_pointer_cast(d_A4.data()));
  h_A4 = d_A4;

  gauss_jordan5<<<1,2*n,n*n*2*sizeof(double)>>>(n, raw_pointer_cast(d_A5.data()));
  h_A5 = d_A5;

/*  double total_distance = 0.0;

  for(int i = 0; i<ACPU.size(); ++i){
    std::cout<<  float_distance(ACPU[i],h_A0[i])<<std::endl;
      std::cout<< float_distance(ACPU[i],h_A1[i])<<std::endl;
      std::cout<< float_distance(ACPU[i],h_A2[i])<<std::endl;
      std::cout<< float_distance(ACPU[i],h_A3[i])<<std::endl;
      std::cout<< float_distance(ACPU[i],h_A4[i])<<std::endl;
      std::cout<< float_distance(ACPU[i],h_A5[i])<<std::endl;
  }




  std::cout<<gmmm(n,A.data(),ACPU.data())<<std::endl
           <<gmmm(n,A.data(),h_A0.data())<<std::endl
           <<gmmm(n,A.data(),h_A1.data())<<std::endl
           <<gmmm(n,A.data(),h_A2.data())<<std::endl
           <<gmmm(n,A.data(),h_A3.data())<<std::endl
           <<gmmm(n,A.data(),h_A4.data())<<std::endl
           <<gmmm(n,A.data(),h_A5.data())<<std::endl;


 // std::cout<<total_distance<<std::endl;
*/

  std::cout<<h_A5<<std::endl;

  return 0;
}
