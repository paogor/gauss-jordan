#include<iostream>
#include<vector>

#include<algorithm>


#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include <boost/timer/timer.hpp>
#include"CUDATIMER.hpp"

#include"gauss-jordanCPU.hpp"
#include"gauss-jordan1.hpp"
#include"gauss-jordan3.hpp"
#include"gauss-jordan5.hpp"



using std::vector;
using thrust::device_vector;
using thrust::host_vector;
using thrust::raw_pointer_cast;


int main(){

  float times_ms[6];

  for(int m = 2; m<65536; m*=2)
  for(int n = 5; n<33; ++n)
  {

  vector<double> A(n*n, 500);

  for(int i = 0; i<n; ++i)
//   for(int j = 0; j<n; ++j)
     A[(i*n)+i] += i*111;


  A.resize( m * n*n);
  for(int i = 1; i<m; ++i)
  std::copy(A.begin(), A.begin()+ n*n, A.begin() + i*(n*n));

  vector<double> ACPU = A;
  host_vector<double> h_A1, h_A3, h_A5, h_A6;



  {
    boost::timer::cpu_timer timer;

    for(int i = 0; i<m; ++i)
      gauss_jordanCPU(n, ACPU.data());

    const boost::timer::cpu_times elapsed_times(timer.elapsed()); // time in ns
    times_ms[0] = float(elapsed_times.wall)/1e6; // time in ms
  }


  {
    device_vector<double> d_A1 = A;
    {
      CUDATIMER t(times_ms + 1);
      gauss_jordan1<<<m,n,n*n*sizeof(double)>>>(n, raw_pointer_cast(d_A1.data()));
    }
    h_A1 = d_A1;
  }


  {
    device_vector<double> d_A3 = A;
    {
      CUDATIMER t(times_ms + 2);
      gauss_jordan3<<<dim3(m,1,1),dim3(n,n,1),(n*n)*sizeof(double)>>>(n, raw_pointer_cast(d_A3.data()));
    }
    h_A3 = d_A3;
  }


  {
    device_vector<double> d_A5 = A;
    {
      CUDATIMER t(times_ms + 3);
      gauss_jordan5<<<m,n,n*n*sizeof(double)>>>(n, raw_pointer_cast(d_A5.data()));
    }
    h_A5 = d_A5;
  }


  std::cout<<n<<", "<<m;
  for(size_t  i = 0; i<4; ++i) std::cout<<", "<<times_ms[i];
  std::cout<<";"<<std::endl;
 
  } // for

  return 0;
}
