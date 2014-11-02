#ifndef __ALG_IF_HPP__ 
#define __ALG_IF_HPP__

template<typename T> inline
__device__ T alg_if(int bool_expr, T if_true, T if_false){

  return T( bool_expr )*if_true + T( 1 - bool_expr )*if_false;

}
#endif
