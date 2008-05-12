/*! \file CudaRandomSequence.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaRandomSequence class.
 */

#include "CudaRandomSequence.h"

CudaRandomSequence
  ::CudaRandomSequence(const unsigned int seed)
    :Parent(seed),mTwister(seed)
{
  ;
} // end CudaRandomSequence::CudaRandomSequence()

void CudaRandomSequence
  ::operator()(const stdcuda::device_ptr<float> &v,
               const size_t n)
{
  mTwister.uniform01(v,n);
} // end CudaRandomSequence::operator()()

