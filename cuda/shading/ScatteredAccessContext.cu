/*! \file ScatteredAccessContext.cpp
 *  \author Jared Hoberock
 *  \brief Inline file for ScatteredAccessContext class.
 */

// turn off exceptions to allow compilation with nvcc
#define BOOST_NO_EXCEPTIONS

#include "ScatteredAccessContext.h"
#include "../include/CudaMaterial.h"
#include <stdcuda/cuda_algorithm.h>
using namespace stdcuda;

__global__ void selectMaterial(const int *stencil,
                               const MaterialHandle *handles,
                               const MaterialHandle h,
                               int *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // AND in the stencil with whether or not we match the handle
  //result[i] = stencil[i] & (handles[i] == h);
  int r = 0; 
  if(stencil[i] && (handles[i] == h))
  {
    r = 1;
  } // end if
  
  result[i] = r;
} // end selectMaterial()

__global__ void selectMaterial(const MaterialHandle *handles,
                               const MaterialHandle h,
                               int *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int r = 0;
  if(handles[i] == h)
  {
    r = 1;
  } // end if

  result[i] = r;
} // end selectMaterial()

void ScatteredAccessContext
  ::evaluateScattering(const device_ptr<const MaterialHandle> &m,
                       const device_ptr<const CudaDifferentialGeometry> &dg,
                       const size_t dgStride,
                       const device_ptr<const int> &stencil,
                       const device_ptr<CudaScatteringDistributionFunction> &f,
                       const size_t n)
{
  vector_dev<int> materialStencil(n);

  const CudaMaterial *material = 0;

  // for each Material, create a new stencil
  // which is the logical AND of stencil and h
  // then, run the Material's kernel with the new stencil
  MaterialHandle h = 0;
  for(MaterialList::const_iterator mPtr = mMaterials->begin();
      mPtr != mMaterials->end();
      ++mPtr, ++h)
  {
    material = dynamic_cast<const CudaMaterial *>(mPtr->get());

    if(material)
    {
      // materialStencil = stencil & (i == h)
      dim3 grid = dim3(1,1,1);
      dim3 block = dim3(n,1,1);
      selectMaterial<<<grid,block>>>(stencil,
                                     m,
                                     h,
                                     &materialStencil[0]);

      // run the shader
      material->evaluateScattering(*this, dg, dgStride, &materialStencil[0], f, n);
    } // end if
  } // end for i
} // end ScatteredAccessContext::evaluateScattering()

void ScatteredAccessContext
  ::evaluateEmission(const device_ptr<const MaterialHandle> &m,
                     const device_ptr<const CudaDifferentialGeometry> &dg,
                     const size_t dgStride,
                     const device_ptr<const int> &stencil,
                     const device_ptr<CudaScatteringDistributionFunction> &f,
                     const size_t n)
{
  vector_dev<int> materialStencil(n);

  const CudaMaterial *material = 0;

  // for each Material, create a new stencil
  // which is the logical AND of stencil and h
  // then, run the Material's kernel with the new stencil
  MaterialHandle h = 0;
  for(MaterialList::const_iterator mPtr = mMaterials->begin();
      mPtr != mMaterials->end();
      ++mPtr, ++h)
  {
    material = dynamic_cast<const CudaMaterial *>(mPtr->get());

    if(material)
    {
      // materialStencil = stencil & (i == h)
      dim3 grid = dim3(1,1,1);
      dim3 block = dim3(n,1,1);
      selectMaterial<<<grid,block>>>(stencil,
                                     m,
                                     h,
                                     &materialStencil[0]);

      // run the shader
      material->evaluateEmission(*this, dg, dgStride, &materialStencil[0], f, n);
    } // end if
  } // end for i
} // end ScatteredAccessContext::evaluateEmission()

void ScatteredAccessContext
  ::evaluateSensor(const device_ptr<const MaterialHandle> &m,
                   const device_ptr<const CudaDifferentialGeometry> &dg,
                   const size_t dgStride,
                   const device_ptr<CudaScatteringDistributionFunction> &f,
                   const size_t n)
{
  vector_dev<int> materialStencil(n);
  const CudaMaterial *material = 0;

  // for each Material, create a new stencil
  // which is the logical AND of stencil and h
  // then, run the Material's kernel with the new stencil
  MaterialHandle h = 0;
  for(MaterialList::const_iterator mPtr = mMaterials->begin();
      mPtr != mMaterials->end();
      ++mPtr, ++h)
  {
    material = dynamic_cast<const CudaMaterial*>(mPtr->get());

    if(material)
    {
      // materialStencil = (i == h)
      dim3 grid = dim3(1,1,1);
      dim3 block = dim3(n,1,1);
      selectMaterial<<<grid,block>>>(m,h,&materialStencil[0]);

      // run the shader
      // normal stride
      material->evaluateSensor(*this, dg, dgStride, &materialStencil[0], f, n);
    } // end if
  } // end for
} // end ScatteredAccessContext::evaluateSensor()

