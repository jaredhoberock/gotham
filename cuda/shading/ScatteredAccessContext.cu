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

__global__ void selectMaterial(const bool *stencil,
                               const MaterialHandle *handles,
                               const MaterialHandle h,
                               bool *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // AND in the stencil with whether or not we match the handle
  //result[i] = stencil[i] & (handles[i] == h);
  bool r = false; 
  if(stencil[i] && (handles[i] == h))
  {
    r = true;
  } // end if
  
  result[i] = r;
} // end selectMaterial()

__global__ void selectMaterial(const MaterialHandle *handles,
                               const MaterialHandle h,
                               bool *result)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool r = false;
  if(handles[i] == h)
  {
    r = true;
  } // end if

  result[i] = r;
} // end selectMaterial()

void ScatteredAccessContext
  ::evaluateScattering(const device_ptr<const MaterialHandle> &m,
                       const CudaDifferentialGeometryArray &dg,
                       const device_ptr<const bool> &stencil,
                       const device_ptr<CudaScatteringDistributionFunction> &f,
                       const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  vector_dev<bool> materialStencil(n);

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

      if(gridSize)
        selectMaterial<<<gridSize,BLOCK_SIZE>>>(stencil,
                                                m,
                                                h,
                                                &materialStencil[0]);
      if(n%BLOCK_SIZE)
        selectMaterial<<<1,n%BLOCK_SIZE>>>(stencil + gridSize*BLOCK_SIZE,
                                           m + gridSize*BLOCK_SIZE,
                                           h,
                                           &materialStencil[gridSize*BLOCK_SIZE]);

      // run the shader
      material->evaluateScattering(*this, dg, &materialStencil[0], f, n);
    } // end if
  } // end for i
} // end ScatteredAccessContext::evaluateScattering()

void ScatteredAccessContext
  ::evaluateEmission(const device_ptr<const MaterialHandle> &m,
                     const CudaDifferentialGeometryArray &dg,
                     const device_ptr<const bool> &stencil,
                     const device_ptr<CudaScatteringDistributionFunction> &f,
                     const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  vector_dev<bool> materialStencil(n);

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
      if(gridSize)
        selectMaterial<<<gridSize,BLOCK_SIZE>>>(stencil,
                                                m,
                                                h,
                                                &materialStencil[0]);
      if(n%BLOCK_SIZE)
        selectMaterial<<<1,n%BLOCK_SIZE>>>(stencil + gridSize*BLOCK_SIZE,
                                           m + gridSize*BLOCK_SIZE,
                                           h,
                                           &materialStencil[gridSize*BLOCK_SIZE]);

      // run the shader
      material->evaluateEmission(*this, dg, &materialStencil[0], f, n);
    } // end if
  } // end for i
} // end ScatteredAccessContext::evaluateEmission()

void ScatteredAccessContext
  ::evaluateSensor(const device_ptr<const MaterialHandle> &m,
                   const CudaDifferentialGeometryArray &dg,
                   const device_ptr<CudaScatteringDistributionFunction> &f,
                   const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  vector_dev<bool> materialStencil(n);
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
      if(gridSize)
        selectMaterial<<<gridSize,BLOCK_SIZE>>>(m,h,&materialStencil[0]);
      if(n%BLOCK_SIZE)
        selectMaterial<<<1,n%BLOCK_SIZE>>>(m + gridSize*BLOCK_SIZE,
                                           h,
                                           &materialStencil[gridSize*BLOCK_SIZE]);

      // run the shader
      // normal stride
      material->evaluateSensor(*this, dg, &materialStencil[0], f, n);
    } // end if
  } // end for
} // end ScatteredAccessContext::evaluateSensor()

