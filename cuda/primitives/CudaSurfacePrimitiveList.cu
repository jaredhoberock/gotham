/*! \file CudaSurfacePrimitiveList.cu
 *  \author Jared Hoberock
 *  \brief Implementation of CudaSurfacePrimitiveList class.
 */

#include "CudaSurfacePrimitiveList.h"
#include <stdcuda/stride_cast.h>

using namespace stdcuda;

CudaSurfacePrimitiveList
  ::~CudaSurfacePrimitiveList(void)
{
  ;
} // end CudaSurfacePrimitiveList::~CudaSurfacePrimitiveList()

static __global__ void kernel(const PrimitiveHandle *prims,
                              const MaterialHandle *primToMaterial,
                              MaterialHandle *materials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  materials[i] = primToMaterial[prims[i]];
} // end getMaterialHandlesKernel()

void CudaSurfacePrimitiveList
  ::getMaterialHandles(const device_ptr<const PrimitiveHandle> &prims,
                       const device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  if(gridSize)
    kernel<<<gridSize,BLOCK_SIZE>>>(prims,
                                    &mPrimitiveHandleToMaterialHandle[0],
                                    materials);
  if(n%BLOCK_SIZE)
    kernel<<<1,n%BLOCK_SIZE>>>(prims + gridSize*BLOCK_SIZE,
                               &mPrimitiveHandleToMaterialHandle[0],
                               materials + gridSize*BLOCK_SIZE);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

static __global__ void kernel(const PrimitiveHandle *prims,
                              const bool *stencil,
                              const MaterialHandle *primToMaterial,
                              MaterialHandle *materials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    materials[i] = primToMaterial[prims[i]];
  } // end if
} // end kernel()

void CudaSurfacePrimitiveList
  ::getMaterialHandles(const device_ptr<const PrimitiveHandle> &prims,
                       const device_ptr<const bool> &stencil,
                       const device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  if(gridSize)
    kernel<<<gridSize,BLOCK_SIZE>>>(prims,
                                    stencil,
                                    &mPrimitiveHandleToMaterialHandle[0],
                                    materials);
  if(n%BLOCK_SIZE)
    kernel<<<1,n%BLOCK_SIZE>>>(prims + gridSize*BLOCK_SIZE,
                               stencil + gridSize*BLOCK_SIZE,
                               &mPrimitiveHandleToMaterialHandle[0],
                               materials + gridSize*BLOCK_SIZE);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

static __global__ void kernel(const PrimitiveHandle *prims,
                              const int primStride,
                              const bool *stencil,
                              const MaterialHandle *primToMaterial,
                              MaterialHandle *materials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    const char *ptr = reinterpret_cast<const char*>(prims) + i*primStride;
    PrimitiveHandle p = *reinterpret_cast<const PrimitiveHandle*>(ptr);
    materials[i] = primToMaterial[p];
  } // end if
} // end kernel()

void CudaSurfacePrimitiveList
  ::getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                       const size_t primStride,
                       const stdcuda::device_ptr<const bool> &stencil,
                       const stdcuda::device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / BLOCK_SIZE;

  if(gridSize)
    kernel<<<gridSize,BLOCK_SIZE>>>(prims,
                                    primStride,
                                    stencil,
                                    &mPrimitiveHandleToMaterialHandle[0],
                                    materials);
  if(n%BLOCK_SIZE)
    kernel<<<1,n%BLOCK_SIZE>>>(stride_cast(prims.get(),gridSize*BLOCK_SIZE,primStride),
                               primStride,
                               stencil + gridSize*BLOCK_SIZE,
                               &mPrimitiveHandleToMaterialHandle[0],
                               materials + gridSize*BLOCK_SIZE);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

