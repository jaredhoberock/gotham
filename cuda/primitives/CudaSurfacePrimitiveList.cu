/*! \file CudaSurfacePrimitiveList.cu
 *  \author Jared Hoberock
 *  \brief Implementation of CudaSurfacePrimitiveList class.
 */

#include "CudaSurfacePrimitiveList.h"

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
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  kernel<<<grid,block>>>(prims, &mPrimitiveHandleToMaterialHandle[0], materials);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

static __global__ void kernel(const PrimitiveHandle *prims,
                              const int *stencil,
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
                       const device_ptr<const int> &stencil,
                       const device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  kernel<<<grid,block>>>(prims,
                         stencil,
                         &mPrimitiveHandleToMaterialHandle[0],
                         materials);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

static __global__ void kernel(const PrimitiveHandle *prims,
                              const int primStride,
                              const int *stencil,
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
                       const stdcuda::device_ptr<const int> &stencil,
                       const stdcuda::device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  dim3 grid(1,1,1);
  dim3 block(n,1,1);

  kernel<<<grid,block>>>(prims,
                         primStride,
                         stencil,
                         &mPrimitiveHandleToMaterialHandle[0],
                         materials);
} // end CudaSurfacePrimitiveList::getMaterialHandles()


