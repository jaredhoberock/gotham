/*! \file cudaGetMaterialHandles.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaGetMaterialHandles functions.
 */

#include "cudaGetMaterialHandles.h"

static __global__ void k(const PrimitiveHandle *prims,
                         const int *stencil,
                         const MaterialHandle *primToMaterial,
                         MaterialHandle *materials)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    materials[i] = primToMaterial[prims[i]];
  } // end if
} // end k()

void getMaterialHandles(const PrimitiveHandle *prims,
                        const int *stencil,
                        const MaterialHandle *primToMaterial,
                        MaterialHandle *materials,
                        const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  k<<<grid,block>>>(prims, stencil, primToMaterial, materials);
} // end getMaterialHandles()

static __global__ void ks(const PrimitiveHandle *prims,
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
} // end k()

void getMaterialHandlesWithStride(const PrimitiveHandle *prims,
                                  const size_t primStride,
                                  const int *stencil,
                                  const MaterialHandle *primToMaterial,
                                  MaterialHandle *materials,
                                  const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  ks<<<grid,block>>>(prims, primStride, stencil, primToMaterial, materials);
} // end getMaterialHandles()

