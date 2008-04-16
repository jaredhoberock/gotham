/*! \file CudaSurfacePrimitiveList.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaSurfacePrimitiveList class.
 */

#include "CudaSurfacePrimitiveList.h"
#include <iostream>
#include "cudaGetMaterialHandles.h"

void CudaSurfacePrimitiveList
  ::getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                       const stdcuda::device_ptr<const int> &stencil,
                       const stdcuda::device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  ::getMaterialHandles(prims,
                       stencil,
                       &mPrimitiveHandleToMaterialHandle[0],
                       materials,
                       n);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

void CudaSurfacePrimitiveList
  ::getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                       const size_t primStride,
                       const stdcuda::device_ptr<const int> &stencil,
                       const stdcuda::device_ptr<MaterialHandle> &materials,
                       const size_t n) const
{
  getMaterialHandlesWithStride(prims,
                               primStride,
                               stencil,
                               &mPrimitiveHandleToMaterialHandle[0],
                               materials,
                               n);
} // end CudaSurfacePrimitiveList::getMaterialHandles()

