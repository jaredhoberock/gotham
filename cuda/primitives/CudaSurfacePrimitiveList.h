/*! \file CudaSurfacePrimitiveList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SurfacePrimitiveList
 *         that resides on a CUDA device.
 */

#pragma once

#include <stdcuda/vector_dev.h>
#include "../../primitives/PrimitiveHandle.h"
#include "../../shading/MaterialHandle.h"
#include "../geometry/CudaDifferentialGeometry.h"

class CudaSurfacePrimitiveList
{
  public:
    /*! Null destructor does nothing.
     */
    virtual ~CudaSurfacePrimitiveList(void);

    /*! This method should be implemented in a child
     *  class to create mPrimitiveHandleToMaterialHandle.
     */
    virtual void finalize(void) = 0;

    /*! This method fills requests for MaterialHandles
     *  given a set of PrimitiveHandles.
     *  \param prims The list of PrimitiveHandles of interest,
     *               presumably belonging to this CudaSurfacePrimitiveList.
     *  \param materials The list of MaterialHandles is returned here.
     *  \param n The length of each list.
     */
    virtual void getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                                    const stdcuda::device_ptr<MaterialHandle> &materials,
                                    const size_t n) const;

    /*! This method fills requests for MaterialHandles
     *  given a set of PrimitiveHandles.
     *  \param prims The list of PrimitiveHandles of interest,
     *               presumably belonging to this CudaSurfacePrimitiveList.
     *  \param stencil A stencil to control which requests get processed.
     *  \param materials The list of MaterialHandles is returned here.
     *  \param n The length of each list.
     */
    virtual void getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<MaterialHandle> &materials,
                                    const size_t n) const;

    /*! This method fills requests for MaterialHandles
     *  given a set of PrimitiveHandles.
     *  \param prims The list of PrimitiveHandles of interest,
     *               presumably belonging to this CudaSurfacePrimitiveList.
     *  \param primStride The size in bytes between elements of the prims list.
     *  \param stencil A stencil to control which requests get processed.
     *  \param materials The list of MaterialHandles is returned here.
     *  \param n The length of each list.
     */
    virtual void getMaterialHandles(const stdcuda::device_ptr<const PrimitiveHandle> &prims,
                                    const size_t primStride,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<MaterialHandle> &materials,
                                    const size_t n) const;

    /*! This method samples the surface area of this CudaSurfacePrimitiveList
     *  in a SIMD fashion.
     *  \param u A list of points in [0,1)^4
     *  \param prims The primitive sampled will be returned to this list.
     *  \param dg The sampled geometry will be returned to this list.
     *  \param pdf The pdf of each sampled point will be returned to this list.
     *  \param n The length of each list.
     */
    virtual void sampleSurfaceArea(const stdcuda::device_ptr<const float4> &u,
                                   const stdcuda::device_ptr<PrimitiveHandle> &prims,
                                   const stdcuda::device_ptr<CudaDifferentialGeometry> &dg,
                                   const stdcuda::device_ptr<float> &pdf,
                                   const size_t n) const = 0;

  protected:
    /*! This maps a PrimitiveHandle to a MaterialHandle.
     */
    stdcuda::vector_dev<MaterialHandle> mPrimitiveHandleToMaterialHandle;
}; // end CudaSurfacePrimitiveList

