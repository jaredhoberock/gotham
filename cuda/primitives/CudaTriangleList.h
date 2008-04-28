/*! \file CudaTriangleList
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CudaSurfacePrimitiveList
 *         which is itself a list of triangles.
 */

#pragma once

#include "CudaSurfacePrimitiveList.h"
#include <cudaaliastable/CudaAliasTable.h>

class CudaTriangleList
  : public CudaSurfacePrimitiveList
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaSurfacePrimitiveList Parent;

    /*! This method creates mSurfaceAreaPdf.
     */
    virtual void finalize(void);

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
                                   const size_t n) const;

    typedef CudaAliasTable<unsigned int, float> TriangleTable;

  protected:
    // these are per-triangle lists of data
    // XXX much of this is redundant and should be per-primitive
    stdcuda::vector_dev<float3> mFirstVertex;
    stdcuda::vector_dev<float3> mSecondVertex;
    stdcuda::vector_dev<float3> mThirdVertex;
    stdcuda::vector_dev<float3>       mGeometricNormalDevice;
    stdcuda::vector_dev<float2>       mFirstVertexParmsDevice;
    stdcuda::vector_dev<float2>       mSecondVertexParmsDevice;
    stdcuda::vector_dev<float2>       mThirdVertexParmsDevice;
    stdcuda::vector_dev<float>           mPrimitiveInvSurfaceAreaDevice;
    stdcuda::vector_dev<PrimitiveHandle> mPrimitiveHandlesDevice;

    TriangleTable mSurfaceAreaPdf;

    // The total surface area over all triangles
    float mTotalSurfaceArea;
}; // end CudaTriangleList

