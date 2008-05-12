/*! \file CudaDifferentialGeometryVector.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CudaDifferentialGeometryArray
 *         which is allocated by vector_dev.
 */

#pragma once

#include "CudaDifferentialGeometryArray.h"
#include <stdcuda/vector_dev.h>

class CudaDifferentialGeometryVector
  : public CudaDifferentialGeometryArray
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaDifferentialGeometryArray Parent;

    /*! Constructor accepts a size parameter.
     *  \param n The size in elements of this CudaDifferentialGeometryVector.
     */
    inline CudaDifferentialGeometryVector(const size_t n);

    /*! This method resizes this CudaDifferentialGeometryVector.
     *  \param n The size in elements of this CudaDifferentialGeometryVector.
     */
    inline void resize(const size_t n);

  protected:
    // storage for each array
    stdcuda::vector_dev<Point> mPointsVector;
    stdcuda::vector_dev<Normal> mNormalsVector;
    stdcuda::vector_dev<Vector> mTangentsVector;
    stdcuda::vector_dev<Vector> mBinormalsVector;
    stdcuda::vector_dev<ParametricCoordinates> mParametricCoordinatesVector;
    stdcuda::vector_dev<float> mSurfaceAreasVector;
    stdcuda::vector_dev<float> mInverseSurfaceAreasVector;
    stdcuda::vector_dev<Vector> mDPDUsVector;
    stdcuda::vector_dev<Vector> mDPDVsVector;
    stdcuda::vector_dev<Vector> mDNDUsVector;
    stdcuda::vector_dev<Vector> mDNDVsVector;
}; // end CudaDifferentialGeometryVector

#include "CudaDifferentialGeometryVector.inl"

