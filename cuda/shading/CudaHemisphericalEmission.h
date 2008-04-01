/*! \file CudaHemisphericalEmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a device object
 *         equivalent to HemisphericalEmission.
 */

#pragma once

#include <spectrum/Spectrum.h>

class CudaHemisphericalEmission
{
  public:
    /*! Constructor accepts a radiance.
     *  \param r Sets mRadiance.
     */
    inline __host__ __device__ CudaHemisphericalEmission(const Spectrum &r);

  //protected:
    Spectrum mRadiance;
}; // end CudaHemisphericalEmission

#include "CudaHemisphericalEmission.inl"

