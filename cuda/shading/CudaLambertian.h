/*! \file CudaLambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a device struct
 *         equivalent to Lambertian.
 */

#pragma once

#include <spectrum/Spectrum.h>

class CudaLambertian
{
  public:
    /*! Constructor accepts an albedo.
     *  \param albedo Sets mAlbedo.
     */
    inline __host__ __device__ CudaLambertian(const Spectrum &albedo);

  //protected:
    Spectrum mAlbedo;
    Spectrum mAlbedoOverPi;
}; // end CudaLambertian

#include "CudaLambertian.inl"

