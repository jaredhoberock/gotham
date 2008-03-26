/*! \file CudaLambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a device struct
 *         equivalent to Lambertian.
 */

#pragma once

#include <spectrum/Spectrum.h>

class CudaLambertian
{
  Spectrum mAlbedo;
  Spectrum mAlbedoOverPi;
}; // end CudaLambertian

