/*! \file CudaLambertian.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaLambertian.h.
 */

#include "CudaLambertian.h"

#ifndef INV_PI
#define INV_PI 0.318309886f
#endif // INV_PI

CudaLambertian
  ::CudaLambertian(const Spectrum &albedo)
    :mAlbedo(albedo),mAlbedoOverPi(albedo * INV_PI)
{
  ;
} // end CudaLambertian::CudaLambertian()

