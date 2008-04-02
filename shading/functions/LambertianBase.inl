/*! \file LambertianBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for LambertianBase.h.
 */

#include "LambertianBase.h"
#include "../areSameHemisphere.h"

#ifndef INV_PI
#define INV_PI 0.318309886f
#endif // INV_PI

template<typename V3, typename S3, typename DG>
  LambertianBase<V3,S3,DG>
    ::LambertianBase(const Spectrum &albedo)
      :mAlbedo(albedo)
{
  mAlbedoOverPi = albedo;

  // XXX god this is so shitty but we have to do it to be
  //     compatible with CUDA vectors
  ((float*)&mAlbedoOverPi)[0] *= INV_PI;
  ((float*)&mAlbedoOverPi)[1] *= INV_PI;
  ((float*)&mAlbedoOverPi)[2] *= INV_PI;
} // end LambertianBase::LambertianBase()

template<typename V3, typename S3, typename DG>
  S3 LambertianBase<V3,S3,DG>
    ::evaluate(const V3 &wo,
               const DG &dg,
               const V3 &wi) const
{
  // XXX god this is so shitty but we have to do it to be
  //     compatible with CUDA vectors
  S3 result;
  ((float*)&result)[0] = 0;
  ((float*)&result)[1] = 0;
  ((float*)&result)[2] = 0;

  if(areSameHemisphere(wi, dg.getNormal(), wo))
  {
    result = mAlbedoOverPi;
  } // end if

  return result;
} // end LambertianBase::evaluate()

template<typename V3, typename S3, typename DG>
  const S3 &LambertianBase<V3,S3,DG>
    ::getAlbedo(void) const
{
  return mAlbedo;
} // end LambertianBase::getAlbedo()

