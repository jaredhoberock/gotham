/*! \file HemisphericalEmissionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for HemisphericalEmissionBase.h.
 */

#include "HemisphericalEmissionBase.h"

#ifndef INV_PI
#define INV_PI 0.318309886f
#endif // INV_PI

#ifndef PI
#define PI 3.14159265f
#endif // PI

template<typename V3, typename S3, typename DG>
  HemisphericalEmissionBase<V3,S3,DG>
    ::HemisphericalEmissionBase(const S3 &radiosity)
      :mRadiance(radiosity)
{
  // XXX god this is so shitty but we have to do it to be
  //     compatible with CUDA vectors
  ((float*)&mRadiance)[0] *= INV_PI;
  ((float*)&mRadiance)[1] *= INV_PI;
  ((float*)&mRadiance)[2] *= INV_PI;
} // end HemisphericalEmissionBase::HemisphericalEmissionBase()

template<typename V3, typename S3, typename DG>
  const S3 &HemisphericalEmissionBase<V3,S3,DG>
    ::getRadiance(void) const
{
  return mRadiance;
} // end HemisphericalEmissionBase::getRadiance()

template<typename V3, typename S3, typename DG>
  const S3 HemisphericalEmissionBase<V3,S3,DG>
    ::getRadiosity(void) const
{
  S3 result = getRadiance();

  // XXX god this is so shitty but we have to do it to be
  //     compatible with CUDA vectors
  ((float*)&result)[0] *= PI;
  ((float*)&result)[1] *= PI;
  ((float*)&result)[2] *= PI;

  return result;
} // end HemisphericalEmissionBase::getRadiosity()

template<typename V3, typename S3, typename DG>
  S3 HemisphericalEmissionBase<V3,S3,DG>
    ::evaluate(const V3 &wo, const DG &dg) const
{
  S3 result;

  // XXX god this is so shitty but we have to do it to be
  //     compatible with CUDA vectors
  ((float*)&result)[0] = 0;
  ((float*)&result)[1] = 0;
  ((float*)&result)[2] = 0;

  // are we pointing in the same direction?
  if(dot(wo,dg.getNormal()) > 0)
  {
    result = mRadiance;
  } // end if

  return result;
} // end HemisphericalEmissionBase::evaluate()

