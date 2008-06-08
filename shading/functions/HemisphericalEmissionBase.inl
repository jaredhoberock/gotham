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

template<typename V3, typename S3>
  HemisphericalEmissionBase<V3,S3>
    ::HemisphericalEmissionBase(const S3 &radiosity)
      :mRadiance(radiosity)
{
  mRadiance.x *= INV_PI;
  mRadiance.y *= INV_PI;
  mRadiance.z *= INV_PI;
} // end HemisphericalEmissionBase::HemisphericalEmissionBase()

template<typename V3, typename S3>
  const S3 &HemisphericalEmissionBase<V3,S3>
    ::getRadiance(void) const
{
  return mRadiance;
} // end HemisphericalEmissionBase::getRadiance()

template<typename V3, typename S3>
  const S3 HemisphericalEmissionBase<V3,S3>
    ::getRadiosity(void) const
{
  S3 result = PI * getRadiance();

  return result;
} // end HemisphericalEmissionBase::getRadiosity()

template<typename V3, typename S3>
  S3 HemisphericalEmissionBase<V3,S3>
    ::evaluate(const V3 &wo,
               const V3 &normal) const
{
  S3 result;

  result.x = 0;
  result.y = 0;
  result.z = 0;

  // are we pointing in the same direction?
  if(dot(wo,normal) > 0)
  {
    result = mRadiance;
  } // end if

  return result;
} // end HemisphericalEmissionBase::evaluate()

template<typename V3, typename S3>
  S3 HemisphericalEmissionBase<V3,S3>
    ::evaluate(const V3 &wo,
               const V3 &point,
               const V3 &tangent,
               const V3 &binormal,
               const V3 &normal) const
{
  return evaluate(wo,normal);
} // end HemisphericalEmissionBase::evaluate()

