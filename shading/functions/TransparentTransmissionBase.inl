/*! \file TransparentTransmissionBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for TransparentTransmissionBase.h.
 */

#include "TransparentTransmissionBase.h"

template<typename V3, typename S3, typename Boolean>
  TransparentTransmissionBase<V3,S3,Boolean>
    ::TransparentTransmissionBase(const Spectrum &transmittance)
      :mTransmittance(transmittance)
{
  ;
} // end TransparentTransmissionBase::TransparentTransmissionBase()

template<typename V3, typename S3, typename Boolean>
  S3 TransparentTransmissionBase<V3,S3,Boolean>
    ::evaluate(void) const
{
  S3 result;
  result.x = 0;
  result.y = 0;
  result.z = 0;
  return result;
} // end TransparentTransmissionBase::evaluate()

template<typename V3, typename S3, typename Boolean>
  S3 TransparentTransmissionBase<V3,S3,Boolean>
    ::evaluate(const Vector &wo,
               const Vector &normal,
               const Vector &wi) const
{
  return evaluate();
} // end TransparentTransmissionBase::evaluate()

template<typename V3, typename S3, typename Boolean>
  S3 TransparentTransmissionBase<V3,S3,Boolean>
    ::sample(const Vector &wo,
             const Vector &normal,
             Vector &wi,
             float &pdf,
             Boolean &delta,
             unsigned int &component) const
{
  delta = true;
  component = 0;

  // we have to do this one component at a time
  // because we can't define this operator for CUDA vector types
  wi.x = -wo.x;
  wi.y = -wo.y;
  wi.z = -wo.z;

  pdf = 1.0f;

  return mTransmittance / fabs(dot(normal,wi));
} // end TransparentTransmissionBase::sample()

template<typename V3, typename S3, typename Boolean>
  S3 TransparentTransmissionBase<V3,S3,Boolean>
    ::sample(const Vector &wo,
             const Vector &point,
             const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             Boolean &delta,
             unsigned int &component) const
{
  return sample(wo,normal,wi,pdf,delta,component);
} // end TransparentTransmissionBase::sample()

template<typename V3, typename S3, typename Boolean>
  float TransparentTransmissionBase<V3,S3,Boolean>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return 0;
} // end TransparentTransmissionBase::evaluatePdf()


