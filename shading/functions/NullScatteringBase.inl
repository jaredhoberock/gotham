/*! \file NullScatteringBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for NullScatteringBase.h.
 */

#include "NullScatteringBase.h"
#include "../../geometry/Mappings.h"

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::evaluate(void) const
{
  S3 result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  return result;
} // end NullScatteringBase::evaluate()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::evaluate(const Vector &wo,
               const Vector &normal,
               const Vector &wi) const
{
  return evaluate();
} // end NullScatteringBase::evaluate()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::evaluate(const Vector &wo,
               const Vector &point,
               const Vector &tangent,
               const Vector &binormal,
               const Vector &normal) const
{
  return evaluate();
} // end NullScatteringBase::evaluate()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::sample(const Vector &point,
             const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             bool &delta,
             unsigned int &component) const
{
  component = 0;
  return sample(tangent,binormal,normal,u0,u1,u2,wi,pdf,delta);
} // end NullScatteringBase::sample()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
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
             bool &delta,
             unsigned int &component) const
{
  return sample(point,tangent,binormal,normal,u0,u1,u2,wi,pdf,delta,component);
} // end NullScatteringBase::sample()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::sample(const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             bool &delta,
             unsigned int &component) const
{
  delta = false;
  component = 0;
  Mappings<V3>::unitSquareToCosineHemisphere(u0, u1, tangent, binormal, normal, wi, pdf);
  return evaluate();
} // end NullScatteringBase::sample()

template<typename V3, typename S3>
  S3 NullScatteringBase<V3,S3>
    ::sample(const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &wi,
             float &pdf,
             bool &delta) const
{
  delta = false;
  Mappings<V3>::unitSquareToCosineHemisphere(u0, u1, tangent, binormal, normal, wi, pdf);
  return evaluate();
} // end NullScatteringBase::sample()

template<typename V3, typename S3>
  float NullScatteringBase<V3,S3>
    ::evaluatePdf(const Vector &wo,
                  const Vector &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal,
                  const Vector &wi) const
{
  return Mappings<V3>::evaluateCosineHemispherePdf(wi,normal);
} // end NullScatteringBase::evaluatePdf()

