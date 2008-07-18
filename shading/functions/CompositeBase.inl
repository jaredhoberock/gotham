/*! \file CompositeBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CompositeBase.h.
 */

#include "CompositeBase.h"

template<typename V3, typename S3,
         typename F0, typename F1,
         typename Boolean>
  CompositeBase<V3,S3,F0,F1,Boolean>
    ::CompositeBase(const F0 &c0, const F1 &c1)
      :mComponent0(c0),mComponent1(c1)
{
  ;
} // end CompositeBase::CompositeBase()

template<typename V3, typename S3,
         typename F0, typename F1,
         typename Boolean>
  S3 CompositeBase<V3,S3,F0,F1,Boolean>
    ::evaluate(const Vector &wo,
               const Vector &normal,
               const Vector &wi) const
{
  return mComponent0.evaluate(wo,normal,wi) + mComponent1.evaluate(wo,normal,wi);
} // end CompositeBase::evaluate()

template<typename V3, typename S3,
         typename F0, typename F1,
         typename Boolean>
  S3 CompositeBase<V3,S3,F0,F1,Boolean>
    ::sample(const Vector &wo,
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
  // select component
  unsigned int i = u2 * NUM_COMPONENTS;
  i = (i >= NUM_COMPONENTS) ? NUM_COMPONENTS - 1 : i;

  Spectrum result;

  // sample it
  if(i == 0)
  {
    result = mComponent0.sample(wo,tangent,binormal,normal,u0,u1,u2,wi,pdf,delta,component);
  } // end if
  else
  {
    result = mComponent1.sample(wo,tangent,binormal,normal,u0,u1,u2,wi,pdf,delta,component);
  } // end if

  component = i;

  // average other components' pdfs if this component isn't specular
  if(!delta)
  {
    // sum over techniques
    if(component == 0)
    {
      pdf += mComponent1.evaluatePdf(wo,tangent,binormal,normal,wi);
    } // end if
    else
    {
      pdf += mComponent0.evaluatePdf(wo,tangent,binormal,normal,wi);
    } // end else
  } // end if

  // finally, divide by the number of components
  // What happens if the component we picked was a delta function?
  // We don't average in the others' pdfs, but what we are left with
  // is a delta pdf represented with 1.0 multiplied by the discrete
  // probability of choosing the component we did: 1.0 / mSize
  pdf /= NUM_COMPONENTS;

  // evaluate the other components and accumulate their results
  if(component == 0)
  {
    result += mComponent1.evaluate(wo,normal,wi);
  } // end if
  else
  {
    result += mComponent0.evaluate(wo,normal,wi);
  } // end else

  return result;
} // end CompositeBase::sample()

