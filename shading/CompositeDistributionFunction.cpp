/*! \file CompositeDistributionFunction.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CompositeDistributionFunction class.
 */

#include "CompositeDistributionFunction.h"
#include <bittricks/bittricks.h>

Spectrum CompositeDistributionFunction
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  Spectrum result(0,0,0);
  size_t i = 0;
  for(const_iterator f = begin();
      i != mSize;
      ++i, ++f)
  {
    result += (*f)->evaluate(wo,dg,wi);
  } // end for f

  return result;
} // end CompositeDistributionFunction::evaluate()

Spectrum CompositeDistributionFunction
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector3 &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &index) const
{
  size_t i = std::min<size_t>(mSize - 1, ifloor(u2 * mSize));

  // select component
  const ScatteringDistributionFunction *sampleMe = (*this)[i];

  // sample it
  Spectrum result = sampleMe->sample(wo, dg, u0, u1, u2, wi, pdf, delta, index);
  index = i;

  // average other components' pdfs if this component isn't specular
  if(!delta)
  {
    // sum over techniques
    for(size_t j = 0; j < mSize; ++j)
    {
      if(i != j)
      {
        pdf += (*this)[j]->evaluatePdf(wi,dg,wo);
      } // end if
    } // end for m
  } // end if

  // finally, divide by the number of components
  // What happens if the component we picked was a delta function?
  // We don't average in the others' pdfs, but what we are left with
  // is a delta pdf represented with 1.0 multiplied by the discrete
  // probability of choosing the component we did: 1.0 / mSize
  pdf /= mSize;

  // evaluate the components and sum their results
  for(size_t j = 0; j < mSize; ++j)
  {
    if(i != i) result += evaluate(wo, dg, wi);
  } // end for i

  return result;
} // end CompositeDistributionFunction::sample()

ScatteringDistributionFunction *CompositeDistributionFunction
  ::clone(FunctionAllocator &allocator) const
{
  ScatteringDistributionFunction *result = Parent0::clone(allocator);
  if(result != 0)
  {
    // clone the components
    for(size_t i = 0; i < mSize; ++i)
    {
      ScatteringDistributionFunction *component = (*this)[i]->clone(allocator);
      if(component == 0) return 0;

      (*static_cast<CompositeDistributionFunction*>(result))[i] = component;
    } // end for i
  } // end if

  return result;
} // end CompositeDistributionFunction::clone()

bool CompositeDistributionFunction
  ::isSpecular(void) const
{
  for(size_t i = 0; i < mSize; ++i)
    if(!(*this)[i]->isSpecular()) return false;

  return true;
} // end CompositeDistributionFunction::isSpecular()

Spectrum CompositeDistributionFunction
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  pdf = 0;
  Spectrum result(Spectrum::black());

  // sum the non-delta functions into the result & pdf
  size_t i = 0;
  float temp;
  for(const_iterator f = begin();
      i != mSize;
      ++i, ++f)
  {
    if(!(*f)->isSpecular())
    {
      result += (*f)->evaluate(wo,dg,wi, delta, component, temp);
      pdf += temp;
    } // end if
  } // end for f

  // if we know that wi came from a delta function,
  // then we need to include that component when we compute
  // our results
  if(delta)
  {
    // evaluate the delta function
    // bash what the previous stuff did with the pdf
    result += (*this)[component]->evaluate(wo, dg, wi, delta, component, pdf);

    // pdf should equal 1 now
  } // end if

  // divide pdf by the number of components
  pdf /= mSize;
  return result;
} // end ComponentIndex::evaluate()

float CompositeDistributionFunction
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi,
                const bool delta,
                const ComponentIndex component) const
{
  float result = 0;
  if(delta)
  {
    result = (mSize != 0) ? 1.0f / mSize : 0;
  } // end if
  else
  {
    result = evaluatePdf(wo,dg,wi);
  } // end else

  return result;
} // end CompositeDistributionFunction::evaluatePdf()

float CompositeDistributionFunction
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi) const
{
  float result = 0;

  // take the average of all components' pdf
  for(size_t i = 0; i < mSize; ++i)
  {
    result += (*this)[i]->evaluatePdf(wo,dg,wi);
  } // end for i

  return (mSize != 0) ? result / mSize : 0;
} // end evaluatePdf()

