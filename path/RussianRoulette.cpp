/*! \file RussianRoulette.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RussianRoulette classes.
 */

#include "RussianRoulette.h"
#include <spectrum/Spectrum.h>
#include "../geometry/DifferentialGeometry.h"
#include "../geometry/Vector.h"

float AlwaysRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  return 1.0f;
} // end AlwaysRoulette::operator()()

ConstantRoulette
  ::ConstantRoulette(const float continueProbability)
    :mContinueProbability(continueProbability)
{
  ;
} // end ConstantRoulette::ConstantRoulette()

float ConstantRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  return mContinueProbability;
} // end AlwaysRoulette::operator()()

float ConstantRoulette
  ::getContinueProbability(void) const
{
  return mContinueProbability;
} // end ConstantRoulette::getContinueProbability()

float ConstantAndAlwaysAfterDeltaRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  return fromDelta ? 1.0f : Parent::operator()(i,f,dg,w,pdf,fromDelta);
} // end AlwaysRoulette::operator()()

ConstantAndAlwaysAfterDeltaRoulette
  ::ConstantAndAlwaysAfterDeltaRoulette(const float continueProbability)
     :Parent(continueProbability)
{
  ;
} // end ConstantAndAlwaysAfterDeltaRoulette::ConstantAndAlwaysAfterDeltaRoulette()

float LuminanceRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  float fs = f.luminance();
  float psaPdf = pdf / dg.getNormal().absDot(w);
  
  // we convert pdf to projected solid angle pdf before dividing fs
  return std::min(1.0f, fs / psaPdf);
} // end AlwaysRoulette::operator()()

float MaxOverSpectrumRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  float fs = f.maxElement();
  float psaPdf = pdf / dg.getNormal().absDot(w);
  
  // we convert pdf to projected solid angle pdf before dividing fs
  return std::min(1.0f, fs / psaPdf);
} // end MaxOverSpectrumRoulette::operator()()

float OnlyAfterDeltaRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  return (i < 2 || fromDelta) ? 1.0f : 0.0f;
}; // end OnlyAfterDeltaRoulette::operator()()

