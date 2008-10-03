/*! \file RussianRoulette.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RussianRoulette classes.
 */

#include "RussianRoulette.h"
#include "../include/detail/Spectrum.h"
#include "../include/detail/DifferentialGeometry.h"
#include "../include/detail/Vector.h"

float RussianRoulette
  ::operator()(void) const
{
  return 1.0f;
} // end RussianRoulette::operator()()

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
  ::operator()(void) const
{
  return mContinueProbability;
} // end ConstantRoulette::operator()()

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

KelemenRoulette
  ::KelemenRoulette(const float continueProbability)
    :Parent(continueProbability)
{
  ;
} // end KelemenRoulette::KelemenRoulette()

float KelemenRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  if(i == 1) return mContinueProbability;
  if(fromDelta) return 1.0f;

  // return MaxOverSpectrumRoulette
  float fs = f.maxElement();
  float psaPdf = pdf / dg.getNormal().absDot(w);
  
  // we convert pdf to projected solid angle pdf before dividing fs
  return std::min(1.0f, fs / psaPdf);
} // end KelemenRoulette::operator()()

VeachRoulette
  ::VeachRoulette(const size_t minimalSubpathLength)
    :Parent(),mMinimalSubpathLength(std::max<size_t>(1,minimalSubpathLength))
{
  ;
} // end VeachRoulette::VeachRoulette()

float VeachRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  if(i < mMinimalSubpathLength || fromDelta) return 1.0f;

  // return MaxOverSpectrumRoulette
  float fs = f.maxElement();
  float psaPdf = pdf / dg.getNormal().absDot(w);
  
  // we convert pdf to projected solid angle pdf before dividing fs
  return std::min(1.0f, fs / psaPdf);
} // end VeachRoulette::operator()()

ModifiedKelemenRoulette
  ::ModifiedKelemenRoulette(const float beginProbability)
    :Parent(),mBeginProbability(beginProbability)
{
  ;
} // end ModifiedKelemenRoulette::ModifiedKelemenRoulette()

float ModifiedKelemenRoulette
  ::operator()(void) const
{
  return mBeginProbability;
} // end ModifiedKelemenRoulette::operator()()

float ModifiedKelemenRoulette
  ::operator()(const unsigned int i,
               const Spectrum &f,
               const DifferentialGeometry &dg,
               const Vector &w,
               const float &pdf,
               const bool fromDelta) const
{
  if(fromDelta) return 1.0f;

  // return MaxOverSpectrumRoulette
  float fs = f.maxElement();
  float psaPdf = pdf / dg.getNormal().absDot(w);
  
  // the value we clamp to has a very strong effect on variance
  // for the Simple*RussianRouletteSamplers
  // For scenes similar to the cornell box, most of the energy is
  // concentrated in the shorter paths, so always lengthening the
  // Path (by clamping to 1.0) isn't necessarily a good form of
  // importance sampling
  // Perhaps a better function would check how the Path throughput
  // changes as the Path is lengthened?
  //
  // On the other hand, we are most interested in this function as it applies
  // to finding long paths in difficult sampling conditions.
  //
  // For now, use a form of defensive sampling for this roulette function:
  // 0.5 instead of 1.0
  // we always want there to be at least a small chance
  // for roulette to kill a path
  return std::min(0.5f, fs / psaPdf);
} // end ModifiedKelemenRoulette::operator()()

