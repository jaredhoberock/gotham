/*! \file EqualVisitImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of EqualVisitImportance class.
 */

#include "EqualVisitImportance.h"
#include "../records/RandomAccessFilm.h"
#include "../renderers/MetropolisRenderer.h"

EqualVisitImportance
  ::EqualVisitImportance(const bool doInterpolate)
    :Parent(),mDoInterpolate(doInterpolate)
{
  ;
} // end EqualVisitImportance::EqualVisitImportance()

void EqualVisitImportance
  ::preprocess(const boost::shared_ptr<RandomSequence> &r,
               const boost::shared_ptr<const Scene> &scene,
               const boost::shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  // very important that this assignment occur first
  mAcceptance = renderer.getAcceptanceImage();

  const_cast<RenderFilm*>(mAcceptance)->fill(Spectrum(1,1,1));

  mConstantImportance.preprocess(r,scene,mutator,renderer);
  Parent::preprocess(r,scene,mutator,renderer);
} // end EqualVisitImportance::preprocess()

float EqualVisitImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  if(results.empty()) return 0;

  // XXX don't actually know if this business is correct yet
  float result = mConstantImportance(x,xPath,results);
  // divide by constant's normalization constant
  // rationale: dividing some integrand by a unitless
  // ratio makes no sense
  // instead, make the function unitless
  result *= mConstantImportance.getInvNormalizationConstant();

  // how many accepts has each pixel received, on average?
  float totalAccepts = mAcceptance->getSum()[0];
  if(totalAccepts != 0)
  {
    float numPixels = mAcceptance->getWidth() * mAcceptance->getHeight();
    float avg = totalAccepts / numPixels;

    // look up the number of visits this point has received
    float visits;
    if(mDoInterpolate)
    {
      // interpolating visits makes for a much smoother result
      visits = mAcceptance->bilerp(x[0][0], x[0][1])[0];
    } // end if
    else
    {
      visits = mAcceptance->pixel(x[0][0], x[0][1])[0];
    } // end else

    // we need to return the ratio of the number of times
    // accepted over the average acceptance rate
    // or something like that
    result *= (avg / visits);
  } // end if

  return result;
} // end EqualVisitImportance::evaluate()

