/*! \file EqualVisitImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of EqualVisitImportance class.
 */

#include "EqualVisitImportance.h"
#include "../films/RandomAccessFilm.h"
#include "../renderers/MetropolisRenderer.h"

void EqualVisitImportance
  ::preprocess(const boost::shared_ptr<RandomSequence> &r,
               const boost::shared_ptr<const Scene> &scene,
               const boost::shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  mAcceptance = renderer.getAcceptanceImage();
} // end EqualVisitImportance::preprocess()

float EqualVisitImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Spectrum &f)
{
  if(f.isBlack()) return 0;

  // how many accepts has each pixel received, on average?
  float totalAccepts = mAcceptance->getSum()[0];
  if(totalAccepts == 0) return 1.0f;

  float numPixels = mAcceptance->getWidth() * mAcceptance->getHeight();

  float avg = totalAccepts / numPixels;

  // we need to return the ratio of the number of times
  // accepted over the average acceptance rate
  // or something like that
  return avg / mAcceptance->pixel(x[0][0], x[0][1])[0];
} // end EqualVisitImportance::operator()()

