/*! \file NormalizedImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of NormalizedImportance class.
 */

#include "NormalizedImportance.h"
#include <stratifiedsequence/StratifiedSequence.h>
#include "../path/KelemenSampler.h"
#include "../path/Path.h"
#include "../shading/ScatteringDistributionFunction.h"
#include <gpcpu/Vector.h>

using namespace boost;
using namespace gpcpu;

void NormalizedImportance
  ::preprocess(const shared_ptr<RandomSequence> &r,
               const shared_ptr<const Scene> &scene,
               const shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  const unsigned int n = 10000;

  unsigned int sx = 4;
  unsigned int sy = 4;
  unsigned int spp = sx*sy;
  unsigned int w = static_cast<unsigned int>(sqrtf(static_cast<float>(n)/spp));
  unsigned int h = (n/spp) / w;
  float weight = 1.0f / spp;

  mEstimate.resize(w,h);
  mEstimate.fill(0);

  PathSampler::HyperPoint x;
  Path xPath;
  Spectrum L;

  float px, py;

  typedef std::vector<PathSampler::Result> ResultList;
  ResultList resultList;

  // XXX fix this
  PathSampler *sampler = const_cast<PathSampler*>(mutator->getSampler());
  StratifiedSequence sequence(0, 1.0f, 0, 1.0f, w*sx, h*sy);
  while(sequence(px, py, (*r)(), (*r)()))
  {
    PathSampler::constructHyperPoint(*r, x);
    x[0][0] = px;
    x[0][1] = py;

    // create a Path
    if(sampler->constructPath(*scene, x, xPath))
    {
      // evaluate the Path
      resultList.clear();
      L = mutator->evaluate(xPath, resultList);

      // deposit the luminance
      mEstimate.element(x[0][0], x[0][1]) += weight * L.luminance();
    } // end if

    ScatteringDistributionFunction::mPool.freeAll();
  } // end while

  // set each pixel to 1 over itself
  for(unsigned int j = 0; j < mEstimate.getDimensions()[1]; ++j)
  {
    for(unsigned int i = 0; i < mEstimate.getDimensions()[0]; ++i)
    {
      float &p = mEstimate.raster(i,j);

      // note that this is not biased, it's defensive sampling
      p = 1.0f / std::max(p, 0.01f);
    } // end for i
  } // end for i
} // end NormalizedImportance::preprocess()

float NormalizedImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Spectrum &f)
{
  // return f's luminance scaled by 1.0 / estimate
  return f.luminance() * mEstimate.element(x[0][0], x[0][1]);
} // end NormalizedImportance::operator()()

