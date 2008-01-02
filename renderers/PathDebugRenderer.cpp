/*! \file PathDebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PathDebugRenderer class.
 */

#include "PathDebugRenderer.h"
#include "../path/Path.h"
#include "../primitives/Scene.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../records/RenderFilm.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../geometry/Ray.h"
#include "../path/KajiyaSampler.h"
#include "../path/SimpleBidirectionalSampler.h"
#include <stratifiedsequence/StratifiedSequence.h>
#include <hilbertsequence/HilbertSequence.h>

using namespace boost;
using namespace gpcpu;

PathDebugRenderer
  ::PathDebugRenderer(void)
    :Parent()
{
  ;
} // end PathDebugRenderer::PathDebugRenderer()

PathDebugRenderer
  ::PathDebugRenderer(shared_ptr<const Scene> s,
                      shared_ptr<Record> r,
                      const shared_ptr<RandomSequence> &sequence,
                      const shared_ptr<PathSampler> &sampler)
    :Parent(s,r,sequence),mSampler(sampler)
{
  ;
} // end PathDebugRenderer::PathDebugRenderer()

PathDebugRenderer
  ::PathDebugRenderer(const shared_ptr<RandomSequence> &s,
                      const shared_ptr<PathSampler> &sampler)
    :Parent(s),mSampler(sampler)
{
  ;
} // end PathDebugRenderer::PathDebugRenderer()

void PathDebugRenderer
  ::kernel(ProgressCallback &progress)
{
  // shorthand
  RandomSequence &z = *mRandomSequence.get();

  Path p;
  float2 uv(0,0);

  // XXX TODO kill this grossness
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  const Scene *s = mScene.get();

  HilbertSequence sequence(0, 1.0f, 0, 1.0f,
                           film->getWidth() * mSamplesPerPixel,
                           film->getHeight() * mSamplesPerPixel);

  std::vector<PathSampler::Result> results;
  PathSampler::HyperPoint x;
  float px, py;

  // initialize the HaltCriterion
  // before we start rendering
  mHalt->init(this, &progress);

  // main loop
  while(!(*mHalt)())
  {
    // get a new film plane sample
    sequence(px, py, z(), z());

    Spectrum L(0.1f, 0.1f, 0.1f);

    // construct a hyperpoint
    mSampler->constructHyperPoint(z, x);

    // replace the first two coords with
    // pixel samples
    x[0][0] = px;
    x[0][1] = py;

    // sample a path
    if(mSampler->constructPath(*s, x, p))
    {
      results.clear();
      mSampler->evaluate(*s, p, results);
      mRecord->record(1.0f, x, p, results);
    } // end if

    // purge all malloc'd memory for this sample
    ScatteringDistributionFunction::mPool.freeAll();

    ++mNumSamples;
  } // end for i
} // end PathDebugRenderer::kernel()

void PathDebugRenderer
  ::setSampler(const boost::shared_ptr<PathSampler> &s)
{
  mSampler = s;
} // end PathDebugRenderer::setSampler()

void PathDebugRenderer
  ::preprocess(void)
{
  Parent::preprocess();
} // end PathDebugRenderer::preprocess()

void PathDebugRenderer
  ::postprocess(void)
{
  Parent::postprocess();

  // XXX TODO kill this nastiness
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  if(film)
  {
    // compute the average pixel value
    Spectrum avg(0,0,0);
    for(size_t j = 0; j < film->getHeight(); ++j)
    {
      for(size_t i = 0; i < film->getWidth(); ++i)
      {
        avg += film->raster(i,j);
      } // end for i
    } // end for j

    std::cout << "Average pixel value: " << avg / (film->getWidth() * film->getHeight()) << std::endl;
  } // end if
} // end PathDebugRenderer::postprocess()

