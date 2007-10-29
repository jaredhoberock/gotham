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

  unsigned int totalPixels = film->getWidth() * film->getHeight();
  unsigned int totalSamples = (mSamplesPerPixel * mSamplesPerPixel) * totalPixels;
  float invSpp = 1.0f / (mSamplesPerPixel * mSamplesPerPixel);
  float invTotalSamples = 1.0f / totalSamples;

  const Scene *s = mScene.get();

  HilbertSequence sequence(0, 1.0f, 0, 1.0f,
                           film->getWidth() * mSamplesPerPixel,
                           film->getHeight() * mSamplesPerPixel);

  std::vector<PathSampler::Result> results;
  PathSampler::HyperPoint x;
  float px, py;

  progress.restart(totalSamples);
  while(sequence(px,py, z(), z()))
  {
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
      mRecord->record(invSpp, x, p, results);
    } // end if

    // purge all malloc'd memory for this sample
    ScatteringDistributionFunction::mPool.freeAll();

    ++progress;
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

#ifdef WIN32
#define OPENEXR_DLL
#endif // WIN32

#include <halfLimits.h>
#include <ImfRgbaFile.h>
#include <ImfRgba.h>

static void writeEXR(const char *filename, const RandomAccessFilm &image)
{
  size_t w = image.getWidth();
  size_t h = image.getHeight();

  // convert image to rgba
  std::vector<Imf::Rgba> pixels(image.getWidth() * image.getHeight());
  for(unsigned int y = 0; y < image.getHeight(); ++y)
  {
    for(unsigned int x = 0; x < image.getWidth(); ++x)
    {
      // flip the image because EXR is stored ass-backwards
      pixels[(h-1-y)*w + x].r = image.raster(x,y)[0];
      pixels[(h-1-y)*w + x].g = image.raster(x,y)[1];
      pixels[(h-1-y)*w + x].b = image.raster(x,y)[2];
      pixels[(h-1-y)*w + x].a = 1.0f;
    } // end for
  } // end for

  Imf::RgbaOutputFile file(filename, image.getWidth(), image.getHeight(), Imf::WRITE_RGBA);
  file.setFrameBuffer(&pixels[0], 1, image.getWidth());
  file.writePixels(image.getHeight());
} // end writeEXR()

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

