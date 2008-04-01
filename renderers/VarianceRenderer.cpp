/*! \file VarianceRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of VarianceRecord class.
 */

#include "VarianceRenderer.h"
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

VarianceRenderer
  ::VarianceRenderer(void)
{
  ;
} // end VarianceRenderer::VarianceRenderer()

VarianceRenderer
  ::VarianceRenderer(const boost::shared_ptr<RandomSequence> &s,
                     const boost::shared_ptr<PathSampler> &sampler)
    :Parent(s,sampler)
{
  ;
} // end VarianceRenderer::VarianceRenderer()

void VarianceRenderer
  ::setVarianceRecord(const boost::shared_ptr<Record> &r)
{
  mVarianceRecord = r;
} // end VarianceRenderer::setVarianceRecord()

void VarianceRenderer
  ::preprocess(void)
{
  mVarianceRecord->preprocess();

  mSamplesImage.resize(static_cast<RenderFilm*>(mVarianceRecord.get())->getWidth(),
                       static_cast<RenderFilm*>(mVarianceRecord.get())->getHeight());
  mSamplesImage.fill(0);

  Parent::preprocess();
} // end VarianceRenderer::preprocess()

void VarianceRenderer
  ::postprocess(void)
{
  Parent::postprocess();

  RenderFilm *v = dynamic_cast<RenderFilm*>(mVarianceRecord.get());

  // divide variance by spp-1
  for(size_t j = 0; j < v->getHeight(); ++j)
  {
    for(size_t i = 0; i < v->getWidth(); ++i)
    {
      static_cast<RandomAccessFilm*>(v)->raster(i,j) /= (mSamplesImage.raster(i,j) - 1.0f);
    } // end for i
  } // end for j

  RenderFilm normalizedVariance = *v;
  RenderFilm rms = *v;
  RenderFilm normalizedRms = *v;

  // rescale variance image so it has 1/2 mean luminance
  float s = 0.5f / v->computeMean().luminance();
  v->scale(Spectrum(s,s,s));
  v->postprocess();

  RenderFilm *mean = dynamic_cast<RenderFilm*>(mRecord.get());

  //// create a 1/2 scale mean
  //RandomAccessFilm lowRes(mean->getWidth()/2, mean->getHeight()/2);
  //for(size_t j = 0; j < lowRes.getHeight(); ++j)
  //{
  //  for(size_t i = 0; i < lowRes.getWidth(); ++i)
  //  {
  //    lowRes.raster(i,j) = mean->raster(2*i,    2*j)
  //                       + mean->raster(2*i+1,  2*j)
  //                       + mean->raster(2*i+1,  2*j+1)
  //                       + mean->raster(2*i,    2*j+1);
  //    lowRes.raster(i,j) /= 4;
  //  } // end for i
  //} // end for j
  // create a 1/2 scale mean
  RandomAccessFilm lowRes(mean->getWidth()/4, mean->getHeight()/4);
  for(size_t j = 0; j < lowRes.getHeight(); ++j)
  {
    for(size_t i = 0; i < lowRes.getWidth(); ++i)
    {
      lowRes.raster(i,j) = mean->raster(4*i,    4*j)
                         + mean->raster(4*i+1,  4*j)
                         + mean->raster(4*i+2,  4*j)
                         + mean->raster(4*i+3,  4*j)
                         + mean->raster(4*i,    4*j+1)
                         + mean->raster(4*i+1,  4*j+1)
                         + mean->raster(4*i+2,  4*j+1)
                         + mean->raster(4*i+3,  4*j+1)
                         + mean->raster(4*i,    4*j+2)
                         + mean->raster(4*i+1,  4*j+2)
                         + mean->raster(4*i+2,  4*j+2)
                         + mean->raster(4*i+3,  4*j+2)
                         + mean->raster(4*i,    4*j+3)
                         + mean->raster(4*i+1,  4*j+3)
                         + mean->raster(4*i+2,  4*j+3)
                         + mean->raster(4*i+3,  4*j+3);
      lowRes.raster(i,j) /= 16;
    } // end for i
  } // end for j

  for(size_t j = 0; j < mean->getHeight(); ++j)
  {
    for(size_t i = 0; i < mean->getWidth(); ++i)
    {
      static_cast<RandomAccessFilm&>(normalizedVariance).raster(i,j) /= mean->raster(i,j);

      const Spectrum rmsPixel = rms.raster(i,j);
      static_cast<RandomAccessFilm&>(rms).raster(i,j) = Spectrum(sqrtf(rmsPixel[0]), sqrtf(rmsPixel[1]), sqrtf(rmsPixel[2]));

      Spectrum denom = mean->raster(i,j);
      //Spectrum denom = lowRes.raster(i/4,j/4);
      denom += Spectrum(1.0/256, 1.0/256, 1.0/256);
      static_cast<RandomAccessFilm&>(normalizedRms).raster(i,j) = rms.raster(i,j) / denom;
    } // end for i
  } // end for j

  // scale all images so that they have 1/2 mean luminance
  s = 0.5f / normalizedVariance.computeMean().luminance();
  //normalizedVariance.scale(Spectrum(s,s,s));
  normalizedVariance.writeEXR("normalized-variance.exr");

  s = 0.5f / rms.computeMean().luminance();
  //rms.scale(Spectrum(s,s,s));
  rms.writeEXR("rms.exr");

  s = 0.5f / normalizedRms.computeMean().luminance();
  //normalizedRms.scale(Spectrum(s,s,s));
  normalizedRms.writeEXR("normalized-rms.exr");

  RenderFilm adHocError = *mean;

  for(size_t j = 0; j < mean->getHeight(); ++j)
  {
    for(size_t i = 0; i < mean->getWidth(); ++i)
    {
      Spectrum s2(Spectrum::black());
      Spectrum middle = mean->raster(i,j);
      // XXX apply tone map here
      //middle[0] = std::min(1.0f, middle[0]);
      //middle[1] = std::min(1.0f, middle[1]);
      //middle[2] = std::min(1.0f, middle[2]);

      unsigned int n = 0;
      if(i != 0)
      {
        Spectrum left = mean->raster(i-1,j);
        // XXX apply tone map here
        //left[0] = std::min(1.0f, left[0]);
        //left[1] = std::min(1.0f, left[1]);
        //left[2] = std::min(1.0f, left[2]);

        Spectrum diff = middle - left;
        diff *= diff;
        s2 += diff;
        ++n;
      } // end if

      if(i + 1 != mean->getWidth())
      {
        Spectrum right = mean->raster(i+1,j);
        // XXX apply tone map here
        //right[0] = std::min(1.0f, right[0]);
        //right[1] = std::min(1.0f, right[1]);
        //right[2] = std::min(1.0f, right[2]);
        
        Spectrum diff = middle - right;
        diff *= diff;
        s2 += diff;
        ++n;
      } // end if

      if(j != 0)
      {
        Spectrum bottom = mean->raster(i,j-1);
        // XXX apply tone map here
        //bottom[0] = std::min(1.0f, bottom[0]);
        //bottom[1] = std::min(1.0f, bottom[1]);
        //bottom[2] = std::min(1.0f, bottom[2]);

        Spectrum diff = middle - bottom;
        diff *= diff;
        s2 += diff;
        ++n;
      } // end if

      if(j + 1 != mean->getHeight())
      {
        Spectrum top = mean->raster(i,j+1);
        // XXX apply tone map here
        //top[0] = std::min(1.0f, top[0]);
        //top[1] = std::min(1.0f, top[1]);
        //top[2] = std::min(1.0f, top[2]);

        Spectrum diff = middle - top;
        diff *= diff;
        s2 += diff;
        ++n;
      } // end if

      s2 /= n;

      //s2[0] = sqrtf(s2[0]);
      //s2[1] = sqrtf(s2[1]);
      //s2[2] = sqrtf(s2[2]);

      static_cast<RandomAccessFilm&>(adHocError).raster(i,j) = s2;
    } // end for i
  } // end for j

  adHocError.writeEXR("adhoc.exr");
} // end VarianceRenderer::postprocess()

void VarianceRenderer
  ::kernel(ProgressCallback &progress)
{
  // shorthand
  RandomSequence &z = *mRandomSequence.get();

  Path p;

  // XXX TODO kill this grossness
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  // XXX TODO kill this grossness
  RenderFilm *variance = dynamic_cast<RenderFilm*>(mVarianceRecord.get());
  variance->resize(film->getWidth(), film->getHeight());
  variance->fill(Spectrum::black());

  unsigned int totalPixels = film->getWidth() * film->getHeight();
  unsigned int totalSamples = (mSamplesPerPixel * mSamplesPerPixel) * totalPixels;

  const Scene *s = mScene.get();

  HilbertSequence sequence(0, 1.0f, 0, 1.0f,
                           film->getWidth() * mSamplesPerPixel,
                           film->getHeight() * mSamplesPerPixel);

  std::vector<PathSampler::Result> results;
  PathSampler::HyperPoint x;
  float px, py;

  unsigned int n = 0;

  gpcpu::size2 raster, oldRaster(0, 0); 

  std::vector<Spectrum> samples;

  progress.restart(totalSamples);
  while(sequence(px,py, z(), z()))
  {
    variance->getRasterPosition(px, py, raster[0], raster[1]);
    raster[0] = std::min<size_t>(raster[0], film->getWidth() - 1);
    raster[1] = std::min<size_t>(raster[1], film->getHeight() - 1);

    //if(raster[0] != oldRaster[0] || raster[1] != oldRaster[1])
    //{
    //  depositVariance(oldRaster[0], oldRaster[1], n, samples);
    //  samples.clear();
    //  n = 0;
    //  oldRaster[0] = raster[0];
    //  oldRaster[1] = raster[1];
    //} // end if

    // construct a hyperpoint
    mSampler->constructHyperPoint(z, x);

    // replace the first two coords with
    // pixel samples
    x[0][0] = px;
    x[0][1] = py;

    // get the old mean
    Spectrum oldMean = dynamic_cast<const RandomAccessFilm*>(mRecord.get())->pixel(px,py);

    // sample a path
    results.clear();
    if(mSampler->constructPath(*s, *mShadingContext, x, p))
    {
      mSampler->evaluate(*s, p, results);
      mRecord->record(1.0f, x, p, results);

      // XXX BUG this won't work with all samplers
      Spectrum L(Spectrum::black());
      for(size_t i = 0; i < results.size(); ++i)
      {
        L += results[i].mThroughput * results[i].mWeight / results[i].mPdf;
      } // end for i

      //samples.push_back(L);
    } // end if
    
    depositVariance(x, p, results, oldMean);

    ++n;

    // purge all malloc'd memory for this sample
    mShadingContext->freeAll();

    ++mNumSamples;
    ++progress;
  } // end for i

  //if(oldRaster[0] != UINT_MAX
  //   && (raster[0] != oldRaster[0] || raster[1] != oldRaster[1]))
  //{
  //  depositVariance(oldRaster[0], oldRaster[1], n, samples);
  //  samples.clear();
  //  n = 0;
  //} // end if
} // end VarianceRenderer::kernel()

void VarianceRenderer
  ::depositVariance(const size_t px,
                    const size_t py,
                    const size_t n,
                    const std::vector<Spectrum> &samples)
{
  // XXX TODO kill this grossness
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  // XXX TODO kill this grossness
  RenderFilm *variance = dynamic_cast<RenderFilm*>(mVarianceRecord.get());

  Spectrum s2;
  Spectrum mean = film->raster(px, py) / n;

  // account for samples which were zeros
  s2 = mean * mean;
  s2 *= static_cast<float>(n - samples.size());

  // sum the squares of distances per channel
  // for each non-zero sample
  for(size_t i = 0; i != samples.size(); ++i)
  {
    const Spectrum &x = samples[i];
    Spectrum del = x - mean;
    s2 += del * del;
  } // end for i

  // divide
  s2 /= n;

  // record
  variance->deposit(px, py, s2);
} // end VarianceRenderer::depositVariance()

void VarianceRenderer
  ::depositVariance(const PathSampler::HyperPoint &x,
                    const Path &xPath,
                    const std::vector<PathSampler::Result> &results,
                    Spectrum oldMean)
{
  // XXX BUG this won't work with all samplers
  float px = x[0][0];
  float py = x[0][1];

  Spectrum sample(Spectrum::black());
  for(size_t i = 0; i != results.size(); ++i)
  {
    sample += results[i].mWeight * results[i].mThroughput / results[i].mPdf;
  } // end for i

  // tonemap
  sample[0] = std::min(sample[0], 1.0f);
  sample[1] = std::min(sample[1], 1.0f);
  sample[2] = std::min(sample[2], 1.0f);

  // from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Algorithm_III
  // update n
  float n = (mSamplesImage.element(px,py) += 1.0f);

  // get the current mean
  Spectrum mean = dynamic_cast<const RandomAccessFilm*>(mRecord.get())->pixel(px,py) / n;

  // tonemap
  mean[0] = std::min(mean[0], 1.0f);
  mean[1] = std::min(mean[1], 1.0f);
  mean[2] = std::min(mean[2], 1.0f);

  // compute delta from the old mean
  
  // tonemap
  oldMean[0] = std::min(oldMean[0], 1.0f);
  oldMean[1] = std::min(oldMean[1], 1.0f);
  oldMean[2] = std::min(oldMean[2], 1.0f);

  Spectrum delta;
  if(n-1 > 0)
  {
    delta = sample - oldMean/(n-1);
  } // end if
  else
  {
    delta = sample;
  } // end else

  // sum
  //std::cerr << "px: " << px << std::endl;
  //std::cerr << "py: " << py << std::endl;
  size_t rx, ry;
  RenderFilm *vFilm = dynamic_cast<RenderFilm*>(mVarianceRecord.get());
  vFilm->getRasterPosition(px, py, rx, ry);
  rx = std::min<size_t>(rx, vFilm->getWidth()-1);
  ry = std::min<size_t>(ry, vFilm->getHeight()-1);

  vFilm->deposit(rx,ry, delta * (sample - mean));
} // end VarianceRenderer::depositVariance()

