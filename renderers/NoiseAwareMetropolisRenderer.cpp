/*! \file NoiseAwareMetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of NoiseAwareMetropolisRenderer class.
 */

#include "NoiseAwareMetropolisRenderer.h"
#include "../records/VarianceFilm.h"
#include "../importance/LuminanceImportance.h"
#include "../importance/EstimateImportance.h"
#include "../importance/TargetImportance.h"

NoiseAwareMetropolisRenderer
  ::NoiseAwareMetropolisRenderer(void)
    :Parent()
{
  ;
} // end NoiseAwareMetropolisRenderer::NoiseAwareMetropolisRenderer()

NoiseAwareMetropolisRenderer
  ::NoiseAwareMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance,
                                 const float varianceExponent)
    :Parent(sequence,m,importance),
     mVarianceExponent(varianceExponent)
{
  ;
} // end NoiseAwareMetropolisRenderer::NoiseAwareMetropolisRenderer()

float NoiseAwareMetropolisRenderer
  ::updateImportance(const float bLuminance,
                     const float w,
                     const float h,
                     const PathSampler::HyperPoint &x,
                     const Path &xPath,
                     const std::vector<PathSampler::Result> &xResults,
                     float &ix,
                     float &xPdf)
{
  using namespace boost;
  LuminanceImportance *lumImportance = dynamic_cast<LuminanceImportance*>(mImportance.get());
  if(lumImportance != 0)
  {
    // we don't have an estimate yet
    return Parent::updateImportance(bLuminance, w, h, x, xPath, xResults, ix, xPdf);
  } // end if

  // prepare the current estimate
  shared_ptr<RenderFilm> current = dynamic_pointer_cast<RenderFilm,Record>(mRecord);
  float s;

  shared_ptr<RandomAccessFilm> lowResEstimate(new RandomAccessFilm(static_cast<size_t>(ceilf(w)),
                                                                   static_cast<size_t>(ceilf(h))));
  current->resample(*lowResEstimate);

  // scale estimate so it has mean luminance equal to bLuminance
  s = bLuminance / lowResEstimate->computeMean().luminance();
  lowResEstimate->scale(Spectrum(s,s,s));

  VarianceFilm *varianceFilm = dynamic_cast<VarianceFilm*>(mRecord.get());
  if(varianceFilm == 0)
  {
    // we don't have a variance estimate yet, so on the upcoming round, we
    // need to start making one
    varianceFilm = new VarianceFilm(*current, lowResEstimate);

    varianceFilm->getVariance().setFilename(std::string("variance.exr"));

    // preprocess
    varianceFilm->getVariance().preprocess();
    
    // reset the record to be a varianceFilm
    mRecord.reset(varianceFilm);

    // handoff to parent
    return Parent::updateImportance(bLuminance, w, h, x, xPath, xResults, ix, xPdf);
  } // end if
  else
  {
    // replace the mean estimate with a newer one
    varianceFilm->setMeanEstimate(lowResEstimate);
  } // end else

  // make a temp of variance
  RandomAccessFilm tempVar = varianceFilm->getVariance();

  // scale temp var by 1/spp to make it a legit estimate
  float spp = static_cast<float>(mNumSamples) / (tempVar.getWidth() * tempVar.getHeight());
  float invSpp = 1.0f / spp;
  tempVar.scale(Spectrum(invSpp,invSpp,invSpp));

  // prepare target importance
  RandomAccessFilm target(static_cast<size_t>(ceilf(w)),
                          static_cast<size_t>(ceilf(h)));
  prepareTargetImportance(*lowResEstimate, tempVar, target);

  // replace the current importance with a new one
  mImportance.reset(new TargetImportance(*lowResEstimate, target, mTargetFilename));
  
  // update invB
  // XXX we shouldn't have to use a separate sequence
  RandomSequence seq(13u);
  float invB = mImportance->estimateNormalizationConstant(seq, mScene, mMutator, 10000);
  invB = 1.0f / invB;

  // update x's importance & pdf
  // compute importance
  ix = (*mImportance)(x, xPath, xResults);

  // compute pdf of x
  xPdf = ix * invB;

  return invB;
} // end NoiseAwareMetropolisRenderer::updateImportance()

void NoiseAwareMetropolisRenderer
  ::prepareTargetImportance(const RandomAccessFilm &mean,
                            const RandomAccessFilm &variance,
                            RandomAccessFilm &target) const
{
  //char buf[256];

  // resample the current estimate of variance into target
  variance.resample(target);

  //sprintf(buf, "variance-%dx%d.exr", target.getWidth(), target.getHeight());
  ////target.writeEXR(buf);

  // create some function of the variance -- call it rms
  target.applyPow(mVarianceExponent);
  //sprintf(buf, "rms-%dx%d.exr", target.getWidth(), target.getHeight());
  //target.writeEXR(buf);

  RandomAccessFilm meanCopy = mean;

  // turn mean into tvi
  RandomAccessFilm tvi = meanCopy;
  RandomAccessFilm down(tvi.getWidth(),tvi.getHeight());
  tvi.resample(down);
  down.resample(tvi);
  tvi.applyThresholdVersusIntensityFunction();
  //sprintf(buf, "tvi-%dx%d.exr", target.getWidth(), target.getHeight());
  //tvi.writeEXR(buf);

  // divide target by tvi
  target.divideLuminance(tvi, 0);
  RandomAccessFilm visualError = target;
  float meanTargetLum = target.computeMean().luminance();
  float s = 0.5f / meanTargetLum;
  visualError.scale(Spectrum(s,s,s));
  //sprintf(buf, "visual-error-%dx%d.exr", target.getWidth(), target.getHeight());
  //visualError.writeEXR(buf);
  
  // make it so the ratio of the max / min == 2
  float low = target.computeLuminancePercentileIgnoreZero(0.05f);
  float high = target.computeLuminancePercentileIgnoreZero(0.95f);
  float r = high / low;
  if(r > 1.0f)
  {
    //float power = 1.0 / log10f(r);
    float power = log10f(2.0f) / log10f(r);
    target.applyPow(power);

    RandomAccessFilm scaledVisualError = target;
    float meanTargetLum = target.computeMean().luminance();
    float s = 0.5f / meanTargetLum;
    scaledVisualError.scale(Spectrum(s,s,s));
    //sprintf(buf, "restricted-visual-error-%dx%d.exr", target.getWidth(), target.getHeight());
    //scaledVisualError.writeEXR(buf);
  } // end if

  // massage normalized rms
  // remove extreme lows
  // remove everything below the 5th percentile
  // and above the 95th percentile
  low = target.computeLuminancePercentileIgnoreZero(0.05f);
  high = target.computeLuminancePercentileIgnoreZero(0.95f);
  target.clampLuminance(low, high);
  //sprintf(buf, "removed-extremes-%dx%d.exr", target.getWidth(), target.getHeight());
  //target.writeEXR(buf);

  // for every zero pixel in the mean, make that pixel very important in the target
  for(size_t y = 0; y != mean.getHeight(); ++y)
  {
    for(size_t x = 0; x != mean.getWidth(); ++x)
    {
      if(mean.raster(x,y).luminance() == 0)
      {
        target.raster(x,y).setLuminance(high);
      } // end if
    } // end for x
  } // end for y

  //sprintf(buf, "removed-zeroes-%dx%dx.exr", target.getWidth(), target.getHeight());
  //target.writeEXR(buf);

  // filter target
  target.bilateralFilter(1.0f, 100.0f, target);
  //sprintf(buf, "filtered-tonemapped-normalized-rms-%dx%d.exr", target.getWidth(), target.getHeight());
  //target.writeEXR(buf);

  //// scale the target so that it has mean luminance 0.5
  //float meanLum = target.computeMean().luminance();
  //target.scale(Spectrum(1.0f / (0.299f * meanLum),
  //                      1.0f / (0.587f * meanLum),
  //                      1.0f / (0.114f * meanLum)));

  //target.scale(Spectrum(0.299f * 0.5f,
  //                      0.578f * 0.5f,
  //                      0.114f * 0.5f));

  //sprintf(buf, "target-%dx%d.exr", target.getWidth(), target.getHeight());
  //target.writeEXR(buf);
} // end NoiseAwareMetropolisRenderer::prepareTargetImportance()

void NoiseAwareMetropolisRenderer
  ::setTargetFilename(const std::string &filename)
{
  mTargetFilename = filename;
} // end NoiseAwareMetropolisRenderer::setTargetFilename()

