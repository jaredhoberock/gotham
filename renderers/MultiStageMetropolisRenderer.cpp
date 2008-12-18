/*! \file MultiStageMetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MultiStageMetropolisRenderer class.
 */

#include "MultiStageMetropolisRenderer.h"
#include "../importance/LuminanceImportance.h"
#include "../importance/EstimateImportance.h"
#include "../mutators/KelemenMutator.h"

MultiStageMetropolisRenderer
  ::MultiStageMetropolisRenderer(void)
    :Parent()
{
  ;
} // end MultiStageMetropolisRenderer::MultiStageMetropolisRenderer()

MultiStageMetropolisRenderer
  ::MultiStageMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance)
    :Parent(sequence,m,importance),mRecursionScale(0.5f)
{
  ;
} // end MultiStageMetropolisRenderer::MultiStageMetropolisRenderer()
                                
void MultiStageMetropolisRenderer
  ::kernel(ProgressCallback &progress)
{
  using namespace boost;

  // XXX TODO kill this nastiness
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  PathSampler::HyperPoint x, y;
  Path xPath, yPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults;

  // estimate 'canonical' normalization constant for luminance
  // XXX this is useful to have around in general
  // XXX MetropolisRenderer should do this in preprocess
  LuminanceImportance luminanceImportance;

  // use our own random sequence so we always get the same result here
  // this is particularly important for difficult scenes where it is hard
  // to agree on an estimate
  // XXX we shouldn't have to do this
  RandomSequence seq(13u);
  float bLuminance = luminanceImportance.estimateNormalizationConstant(seq, mScene, mShadingContext, mMutator, 10000);

  // now integrate the importance function that we are going to be using
  // XXX we should really choose a init path proportional to this function, not the luminance function
  //size_t initialPathIndex;
  //float b = mImportance->estimateNormalizationConstant(mSeedPoints, mSeedPaths, mSeedResults);
  //x = mSeedPoints[initialPathIndex];
  //mSeedPaths[initialPathIndex].clone(xPath, mLocalPool); 
  float b = mImportance->estimateNormalizationConstant(mRandomSequence, mScene, mShadingContext, mMutator,
                                                       10000, mLocalPool, x, xPath);

  float invB = 1.0f / b;

  // we owe these rays at the end
  unsigned long int owedRays = mScene->getRaysCast();

  // reset rays cast to 0
  mScene->setRaysCast(0);

  // evaluate the initial seed
  Spectrum f, g;
  f = mMutator->evaluate(xPath,xResults);
  float ix = (*mImportance)(x, xPath, xResults), iy = 0;
  float xPdf = ix * invB, yPdf = 0;

  // XXX this is still pretty gross
  float pLargeStep = 0;
  try
  {
    pLargeStep = dynamic_cast<KelemenMutator&>(*mMutator).getLargeStepProbability();
  } // end try
  catch(...)
  {
    ;
  } // end catch

  PathToImage mapToImage;

  float a;
  int whichMutation;

  // initialize the HaltCriterion
  // before we start rendering
  mHalt->init(this, &progress);

  // find the terminal stage of recursion
  float recurseWidth  = dynamic_cast<RenderFilm*>(mRecord.get())->getWidth();
  float recurseHeight = dynamic_cast<RenderFilm*>(mRecord.get())->getHeight();
  float recurseTarget = progress.expected_count();

  while(   recurseWidth  * mRecursionScale >= 2
        && recurseHeight * mRecursionScale >= 2)
  {
    recurseWidth  *= mRecursionScale;
    recurseHeight *= mRecursionScale;
    recurseTarget *= mRecursionScale * mRecursionScale;
  } // end while

  // clamp target effort to something reasonable
  float actualTarget = std::max(10000.0f + progress.count(), recurseTarget);

  // after the halfway point, do regular steps
  float targetStep = 0;
  float widthStep = 0;
  float heightStep = 0;

  size_t currentEstimate = 0;

  // main loop
  while(!(*mHalt)())
  {
    if(progress.count() > actualTarget)
    {
      long unsigned int r = mScene->getRaysCast();

      // these rays are "free"
      invB = updateImportance(bLuminance,
                              recurseWidth, recurseHeight,
                              x, xPath, xResults,
                              ix, xPdf);
      // add these to the total at the end
      long unsigned int rayDifference = mScene->getRaysCast() - r;
      owedRays += rayDifference;
      mScene->setRaysCast(r);

      if(recurseWidth >= 0.4f * film->getWidth()
        || recurseHeight >= 0.4f * film->getHeight())
      {
        if(targetStep == 0)
        {
          // from now on, do 10 more steps
          targetStep = (progress.expected_count() - progress.count()) / 10.0f;
          widthStep = (film->getWidth() - recurseWidth) / 10.0f;
          heightStep = (film->getHeight() - recurseHeight) / 10.0f;
        } // end if

        // after the halfway point, proceed regularly
        recurseWidth += widthStep;
        recurseHeight += heightStep;
        recurseTarget += targetStep;
      } // end if
      else
      {
        // update new target
        recurseWidth  /= mRecursionScale;
        recurseHeight /= mRecursionScale;
        recurseTarget /= (mRecursionScale * mRecursionScale);
      } // end else

      // clamp target effort to something reasonable
      actualTarget = std::max(10000.0f, recurseTarget);

      // update the index of the estimate we're on
      ++currentEstimate;
    } // end if

    // mutate
    whichMutation = (*mMutator)(x,xPath,y,yPath);

    // evaluate
    if(whichMutation != -1)
    {
      yResults.clear();
      g = mMutator->evaluate(yPath, yResults);

      // compute importance
      iy = (*mImportance)(y, yPath, yResults);

      // compute pdf of y
      yPdf = iy * invB;
    } // end if
    else
    {
      iy = 0;
    } // end else

    // calculate accept probability
    a = mMutator->evaluateTransitionRatio(whichMutation, x, xPath, ix, y, yPath, iy);
    a = std::min<float>(1.0f, a * iy/ix);

    if(ix > 0)
    {
      // record x
      float xWeight = (1.0f - a) / (xPdf+pLargeStep);
      mRecord->record(xWeight, x, xPath, xResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      gpcpu::float2 pixel;
      mapToImage(xResults[0], x, xPath, pixel[0], pixel[1]);
      mAcceptanceImage.deposit(pixel[0], pixel[1], Spectrum(1.0f - a, 1.0f - a, 1.0f - a));
    } // end if

    if(iy > 0)
    {
      // record y
      float yWeight = (a + float(whichMutation))/(yPdf + pLargeStep);
      mRecord->record(yWeight, y, yPath, yResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      gpcpu::float2 pixel;
      mapToImage(yResults[0], y, yPath, pixel[0], pixel[1]);
      mAcceptanceImage.deposit(pixel[0], pixel[1], Spectrum(a, a, a));
      mProposalImage.deposit(pixel[0], pixel[1], Spectrum::white());
    } // end if

    // accept?
    if((*mRandomSequence)() < a)
    {
      ++mNumAccepted;
      x = y;

      // safely copy the path
      copyPath(xPath, yPath);

      f = g;
      ix = iy;
      xResults = yResults;
      xPdf = yPdf;
    } // end if

    // purge all malloc'd memory for this sample
    mShadingContext->freeAll();

    // update the number of samples we've taken so far
    ++mNumSamples;
  } // end for i

  // pay back the rays we owe
  mScene->setRaysCast(mScene->getRaysCast() + owedRays);

  // purge the local store
  mLocalPool.freeAll();
} // end TargetRaysRenderer::kernel()

float MultiStageMetropolisRenderer
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
  shared_ptr<RenderFilm> current = dynamic_pointer_cast<RenderFilm,Record>(mRecord);
  float s;

  //RenderFilm temp = *current;
  //s = bLuminance / temp.computeMean().luminance();
  //temp.scale(Spectrum(s,s,s));

  char buf[32];
  //sprintf(buf, "estimate-%d.exr", currentEstimate);
  //temp.writeEXR(buf);

  // update the estimate by resampling the current record
  // into a lower res image
  // round up for these dimensions
  shared_ptr<RenderFilm> lowResEstimate(new RenderFilm(static_cast<size_t>(ceilf(w)),
                                                       static_cast<size_t>(ceilf(h))));
  current->resample(*lowResEstimate);

  // scale estimate so it has mean luminance equal to bLuminance
  s = bLuminance / lowResEstimate->computeMean().luminance();
  lowResEstimate->scale(Spectrum(s,s,s));

  sprintf(buf, "lowres-estimate-%dx%d.exr", lowResEstimate->getWidth(), lowResEstimate->getHeight());
  lowResEstimate->writeEXR(buf);

  // replace the current importance with a new one
  mImportance.reset(new EstimateImportance(*lowResEstimate));

  // preprocess the importance and grab b
  mImportance->preprocess(mRandomSequence, mScene, mShadingContext, mMutator, *this);
  //mImportance->preprocess(mSeedPoints, mSeedPaths, mSeedResults);
  float invB = mImportance->getInvNormalizationConstant();

  // update x's importance & pdf
  // compute importance
  ix = (*mImportance)(x, xPath, xResults);

  // compute pdf of x
  xPdf = ix * invB;

  return invB;
} // end MultiStageMetropolisRenderer::updateImportance()

