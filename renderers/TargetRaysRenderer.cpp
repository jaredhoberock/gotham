/*! \file TargetRaysRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TargetRaysRenderer class.
 */

#include "TargetRaysRenderer.h"
#include "../mutators/KelemenMutator.h"
#include "../path/PathToImage.h"

using namespace gpcpu;

TargetRaysRenderer
  ::TargetRaysRenderer(void)
    :Parent()
{
  ;
} // end TargetRaysRenderer::TargetRaysRenderer()

TargetRaysRenderer
  ::TargetRaysRenderer(const boost::shared_ptr<RandomSequence> &s,
                       const boost::shared_ptr<PathMutator> &mutator,
                       const boost::shared_ptr<ScalarImportance> &importance,
                       const Target target)
    :Parent(s,mutator,importance),mRayTarget(target)
{
  ;
} // end TargetRaysRenderer::TargetRaysRenderer()

TargetRaysRenderer
  ::TargetRaysRenderer(boost::shared_ptr<const Scene> &s,
                       boost::shared_ptr<Record> &r,
                       const boost::shared_ptr<RandomSequence> &sequence,
                       const boost::shared_ptr<PathMutator> &m,
                       const boost::shared_ptr<ScalarImportance> &i,
                       const Target target)
    :Parent(s,r,sequence,m,i),mRayTarget(target)
{
  ;
} // end TargetRaysRenderer::TargetRaysRenderer()

void TargetRaysRenderer
  ::kernel(ProgressCallback &progress)
{
  PathSampler::HyperPoint x, y;
  Path xPath, yPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults;

  // estimate normalization constant and pick a seed
  float b = mImportance->estimateNormalizationConstant(mRandomSequence, mScene, mMutator,
                                                       10000, mLocalPool, x, xPath);
  float invB = 1.0f / b;

  // initial seed
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

  Target oldRays;
  progress.restart(mRayTarget);
  while(progress.count() < progress.expected_count())
  {
    oldRays = mScene->getRaysCast();

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
    ScatteringDistributionFunction::mPool.freeAll();

    ++mNumSamples;
    progress += mScene->getRaysCast() - oldRays;
  } // end for i

  // purge the local store
  mLocalPool.freeAll();
} // end TargetRaysRenderer::kernel()

