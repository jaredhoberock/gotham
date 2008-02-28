/*! \file ExperimentalMetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ExperimentalMetropolisRenderer class.
 */

#include "ExperimentalMetropolisRenderer.h"
#include "../path/KelemenSampler.h"
#include "../path/KajiyaSampler.h"
#include <aliastable/AliasTable.h>
#include "../path/Path.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../mutators/KelemenMutator.h"
#include "../importance/LuminanceImportance.h"
#include "../path/PathToImage.h"

#include <gpcpu/Vector.h>
#include <boost/progress.hpp>

using namespace boost;
using namespace gpcpu;

ExperimentalMetropolisRenderer
  ::ExperimentalMetropolisRenderer(void)
    :Parent()
{
  ;
} // end ExperimentalMetropolisRenderer::ExperimentalMetropolisRenderer()

ExperimentalMetropolisRenderer
  ::ExperimentalMetropolisRenderer(const boost::shared_ptr<RandomSequence> &s,
                                   const boost::shared_ptr<PathMutator> &mutator,
                                   const boost::shared_ptr<ScalarImportance> &importance)
    :Parent(s,mutator,importance)
{
  ;
} // end ExperimentalMetropolisRenderer::ExperimentalMetropolisRenderer()

void ExperimentalMetropolisRenderer
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
  mMutator->evaluate(xPath,xResults);
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

  float a;
  int whichMutation;

  PathToImage mapToImage;

  // initialize the HaltCriterion
  // before we start rendering
  mHalt->init(this, &progress);

  mConsecutiveRejections = 0;

  // main loop
  while(!(*mHalt)())
  {
    // mutate
    whichMutation = (*mMutator)(x,xPath,y,yPath);

    // evaluate
    if(whichMutation != -1)
    {
      yResults.clear();
      mMutator->evaluate(yPath, yResults);

      // compute importance
      iy = (*mImportance)(y, yPath, yResults);

      // compute pdf of y
      yPdf = iy * invB;
    } // end if
    else
    {
      iy = 0;
    } // end else

    // recompute x
    ix = (*mImportance)(x, xPath, xResults);

    // calculate accept probability
    a = 0;
    if(iy > 0)
    {
      a = mMutator->evaluateTransitionRatio(whichMutation, x, xPath, ix, y, yPath, iy);
      a = std::min<float>(1.0f, a * iy/ix);
    } // end if

    float w = 1.0f - a;
    if(iy > 0)
    {
      w *= expf(-mConsecutiveRejections);
    } // end if

    if(ix > 0)
    {
      // XXX TODO PERF: all this is redundant
      //          idea: accumulate x's weight and only deposit
      //                when a sample is finally rejected
      // record x
      //float xWeight = w * (1.0f - a) / (xPdf+pLargeStep);
      float xWeight = w / (xPdf+pLargeStep);
      mRecord->record(xWeight, x, xPath, xResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      gpcpu::float2 pixel;
      float xu, xv;
      mapToImage(xResults[0], x, xPath, xu, xv);
      mAcceptanceImage.deposit(xu, xv, Spectrum(1.0f - a, 1.0f - a, 1.0f - a));
    } // end if

    if(iy > 0)
    {
      // record y
      //float yWeight = (a + float(whichMutation))/(yPdf + pLargeStep);
      float yWeight = ((1.0f - w) + float(whichMutation))/(yPdf + pLargeStep);
      mRecord->record(yWeight, y, yPath, yResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      float yu, yv;
      mapToImage(yResults[0], y, yPath, yu, yv);
      mAcceptanceImage.deposit(yu, yv, Spectrum(a, a, a));

      // add to the proposal image
      mProposalImage.deposit(yu, yv, Spectrum::white());
    } // end if

    // accept?
    if((*mRandomSequence)() < a)
    {
      ++mNumAccepted;
      x = y;

      // safely copy the path
      copyPath(xPath, yPath);

      ix = iy;
      xResults = yResults;
      xPdf = yPdf;

      mConsecutiveRejections = 0;
    } // end if
    else
    {
      mConsecutiveRejections += 1.0f;
    } // end else

    // purge all malloc'd memory for this sample
    ScatteringDistributionFunction::mPool.freeAll();

    ++mNumSamples;
  } // end for i

  // purge the local store
  mLocalPool.freeAll();
} // end ExperimentalMetropolisRenderer::kernel()

