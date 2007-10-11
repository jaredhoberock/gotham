/*! \file TargetRaysRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TargetRaysRenderer class.
 */

#include "TargetRaysRenderer.h"
#include "../mutators/KelemenMutator.h"

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
                       const unsigned int target)
    :Parent(s,mutator,importance),mNumSamplesTaken(0),mRayTarget(target)
{
  ;
} // end TargetRaysRenderer::TargetRaysRenderer()

TargetRaysRenderer
  ::TargetRaysRenderer(boost::shared_ptr<const Scene> &s,
                       boost::shared_ptr<RenderFilm> &f,
                       const boost::shared_ptr<RandomSequence> &sequence,
                       const boost::shared_ptr<PathMutator> &m,
                       const boost::shared_ptr<ScalarImportance> &i,
                       const unsigned int target)
    :Parent(s,f,sequence,m,i),mNumSamplesTaken(0),mRayTarget(target)
{
  ;
} // end TargetRaysRenderer::TargetRaysRenderer()

void TargetRaysRenderer
  ::kernel(ProgressCallback &progress)
{
  unsigned int totalPixels = mFilm->getWidth() * mFilm->getHeight();

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

  float a;
  int whichMutation;

  unsigned int oldRays;
  progress.restart(mRayTarget);
  mNumSamplesTaken = 0;
  while(progress.count() < progress.expected_count())
  {
    oldRays = mScene->getRaysCast();

    // mutate
    whichMutation = (*mMutator)(x,xPath,y,yPath);
    ++mNumProposed;

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

    // recompute x
    ix = (*mImportance)(x, xPath, xResults);
    xPdf = ix * invB;

    // calculate accept probability
    a = mMutator->evaluateTransitionRatio(whichMutation, x, xPath, ix, y, yPath, iy);
    a = std::min<float>(1.0f, a * iy/ix);

    if(ix > 0)
    {
      // record x
      float xWeight = (1.0f - a) / (xPdf+pLargeStep);
      for(ResultList::const_iterator r = xResults.begin();
          r != xResults.end();
          ++r)
      {
        Spectrum deposit;
        float2 pixel;
        if(r->mEyeLength == 1)
        {
          unsigned int endOfLightPath = xPath.getSubpathLengths().sum() - r->mLightLength;
          ::Vector w = xPath[endOfLightPath].mDg.getPoint();
          w -= xPath[0].mDg.getPoint();
          xPath[0].mSensor->invert(w, xPath[0].mDg,
                                   pixel[0], pixel[1]);
        } // end if
        else
        {
          xPath[0].mSensor->invert(xPath[0].mToNext,
                                   xPath[0].mDg,
                                   pixel[0], pixel[1]);
        } // end else

        // each sample contributes (1/spp) * MIS weight * MC weight * f / pdf
        deposit = xWeight * r->mWeight * r->mThroughput / r->mPdf;
        mFilm->deposit(pixel[0], pixel[1], deposit);

        //// add to the acceptance image
        //mAcceptanceImage.pixel(pixel[0], pixel[1]) += (1.0f - a);
      } // end for r

      // add to the acceptance image
      mAcceptanceImage.deposit(x[0][0], x[0][1], static_cast<Spectrum>(1.0f - a));
    } // end if

    if(iy > 0)
    {
      // record y
      float yWeight = (a + float(whichMutation))/(yPdf + pLargeStep);
      for(ResultList::const_iterator ry = yResults.begin();
          ry != yResults.end();
          ++ry)
      {
        Spectrum deposit;
        float2 pixel;
        if(ry->mEyeLength == 1)
        {
          unsigned int endOfLightPath = yPath.getSubpathLengths().sum() - ry->mLightLength;
          ::Vector w = yPath[endOfLightPath].mDg.getPoint();
          w -= yPath[0].mDg.getPoint();
          yPath[0].mSensor->invert(w, yPath[0].mDg,
                                   pixel[0], pixel[1]);
        } // end if
        else
        {
          yPath[0].mSensor->invert(yPath[0].mToNext,
                                   yPath[0].mDg,
                                   pixel[0], pixel[1]);
        } // end else

        // each sample contributes (1/spp) * MIS weight * MC weight * f
        deposit = yWeight * ry->mWeight * ry->mThroughput / ry->mPdf;
        mFilm->deposit(pixel[0], pixel[1], deposit);

        //// add to the acceptance image
        //mAcceptanceImage.pixel(pixel[0], pixel[1]) += a;
      } // end for r

      // add to the acceptance image
      mAcceptanceImage.deposit(y[0][0], y[0][1], static_cast<Spectrum>(a));
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

    ++mNumSamplesTaken;
    progress += mScene->getRaysCast() - oldRays;
  } // end for i

  // purge the local store
  mLocalPool.freeAll();
} // end TargetRaysRenderer::kernel()

void TargetRaysRenderer
  ::postprocess(void)
{
  // need to output a newline since progress_display
  // only does this when count() == expected_count()
  std::cout << std::endl;

  // scale film by 1/spp
  float spp = static_cast<float>(mNumSamplesTaken) / (mFilm->getWidth() * mFilm->getHeight());
  float invSpp = 1.0f / spp;
  mFilm->scale(Spectrum(invSpp, invSpp, invSpp));
  
  Parent::postprocess();
} // end TargetRaysRenderer::postprocess()

