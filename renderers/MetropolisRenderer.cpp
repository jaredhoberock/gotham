/*! \file MetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MetropolisRenderer class.
 */

#include "MetropolisRenderer.h"
#include "../path/KelemenSampler.h"
#include "../path/KajiyaSampler.h"
#include <aliastable/AliasTable.h>
#include "../films/RandomAccessFilm.h"
#include "../path/Path.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../mutators/KelemenMutator.h"

#include <gpcpu/Vector.h>
#include <boost/progress.hpp>

using namespace boost;
using namespace gpcpu;

MetropolisRenderer
  ::MetropolisRenderer(void)
     :Parent()
{
  ;
} // end MetropolisRenderer::MetropolisRenderer()

MetropolisRenderer
  ::MetropolisRenderer(const shared_ptr<RandomSequence> &s,
                       const shared_ptr<PathMutator> &mutator,
                       const shared_ptr<ScalarImportance> &importance)
    :Parent(),mMutator(mutator),mImportance(importance)
{
  setRandomSequence(s);
} // end MetropolisRenderer::MetropolisRenderer()

MetropolisRenderer
  ::MetropolisRenderer(shared_ptr<const Scene> &s,
                       shared_ptr<RenderFilm> &f,
                       const shared_ptr<RandomSequence> &sequence,
                       const shared_ptr<PathMutator> &m,
                       const shared_ptr<ScalarImportance> &i)
    :Parent(s,f,sequence),mMutator(m),mImportance(i)
{
  ;
} // end MetropolisRenderer::MetropolisRenderer()

void MetropolisRenderer
  ::setMutator(shared_ptr<PathMutator> &m)
{
  mMutator = m;
} // end MetropolisRenderer::setMutator()

void MetropolisRenderer
  ::setImportance(shared_ptr<ScalarImportance> &i)
{
  mImportance = i;
} // end MetropolisRenderer::setImportance()

void MetropolisRenderer
  ::setScene(const shared_ptr<const Scene> &s)
{
  Parent::setScene(s);
  mMutator->setScene(s);
} // end MetropolisRenderer::setScene()

float MetropolisRenderer
  ::estimateNormalizationConstant(const size_t n,
                                  PathSampler::HyperPoint &x,
                                  Path &xPath)
{
  float result = 0;
  Spectrum L;

  // estimate b
  typedef std::vector<PathSampler::HyperPoint> SeedList;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList resultList;
  SeedList seeds;
  std::vector<float> seedImportance;
  float I;

  // XXX fix this
  PathSampler *sampler = const_cast<PathSampler*>(mMutator->getSampler());
  for(size_t i = 0; i < n; ++i)
  {
    PathSampler::constructHyperPoint(*mRandomSequence, x);

    // create a Path
    if(sampler->constructPath(*mScene, x, xPath))
    {
      // evaluate the Path
      resultList.clear();
      L = mMutator->evaluate(xPath, resultList);

      I = (*mImportance)(x, L);
      result += I;

      seeds.push_back(x);
      seedImportance.push_back(I);
    } // end if

    // free all integrands allocated in this sample
    ScatteringDistributionFunction::mPool.freeAll();
  } // end for i

  // pick a seed
  AliasTable<PathSampler::HyperPoint> aliasTable;
  aliasTable.build(seeds.begin(), seeds.end(),
                   seedImportance.begin(), seedImportance.end());
  x = aliasTable((*mRandomSequence)());
  Path temp;
  sampler->constructPath(*mScene, x, temp);

  // copy x's integrands to the local store
  copyPath(xPath, temp);

  // free all integrands that were allocated in the estimate
  ScatteringDistributionFunction::mPool.freeAll();
  
  return result / n;
} // end MetropolisRenderer::estimateNormalizationConstant()

void MetropolisRenderer
  ::kernel(ProgressCallback &progress)
{
  unsigned int totalPixels = mFilm->getWidth() * mFilm->getHeight();
  unsigned int totalSamples = (mSamplesPerPixel * mSamplesPerPixel) * totalPixels;
  float invSpp = 1.0f / (mSamplesPerPixel * mSamplesPerPixel);

  PathSampler::HyperPoint x, y;
  Path xPath, yPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults;

  // estimate normalization constant and pick a seed
  float b = estimateNormalizationConstant(10000, x, xPath);
  float invB = 1.0f / b;

  // initial seed
  Spectrum f, g;
  f = mMutator->evaluate(xPath,xResults);
  float ix = (*mImportance)(x, f), iy = 0;
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

  progress.restart(totalSamples);
  for(size_t i = 0; i < totalSamples; ++i)
  {
    // mutate
    whichMutation = (*mMutator)(x,xPath,y,yPath);

    // evaluate
    if(whichMutation != -1)
    {
      yResults.clear();
      g = mMutator->evaluate(yPath, yResults);

      // compute importance
      iy = (*mImportance)(y, g);

      // compute pdf of y
      yPdf = iy * invB;
    } // end if
    else
    {
      iy = 0;
    } // end else

    // recompute x
    ix = (*mImportance)(x,f);
    xPdf = ix * invB;

    // calculate accept probability
    a = mMutator->evaluateTransitionRatio(whichMutation, x, xPath, ix, y, yPath, iy);
    a = std::min<float>(1.0f, a * iy/ix);

    if(ix > 0)
    {
      // record x
      float xWeight = invSpp * (1.0f - a) / (xPdf+pLargeStep);
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
      float yWeight = invSpp * (a + float(whichMutation))/(yPdf + pLargeStep);
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

    ++progress;
  } // end for i

  // purge the local store
  mLocalPool.freeAll();
} // end MetropolisRenderer::kernel()

void MetropolisRenderer
  ::copyPath(Path &dst, const Path &src)
{
  // free the local store
  mLocalPool.freeAll();

  // clone src
  src.clone(dst, mLocalPool);
} // end MetropolisRenderer::copyPath()

void MetropolisRenderer
  ::setRandomSequence(const shared_ptr<RandomSequence> &s)
{
  Parent::setRandomSequence(s);
  mMutator->setRandomSequence(s);
} // end MetropolisRenderer::setRandomSequence()

void MetropolisRenderer
  ::preprocess(void)
{
  Parent::preprocess();

  // zero the accepted count
  mNumAccepted = 0;

  // zero the acceptance image
  mAcceptanceImage.resize(mFilm->getWidth(), mFilm->getHeight());
  mAcceptanceImage.fill(Spectrum::black());

  // preprocess the mutator
  mMutator->preprocess();

  // preprocess the scalar importance
  mImportance->preprocess(mRandomSequence, mScene, mMutator, *this);
} // end MetropolisRenderer::preprocess()

std::string MetropolisRenderer
  ::getRenderParameters(void) const
{
  std::string result = Parent::getRenderParameters();

  // append the name of the path sampler
  result += typeid(*mMutator->getSampler()).name();
  result += '-';

  // append the name of the mutator
  result += typeid(*mMutator).name();
  result += '-';

  // append the name of the importance
  result += typeid(*mImportance).name();

  return result;
} // end MetropolisRenderer::getRenderParameters()

void MetropolisRenderer
  ::postRenderReport(const double elapsed) const
{
  Parent::postRenderReport(elapsed);

  unsigned int totalPixels = mFilm->getWidth() * mFilm->getHeight();
  unsigned int totalSamples = (mSamplesPerPixel * mSamplesPerPixel) * totalPixels;
  std::cout << "Proposal acceptance rate: " << static_cast<float>(mNumAccepted) / totalSamples << std::endl;
} // end MetropolisRenderer::postRenderReport()

void MetropolisRenderer
  ::postprocess(void)
{
  Parent::postprocess();

  mMutator->postprocess();

  mAcceptanceImage.writeEXR("acceptance.exr");
} // end MetropolisRenderer::postprocess()

RenderFilm *MetropolisRenderer
  ::getAcceptanceImage(void)
{
  return &mAcceptanceImage;
} // end MetropolisRenderer::getAcceptanceImage()

