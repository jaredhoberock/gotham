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
#include "../importance/LuminanceImportance.h"
#include "../path/PathToImage.h"

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

  PathToImage mapToImage;

  progress.restart(totalSamples);
  for(size_t i = 0; i < totalSamples; ++i)
  {
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
      // XXX TODO PERF: all this is redundant
      //          idea: accumulate x's weight and only deposit
      //                when a sample is finally rejected
      // record x
      float xWeight = invSpp * (1.0f - a) / (xPdf+pLargeStep);
      for(ResultList::const_iterator r = xResults.begin();
          r != xResults.end();
          ++r)
      {
        Spectrum deposit;
        float2 pixel;

        // map the result to a location in the image
        mapToImage(*r,x,xPath,pixel[0],pixel[1]);

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

        // map the result to a location in the image
        mapToImage(*ry, y, yPath, pixel[0], pixel[1]);

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

  // zero the proposed/accepted count
  mNumAccepted = mNumProposed = 0;

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

  std::cout << "Proposal acceptance rate: " << static_cast<float>(mNumAccepted) / mNumProposed << std::endl;
} // end MetropolisRenderer::postRenderReport()

void MetropolisRenderer
  ::postprocess(void)
{
  // estimate the canonical normalization constant
  LuminanceImportance luminance;
  float b = luminance.estimateNormalizationConstant(mRandomSequence, mScene, mMutator, 10000);
  // rescale image so that it has b mean luminance
  float s = b / mFilm->computeMean().luminance();
  mFilm->scale(Spectrum(s,s,s));

  Parent::postprocess();

  mMutator->postprocess();

  // rescale acceptance image so it has 1/2 mean luminance
  Spectrum avg = mAcceptanceImage.getSum() / (mAcceptanceImage.getWidth() * mAcceptanceImage.getHeight());
  s = 0.5f / avg.luminance();
  mAcceptanceImage.scale(Spectrum(s,s,s));
  mAcceptanceImage.writeEXR("acceptance.exr");
} // end MetropolisRenderer::postprocess()

RenderFilm *MetropolisRenderer
  ::getAcceptanceImage(void)
{
  return &mAcceptanceImage;
} // end MetropolisRenderer::getAcceptanceImage()

