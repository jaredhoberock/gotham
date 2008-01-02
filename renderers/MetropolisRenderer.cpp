/*! \file MetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MetropolisRenderer class.
 */

#include "MetropolisRenderer.h"
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
                       shared_ptr<Record> &r,
                       const shared_ptr<RandomSequence> &sequence,
                       const shared_ptr<PathMutator> &m,
                       const shared_ptr<ScalarImportance> &i)
    :Parent(s,r,sequence),mMutator(m),mImportance(i)
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

    if(ix > 0)
    {
      // XXX TODO PERF: all this is redundant
      //          idea: accumulate x's weight and only deposit
      //                when a sample is finally rejected
      // record x
      float xWeight = (1.0f - a) / (xPdf+pLargeStep);
      //float xWeight = (1.0f - a) / xPdf;
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
      float yWeight = (a + float(whichMutation))/(yPdf + pLargeStep);
      //float yWeight = a / yPdf;
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
    } // end if

    // purge all malloc'd memory for this sample
    ScatteringDistributionFunction::mPool.freeAll();

    ++mNumSamples;
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

  // XXX kill this nastiness somehow
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  if(film != 0)
  {
    // zero the acceptance image
    mAcceptanceImage.resize(film->getWidth(), film->getHeight());
    mAcceptanceImage.preprocess();

    // zero the proposal image
    mProposalImage.resize(film->getWidth(), film->getHeight());
    mProposalImage.preprocess();
  } // end if

  // preprocess the mutator
  mMutator->preprocess();

  // preprocess the scalar importance
  mImportance->preprocess(mRandomSequence, mScene, mMutator, *this);
} // end MetropolisRenderer::preprocess()

void MetropolisRenderer
  ::postRenderReport(const double elapsed) const
{
  Parent::postRenderReport(elapsed);

  std::cout << "Proposal acceptance rate: " << static_cast<float>(mNumAccepted) / mNumSamples << std::endl;
} // end MetropolisRenderer::postRenderReport()

void MetropolisRenderer
  ::postprocess(void)
{
  // estimate the canonical normalization constant
  LuminanceImportance luminance;

  // rescale image so that it has b mean luminance
  // XXX TODO kill this nastiness
  //          should Record provide these hooks?
  //          Or should we only do this when Record is an image?
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  if(film)
  {
    // use our own random sequence so we always get the same result here
    // this is particularly important for difficult scenes where it is hard
    // to agree on an estimate
    RandomSequence seq(13u);

    float b = luminance.estimateNormalizationConstant(seq, mScene, mMutator, 10000);
    float s = b / film->computeMean().luminance();
    film->scale(Spectrum(s,s,s));
  } // end if

  // jump over MonteCarloRenderer's postprocess
  // we don't need to scale by 1 / sp
  Parent::Parent::postprocess();

  mMutator->postprocess();

  // rescale acceptance image so it has 1/2 mean luminance
  float s = 0.5f / mAcceptanceImage.computeMean().luminance();
  mAcceptanceImage.scale(Spectrum(s,s,s));
  mAcceptanceImage.postprocess();

  // rescale proposal image so it has 1/2 mean luminance
  s = 0.5f / mProposalImage.computeMean().luminance();
  mProposalImage.scale(Spectrum(s,s,s));
  mProposalImage.postprocess();
} // end MetropolisRenderer::postprocess()

RenderFilm *MetropolisRenderer
  ::getAcceptanceImage(void)
{
  return &mAcceptanceImage;
} // end MetropolisRenderer::getAcceptanceImage()

void MetropolisRenderer
  ::setAcceptanceFilename(const std::string &filename)
{
  mAcceptanceImage.setFilename(filename);
} // end MetropolisRenderer::setAcceptanceFilename()

void MetropolisRenderer
  ::setProposalFilename(const std::string &filename)
{
  mProposalImage.setFilename(filename);
} // end MetropolisRenderer::setProposalFilename()

