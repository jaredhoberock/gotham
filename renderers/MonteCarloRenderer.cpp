/*! \file MonteCarloRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MonteCarloRenderer class.
 */

#include "MonteCarloRenderer.h"
#include "../records/RenderFilm.h"

void MonteCarloRenderer
  ::setRandomSequence(const boost::shared_ptr<RandomSequence> &sequence)
{
  mRandomSequence = sequence;
} // end MonteCarloRenderer::setRandomSequence()

void MonteCarloRenderer
  ::preprocess(void)
{
  Parent::preprocess();

  mNumSamples = 0;

  // introduce ourself to the HaltCriterion
  mHalt->setRenderer(this);
} // end MonteCarloRenderer::preprocess()

void MonteCarloRenderer
  ::postprocess(void)
{
  // rescale image by 1 / spp
  // XXX TODO kill this nastiness
  //          should Record provide these hooks?
  //          Or should we only do this when Record is an image?
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  if(film)
  {
    float spp = mNumSamples;
    spp /= (film->getWidth() * film->getHeight());
    float invSpp = 1.0f / spp;

    film->scale(Spectrum(invSpp,invSpp,invSpp));
  } // end if

  // hand off to Parent
  Parent::postprocess();
} // end MonteCarloRenderer::postprocess()

unsigned long MonteCarloRenderer
  ::getNumSamples(void) const
{
  return mNumSamples;
} // end MonteCarloRenderer::getNumSamples()

void MonteCarloRenderer
  ::setHaltCriterion(const boost::shared_ptr<HaltCriterion> &h)
{
  mHalt = h;
} // end MonteCarloRenderer::setHaltCriterion()

