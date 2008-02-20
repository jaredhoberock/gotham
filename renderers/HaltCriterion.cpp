/*! \file HaltCriterion.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HaltCriterion classes.
 */

#include "HaltCriterion.h"
#include "MonteCarloRenderer.h"
#include "../primitives/Scene.h"

void HaltCriterion
  ::init(const MonteCarloRenderer *r,
         Renderer::ProgressCallback *p)
{
  mRenderer = r;
  mProgress = p;
} // end HaltCriterion::setRenderer()

const MonteCarloRenderer *HaltCriterion
  ::getRenderer(void) const
{
  return mRenderer;
} // end HaltCriterion::getRenderer()

void TargetCriterion
  ::init(const MonteCarloRenderer *r,
         Renderer::ProgressCallback *p)
{
  Parent::init(r,p);

  p->restart(getTarget());
  
  mPrevious = 0;
} // end TargetCriterion::init()

void TargetCriterion
  ::setTarget(const TargetCriterion::Target t)
{
  mTarget = t;
} // end TargetCriterion::setTarget()

TargetCriterion::Target TargetCriterion
  ::getTarget(void) const
{
  return mTarget;
} // end TargetCriterion::getTarget()

bool TargetSampleCount
  ::operator()(void)
{
  Target currentSamples = mRenderer->getNumSamples();

  // update progress
  *mProgress += currentSamples - mPrevious;

  // update previous
  mPrevious = currentSamples;
  return currentSamples >= getTarget();
} // end TargetSampleCount::operator()()

TargetRayCount
  ::TargetRayCount(const Target t)
    :Parent()
{
  setTarget(t);
} // end TargetRayCount::TargetRayCount()

bool TargetRayCount
  ::operator()(void)
{
  Target currentRays = mRenderer->getScene()->getRaysCast();

  // update progress
  *mProgress += currentRays - mPrevious;

  // update previous
  mPrevious = currentRays;

  return currentRays >= getTarget();
} // end TargetRayCount::operator()()

void HaltCriterion
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["renderer:target:function"] = "samples";
} // end HaltCriterion::getDefaultAttributes()

HaltCriterion *HaltCriterion
  ::createCriterion(Gotham::AttributeMap &attr)
{
  using namespace boost;

  std::string targetFunctionName = attr["renderer:target:function"];

  // count the number of pixels
  size_t width = lexical_cast<size_t>(attr["record:width"]);
  size_t height = lexical_cast<size_t>(attr["record:height"]);

  size_t numPixels = width * height;

  // default target to the number of pixels
  TargetCriterion::Target target = numPixels;
  Gotham::AttributeMap::const_iterator a = attr.find("renderer:target:count");
  if(a != attr.end())
  {
    any val = a->second;
    target = atol(boost::any_cast<std::string>(val).c_str());
  } // end if

  // check if we specified spp
  // this automatically overrides the function name
  a = attr.find("renderer:spp");
  if(a != attr.end())
  {
    any val = a->second;
    unsigned int spp = atoi(any_cast<std::string>(val).c_str());

    // setting samples per pixel automatically overrides the target function
    targetFunctionName = "samples";

    // remember we actually use the square of this value
    target = spp * spp * numPixels;
  } // end if

  HaltCriterion *result = 0;

  // create a HaltCriterion
  if(targetFunctionName == "samples")
  {
    TargetCriterion *r = new TargetSampleCount();
    r->setTarget(target);
    result = r;
  } // end if
  else if(targetFunctionName == "rays")
  {
    result = new TargetRayCount(target);
  } // end if
  else if(targetFunctionName == "photons")
  {
    TargetCriterion *r = new TargetPhotonCount();
    r->setTarget(target);
    result = r;
  } // end else if
  else
  {
    std::cerr << "Warning: unknown target function \"" << targetFunctionName << "\"." << std::endl;
    TargetCriterion *r = new TargetSampleCount();
    r->setTarget(target);
    result = r;
  } // end else if

  return result;
} // end HaltCriterion::createCriterion()

void TargetPhotonCount
  ::init(const MonteCarloRenderer *r,
         Renderer::ProgressCallback *p)
{
  Parent::init(r,p);

  mPhotons = dynamic_cast<const PhotonRecord*>(r->getRecord().get());
  if(mPhotons == 0)
  {
    std::cerr << "Error: the current record is not a photon map!" << std::endl;
    exit(-1);
  } // end if
} // end TargetPhotonCount::init()

bool TargetPhotonCount
  ::operator()(void)
{
  Target currentPhotons = mPhotons->size();

  // update progress
  *mProgress += currentPhotons - mPrevious;

  // update previous
  mPrevious = currentPhotons;

  return currentPhotons >= getTarget();
} // end TargetPhotonCount::operator()()

