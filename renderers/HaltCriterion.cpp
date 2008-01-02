/*! \file HaltCriterion.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HaltCriterion classes.
 */

#include "HaltCriterion.h"
#include "MonteCarloRenderer.h"
#include "../primitives/Scene.h"

void HaltCriterion
  ::setRenderer(const MonteCarloRenderer *r)
{
  mRenderer = r;
} // end HaltCriterion::setRenderer()

const MonteCarloRenderer *HaltCriterion
  ::getRenderer(void) const
{
  return mRenderer;
} // end HaltCriterion::getRenderer()

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
  return mRenderer->getNumSamples() >= getTarget();
} // end TargetSampleCount::operator()()

bool TargetRayCount
  ::operator()(void)
{
  return mRenderer->getScene()->getRaysCast() >= getTarget();
} // end TargetRayCount::operator()()

HaltCriterion *HaltCriterion
  ::createCriterion(const Gotham::AttributeMap &attr)
{
  using namespace boost;

  std::string targetFunctionName = "samples";
  Gotham::AttributeMap::const_iterator a = attr.find("renderer::target::function");
  if(a != attr.end())
  {
    any val = a->second;
    targetFunctionName = boost::any_cast<std::string>(val);
  } // end if

  TargetCriterion::Target target = 0;
  a = attr.find("renderer::target::count");
  if(a != attr.end())
  {
    any val = a->second;
    target = atol(boost::any_cast<std::string>(val).c_str());
  } // end if

  // check if we specified spp
  // this automatically overrides the function name
  a = attr.find("renderer::spp");
  if(a != attr.end())
  {
    any val = a->second;
    unsigned int spp = atoi(any_cast<std::string>(val).c_str());

    // setting samples per pixel automatically overrides the target function
    targetFunctionName = "samples";

    // the total sample count is the spp * total pixel count
    // XXX these defaults really should not be hard-coded here
    // image width
    size_t width = 512;
    a = attr.find("record::width");
    if(a != attr.end())
    {
      any val = a->second;
      width = atoi(any_cast<std::string>(val).c_str());
    } // end if

    // image height
    size_t height = 512;
    a = attr.find("record::height");
    if(a != attr.end())
    {
      any val = a->second;
      height = atoi(any_cast<std::string>(val).c_str());
    } // end if

    target = spp * width * height;
  } // end if

  HaltCriterion *result = 0;

  // create a HaltCriterion
  if(targetFunctionName == "samples")
  {
    result = new TargetSampleCount();
  } // end if
  else if(targetFunctionName == "rays")
  {
    result = new TargetRayCount();
  } // end if
  else
  {
    std::cerr << "Warning: unknown target function \"" << targetFunctionName << "\"." << std::endl;
    result = new TargetSampleCount();
  } // end else if

  return result;
} // end HaltCriterion::createCriterion()

