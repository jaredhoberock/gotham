/*! \file HaltCriterion.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HaltCriterion classes.
 */

#include "HaltCriterion.h"
#include "MonteCarloRenderer.h"
#include "../primitives/Scene.h"
#include <boost/lexical_cast.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>
using namespace boost;
using namespace boost::python;

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

void TargetPixelSampleCount
  ::setStrata(const size_t xStrata,
              const size_t yStrata)
{
  mXStrata = xStrata;
  mYStrata = yStrata;
} // end TargetPixelSampleCount::setStrata()

size_t TargetPixelSampleCount
  ::getXStrata(void) const
{
  return mXStrata;
} // end TargetPixelSampleCount::getXStrata()

size_t TargetPixelSampleCount
  ::getYStrata(void) const
{
  return mYStrata;
} // end TargetPixelSampleCount::getYStrata()

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
  boost::tuple<size_t,size_t> spp(2,2);
  a = attr.find("renderer:spp");
  if(a != attr.end())
  {
    any val = a->second;
    try
    {
      // try to convert a tuple by evaluating the python expression
      python::tuple temp = extract<python::tuple>(eval(a->second.c_str()));
      size_t sx = extract<size_t>(temp[0]);
      size_t sy = extract<size_t>(temp[1]);
      spp = boost::make_tuple(sx,sy);
    } // end try
    catch(...)
    {
      try
      {
        // try to convert a single integer
        size_t xStrata = lexical_cast<size_t>(a->second);
        spp = boost::tuple<size_t,size_t>(xStrata,xStrata);
      } // end try
      catch(bad_lexical_cast &e)
      {
        std::cerr << "HaltCriterion::createCriterion(): Warning: Couldn't interpret " << a->second << " as samples per pixel (xStrata,yStrata)." << std::endl;
      } // end catch
    } // end catch

    // setting samples per pixel automatically overrides the target function
    targetFunctionName = "samples";

    // target based on number of strata per pixel
    target = spp.get<0>() * spp.get<1>() * numPixels;
  } // end if

  HaltCriterion *result = 0;

  // create a HaltCriterion
  if(targetFunctionName == "samples")
  {
    TargetPixelSampleCount *r = new TargetPixelSampleCount();
    r->setTarget(target);
    r->setStrata(spp.get<0>(), spp.get<1>());
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

