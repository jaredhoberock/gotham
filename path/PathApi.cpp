/*! \file PathApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PathApi class.
 */

#include "PathApi.h"
#include "KajiyaSampler.h"
#include "ArvoKirkSampler.h"
#include "SimpleBidirectionalSampler.h"
#include "SimpleForwardRussianRouletteSampler.h"
#include "SimpleBidirectionalRussianRouletteSampler.h"
#include "KelemenSampler.h"
#include "WhittedSampler.h"
#include "ShirleySampler.h"
#include "RussianRoulette.h"

PathSampler *PathApi
  ::sampler(const Gotham::AttributeMap &attr)
{
  PathSampler *result = 0;
  std::string samplerName = "kajiya";

  // fish out the parameters
  Gotham::AttributeMap::const_iterator a = attr.find("path::sampler");
  if(a != attr.end())
  {
    boost::any val = a->second;
    samplerName = boost::any_cast<std::string>(val);
  } // end if

  unsigned int maxLength = UINT_MAX;
  a = attr.find("path::maxlength");
  if(a != attr.end())
  {
    boost::any val = a->second;
    maxLength = atoi(boost::any_cast<std::string>(val).c_str());
  } // end if

  std::string rrFunction = "always";
  a = attr.find("path::russianroulette::function");
  if(a != attr.end())
  {
    boost::any val = a->second;
    rrFunction = boost::any_cast<std::string>(val);
  } // end if

  float continueProbability = 1.0f;
  a = attr.find("path::russianroulette::continueprobability");
  if(a != attr.end())
  {
    boost::any val = a->second;
    continueProbability = static_cast<float>(atof(boost::any_cast<std::string>(val).c_str()));
  } // end if

  size_t minimumSubpathLength = 3;
  a = attr.find("path::russianroulette::minimumsubpathlenth");
  if(a != attr.end())
  {
    boost::any val = a->second;
    minimumSubpathLength = atoi(boost::any_cast<std::string>(val).c_str());
  } // end if

  // create the russian roulette
  boost::shared_ptr<RussianRoulette> rr(new AlwaysRoulette());
  if(rrFunction == "always")
  {
    rr.reset(new AlwaysRoulette());
  } // end if
  else if(rrFunction == "constant")
  {
    rr.reset(new ConstantRoulette(continueProbability));
  } // end else if
  else if(rrFunction == "constantandalwaysafterdelta")
  {
    rr.reset(new ConstantAndAlwaysAfterDeltaRoulette(continueProbability));
  } // end else if
  else if(rrFunction == "luminance")
  {
    rr.reset(new LuminanceRoulette());
  } // end else if
  else if(rrFunction == "maxoverspectrum")
  {
    rr.reset(new MaxOverSpectrumRoulette());
  } // else if
  else if(rrFunction == "onlyafterdelta")
  {
    rr.reset(new OnlyAfterDeltaRoulette());
  } // end else if
  else if(rrFunction == "kelemen")
  {
    rr.reset(new KelemenRoulette(continueProbability));
  } // end else if
  else if(rrFunction == "veach")
  {
    rr.reset(new VeachRoulette(minimumSubpathLength));
  } // end else if
  else
  {
    std::cerr << "Warning: unknown Russian roulette function \"" << rrFunction << "\"." << std::endl;
  } // end else

  // create the sampler
  // keep this list alphabetized
  if(samplerName == "arvokirk")
  {
    result = new ArvoKirkSampler(rr, maxLength-1);
  } // end else if
  else if(samplerName == "kajiya")
  {
    result = new KajiyaSampler(maxLength-1);
  } // end if
  else if(samplerName == "kelemen")
  {
    result = new KelemenSampler(rr, maxLength);
  } // end else if
  else if(samplerName == "shirley")
  {
    result = new ShirleySampler(maxLength-1);
  } // end else if
  else if(samplerName == "simplebidirectional")
  {
    result = new SimpleBidirectionalSampler(maxLength);
  } // end else if
  else if(samplerName == "simplebidirectionalrussianroulette")
  {
    result = new SimpleBidirectionalRussianRouletteSampler(maxLength);
  } // end else if
  else if(samplerName == "simpleforwardrussianroulette")
  {
    result = new SimpleForwardRussianRouletteSampler(rr, maxLength-1);
  } // end else if
  else if(samplerName == "whitted")
  {
    result = new WhittedSampler(maxLength-1);
  } // end else if
  else
  {
    std::cerr << "Warning: unknown sampler \"" << samplerName << "\"." << std::endl;
    result = new KajiyaSampler(maxLength-1);
  } // end else

  return result;
} // end PathApi::sampler()

