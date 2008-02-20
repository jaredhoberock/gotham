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
#include "JensenSampler.h"
#include "RussianRoulette.h"

using namespace boost;

void PathApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["path:sampler"] = "kajiya";
  attr["path:maxlength"] = "4";
  attr["path:russianroulette:function"] = "always";
  attr["path:russianroulette:continueprobability"] = "1.0";
  attr["path:russianroulette:minimumsubpathlength"] = "3";
  attr["path:finalgathersamples"] = "64";
  attr["path:estimatephotons"] = "50";
} // end PathApi::getDefaultAttributes()

PathSampler *PathApi
  ::sampler(Gotham::AttributeMap &attr,
            const Gotham::PhotonMaps &photonMaps)
{
  PathSampler *result = 0;
  std::string samplerName = "kajiya";

  // fish out the parameters
  samplerName = attr["path:sampler"];

  size_t maxLength = lexical_cast<size_t>(attr["path:maxlength"]);

  std::string rrFunction = attr["path:russianroulette:function"];

  float continueProbability = lexical_cast<float>(attr["path:russianroulette:continueprobability"]);

  size_t minimumSubpathLength = lexical_cast<size_t>(attr["path:russianroulette:minimumsubpathlength"]);

  float numStrata = lexical_cast<float>(attr["path:finalgathersamples"]);

  // figure out x & y
  size_t xStrata = static_cast<size_t>(floorf(sqrtf(numStrata)));
  size_t yStrata = static_cast<size_t>(ceilf(sqrtf(numStrata)));

  size_t estimatePhotons = lexical_cast<size_t>(attr["path:estimatephotons"]);

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
  else if(rrFunction == "kelemen")
  {
    rr.reset(new KelemenRoulette(continueProbability));
  } // end else if
  else if(rrFunction == "luminance")
  {
    rr.reset(new LuminanceRoulette());
  } // end else if
  else if(rrFunction == "maxoverspectrum")
  {
    rr.reset(new MaxOverSpectrumRoulette());
  } // else if
  else if(rrFunction == "modifiedkelemen")
  {
    rr.reset(new ModifiedKelemenRoulette(continueProbability));
  } // end else if
  else if(rrFunction == "onlyafterdelta")
  {
    rr.reset(new OnlyAfterDeltaRoulette());
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
  else if(samplerName == "jensen")
  {
    JensenSampler *js = new JensenSampler();

    js->setFinalGatherPhotons(estimatePhotons);
    js->setFinalGatherStrata(xStrata, yStrata);

    // fetch the global photon map
    Gotham::PhotonMaps::const_iterator global = photonMaps.find(std::string("global"));
    if(global != photonMaps.end())
    {
      js->setGlobalMap(global->second);
    } // end if

    result = js;
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
    result = new SimpleBidirectionalRussianRouletteSampler(rr, maxLength);
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

