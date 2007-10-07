/*! \file MutatorApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MutatorApi class.
 */

#include "MutatorApi.h"
#include "../path/PathApi.h"
#include "KelemenMutator.h"
#include "SeededMutator.h"
#include "StratifiedMutator.h"
#include <gpcpu/Vector.h>
using namespace boost;
using namespace gpcpu;

PathMutator *MutatorApi
  ::mutator(const Gotham::AttributeMap &attr)
{
  PathMutator *result = 0;
  std::string mutatorName = "kelemen";

  // fish out the parameters
  Gotham::AttributeMap::const_iterator a = attr.find("mutator::strategy");
  if(a != attr.end())
  {
    any val = a->second;
    mutatorName = boost::any_cast<std::string>(val);
  } // end if

  float largeStepProbability = 0.5f;
  a = attr.find("mutator::largestep");
  if(a != attr.end())
  {
    any val = a->second;
    largeStepProbability = static_cast<float>(atof(any_cast<std::string>(val).c_str()));
  } // end if

  unsigned int w = 512,h = 512;
  a = attr.find("film::width");
  if(a != attr.end())
  {
    any val = a->second;
    w = atoi(any_cast<std::string>(val).c_str());
  } // end if

  a = attr.find("film::height");
  if(a != attr.end())
  {
    any val = a->second;
    h = atoi(any_cast<std::string>(val).c_str());
  } // end if

  size_t numSeeds = 512;
  a = attr.find("mutator::targetseeds");
  if(a != attr.end())
  {
    any val = a->second;
    numSeeds = atoi(any_cast<std::string>(val).c_str());
  } // end if

  // create the mutator
  if(mutatorName == "kelemen")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));
    result = new KelemenMutator(largeStepProbability, sampler);
  } // end if
  else if(mutatorName == "seeded")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));
    result = new SeededMutator(largeStepProbability, sampler, numSeeds);
  } // end else if
  else if(mutatorName == "stratified")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));
    result = new StratifiedMutator(largeStepProbability, sampler, uint2(w,h));
  } // end else if
  else
  {
    std::cerr << "Warning: unknown mutation strategy \"" << mutatorName << "\"." << std::endl;

    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));
    result = new KelemenMutator(largeStepProbability, sampler);
  } // end else if

  return result;
} // end MutatorApi::mutator()

