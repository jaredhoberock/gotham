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

void MutatorApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["mutator:strategy"]    = std::string("kelemen");
  attr["mutator:largestep"]   = std::string("0.5f");
  attr["mutator:targetseeds"] = std::string("512");
} // end MutatorApi::getDefaultAttributes()

PathMutator *MutatorApi
  ::mutator(Gotham::AttributeMap &attr)
{
  PathMutator *result = 0;

  // fish out the parameters
  std::string mutatorName = attr["mutator:strategy"];

  float largeStepProbability = lexical_cast<float>(attr["mutator:largestep"]);

  size_t numSeeds = lexical_cast<size_t>(attr["mutator:targetseeds"]);

  size_t w = lexical_cast<size_t>(attr["record:width"]);
  size_t h = lexical_cast<size_t>(attr["record:height"]);

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

