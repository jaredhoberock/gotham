/*! \file WhittedSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of WhittedSampler class.
 */

#include "WhittedSampler.h"
#include "RussianRoulette.h"

WhittedSampler
  ::WhittedSampler(const unsigned int maxEyeLength)
    :Parent(boost::shared_ptr<RussianRoulette>(new OnlyAfterDeltaRoulette()), maxEyeLength)
{
  ;
} // end WhittedSampler::WhittedSampler()

