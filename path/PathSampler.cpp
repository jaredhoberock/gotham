/*! \file PathSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PathSampler class.
 */

#include "PathSampler.h"

void PathSampler
  ::constructHyperPoint(RandomSequence &s, HyperPoint &x)
{
  for(HyperPoint::iterator coord = x.begin();
      coord != x.end();
      ++coord)
  {
    for(HyperPoint::value_type::iterator subcoord = coord->begin();
        subcoord != coord->end();
        ++subcoord)
    {
      *subcoord = s();
    } // end for subcoord
  } // end for coord
} // end PathSampler::constructHyperPoint()

