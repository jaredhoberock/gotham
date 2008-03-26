/*! \file ArvoKirkSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ArvoKirkSampler class.
 */

#include "ArvoKirkSampler.h"
#include "Path.h"
#include "../geometry/Ray.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "RussianRoulette.h"

ArvoKirkSampler
  ::ArvoKirkSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                    const unsigned int maxEyeLength)
    :Parent(maxEyeLength),mRoulette(roulette)
{
  ;
} // end ArvoKirkSampler::ArvoKirkSampler()

bool ArvoKirkSampler
  ::constructPath(const Scene &scene,
                  const HyperPoint &x,
                  Path &p)
{
  // shorthand
  const RussianRoulette *rr = mRoulette.get();

  // insert a lens vertex
  // reserve the 0th coordinate to choose
  // the film plane
  // XXX remove the need for this
  unsigned int lastPosition = p.insert(0, &scene, scene.getSensors(), false, x[1][0], x[1][1], x[1][2], x[1][3]);

  if(lastPosition == Path::INSERT_FAILED) return false;

  // insert vertices until we miss one or we run out of slots
  float u0 = x[0][0];
  float u1 = x[0][1];
  float u2 = x[0][2];
  float u3 = x[0][3];
  size_t coord = 2;
  while((p.insertRussianRoulette(lastPosition, &scene, true, lastPosition != 0, 
                                 u0, u1, u2, u3, rr))
        < mMaxEyeLength - 1)
  {
    u0 = x[coord][0];
    u1 = x[coord][1];
    u2 = x[coord][2];
    u3 = x[coord][3];
    ++lastPosition;
    ++coord;
  } // end while

  // insert a light vertex at the position just beyond the
  // last eye vertex
  // use the final coordinate to choose the light vertex
  const HyperPoint::value_type &c = x[x.size()-1];
  lastPosition = p.insert(p.getSubpathLengths()[0], &scene, scene.getEmitters(), true,
                          c[0], c[1], c[2], c[3]);

  return lastPosition != Path::INSERT_FAILED;
} // end ArvoKirkSampler::constructPath()

