/*! \file SimpleForwardRussianRouletteSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SimpleForwardRussianRouletteSampler class.
 */

#include "SimpleForwardRussianRouletteSampler.h"
#include "RussianRoulette.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../primitives/SurfacePrimitiveList.h"

SimpleForwardRussianRouletteSampler
  ::SimpleForwardRussianRouletteSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                                        const size_t maxEyeLength)
    :mRoulette(roulette),mMaxEyeLength(std::min<size_t>(maxEyeLength, Path::static_size - 1))
{
  ;
} // end SimpleForwardRussianRouletteSampler::init()

bool SimpleForwardRussianRouletteSampler
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
  if(p.insert(0, scene.getSensors(), false, x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

  unsigned int lastPosition = 0;

  // insert vertices until we miss one or we run out of slots
  float u0 = x[0][0];
  float u1 = x[0][1];
  float u2 = x[0][2];
  float u3 = x[0][3];
  size_t coord = 2;
  while((p.insertRussianRouletteWithTermination(lastPosition, &scene, true, lastPosition != 0, 
                                                u0, u1, u2, u3, rr))
        < mMaxEyeLength - 2)
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
  if(p.insert(p.getSubpathLengths()[0], scene.getEmitters(), true,
              c[0], c[1], c[2], c[3]) == Path::NULL_VERTEX) return false;

  return true;
} // end SimpleForwardRussianRouletteSampler::constructPath()

void SimpleForwardRussianRouletteSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  size_t totalLength = p.getSubpathLengths().sum();
  size_t eyeLength = p.getSubpathLengths()[0];
  size_t lightLength = p.getSubpathLengths()[1];

  const PathVertex &e = p[eyeLength-1];
  const PathVertex &l = p[eyeLength];

  // don't connect specular surfaces to anything
  if(e.mScattering->isSpecular() || l.mScattering->isSpecular()) return;

  Spectrum L = e.mThroughput * l.mThroughput;

  // XXX PERF: make compute throughput take the connection vector and geometric term
  L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                           lightLength, l);

  // compute the weight before casting the ray
  if(!L.isBlack()
     && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint())))
  {
    // add a new result
    results.resize(results.size() + 1);
    Result &r = results.back();

    // multiply by the connection throughput
    r.mThroughput = L;

    // set pdf, weight, and (s,t)
    r.mPdf = e.mAccumulatedPdf * l.mAccumulatedPdf;
    r.mWeight = 1.0f;
    r.mEyeLength = eyeLength;
    r.mLightLength = lightLength;
  } // end if
} // end SimpleForwardRussianRouletteSampler::evaluate()

