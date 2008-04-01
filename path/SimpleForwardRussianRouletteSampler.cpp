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
#include "../shading/HemisphericalEmission.h"

SimpleForwardRussianRouletteSampler
  ::SimpleForwardRussianRouletteSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                                        const size_t maxEyeLength)
    :mMaxEyeLength(std::min<size_t>(maxEyeLength, Path::static_size - 1)),
     mRoulette(roulette)
{
  ;
} // end SimpleForwardRussianRouletteSampler::init()

bool SimpleForwardRussianRouletteSampler
  ::constructPath(const Scene &scene,
                  ShadingContext &context,
                  const HyperPoint &x,
                  Path &p)
{
  // clear the Path to begin
  p.clear();

  // shorthand
  const RussianRoulette *rr = mRoulette.get();

  // insert a lens vertex
  // reserve the 0th coordinate to choose
  // the film plane
  // XXX remove the need for this
  if(p.insert(0, &scene, context, scene.getSensors(), false, x[1][0], x[1][1], x[1][2], x[1][3]) == Path::INSERT_FAILED) return false;

  unsigned int lastPosition = 0;

  // insert vertices until we miss one or we run out of slots
  float u0 = x[0][0];
  float u1 = x[0][1];
  float u2 = x[0][2];
  float u3 = x[0][3];
  size_t coord = 2;
  gpcpu::float2 &termination = p.getTerminationProbabilities();
  do
  {
    lastPosition =
      p.insertRussianRouletteWithTermination(lastPosition, &scene, context, true, lastPosition != 0, 
                                             u0, u1, u2, u3, rr, termination[0]);
    u0 = x[coord][0];
    u1 = x[coord][1];
    u2 = x[coord][2];
    u3 = x[coord][3];
    ++coord;
  } // end do
  while(lastPosition < mMaxEyeLength - 1);

  // if we don't find a vertex when we said we would look,
  // we MUST return a failure; otherwise, we will bias results
  // towards shorter paths
  if(lastPosition == Path::INSERT_FAILED) return false;

  // if we terminated because we hit mMaxEyeLength, set termination to 1
  if(p.getSubpathLengths()[0] == mMaxEyeLength) termination[0] = 1.0f;

  // don't insert a light vertex under the following conditions:
  // the end of the eye path is an emitter, and the last eye vertex was
  // sampled from a delta function
  // otherwise, we can't sample such paths correctly
  // XXX DESIGN this check for an emitter is ugly we should just add some isEmitter
  //     method to the ScatteringDistributionFunction class
  const PathVertex &lastEyeVert = p[p.getSubpathLengths()[0]-1];

  if(lastEyeVert.mFromDelta
     && dynamic_cast<const HemisphericalEmission*>(lastEyeVert.mEmission) != 0)
  {
    // do nothing
  } // end if
  else
  {
    // insert a light vertex at the position just beyond the
    // last eye vertex
    // use the final coordinate to choose the light vertex
    const HyperPoint::value_type &c = x[x.size()-1];
    if(p.insert(p.getSubpathLengths()[0], &scene, context, scene.getEmitters(), true,
                c[0], c[1], c[2], c[3]) == Path::INSERT_FAILED) return false;
  } // end else

  termination[1] = 1.0f;

  return true;
} // end SimpleForwardRussianRouletteSampler::constructPath()

void SimpleForwardRussianRouletteSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  size_t eyeLength = p.getSubpathLengths()[0];
  size_t lightLength = p.getSubpathLengths()[1];

  const PathVertex &e = p[eyeLength-1];
  Spectrum L = e.mThroughput;
  float pdf = e.mAccumulatedPdf;

  bool addResult;
  if(lightLength == 0)
  {
    // evaluate emission at the end of the eye path
    L *= e.mEmission->evaluate(e.mToPrev, e.mDg);

    // add a result to the list if it isn't black
    addResult = !L.isBlack();
  } // end if
  else
  {
    const PathVertex &l = p[eyeLength];

    // don't connect specular surfaces to anything
    if(e.mScattering->isSpecular() || l.mScattering->isSpecular()) return;

    // modulate by the light subpath's throughput
    L *= l.mThroughput;

    // modulate pdf by the light subpath's pdf
    pdf *= l.mAccumulatedPdf;

    // compute the throughput of the connection
    // XXX PERF: make compute throughput take the connection vector and geometric term
    L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                             lightLength, l);

    // add a result if the throughput isn't black and it is not shadowed
    addResult = !L.isBlack() && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint()));
  } // end else

  if(addResult)
  {
    // add a new result
    results.resize(results.size() + 1);
    Result &r = results.back();

    // multiply by the connection throughput
    r.mThroughput = L;

    // set pdf, weight, and (s,t)
    // compute the pdf by muliplying the pdfs of generating each subpath times the
    // probability of ending the eye path where we did.
    r.mPdf = pdf * p.getTerminationProbabilities()[0];

    r.mWeight = 1.0f;
    r.mEyeLength = eyeLength;
    r.mLightLength = lightLength;
  } // end if
} // end SimpleForwardRussianRouletteSampler::evaluate()

