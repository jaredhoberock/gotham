/*! \file SimpleBidirectionalRussianRouletteSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SimpleBidirectionalRussianRouletteSampler class.
 */

#include "SimpleBidirectionalRussianRouletteSampler.h"
#include "Path.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../primitives/SurfacePrimitiveList.h"

SimpleBidirectionalRussianRouletteSampler
  ::SimpleBidirectionalRussianRouletteSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                                              const size_t maxLength)
    :mMaxPathLength(maxLength),
     mRoulette(roulette)
{
  ;
} // end SimpleBidirectionalRussianRouletteSampler::init()

bool SimpleBidirectionalRussianRouletteSampler
  ::constructPath(const Scene &scene,
                  ShadingContext &context,
                  const HyperPoint &x,
                  Path &p)
{
  // clear the Path to begin
  p.clear();

  // shorthand
  const RussianRoulette *roulette = mRoulette.get();

  gpcpu::float2 &termination = p.getTerminationProbabilities();
  size_t justAdded[2];

  // insert an eye vertex
  float rr = (*roulette)();
  if(x[2][4] < rr)
  {
    // if we don't find a vertex when we said we would look,
    // we MUST return a failure; otherwise, we will bias results
    // towards shorter paths
    // XXX TODO make a similar insert() method which accepts a RussianRoulette object
    justAdded[0] = p.insert(0, &scene, context, scene.getSensors(), false, x[2][0], x[2][1], x[2][2], x[2][3]);
    if(justAdded[0] == Path::INSERT_FAILED) return false;
    p[justAdded[0]].mPdf *= rr;
    p[justAdded[0]].mAccumulatedPdf *= rr;
  } // end if
  else
  {
    justAdded[0] = Path::ROULETTE_TERMINATED;
    termination[0] = 1.0f - rr;
  } // end else

  // insert a light vertex
  rr = (*roulette)();
  if(x[1][4] < rr)
  {
    // if we don't find a vertex when we said we would look,
    // we MUST return a failure; otherwise, we will bias results
    // towards shorter paths
    // XXX TODO make a similar insert() method which accepts a RussianRoulette object
    justAdded[1] = p.insert(p.size()-1, &scene, context, scene.getEmitters(), true, x[1][0], x[1][1], x[1][2], x[1][3]);
    if(justAdded[1] == Path::INSERT_FAILED) return false;
    p[justAdded[1]].mPdf *= rr;
    p[justAdded[1]].mAccumulatedPdf *= rr;
  } // end if
  else
  {
    justAdded[1] = Path::ROULETTE_TERMINATED;
    termination[1] = 1.0f - rr;
  } // end else

  size_t i = 2;
  bool subpath = false;
  while(p.getSubpathLengths().sum() <= (mMaxPathLength-1)
        && i < x.size())
  {
    if(justAdded[subpath] <= Path::INSERT_SUCCESS)
    {
      if(i == 2)
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, context, true, false,
                                                                    x[0][0], x[0][1], x[0][2], x[0][3], roulette, termination[subpath]);
      } // end if
      else if(i == 3)
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, context, false, false,
                                                                    x[i][0], x[i][1], x[i][2], x[i][3], roulette, termination[subpath]);
      } // end else if
      else
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, context, !subpath, true,
                                                                    x[i][0], x[i][1], x[i][2], x[i][3], roulette, termination[subpath]);
      } // end else
    } // end if

    // if we don't find a vertex when we said we would look,
    // we MUST return a failure; otherwise, we will bias results
    // towards shorter paths
    if(justAdded[subpath] == Path::INSERT_FAILED) return false;

    ++i;
    subpath = !subpath;
  } // end while

  // if we had to terminate due to running out of room,
  // set the subpath's termination probabilities to 1
  if(p.getSubpathLengths().sum() == mMaxPathLength || i == x.size())
  {
    // only do this if we were planning on continuing the subpath
    if(justAdded[0] <= Path::INSERT_SUCCESS) termination[0] = 1.0f;
    if(justAdded[1] <= Path::INSERT_SUCCESS) termination[1] = 1.0f;
  } // end if

  // shuffle the light path so its last light vertex
  // immediately follows the last eye vertex
  size_t j = p.size() - p.getSubpathLengths()[1];
  for(size_t i = p.getSubpathLengths()[0];
      j != p.size();
      ++i, ++j)
  {
    p[i] = p[j];
  } // end for i

  // connect the two subpaths if both of them exist
  if(p.getSubpathLengths()[0] != 0 && p.getSubpathLengths()[1] != 0)
    p.connect(p[p.getSubpathLengths()[0]-1], p[p.getSubpathLengths()[0]]);

  return p.getSubpathLengths().sum() != 0;
} // end SimpleBidirectionalRussianRouletteSampler::constructPath()

void SimpleBidirectionalRussianRouletteSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  // shorthand
  const RussianRoulette *roulette = mRoulette.get();

  size_t eyeLength = p.getSubpathLengths()[0];
  size_t lightLength = p.getSubpathLengths()[1];

  float w;
  float pdf;
  Spectrum L;
  bool addResult = false;

  if(eyeLength != 0 && lightLength != 0)
  {
    const PathVertex &e = p[eyeLength-1];
    const PathVertex &l = p[eyeLength];

    // compute pdf of the entire Path
    pdf = e.mAccumulatedPdf * l.mAccumulatedPdf;

    // don't connect specular surfaces to anything
    // if we used the correct RussianRoulette object, this should never happen
    if(e.mScattering->isSpecular() || l.mScattering->isSpecular()) return;

    // start with the throughput of the connection
    // XXX PERF: make compute throughput take the connection vector and geometric term
    L = p.computeThroughputAssumeVisibility(eyeLength, e,
                                             lightLength, l);

    // modulate by subpaths' throughput
    L *= e.mThroughput * l.mThroughput;

    // compute MIS weight
    w = Path::computePowerHeuristicWeight(scene, 0, 0, &l, lightLength, p.getSubpathLengths()[1],
                                          &e, eyeLength, p.getSubpathLengths()[0],
                                          e.mToNext, e.mNextGeometricTerm, *roulette);

    // add a result if the path carries throughput
    addResult = w != 0 && !L.isBlack() && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint()));
  } // end if
  else if(lightLength != 0)
  {
    const PathVertex &l = p[0];

    // compute pdf
    pdf = l.mAccumulatedPdf;

    // evaluate sensor at the end of the light path
    L = l.mThroughput * l.mSensor->evaluate(l.mToNext, l.mDg);

    // compute MIS weight
    w = Path::computePowerHeuristicWeightEyeSubpaths(scene, 0, &l, lightLength, p.getSubpathLengths()[1],
                                                     0, 0, p.getSubpathLengths()[0],
                                                     *roulette);
    w = 1.0f / (w + 1.0f);

    // add a result if L isn't black & w != 0
    addResult = !L.isBlack() && w != 0;
  } // end else if
  else if(eyeLength != 0)
  {
    const PathVertex &e = p[eyeLength-1];

    // compute pdf
    pdf = e.mAccumulatedPdf;

    // evaluate emission at the end of the eye path
    L = e.mThroughput * e.mEmission->evaluate(e.mToPrev, e.mDg);

    // compute MIS weight
    w = Path::computePowerHeuristicWeightLightSubpaths(scene, 0,
                                                       0, 0, p.getSubpathLengths()[1],
                                                       &e, eyeLength, p.getSubpathLengths()[0],
                                                       *roulette);
    w = 1.0f / (w + 1.0f);

    // add a result if L isn't black & w != 0
    addResult = !L.isBlack() && w != 0;
  } // end else

  if(addResult)
  {
    // add a new result
    results.resize(results.size() + 1);
    Result &r = results.back();

    // multiply by the connection throughput
    r.mThroughput = L;

    // set pdf, weight, and (s,t)
    r.mPdf = pdf * p.getTerminationProbabilities().product();

    r.mWeight = w;
    r.mEyeLength = eyeLength;
    r.mLightLength = lightLength;
  } // end if
} // end SimpleBidirectionalRussianRouletteSampler::evaluate()

