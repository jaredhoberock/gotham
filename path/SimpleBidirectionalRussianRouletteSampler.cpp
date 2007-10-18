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
  ::SimpleBidirectionalRussianRouletteSampler(const size_t maxLength)
    :mRoulette(0.5f),mMaxPathLength(maxLength)
{
  ;
} // end SimpleBidirectionalRussianRouletteSampler::init()

bool SimpleBidirectionalRussianRouletteSampler
  ::constructPath(const Scene &scene,
                  const HyperPoint &x,
                  Path &p)
{
  // shorthand
  const RussianRoulette *roulette = &mRoulette;

  // insert an eye vertex
  if(p.insert(0, scene.getSensors(), false,
              x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;

  // insert a light vertex
  if(p.insert(p.size()-1, scene.getEmitters(), true,
              x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

  size_t i = 2;
  size_t justAdded[2] = {0, p.size()-1};
  bool subpath = false;
  gpcpu::float2 &termination = p.getTerminationProbabilities();
  while(p.getSubpathLengths().sum() < mMaxPathLength
        && i < x.size())
  {
    if(justAdded[subpath] != Path::NULL_VERTEX)
    {
      if(i == 2)
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, true, false,
                                                                    x[0][0], x[0][1], x[0][2], x[0][3], roulette, termination[subpath]);
      } // end if
      else if(i == 3)
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, false, false,
                                                                    x[i][0], x[i][1], x[i][2], x[i][3], roulette, termination[subpath]);
      } // end else if
      else
      {
        justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, !subpath, true,
                                                                    x[i][0], x[i][1], x[i][2], x[i][3], roulette, termination[subpath]);
      } // end else
    } // end if

    ++i;
    subpath = !subpath;
  } // end while

  // if we just terminated due to running out of room,
  // set that subpath's termination probability to 1
  if(p.getSubpathLengths().sum() == mMaxPathLength) termination[!subpath] = 1.0f;

  // shuffle the light path so its last light vertex
  // immediately follows the last eye vertex
  size_t j = p.size() - p.getSubpathLengths()[1];
  for(size_t i = p.getSubpathLengths()[0];
      j != p.size();
      ++i, ++j)
  {
    p[i] = p[j];
  } // end for i

  // connect the two subpaths
  p.connect(p[p.getSubpathLengths()[0]-1], p[p.getSubpathLengths()[0]]);

  return true;
} // end SimpleBidirectionalRussianRouletteSampler::constructPath()

//bool SimpleBidirectionalRussianRouletteSampler
//  ::constructPath(const Scene &scene,
//                  const HyperPoint &x,
//                  Path &p)
//{
//  size_t i = 2;
//  size_t justAdded[2] = {0, p.size()-1};
//  bool subpath = false;
//  if(x[0][4] > 0.5f) subpath = true;
//
//  if(!subpath)
//  {
//    // insert an eye vertex
//    if(p.insert(0, scene.getSensors(), false,
//                x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;
//
//    // insert a light vertex
//    if(p.insert(p.size()-1, scene.getEmitters(), true,
//                x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;
//
//    while(p.getSubpathLengths().sum() < mMaxPathLength
//          && i < x.size())
//    {
//      if(justAdded[subpath] != Path::NULL_VERTEX)
//      {
//        if(i == 2)
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, true, false,
//                                                                      x[0][0], x[0][1], x[0][2], x[0][3], &mRoulette);
//        } // end if
//        else if(i == 3)
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, false, false,
//                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
//        } // end else if
//        else
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, !subpath, true,
//                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
//        } // end else
//      } // end if
//
//      ++i;
//      subpath = !subpath;
//    } // end while
//  } // end if
//  else
//  {
//    // insert a light vertex
//    if(p.insert(p.size()-1, scene.getEmitters(), true,
//                x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;
//
//    // insert an eye vertex
//    if(p.insert(0, scene.getSensors(), false,
//                x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;
//
//    i = 3;
//
//    // XXX can't start from 2
//    // but if we start from 3, we will skip using one of the points
//    //i = 2;
//    while(p.getSubpathLengths().sum() < mMaxPathLength
//          && i < x.size())
//    {
//      if(justAdded[subpath] != Path::NULL_VERTEX)
//      {
//        if(!subpath && p.getSubpathLengths()[0] == 1)
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, true, false,
//                                                                      x[0][0], x[0][1], x[0][2], x[0][3], &mRoulette);
//        } // end if
//        else if(subpath && p.getSubpathLengths()[1] == 1)
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, false, false,
//                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
//        } // end else if
//        else
//        {
//          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, !subpath, true,
//                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
//        } // end else
//      } // end if
//
//      ++i;
//      subpath = !subpath;
//    } // end while
//  } // end else
//
//  // shuffle the light path so its last light vertex
//  // immediately follows the last eye vertex
//  size_t j = p.size() - p.getSubpathLengths()[1];
//  for(size_t i = p.getSubpathLengths()[0];
//      j != p.size();
//      ++i, ++j)
//  {
//    p[i] = p[j];
//  } // end for i
//
//  // connect the two subpaths
//  p.connect(p[p.getSubpathLengths()[0]-1], p[p.getSubpathLengths()[0]]);
//
//  return true;
//} // end SimpleBidirectionalRussianRouletteSampler::constructPath()

void SimpleBidirectionalRussianRouletteSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  // shorthand
  const RussianRoulette *roulette = &mRoulette;

  size_t totalLength = p.getSubpathLengths().sum();
  size_t eyeLength = p.getSubpathLengths()[0];
  size_t lightLength = p.getSubpathLengths()[1];

  float w;
  const PathVertex &e = p[eyeLength-1];
  const PathVertex &l = p[eyeLength];

  // don't connect specular surfaces to anything
  if(e.mScattering->isSpecular() || l.mScattering->isSpecular()) return;

  Spectrum L = e.mThroughput * l.mThroughput;

  // XXX PERF: make compute throughput take the connection vector and geometric term
  L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                           lightLength, l);

  // both subpaths should have at least one vertex
  assert(eyeLength != 0 && lightLength != 0);
  w = Path::computePowerHeuristicWeight(scene, 1, 1, &l, lightLength, p.getSubpathLengths()[1],
                                        &e, eyeLength, p.getSubpathLengths()[0],
                                        e.mToNext, e.mNextGeometricTerm, *roulette);  //// compute the weight before casting the ray
  //if(eyeLength != 0 && lightLength != 0)
  //{
  //  w = Path::computePowerHeuristicWeightWithTermination(scene, &l, lightLength, p.getSubpathLengths()[1],
  //                                        &e, eyeLength, p.getSubpathLengths()[0],
  //                                        e.mToNext, e.mNextGeometricTerm, *roulette);
  //} // end if
  //else if(eyeLength == 0)
  //{
  //  w = Path::computePowerHeuristicWeightWithTerminationEyeSubpaths(scene, &l, lightLength, p.getSubpathLengths()[1],
  //                                                   0, 0, p.getSubpathLengths()[0],
  //                                                   *roulette);
  //  w = 1.0f / (w + 1.0f);
  //} // end else if
  //else
  //{
  //  w = Path::computePowerHeuristicWeightWithTerminationLightSubpaths(scene,
  //                                                     0, 0, p.getSubpathLengths()[1],
  //                                                     &e, eyeLength, p.getSubpathLengths()[0],
  //                                                     *roulette);
  //  w = 1.0f / (w + 1.0f);
  //} // end else

  if(!L.isBlack()
     && w != 0
     && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint())))
  {
    //// implement MIS weights
    //w = 1.0f / (p.getSubpathLengths().sum()-1);

    // add a new result
    results.resize(results.size() + 1);
    Result &r = results.back();

    // multiply by the connection throughput
    r.mThroughput = L;

    // set pdf, weight, and (s,t)
    r.mPdf = e.mAccumulatedPdf * l.mAccumulatedPdf * p.getTerminationProbabilities().product();

    r.mWeight = w;
    r.mEyeLength = eyeLength;
    r.mLightLength = lightLength;
  } // end if
} // end SimpleBidirectionalRussianRouletteSampler::evaluate()

