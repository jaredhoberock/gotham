/*! \file KelemenSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of KelemenSampler class.
 */

#include "KelemenSampler.h"
#include "../geometry/Ray.h"
#include "../shading/Material.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../primitives/Scene.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "RussianRoulette.h"

KelemenSampler
  ::KelemenSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                   const unsigned int maxLength)
     :mMaxPathLength(maxLength),mRoulette(roulette)
{
  ;
} // end KelemenSampler::KelemenSampler()

bool KelemenSampler
  ::constructPathInterleaved(const Scene &scene,
                             const HyperPoint &x,
                             Path &p) const
{
  // shorthand
  const RussianRoulette *roulette = mRoulette.get();

  // insert an eye vertex
  if(p.insert(0, scene.getSensors(), false,
              x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;

  // insert a light vertex
  if(p.insert(p.size()-1, scene.getEmitters(), true,
              x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

  size_t i = 2;
  size_t justAdded[2] = {0, p.size()-1};
  bool subpath = false;
  while(p.getSubpathLengths().sum() < mMaxPathLength
        && i < x.size())
  {
    if(justAdded[subpath] != Path::NULL_VERTEX)
    {
      if(i == 2)
      {
        justAdded[subpath] = p.insertRussianRoulette(justAdded[subpath], &scene, true, false,
                                                     x[0][0], x[0][1], x[0][2], x[0][3], roulette);
      } // end if
      else if(i == 3)
      {
        justAdded[subpath] = p.insertRussianRoulette(justAdded[subpath], &scene, false, false,
                                                     x[i][0], x[i][1], x[i][2], x[i][3], roulette);
      } // end else if
      else
      {
        justAdded[subpath] = p.insertRussianRoulette(justAdded[subpath], &scene, !subpath, true,
                                                     x[i][0], x[i][1], x[i][2], x[i][3], roulette);
      } // end else
    } // end if

    ++i;
    subpath = !subpath;
  } // end while

  return true;
} // end KelemenSampler::constructPathInterleaved()

bool KelemenSampler
  ::constructEyePath(const Scene &scene,
                     const HyperPoint &x,
                     Path &p) const
{
  // shorthand
  const RussianRoulette *roulette = mRoulette.get();

  // insert an eye vertex
  if(p.insert(0, scene.getSensors(), false,
              x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;

  // treat every other coordinate of x as an eye coordinate
  size_t i = 2;
  size_t justAdded = 0;
  while(p.getSubpathLengths().sum() <= mMaxPathLength
        && justAdded != Path::NULL_VERTEX
        && i < x.size())
  {
    if(i == 2)
    {
      justAdded = p.insertRussianRoulette(justAdded, &scene, true, p.getSubpathLengths()[0] != 1,
                                          x[0][0], x[0][1], x[0][2], x[0][3], roulette);
    } // end if
    else
    {
      justAdded = p.insertRussianRoulette(justAdded, &scene, true, true,
                                          x[i][0], x[i][1], x[i][2], x[i][3], roulette);
    } // end else

    i += 2;
  } // end while

  return true;
} // end KelemenSampler::constructEyePath()

bool KelemenSampler
  ::constructLightPath(const Scene &scene,
                       const HyperPoint &x,
                       Path &p) const
{
  // shorthand
  const RussianRoulette *roulette = mRoulette.get();

  // insert a light vertex
  if(p.insert(p.size()-1, scene.getEmitters(), true,
              x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

  // treat every other coordinate of x as a light coordinate
  size_t i = 2;
  size_t justAdded = p.size()-1;
  while(p.getSubpathLengths().sum() <= mMaxPathLength
        && justAdded != Path::NULL_VERTEX
        && i < x.size())
  {
    if(i == 3)
    {
      justAdded = p.insertRussianRoulette(justAdded, &scene, false, false,
                                          x[i][0], x[i][1], x[i][2], x[i][3], roulette);
    } // end if
    else
    {
      justAdded = p.insertRussianRoulette(justAdded, &scene, false, true,
                                          x[i][0], x[i][1], x[i][2], x[i][3], roulette);
    } // end else

    i += 2;
  } // end while

  return true;
} // end KelemenSampler::constructEyePath()

bool KelemenSampler
  ::constructPath(const Scene &scene,
                  const HyperPoint &x,
                  Path &p)
{
  p.clear();
  //constructEyePath(scene, x, p);
  //constructLightPath(scene, x, p);
  if(constructPathInterleaved(scene, x, p))
  {
    // shuffle the light path so its last vertex
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
  } // end if

  return false;
} // end KelemenSampler::constructPath()

void KelemenSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  // shorthand
  const RussianRoulette &roulette = *mRoulette.get();

  size_t totalLength = p.getSubpathLengths().sum();

  float w;
  for(size_t eyeLength = 1;
      eyeLength <= p.getSubpathLengths()[0];
      ++eyeLength)
  {
    // XXX yuck
    PathVertex &e = const_cast<PathVertex&>(p[eyeLength-1]);

    for(size_t lightLength = 1;
        lightLength <= p.getSubpathLengths()[1];
        ++lightLength)
    {
      // connect the eye and light subpaths
      // XXX yuck
      PathVertex &l = const_cast<PathVertex&>(p[totalLength - lightLength]);

      // don't connect specular surfaces to anything
      if(e.mScattering->isSpecular() || l.mScattering->isSpecular()) continue;

      // save the old connection
      Vector toNext = e.mToNext;
      Vector toPrev = l.mToPrev;
      float g = e.mNextGeometricTerm;

      // temporarily connect the vertices
      const_cast<Path&>(p).connect(e, l);

      Spectrum L = e.mThroughput * l.mThroughput;
      // XXX PERF: make compute throughput take the connection vector and geometric term
      L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                               lightLength, l);

      // compute the weight before casting the ray
      if(!L.isBlack()
         && (w = Path::computePowerHeuristicWeight(scene, 0, 0, &l, lightLength, p.getSubpathLengths()[1],
                                                   &e, eyeLength, p.getSubpathLengths()[0],
                                                   e.mToNext, e.mNextGeometricTerm, roulette))
         && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint())))
      {
        // add a new result
        results.resize(results.size() + 1);
        Result &r = results.back();

        // multiply by the connection throughput
        r.mThroughput = L;

        // set pdf, weight, and (s,t)
        r.mPdf = e.mAccumulatedPdf * l.mAccumulatedPdf;
        r.mWeight = w;
        r.mEyeLength = eyeLength;
        r.mLightLength = lightLength;
      } // end if

      // handle eye path of length 0
      if(lightLength > 1)
      {
        Spectrum S = l.mSensor->evaluate(l.mToNext, l.mDg);
        if(!S.isBlack())
        {
          w = Path::computePowerHeuristicWeightEyeSubpaths(scene, 0, &l, lightLength, p.getSubpathLengths()[1],
                                                           0, 0, p.getSubpathLengths()[0],
                                                           roulette);
          w = 1.0f / (w + 1.0f);

          // add a new result
          results.resize(results.size() + 1);
          Result &r = results.back();

          r.mThroughput = S * l.mThroughput;

          // set pdf, weight, and (s,t)
          r.mPdf = l.mAccumulatedPdf;
          r.mWeight = w;
          r.mEyeLength = 0;
          r.mLightLength = lightLength;
        } // end if
      } // end if

      // restore the connection
      e.mToNext = toNext;
      l.mToPrev = toPrev;
      e.mNextGeometricTerm = l.mPreviousGeometricTerm = g;
    } // end for lightLength

    // handle light path of length 0
    if(eyeLength > 1)
    {
      Spectrum E = e.mEmission->evaluate(e.mToPrev, e.mDg);
      if(!E.isBlack())
      {
        w = Path::computePowerHeuristicWeightLightSubpaths(scene, 0,
                                                           0, 0, p.getSubpathLengths()[1],
                                                           &e, eyeLength, p.getSubpathLengths()[0],
                                                           roulette);
        w = 1.0f / (w + 1.0f);

        // add a new result
        results.resize(results.size() + 1);
        Result &r = results.back();

        r.mThroughput = e.mThroughput * E;


        // set pdf, weight, and (s,t)
        r.mPdf = e.mAccumulatedPdf;
        r.mWeight = w;
        r.mEyeLength = eyeLength;
        r.mLightLength = 0;
      } // end if
    } // end if
  } // end for eyeLength
} // end KelemenSampler::evaluate()
