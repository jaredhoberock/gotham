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
  size_t i = 2;
  size_t justAdded[2] = {0, p.size()-1};
  bool subpath = false;
  if(x[0][4] > 0.5f) subpath = true;

  if(!subpath)
  {
    // insert an eye vertex
    if(p.insert(0, scene.getSensors(), false,
                x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;

    // insert a light vertex
    if(p.insert(p.size()-1, scene.getEmitters(), true,
                x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

    while(p.getSubpathLengths().sum() < mMaxPathLength
          && i < x.size())
    {
      if(justAdded[subpath] != Path::NULL_VERTEX)
      {
        if(i == 2)
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, true, false,
                                                                      x[0][0], x[0][1], x[0][2], x[0][3], &mRoulette);
        } // end if
        else if(i == 3)
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, false, false,
                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
        } // end else if
        else
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, !subpath, true,
                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
        } // end else
      } // end if

      ++i;
      subpath = !subpath;
    } // end while
  } // end if
  else
  {
    // insert a light vertex
    if(p.insert(p.size()-1, scene.getEmitters(), true,
                x[2][0], x[2][1], x[2][2], x[2][3]) == Path::NULL_VERTEX) return false;

    // insert an eye vertex
    if(p.insert(0, scene.getSensors(), false,
                x[1][0], x[1][1], x[1][2], x[1][3]) == Path::NULL_VERTEX) return false;

    i = 3;

    // XXX can't start from 2
    // but if we start from 3, we will skip using one of the points
    //i = 2;
    while(p.getSubpathLengths().sum() < mMaxPathLength
          && i < x.size())
    {
      if(justAdded[subpath] != Path::NULL_VERTEX)
      {
        if(!subpath && p.getSubpathLengths()[0] == 1)
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, true, false,
                                                                      x[0][0], x[0][1], x[0][2], x[0][3], &mRoulette);
        } // end if
        else if(subpath && p.getSubpathLengths()[1] == 1)
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, false, false,
                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
        } // end else if
        else
        {
          justAdded[subpath] = p.insertRussianRouletteWithTermination(justAdded[subpath], &scene, !subpath, true,
                                                                      x[i][0], x[i][1], x[i][2], x[i][3], &mRoulette);
        } // end else
      } // end if

      ++i;
      subpath = !subpath;
    } // end while
  } // end else

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

void SimpleBidirectionalRussianRouletteSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
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

  // compute the weight before casting the ray
  if(!L.isBlack()
     && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint())))
  {
    // implement MIS weights
    w = 1.0f / (p.getSubpathLengths().sum()-1);

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
} // end SimpleBidirectionalRussianRouletteSampler::evaluate()

// XXX share the implementation with KelemenSampler
static float computeWeightLightSubpaths(const Scene &scene,
                                        const Path::const_iterator &lLast,
                                        const size_t s,
                                        const size_t lightSubpathLength,
                                        const Path::const_iterator &eLast,
                                        const size_t t,
                                        const size_t eyeSubpathLength,
                                        const RussianRoulette &roulette)
{
  float sum = 0.0f;
  size_t k = s + t;
  float ratio = 1.0f;

  size_t minimumEyeSubpathLength = 1;

  // iteratively adding light vertices
  Path::const_iterator vert = eLast;
  Path::const_iterator nextVert = vert; --nextVert;
  Path::const_iterator prevVert = lLast;
  for(size_t sPrime = s + 1;
      sPrime <= k - minimumEyeSubpathLength;
      ++sPrime, --vert, --nextVert)
  {
    size_t tPrime = k - sPrime;
    float solidAnglePdf = 0;

    // if we find a non-intersectable primitive
    // (like a point light or directional light)
    // break immediately, since the probability
    // of connecting a path there and beyond is 0
    //if(!vert->mPrimitive->isIntersectable()) break;

    bool specularConnection = false;
    // when tPrime is 0, there's no connection to be made
    // so we don't check
    if(tPrime != 0)
    {
      specularConnection |= vert->mScattering->isSpecular();
      specularConnection |= nextVert->mScattering->isSpecular();
    } // end if

    // multiply by probability of point added to light path
    if(sPrime == 1)
    {
      // vertex was added at a light surface
      ratio *= scene.getEmitters()->evaluateSurfaceAreaPdf(vert->mSurface,
                                                           vert->mDg);
    } // end if
    else
    {
      // compute the solid angle pdf
      solidAnglePdf = 0;
      Spectrum f;
      if(sPrime == 2)
      {
        // vertex was added as an emission from a light surface
        solidAnglePdf = prevVert->mEmission->evaluatePdf(prevVert->mToPrev,
                                                         prevVert->mDg);
        f = prevVert->mEmission->evaluate(prevVert->mToPrev, prevVert->mDg);
      } // end else if
      else if(!prevVert->mScattering->isSpecular())
      {
        // XXX PERF: we could add a method which evaluates the integrand & pdf at the same time
        //     which would compute much less redundant work
        solidAnglePdf = prevVert->mScattering->evaluatePdf(prevVert->mToNext,
                                                           prevVert->mDg,
                                                           prevVert->mToPrev);
        f = prevVert->mScattering->evaluate(prevVert->mToNext,
                                           prevVert->mDg,
                                           prevVert->mToPrev);
      } // end if
      else
      {
        solidAnglePdf = 1.0f;
        // XXX handle this case directly:
        //     there needs to be a way to query a specular function
        //     for its reflectance/transmittance
        f = Spectrum::white();
      } // end else

      // XXX technically, if we receive a solidAnglePdf of zero, this means
      //     we should quit altogether: no other sampling strategies will have
      //     a non-zero contribution to the sum
      if(solidAnglePdf != 0)
      {
        ratio *= solidAnglePdf;
      } // end if

      // convert to projected solid angle pdf
      ratio /= prevVert->mDg.getNormal().absDot(prevVert->mToPrev);

      // convert to area product pdf
      ratio *= prevVert->mPreviousGeometricTerm;

      // convert to Russian roulette area product pdf
      // XXX BUG fix this: compute bounceDelta
      ratio *= roulette(sPrime - 1, f, prevVert->mDg, prevVert->mToPrev, solidAnglePdf, false);
    } // end else

    // divide by probability of point removed from eye path
    // use the cached pdf value at the new vertex
    // note that this is not just an optimization: we must do this
    // to account for specular bsdfs since evaluatePdf() always returns 0
    ratio /= (vert->mPdf * vert->mPreviousGeometricTerm);

    if(!specularConnection)
    {
      // power heuristic
      sum += ratio*ratio;
    } // end if

    // update prev vertex
    prevVert = vert;
  } // end for

  return sum;
} // end computeWeightLightSubpaths()

// XXX share the implementation with KelemenSampler
static float computeWeightEyeSubpaths(const Scene &scene,
                                      const Path::const_iterator &lLast,
                                      const size_t s,
                                      const size_t lightSubpathLength,
                                      const Path::const_iterator &eLast,
                                      const size_t t,
                                      const size_t eyeSubpathLength,
                                      const RussianRoulette &roulette)
{
  float sum = 0.0f;
  size_t k = s + t;
  float ratio = 1.0f;

  size_t minimumLightSubpathLength = 1;

  // iteratively adding eye vertices
  Path::const_iterator vert = lLast;
  Path::const_iterator nextVert = vert; ++nextVert;
  Path::const_iterator prevVert = eLast;
  for(size_t tPrime = t + 1;
      tPrime <= k - minimumLightSubpathLength;
      ++tPrime, ++nextVert, ++vert)
  {
    size_t sPrime = k - tPrime;
    float solidAnglePdf = 0;

    // if we find a non-intersectable primitive
    // (like a point light or directional light)
    // break immediately since the probability
    // of connecting a path there and beyond is 0
    //if(!vert->mPrimitive->isIntersectable()) break;

    bool specularConnection = false;

    if(sPrime != 0)
    {
      specularConnection |= vert->mScattering->isSpecular();
      specularConnection |= nextVert->mScattering->isSpecular();
    } // end if

    // multiply by probability of point added to eye path
    if(tPrime == 1)
    {
      // vertex was added as a lens sample
      ratio *= scene.getSensors()->evaluateSurfaceAreaPdf(vert->mSurface,
                                                          vert->mDg);
    } // end if
    else
    {
      solidAnglePdf = 0;
      Spectrum f;
      if(tPrime == 2)
      {
        // vertex was added as a film plane sample
        solidAnglePdf = prevVert->mSensor->evaluatePdf(prevVert->mToNext,
                                                       prevVert->mDg);
        f = prevVert->mSensor->evaluate(prevVert->mToNext,
                                        prevVert->mDg);
      } // end if
      else if(!prevVert->mScattering->isSpecular())
      {
        solidAnglePdf = prevVert->mScattering->evaluatePdf(prevVert->mToPrev,
                                                           prevVert->mDg,
                                                           prevVert->mToNext);
        f = prevVert->mScattering->evaluate(prevVert->mToPrev,
                                            prevVert->mDg,
                                            prevVert->mToNext);
      } // end if
      else
      {
        solidAnglePdf = 1.0f;
        // XXX handle this case directly:
        //     there needs to be a way to query a specular function
        //     for its reflectance/transmittance
        f = Spectrum::white();
      } // end else

      // XXX technically, if we receive a solidAnglePdf of zero, this means
      //     we should quit altogether: no other sampling strategies will have
      //     a non-zero contribution to the sum
      if(solidAnglePdf != 0)
      {
        ratio *= solidAnglePdf;
      } // end if

      // convert to projected solid angle pdf
      ratio /= prevVert->mDg.getNormal().absDot(prevVert->mToNext);

      // convert to area product pdf
      ratio *= prevVert->mNextGeometricTerm;

      // convert to Russian roulette area product pdf
      // XXX BUG: compute fromDelta
      ratio *= roulette(tPrime - 1, f, prevVert->mDg, prevVert->mToNext, solidAnglePdf, false);
    } // end else

    // divide by probability of point removed from light path
    // use the cached pdf value at the new vertex
    // note that this is not just an optimization: we must do this
    // to account for specular bsdfs since evaluatePdf() always returns 0
    ratio /= (vert->mPdf * vert->mNextGeometricTerm);

    // be careful about specular connecting vertices
    // if a connecting vertex (ie, vert or nextVert) is specular,
    // then this path carries no power (since the probability of the other
    // connecting vertex lying on the specular direction is 0)
    // In this case, don't add to the sum

    if(!specularConnection)
    {
      // power heuristic
      sum += ratio*ratio;
    } // end if

    // update prev vertex
    prevVert = vert;
  } // end for tPrime

  return sum;
} // end computeWeightEyeSubpaths()

float SimpleBidirectionalRussianRouletteSampler
  ::computeWeight(const Scene &scene,
                  const Path::const_iterator &lLast,
                  const size_t s,
                  const size_t lightSubpathLength,
                  const Path::const_iterator &eLast,
                  const size_t t,
                  const size_t eyeSubpathLength,
                  const Vector &connection,
                  const float g,
                  const RussianRoulette &roulette) const
{
  float sum = 1.0f;

  sum += computeWeightLightSubpaths
           (scene, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  sum += computeWeightEyeSubpaths
           (scene, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  return 1.0f / sum;
} // end SimpleBidirectionalRussianRouletteSampler::computeWeight()

