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

float computeWeightLightSubpaths(const Scene &scene,
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

  size_t minimumEyeSubpathLength = 0;

  // note this is different from specularConnection
  bool deltaBounce = false;
  ScatteringDistributionFunction::ComponentIndex bounceComponent = 0;

  // iteratively adding light vertices
  Path::const_iterator vert = eLast;
  Path::const_iterator nextVert = vert; --nextVert;
  Path::const_iterator prevVert = lLast;
  for(size_t sPrime = s + 1;
      sPrime <= k;
      ++sPrime, --vert, --nextVert)
  {
    size_t tPrime = k - sPrime;
    float solidAnglePdf = 0;

    // if we find a non-intersectable primitive
    // (like a point light or directional light)
    // break immediately, since the probability
    // of connecting a path there and beyond is 0
    //if(!vert->mPrimitive->isIntersectable()) break;

    // note this is different from deltaBounce
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
      else
      {
        // XXX PERF: we could add a method which evaluates the integrand & pdf at the same time
        //     which would compute much less redundant work
        // XXX DESIGN: rather than taking a boolean deltaBounce, just keep track of which
        //             component the bounce came from?
        solidAnglePdf = prevVert->mScattering->evaluatePdf(prevVert->mToNext,
                                                           prevVert->mDg,
                                                           prevVert->mToPrev,
                                                           deltaBounce,
                                                           bounceComponent);

        // XXX DESIGN: kill this branch by creating an evaluate() method
        //     similar to the evaluatePdf() we just called.
        if(!prevVert->mScattering->isSpecular())
        {
          f = prevVert->mScattering->evaluate(prevVert->mToNext,
                                              prevVert->mDg,
                                              prevVert->mToPrev);
        } // end if
        else
        {
          // XXX this isn't correct in general
          f = Spectrum::white();
        } // end else
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
      ratio *= roulette(sPrime - 1, f, prevVert->mDg, prevVert->mToPrev, solidAnglePdf, deltaBounce);

      // update bounce info
      // not on the first iteration
      if(sPrime != s + 1)
      {
        deltaBounce = prevVert->mFromDelta;
        bounceComponent = prevVert->mFromComponent;
      } // end if
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

float computeWeightEyeSubpaths(const Scene &scene,
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

  // note that this is different from specularConnection
  bool deltaBounce = false;
  ScatteringDistributionFunction::ComponentIndex bounceComponent;

  // iteratively adding eye vertices
  Path::const_iterator vert = lLast;
  Path::const_iterator nextVert = vert; ++nextVert;
  Path::const_iterator prevVert = eLast;
  for(size_t tPrime = t + 1;
      tPrime <= k;
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
      // compute the solid angle pdf
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
      else
      {
        // XXX PERF: we could add a method which evaluates the integrand & pdf at the same time
        //     which would compute much less redundant work
        // XXX DESIGN: rather than taking a boolean deltaBounce, just keep track of which
        //             component the bounce came from?
        solidAnglePdf = prevVert->mScattering->evaluatePdf(prevVert->mToPrev,
                                                           prevVert->mDg,
                                                           prevVert->mToNext,
                                                           deltaBounce,
                                                           bounceComponent);

        // XXX DESIGN: kill this branch by creating an evaluate() method
        //     similar to the evaluatePdf() we just called.
        if(!prevVert->mScattering->isSpecular())
        {
          f = prevVert->mScattering->evaluate(prevVert->mToPrev,
                                              prevVert->mDg,
                                              prevVert->mToNext);
        } // end if
        else
        {
          // XXX this isn't correct in general
          f = Spectrum::white();
        } // end else
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
      ratio *= roulette(tPrime - 1, f, prevVert->mDg, prevVert->mToNext, solidAnglePdf, deltaBounce);

      // update bounce info
      // not on the first iteration
      if(tPrime != t + 1)
      {
        deltaBounce = prevVert->mFromDelta;
        bounceComponent = prevVert->mFromComponent;
      } // end if
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
         && (w = computeWeight(scene, &l, lightLength, p.getSubpathLengths()[1],
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
          w = computeWeightEyeSubpaths(scene, &l, lightLength, p.getSubpathLengths()[1],
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
        w = computeWeightLightSubpaths(scene,
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

float KelemenSampler
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

  // Temporarily set the geometric terms and to/prev vectors
  // at eLast and lLast, since as they are, they correspond
  // to a different path.  We will restore them at the end of the
  // function.
  Vector cacheLightToPrev = lLast->mToPrev;
  Vector cacheEyeToNext = eLast->mToNext;
  float cacheLightPrevG = lLast->mPreviousGeometricTerm;
  float cacheEyeNextG = eLast->mNextGeometricTerm;

  const_cast<PathVertex&>(*lLast).mToPrev = -connection;
  const_cast<PathVertex&>(*lLast).mPreviousGeometricTerm = g;
  const_cast<PathVertex&>(*eLast).mToNext = connection;
  const_cast<PathVertex&>(*eLast).mNextGeometricTerm = g;

  sum += computeWeightLightSubpaths
           (scene, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  sum += computeWeightEyeSubpaths
           (scene, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  // restore data
  const_cast<PathVertex&>(*lLast).mToPrev = cacheLightToPrev;
  const_cast<PathVertex&>(*eLast).mToNext = cacheEyeToNext;
  const_cast<PathVertex&>(*lLast).mPreviousGeometricTerm = cacheLightPrevG;
  const_cast<PathVertex&>(*eLast).mNextGeometricTerm = cacheEyeNextG;

  return 1.0f / sum;
} // end KelemenSampler::computeWeight()

