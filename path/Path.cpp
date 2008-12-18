/*! \file Path.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Path class.
 */

#include "Path.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../primitives/Scene.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../geometry/Ray.h"
#include "RussianRoulette.h"
#include <typeinfo>
#include <string.h>

// XXX refactor this method and the next
unsigned int Path
  ::insert(const unsigned int i,
           const Scene *scene,
           ShadingContext &context,
           const SurfacePrimitive *prim,
           const bool emission,
           const float u0,
           const float u1,
           const float u2)
{
  PathVertex &vert = (*this)[i];

  vert.mSurface = prim;
  // XXX DESIGN: this should probably return true/false for compatibility
  prim->sampleSurfaceArea(u0, u1, u2, vert.mDg, vert.mPdf);

  // initialize the integrand
  vert.mEmission = context.evaluateEmission(prim->getMaterial(), vert.mDg);
  vert.mScattering = context.evaluateScattering(prim->getMaterial(), vert.mDg);
  vert.mSensor = context.evaluateSensor(prim->getMaterial(), vert.mDg);

  // init delta
  // XXX implement delta for surface area pdfs
  vert.mFromDelta = false;

  // init component
  // XXX implement component for surface area pdfs
  vert.mFromComponent = 0;

  // initialize the throughput as white
  vert.mThroughput = Spectrum::white();

  // init the accumulated pdf
  vert.mAccumulatedPdf = vert.mPdf;

  // init the geometric terms to be 1
  vert.mNextGeometricTerm = vert.mPreviousGeometricTerm = 1.0f;

  // the subpath now has a length of 1
  mSubpathLengths[static_cast<unsigned int>(emission)] = 1;
  return i;
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int i,
           const Scene *scene,
           ShadingContext &context,
           const SurfacePrimitiveList *surfaces,
           const bool emission,
           const float u0,
           const float u1,
           const float u2,
           const float u3)
{
  PathVertex &vert = (*this)[i];

  // sample a point from the list
  if(surfaces->sampleSurfaceArea(u0, u1, u2, u3,
                                 &vert.mSurface,
                                 vert.mDg,
                                 vert.mPdf))
  {
    // initialize the integrand
    vert.mEmission = context.evaluateEmission(vert.mSurface->getMaterial(), vert.mDg);
    vert.mScattering = context.evaluateScattering(vert.mSurface->getMaterial(), vert.mDg);
    vert.mSensor = context.evaluateSensor(vert.mSurface->getMaterial(), vert.mDg);

    // init delta
    // XXX implement delta for surface area pdfs
    vert.mFromDelta = false;

    // init component
    // XXX implement component for surface area pdfs
    vert.mFromComponent = 0;

    // initialize the throughput as white
    vert.mThroughput = Spectrum::white();

    // init the accumulated pdf
    vert.mAccumulatedPdf = vert.mPdf;

    // init the geometric terms to be 1
    vert.mNextGeometricTerm = vert.mPreviousGeometricTerm = 1.0f;

    // the subpath now has a length of 1
    mSubpathLengths[static_cast<unsigned int>(emission)] = 1;
    return i;
  } // end if

  return INSERT_FAILED;
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int previous,
           const Scene *scene,
           ShadingContext &context,
           const bool after,
           const bool scatter,
           const float u0,
           const float u1,
           const float u2)
{
  PathVertex &prev = (*this)[previous];

  Vector w;
  float pdf;

  Spectrum f;

  // sample a ray direction
  bool delta = false;
  ScatteringDistributionFunction::ComponentIndex component = 0;
  if(scatter)
  {
    // scatter
    f = prev.mScattering->sample(after ? prev.mToPrev : prev.mToNext,
                                 prev.mDg, u0, u1, u2,
                                 w, pdf, delta, component);
  } // end if
  else
  {
    // emit/sense
    if(after)
      f = prev.mSensor->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
    else
      f = prev.mEmission->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
  } // end else

  unsigned int result = INSERT_FAILED;
  if(!f.isBlack()) result = insert(previous, scene, context, after, w, f, pdf, delta, component);

  return result;
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int previous,
           const Scene *scene,
           ShadingContext &context,
           const bool after,
           const Vector &dir,
           const Spectrum &f,
           float pdf,
           const bool delta,
           const ScatteringDistributionFunction::ComponentIndex component)
{
  unsigned int result = INSERT_FAILED;

  // XXX i kind of think this should be passed as a reference
  const PathVertex &prev = (*this)[previous];

  // cast a Ray
  Ray r(prev.mDg.getPoint(), dir);

  Intersection inter;
  if(scene->intersect(r, inter))
  {
    // evaluate the integrand
    // XXX yuck
    PrimitiveHandle prim = inter.getPrimitive();
    const SurfacePrimitive *surface =
      dynamic_cast<const SurfacePrimitive*>((*scene->getPrimitives())[prim].get());
    const DifferentialGeometry &dg = inter.getDifferentialGeometry();

    ScatteringDistributionFunction *emission = context.evaluateEmission(surface->getMaterial(), dg);
    ScatteringDistributionFunction *scattering = context.evaluateScattering(surface->getMaterial(), dg);
    ScatteringDistributionFunction *sensor = context.evaluateSensor(surface->getMaterial(), dg);

    // convert pdf to projected solid angle measure
    // XXX this division is by zero at the silhouette
    //     this introduces nans, of course
    // BUG #1
    pdf /= prev.mDg.getNormal().absDot(dir);

    result = insert(previous, after, dir, f, pdf, delta, component,
                    surface, emission, scattering, sensor, dg);
  } // end if

  return result;
} // end Path::insert()

// XXX this probably belonds somewhere else
float Path
  ::computeG(const Normal &n0,
             const Vector &w,
             const Normal &n1,
             const float d2)
{
  // prefer absDot to posDot:
  // This agrees with theory: geometric term is defined as the absolute value
  // of this dot product
  // This is necessary to correctly account for refractive transmission where
  // the refractive direction and surface normal are not in different
  // hemispheres
  return n0.absDot(w) * n1.absDot(w) / d2;
} // end computeG()

unsigned int Path
  ::insert(const unsigned int previous,
           const bool after,
           const Vector &dir,
           const Spectrum &f,
           const float pdf,
           const bool delta,
           const ScatteringDistributionFunction::ComponentIndex component,
           const SurfacePrimitive *surface,
           const ScatteringDistributionFunction *emission,
           const ScatteringDistributionFunction *scattering,
           const ScatteringDistributionFunction *sensor,
           const DifferentialGeometry &dg)
{
  // where to insert?
  unsigned int result = after ? (previous+1) : (previous-1);

  PathVertex &v = (*this)[result];

  // init the vertex
  // XXX I kind of think prev should be passed as a reference
  PathVertex &prev = (*this)[previous];
  v.init(surface, dg, emission, scattering, sensor,
         f * prev.mThroughput, prev.mAccumulatedPdf * pdf, pdf, delta, component);

  // increment the subpath length
  mSubpathLengths[static_cast<unsigned int>(!after)]++;

  // set local geometry
  float d2 = (prev.mDg.getPoint() - v.mDg.getPoint()).norm2();
  float g = computeG(prev.mDg.getNormal(), dir, v.mDg.getNormal(), d2);
  if(after)
  {
    prev.mToNext = dir;
    v.mToPrev = -dir;
    v.mPreviousGeometricTerm = prev.mNextGeometricTerm = g;
  } // end if
  else
  {
    prev.mToPrev = dir;
    v.mToNext = -dir;
    v.mNextGeometricTerm = prev.mPreviousGeometricTerm = g;
  } // end else

  return result;
} // end Path::insert()

void PathVertex
  ::init(const SurfacePrimitive *surface,
         const DifferentialGeometry &dg,
         const ScatteringDistributionFunction *emission,
         const ScatteringDistributionFunction *scattering,
         const ScatteringDistributionFunction *sensor,
         const Spectrum &f,
         const float accumulatedPdf,
         const float pdf,
         const bool delta,
         const ScatteringDistributionFunction::ComponentIndex component)
{
  mSurface = surface;
  mDg = dg;
  mEmission = emission;
  mScattering = scattering;
  mSensor = sensor;
  mThroughput = f;
  mAccumulatedPdf = accumulatedPdf;
  mPdf = pdf;
  mFromDelta = delta;
  mFromComponent = component;
} // end PathVertex::init()

void Path
  ::connect(PathVertex &v0,
            PathVertex &v1)
{
  Vector wi = v1.mDg.getPoint() - v0.mDg.getPoint();
  float d2 = wi.dot(wi);

  // XXX inverse square root here?
  float d = sqrtf(d2);
  wi /= d;

  v0.mToNext = wi;
  v1.mToPrev = -wi;

  v0.mNextGeometricTerm = computeG(v0.mDg.getNormal(), wi, v1.mDg.getNormal(), d2);
  v1.mPreviousGeometricTerm = v0.mNextGeometricTerm;
} // end Path::connect()

Spectrum Path
  ::computeThroughputAssumeVisibility(const unsigned int eyeSubpathLength,
                                      const PathVertex &e,
                                      const unsigned int lightSubpathLength,
                                      const PathVertex &l) const
{
  Vector wi = l.mDg.getPoint() - e.mDg.getPoint();
  float d2 = wi.dot(wi);
  // XXX inverse square root here?
  float d = sqrtf(d2);
  wi /= d;
  Spectrum f(1,1,1);

  // evaluate integrand on eye end
  if(eyeSubpathLength == 1)
  {
    f *= e.mSensor->evaluate(wi, e.mDg);
  } // end if
  else
  {
    f *= e.mScattering->evaluate(e.mToPrev, e.mDg, wi);
  } // end else

  // evaluate bsdf on light end
  if(lightSubpathLength == 1)
  {
    f *= l.mEmission->evaluate(-wi, l.mDg);
  } // end if
  else
  {
    f *= l.mScattering->evaluate(l.mToNext, l.mDg, -wi);
  } // end else

  // modulate by geometry term
  float g = computeG(e.mDg.getNormal(), wi, l.mDg.getNormal(), d2);
  f *= g;

  return f;
} // end Path::computeThroughputAssumeVisibility()

const gpcpu::uint2 &Path
  ::getSubpathLengths(void) const
{
  return mSubpathLengths;
} // end Path::getSubpathLengths()

const gpcpu::float2 &Path
  ::getTerminationProbabilities(void) const
{
  return mTerminationProbabilities;
} // end Path::getTerminationProbabilities()

gpcpu::float2 &Path
  ::getTerminationProbabilities(void)
{
  return mTerminationProbabilities;
} // end Path::getTerminationProbabilities()

void Path
  ::clear(void)
{
  memset(this, 0, sizeof(Path));
} // end Path::clear()

unsigned int Path
  ::insertRussianRoulette(const unsigned int previous,
                          const Scene *scene,
                          ShadingContext &context,
                          const bool after,
                          const bool scatter,
                          const float u0,
                          const float u1,
                          const float u2,
                          const float u3,
                          const RussianRoulette *roulette)
{
  PathVertex &prev = (*this)[previous];

  Vector w;
  float pdf;

  Spectrum f;

  // sample a ray direction
  bool delta = false;
  ScatteringDistributionFunction::ComponentIndex component = 0;
  if(scatter)
  {
    // scatter
    f = prev.mScattering->sample(after ? prev.mToPrev : prev.mToNext,
                                 prev.mDg, u0, u1, u2,
                                 w, pdf, delta, component);
  } // end if
  else
  {
    // emit/sense
    // XXX implement delta/component for sensors/emitters
    if(after)
      f = prev.mSensor->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
    else
      f = prev.mEmission->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
  } // end else

  // test russian roulette before casting a ray
  unsigned int newIndex = getSubpathLengths()[static_cast<unsigned int>(!after)];
  float rr = (*roulette)(newIndex, f, prev.mDg, w, pdf, delta);

  unsigned int result = ROULETTE_TERMINATED;
  if(u3 < rr)
  {
    result = insert(previous, scene, context, after, w, f, pdf * rr, delta, component);
  } // end if

  return result;
} // end Path::insert()

unsigned int Path
  ::insertRussianRouletteWithTermination(const unsigned int previous,
                                         const Scene *scene,
                                         ShadingContext &context,
                                         const bool after,
                                         const bool scatter,
                                         const float u0,
                                         const float u1,
                                         const float u2,
                                         const float u3,
                                         const RussianRoulette *roulette,
                                         float &termination)
{
  PathVertex &prev = (*this)[previous];

  Vector w;
  float pdf;

  Spectrum f;

  // sample a ray direction
  bool delta = false;
  ScatteringDistributionFunction::ComponentIndex component;
  if(scatter)
  {
    // scatter
    f = prev.mScattering->sample(after ? prev.mToPrev : prev.mToNext,
                                 prev.mDg, u0, u1, u2,
                                 w, pdf, delta, component);
  } // end if
  else
  {
    // emit/sense
    // XXX implement delta/component for sensor/emission
    if(after)
      f = prev.mSensor->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
    else
      f = prev.mEmission->sample(prev.mDg, u0, u1, u2, w, pdf, delta);
  } // end else

  // test russian roulette before casting a ray
  unsigned int newIndex = getSubpathLengths()[static_cast<unsigned int>(!after)];
  float rr = (*roulette)(newIndex, f, prev.mDg, w, pdf, delta);

  unsigned int result = ROULETTE_TERMINATED;
  if(u3 < rr)
  {
    result = insert(previous, scene, context, after, w, f, pdf * rr, delta, component);
  } // end if

  termination = 1.0f - rr;

  return result;
} // end Path::insertRussianRouletteWithTermination()

Path
  ::Path(void)
{
  clear();
} // end Path::Path()

Path
  ::Path(const Path &p)
{
  std::cerr << "Path::Path(): Disallowed copy constructor called!" << std::endl;

  // do a dumb copy
  *this = p;

  assert(0);
} // end Path::Path()

bool Path
  ::clone(Path &dst, FunctionAllocator &allocator) const
{
  // first do a dumb copy
  dst = *this;

  // now, for each vertex of dst, allocate
  // a new integrand with allocator and copy
  // from src
  for(size_t i = 0; i < dst.getSubpathLengths().sum(); ++i)
  {
    if((*this)[i].mSensor != 0)
    {
      // clone
      dst[i].mSensor = (*this)[i].mSensor->clone(allocator);
      if(dst[i].mSensor == 0) return false;
    } // end if

    if((*this)[i].mScattering != 0)
    {
      // clone
      dst[i].mScattering = (*this)[i].mScattering->clone(allocator);
      if(dst[i].mScattering == 0) return false;
    } // end if

    if((*this)[i].mEmission != 0)
    {
      // clone
      dst[i].mEmission = (*this)[i].mEmission->clone(allocator);
      if(dst[i].mEmission == 0) return false;
    } // end if
  } // end for i

  return true;
} // end Path::clone()

float Path
  ::computePowerHeuristicWeight(const Scene &scene,                                             
                                const size_t minimumLightSubpathLength,
                                const size_t minimumEyeSubpathLength,
                                const const_iterator &lLast,
                                const size_t s,
                                const size_t lightSubpathLength,
                                const const_iterator &eLast,
                                const size_t t,
                                const size_t eyeSubpathLength,
                                const Vector &connection,
                                const float g,
                                const RussianRoulette &roulette)
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

  sum += computePowerHeuristicWeightLightSubpaths
           (scene, minimumEyeSubpathLength, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  sum += computePowerHeuristicWeightEyeSubpaths
           (scene, minimumLightSubpathLength, lLast, s, lightSubpathLength,
            eLast, t, eyeSubpathLength, roulette);

  // restore data
  const_cast<PathVertex&>(*lLast).mToPrev = cacheLightToPrev;
  const_cast<PathVertex&>(*eLast).mToNext = cacheEyeToNext;
  const_cast<PathVertex&>(*lLast).mPreviousGeometricTerm = cacheLightPrevG;
  const_cast<PathVertex&>(*eLast).mNextGeometricTerm = cacheEyeNextG;

  return 1.0f / sum;
} // end Path::computePowerHeuristicWeight()

float Path
  ::computePowerHeuristicWeightLightSubpaths(const Scene &scene,
                                             const size_t minimumEyeSubpathLength,
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

  // note this is different from specularConnection
  bool deltaBounce = false;
  ScatteringDistributionFunction::ComponentIndex bounceComponent = 0;

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
        // XXX DESIGN: rather than taking a boolean deltaBounce, just keep track of which
        //             component the bounce came from?
        // vertex was added as a bounce
        f = prevVert->mScattering->evaluate(prevVert->mToNext,
                                            prevVert->mDg,
                                            prevVert->mToPrev,
                                            deltaBounce,
                                            bounceComponent,
                                            solidAnglePdf);
      } // end else

      // XXX technically, if we receive a solidAnglePdf of zero, this means
      //     we should quit altogether: no other sampling strategies will have
      //     a non-zero contribution to the sum
      if(solidAnglePdf != 0)
      {
        ratio *= solidAnglePdf;
      } // end if

      // convert to projected solid angle pdf
      // BUG #1
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
} // end Path::computePowerHeuristicWeightLightSubpaths()

float Path
  ::computePowerHeuristicWeightEyeSubpaths(const Scene &scene,
                                           const size_t minimumLightSubpathLength,
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
  ScatteringDistributionFunction::ComponentIndex bounceComponent = 0;

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
        // XXX DESIGN: rather than taking a boolean deltaBounce, just keep track of which
        //             component the bounce came from?
        // vertex was added as a bounce
        f = prevVert->mScattering->evaluate(prevVert->mToPrev,
                                            prevVert->mDg,
                                            prevVert->mToNext,
                                            deltaBounce,
                                            bounceComponent,
                                            solidAnglePdf);
      } // end else

      // XXX technically, if we receive a solidAnglePdf of zero, this means
      //     we should quit altogether: no other sampling strategies will have
      //     a non-zero contribution to the sum
      if(solidAnglePdf != 0)
      {
        ratio *= solidAnglePdf;
      } // end if

      // convert to projected solid angle pdf
      // BUG #1
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
} // end Path::computePowerHeuristicWeightEyeSubpaths()

bool Path
  ::isSane(void) const
{
  for(size_t i = 0; i != getSubpathLengths().sum(); ++i)
  {
    const PathVertex &v = operator[](i);
    if(v.mScattering == 0 && v.mEmission == 0 && v.mSensor == 0)
    {
      return false;
    } // end if
  } // end for i

  return true;
} // end Path::isSane()

