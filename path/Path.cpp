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

// XXX refactor this method and the next
unsigned int Path
  ::insert(const unsigned int i,
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
  vert.mEmission = vert.mSurface->getMaterial()->evaluateEmission(vert.mDg);
  vert.mScattering = vert.mSurface->getMaterial()->evaluateScattering(vert.mDg);
  vert.mSensor = vert.mSurface->getMaterial()->evaluateSensor(vert.mDg);

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
    vert.mEmission = vert.mSurface->getMaterial()->evaluateEmission(vert.mDg);
    vert.mScattering = vert.mSurface->getMaterial()->evaluateScattering(vert.mDg);
    vert.mSensor = vert.mSurface->getMaterial()->evaluateSensor(vert.mDg);

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

  return NULL_VERTEX;
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int previous,
           const Scene *scene,
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

  unsigned int result = NULL_VERTEX;
  if(!f.isBlack()) result = insert(previous, scene, after, w, f, pdf, delta, component);

  return result;
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int previous,
           const Scene *scene,
           const bool after,
           const Vector &dir,
           const Spectrum &f,
           float pdf,
           const bool delta,
           const ScatteringDistributionFunction::ComponentIndex component)
{
  unsigned int result = NULL_VERTEX;

  // XXX i kind of think this should be passed as a reference
  const PathVertex &prev = (*this)[previous];

  // cast a Ray
  Ray r(prev.mDg.getPoint(), dir);

  Primitive::Intersection inter;
  if(scene->intersect(r, inter))
  {
    // evaluate the integrand
    // XXX yuck
    const Primitive *prim = inter.getPrimitive();
    const SurfacePrimitive *surface =
      dynamic_cast<const SurfacePrimitive*>(prim);
    const Material *material = surface->getMaterial();
    const DifferentialGeometry &dg = inter.getDifferentialGeometry();

    ScatteringDistributionFunction *emission = material->evaluateEmission(dg);
    ScatteringDistributionFunction *scattering = material->evaluateScattering(dg);
    ScatteringDistributionFunction *sensor = material->evaluateSensor(dg);

    // convert pdf to projected solid angle measure
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

void Path
  ::clear(void)
{
  memset(this, 0, sizeof(Path));
} // end Path::clear()

unsigned int Path
  ::insertRussianRoulette(const unsigned int previous,
                          const Scene *scene,
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

  unsigned int result = NULL_VERTEX;
  if(u3 < rr)
  {
    result = insert(previous, scene, after, w, f, pdf * rr, delta, component);
  } // end if

  return result;
} // end Path::insert()

unsigned int Path
  ::insertRussianRouletteWithTermination(const unsigned int previous,
                                         const Scene *scene,
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

  unsigned int result = NULL_VERTEX;
  if(u3 < rr)
  {
    result = insert(previous, scene, after, w, f, pdf * rr, delta, component);
    if(result == NULL_VERTEX)
    {
      // if we find no intersection, we must reject this sample
      // entirely, otherwise we will overcount shorter length
      // paths, biasing the result
      // XXX have some return value indicating this result
      //     so we can be explicit about it
      (*this)[previous].mThroughput = Spectrum::black();
    } // end if
  } // end if
  else
  {
    // append the termination probability to the previous vertex
    (*this)[previous].mPdf *= (1.0f - rr);
    (*this)[previous].mAccumulatedPdf *= (1.0f - rr);
  } // end else

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
  assert(0);
} // end Path::Path()

Path &Path
  ::operator=(const Path &p)
{
  std::cerr << "Path::Path(): Disallowed assignment operator called!" << std::endl;
  assert(0);
  return *this;
} // end Path::Path()

bool Path
  ::clone(Path &dst, FunctionAllocator &allocator) const
{
  // first do a dumb copy
  memcpy((void*)&dst,
         (const void*)this,
         sizeof(Path));

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

