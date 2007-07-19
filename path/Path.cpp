/*! \file Path.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Path class.
 */

#include "Path.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../primitives/Scene.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../geometry/Ray.h"

void Path
  ::insert(const unsigned int i,
           const SurfacePrimitiveList *surfaces,
           const bool emission,
           const float u0,
           const float u1,
           const float u2,
           const float u3)
{
  PathVertex &vert = emission ? back() : front();

  // sample a point from the list
  surfaces->sampleSurfaceArea(u0, u1, u2, u3,
                              &vert.mSurface,
                              vert.mDg,
                              vert.mPdf);

  // initialize the integrand
  vert.mIntegrand = emission ?
    vert.mSurface->getMaterial()->evaluateEmission(vert.mDg) :
    vert.mSurface->getMaterial()->evaluateSensor(vert.mDg);

  // initialize the throughput as white
  vert.mThroughput = Spectrum::white();

  // init the accumulated pdf
  vert.mAccumulatedPdf = vert.mPdf;

  // the subpath now has a length of 1
  mSubpathLengths[emission] = 1;
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
  if(scatter)
  {
    // scatter
    f = prev.mIntegrand->sample(after ? prev.mToPrev : prev.mToNext,
                                prev.mDg, u0, u1, u2,
                                w, pdf);
  } // end if
  else
  {
    // emit/sense
    f = prev.mIntegrand->sample(prev.mDg, u0, u1, u2, w, pdf);
  } // end else

  return insert(previous, scene, after, w, f, pdf);
} // end Path::insert()

unsigned int Path
  ::insert(const unsigned int previous,
           const Scene *scene,
           const bool after,
           const Vector &dir,
           const Spectrum &f,
           float pdf)
{
  unsigned int result = NULL_VERTEX;

  // XXX i kind of think this should be passed as a reference
  const PathVertex &prev = (*this)[previous];

  // cast a Ray
  Ray r(prev.mDg.getPoint(), dir);
  Primitive::Intersection inter;
  if(scene->intersect(r, inter))
  {
    // convert pdf to projected solid angle measure
    pdf /= prev.mDg.getNormal().absDot(dir);
    result = insert(previous, after, dir, f, pdf,
                    inter.getDifferentialGeometry());
  } // end if

  return result;
} // end Path::insert()

// XXX this probably belonds somewhere else
static float computeG(const Normal &n0,
                      const Vector &w,
                      const Normal &n1,
                      const float d2)
{
  return n0.absDot(w) * n1.absDot(w) / d2;
} // end computeG()

unsigned int Path
  ::insert(const unsigned int previous,
           const bool after,
           const Vector &dir,
           const Spectrum &f,
           const float pdf,
           const DifferentialGeometry &dg)
{
  // where to insert?
  unsigned int result = after ? (previous+1) : (previous-1);

  PathVertex &v = (*this)[result];

  // init the vertex
  // XXX I kind of think prev should be passed as a reference
  PathVertex &prev = (*this)[previous];
  v.init(dg, f * prev.mThroughput, prev.mAccumulatedPdf * pdf, pdf);

  // increment the subpath length
  mSubpathLengths[!after]++;

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
  ::init(const DifferentialGeometry &dg,
         const Spectrum &f,
         const float accumulatedPdf,
         const float pdf)
{
  mDg = dg;
  mThroughput = f;
  mAccumulatedPdf = accumulatedPdf;
  mPdf = pdf;
} // end PathVertex::init()

