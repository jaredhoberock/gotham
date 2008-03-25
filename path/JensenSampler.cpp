/*! \file JensenSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of JensenSampler class.
 */

#include "JensenSampler.h"
#include <stratifiedsequence/StratifiedSequence.h>
#include <hilbertsequence/HilbertSequence.h>
#include "../geometry/Ray.h"
#include "../primitives/SurfacePrimitive.h"

void JensenSampler
  ::setGlobalMap(const boost::shared_ptr<PhotonMap> &pm)
{
  mGlobalMap = pm;
} // end JensenSampler::setGlobalMap()

void JensenSampler
  ::setCausticMap(const boost::shared_ptr<PhotonMap> &pm)
{
  mCausticMap = pm;
} // end JensenSampler::setCausticMap()

void JensenSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  // evaluate direct lighting first
  //Parent::evaluate(scene,p,results);

  if(mGlobalMap.get() != 0)
  {
    evaluateIndirect(scene,p,results);
  } // end if
} // end JensenSampler::evaluate()

void JensenSampler
  ::evaluateIndirect(const Scene &scene,
                     const Path &p,
                     std::vector<Result> &results) const
{
  // don't sample the hemisphere around the sensor
  if(p.getSubpathLengths()[0] < 2) return;

  // get the last PathVertex
  const PathVertex &v = p[p.getSubpathLengths()[0] - 1];

  float maxDist2 = std::numeric_limits<float>::infinity();
  Intersection inter;
  Spectrum L(Spectrum::black());
  Vector wi;
  Spectrum f;
  float pdf;
  bool delta;
  ScatteringDistributionFunction::ComponentIndex component = 0;

  mGather.clear();
  mGlobalMap->rangeQuery(v.mDg.getPoint(),
                         maxDist2, mGather);
  L = estimateRadiance(v.mToPrev, v.mDg,
                       v.mScattering, maxDist2, mGather);

  //float u0, u1;
  //mStratifiedSequence.reset();
  //while(mStratifiedSequence(u0, u1))
  //{
  //  // generate a ray direction
  //  // XXX randomize or stratify u2
  //  f = v.mScattering->sample(v.mToPrev,
  //                            v.mDg,
  //                            u0, u1, 0.5f,
  //                            wi, pdf, delta, component);

  //  // intersect the Scene
  //  Ray ray(v.mDg.getPoint(), wi);
  //  if(scene.intersect(ray, inter))
  //  {
  //    // query photon map at intersection
  //    maxDist2 = std::numeric_limits<float>::infinity();

  //    // XXX this could potentially deallocate the memory
  //    //     allocated in the reserve() below.
  //    mGather.clear();
  //    mGlobalMap->rangeQuery(inter.getDifferentialGeometry().getPoint(),
  //                           maxDist2, mGather);

  //    const Primitive *prim = inter.getPrimitive();

  //    // XXX we should really use our own allocator here
  //    //     we could run out of room by using too many strata
  //    ScatteringDistributionFunction *bsdf = static_cast<const SurfacePrimitive*>(prim)->getMaterial()->evaluateScattering(inter.getDifferentialGeometry());

  //    // convert pdf to projected solid angle measure
  //    pdf /= v.mDg.getNormal().absDot(wi);

  //    // evaluate radiance 
  //    L += f * estimateRadiance(-wi, inter.getDifferentialGeometry(),
  //                              bsdf, maxDist2, mGather) / pdf;
  //  } // end if
  //} // end while

  //// divide by N
  //L *= mInvNumStrata;

  // add a result
  results.resize(results.size() + 1);
  Result &r = results.back();
  r.mThroughput = v.mThroughput * L;
  r.mPdf = v.mAccumulatedPdf;
  r.mWeight = 1.0f;
  r.mEyeLength = p.getSubpathLengths()[0];
  r.mLightLength = 0;
} // end JensenSampler::evaluateIndirect()

Spectrum JensenSampler
  ::estimateRadiance(const Vector &wo,
                     const DifferentialGeometry &dg,
                     const ScatteringDistributionFunction *f,
                     const float maxDist2,
                     const PhotonGatherer &gather) const
{
  Spectrum result(0,0,0);

  // estimate density
  // constant kernel for now
  for(PhotonGatherer::const_iterator p = gather.begin();
      p != gather.end();
      ++p)
  {
    result += f->evaluate(wo, dg, p->second->mWi) * dg.getNormal().absDot(p->second->mWi) * p->second->mPower;
  } // end for i

  //if(gather.size() > 0)
  //{
  //  result /= gather.size();
  //} // end if
  result /= (PI * maxDist2);

  return result;
} // end JensenSampler::estimateRadiance()

void JensenSampler
  ::PhotonGatherer
    ::operator()(const Photon &p,
                 const float d2,
                 float &maxDist2)
{
  ++mPhotonsVisited;

  if(size() < capacity())
  {
    push_back(std::make_pair(d2, &p));

    if(size() == capacity())
    {
      // heapify
      std::make_heap(begin(), end());
      maxDist2 = front().first;
    } // end if
  } // end if
  else
  {
    std::pop_heap(begin(), end());
    back() = std::make_pair(d2, &p);
    std::push_heap(begin(), end());

    // update max distance squared
    maxDist2 = front().first;
  } // end else
} // end PhotonGatherer::operator()()

JensenSampler::PhotonKernel
  ::PhotonKernel(const Point &p, const float &maxDist2)
    :mX(p),mMaxDist2(maxDist2),mInvMaxDist2(1.0f/mMaxDist2)
{
  ;
} // end PhotonKernel::PhotonKernel()

JensenSampler::ConstantKernel
  ::ConstantKernel(const Point &p, const float &maxDist2)
    :Parent(p,maxDist2)
{
  ;
} // end ConstantKernel::ConstantKernel()

float JensenSampler::ConstantKernel
  ::operator()(const Photon &p, const float d2) const
{
  return INV_PI * mInvMaxDist2;
} // end ConstantKernel::operator()()

void JensenSampler
  ::setFinalGatherStrata(const size_t x, const size_t y)
{
  mFinalGatherX = x;
  mFinalGatherY = y;
  mInvNumStrata = 1.0f / (mFinalGatherX * mFinalGatherY);
  mStratifiedSequence.reset(0.0f, 1.0f, 0.0f, 1.0f,
                            mFinalGatherX, mFinalGatherY);
} // end JensenSampler::setFinalGatherStrata()

void JensenSampler
  ::setFinalGatherPhotons(const size_t n)
{
  mGather.reserve(n);
} // end JensenSampler::setFinalGatherPhotons()

