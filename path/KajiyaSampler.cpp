/*! \file KajiyaSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of KajiyaSampler class.
 */

#include "KajiyaSampler.h"
#include "Path.h"
#include "../geometry/Ray.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../shading/Material.h"
#include "../primitives/SurfacePrimitive.h"

KajiyaSampler
  ::KajiyaSampler(const unsigned int maxEyeLength)
{
  mMaxEyeLength = maxEyeLength;
  if(mMaxEyeLength > Path::static_size - 1)
  {
    std::cerr << "Warning, clamping maximum eye subpath length to " << Path::static_size - 1 << "." << std::endl;
    mMaxEyeLength = Path::static_size - 1;
  } // end if

  if(mMaxEyeLength < 1)
  {
    std::cerr << "Warning, clamping maximum eye subpath length to 1." << std::endl;
    mMaxEyeLength = 1;
  } // end if
} // end KajiyaSampler::KajiyaSampler()

void KajiyaSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  gpcpu::uint2 subpathLengths = p.getSubpathLengths();

  const PathVertex &light = p[subpathLengths.sum() - 1];
  Spectrum L;

  // note that we could start at eyeLength == 1 and connect
  // length-2 paths, which would get rid of the special case of eyeLength == 2
  // below.  however, we wouldn't get to take advantage of the stratified sampling
  // over the image plane, and we would also lose the nice property that this PathSampler's
  // Results always touch the same pixel
  for(size_t eyeLength = 2;
      eyeLength <= subpathLengths[0];
      ++eyeLength)
  {
    const PathVertex &e = p[eyeLength - 1];

    if(eyeLength == 2 || e.mFromDelta)
    {
      // evaluate the material's emission
      L = e.mThroughput * e.mEmission->evaluate(e.mToPrev, e.mDg);
      if(L != Spectrum::black())
      {
        // add a new result
        results.resize(results.size() + 1);
        Result &r = results.back();
        r.mThroughput = L;
        r.mPdf = e.mAccumulatedPdf;
        r.mWeight = 1.0f;
        r.mEyeLength = eyeLength;
        r.mLightLength = 0;
      } // end if
    } // end if

    if(!e.mScattering->isSpecular())
    {
      // connect a path
      // compute the throughput of the partial path
      L = e.mThroughput * light.mThroughput;
      
      // XXX make compute throughput take the connection vector and geometric term
      L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                               1, light);

      if(!L.isBlack()
         && !scene.intersect(Ray(e.mDg.getPoint(), light.mDg.getPoint())))
      {
        // add a new result
        results.resize(results.size() + 1);
        Result &r= results.back();

        // multiply by the connection throughput
        r.mThroughput = L;

        // set pdf, weight, and (s,t)
        r.mPdf = e.mAccumulatedPdf * light.mAccumulatedPdf;
        r.mWeight = 1.0f;
        r.mEyeLength = eyeLength;
        r.mLightLength = 1;
      } // end if
    } // end if
  } // end for eyeLength
} // end KajiyaSampler::evaluate()

bool KajiyaSampler
  ::constructPath(const Scene &scene,
                  const HyperPoint &x,
                  Path &p)
{
  // insert a lens vertex
  // reserve the 0th coordinate to choose
  // the film plane
  // XXX remove the need for this
  unsigned int lastPosition = p.insert(0, scene.getSensors(), false, x[1][0], x[1][1], x[1][2], x[1][3]);

  if(lastPosition == Path::INSERT_FAILED) return false;

  // insert vertices until we miss one or we run out of slots
  float u0 = x[0][0];
  float u1 = x[0][1];
  float u2 = x[0][2];
  size_t coord = 2;
  while((p.insert(lastPosition, &scene, true, lastPosition != 0, 
                  u0, u1, u2))
        < mMaxEyeLength - 1)
  {
    u0 = x[coord][0];
    u1 = x[coord][1];
    u2 = x[coord][2];
    ++lastPosition;
    ++coord;
  } // end while

  // insert a light vertex at the position just beyond the
  // last eye vertex
  // use the final coordinate to choose the light vertex
  const HyperPoint::value_type &c = x[x.size()-1];
  lastPosition = p.insert(p.getSubpathLengths()[0], scene.getEmitters(), true,
                          c[0], c[1], c[2], c[3]);

  return lastPosition != Path::INSERT_FAILED;
} // end KajiyaSampler::constructPath()

