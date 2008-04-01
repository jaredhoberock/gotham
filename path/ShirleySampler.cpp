/*! \file ShirleySampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShirleySampler class.
 */

#include "ShirleySampler.h"
#include "Path.h"
#include "../geometry/Ray.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"
#include <stratifiedsequence/StratifiedSequence.h>

ShirleySampler
  ::ShirleySampler(const unsigned int maxEyeLength)
    :Parent(maxEyeLength)
{
  std::cerr << "ShirleySampler::ShirleySampler(): I need access to a ShadingContext! Fix me!" << std::endl;
} // end ShirleySampler::ShirleySampler()

void ShirleySampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  gpcpu::uint2 subpathLengths = p.getSubpathLengths();

  Spectrum L;
  const SurfacePrimitiveList *emitters = scene.getEmitters();

  // note that we could start at eyeLength == 1 and connect
  // length-2 paths, which would get rid of the special case of eyeLength == 2
  // below.  however, we wouldn't get to take advantage of the stratified sampling
  // over the image plane, and we would also lose the nice property that this PathSampler's
  // Results always touch the same pixel
  Path temp;
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
      Spectrum sum(Spectrum::black());

      for(SurfacePrimitiveList::const_iterator luminaire = emitters->begin();
          luminaire != emitters->end();
          ++luminaire)
      {
        StratifiedSequence s(0, 1, 0, 1, 2, 2);
        float u0, u1;
        while(s(u0,u1))
        {
          // XXX BUG we need access to random numbers
          if(temp.insert(0, &scene, *mShadingContext, luminaire->get(), true, u0, u1, 0.5f) != Path::INSERT_FAILED)
          {
            PathVertex &light = temp[0];

            // connect a path
            // compute the throughput of the partial path
            Spectrum L = e.mThroughput * light.mThroughput / light.mPdf;
            
            // XXX make compute throughput take the connection vector and geometric term
            L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                                     1, light);
            if(!L.isBlack()
               && !scene.intersect(Ray(e.mDg.getPoint(), light.mDg.getPoint())))
            {
              sum += 0.25f * L;
            } // end if
          } // end if
        } // end while
      } // end for

      if(!sum.isBlack())
      {
        // add a new result
        results.resize(results.size() + 1);
        Result &r= results.back();

        r.mThroughput = sum;

        // set pdf, weight, and (s,t)
        r.mPdf = e.mAccumulatedPdf;
        r.mWeight = 1.0f;
        r.mEyeLength = eyeLength;
        r.mLightLength = 1;
      } // end if
    } // end if
  } // end for eyeLength
} // end ShirleySampler::evaluate()

