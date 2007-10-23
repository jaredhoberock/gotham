/*! \file EnergyRedistributionRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of EnergyRedistributionRenderer class.
 */

#include "EnergyRedistributionRenderer.h"
#include "../path/KelemenSampler.h"
#include "../path/KajiyaSampler.h"
#include <aliastable/AliasTable.h>
#include "../path/Path.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../mutators/KelemenMutator.h"
#include <hilbertsequence/HilbertSequence.h>
#include <bittricks/bittricks.h>

#include <gpcpu/Vector.h>
#include <boost/progress.hpp>

using namespace boost;
using namespace gpcpu;

EnergyRedistributionRenderer
  ::EnergyRedistributionRenderer(const float mutationsPerSample,
                           const unsigned int chainLength,
                           const boost::shared_ptr<RandomSequence> &s,
                           const boost::shared_ptr<PathMutator> &mutator,
                           const boost::shared_ptr<ScalarImportance> &importance)
    :Parent(s,mutator,importance),
     mMutationsPerSample(mutationsPerSample),
     mChainLength(chainLength)
{
  ;
} // end EnergyRedistributionRenderer::EnergyRedistributionRenderer()

void EnergyRedistributionRenderer
  ::kernel(ProgressCallback &progress)
{
  unsigned int totalPixels = mFilm->getWidth() * mFilm->getHeight();
  unsigned int totalSamples = (mSamplesPerPixel * mSamplesPerPixel) * totalPixels;
  float oneOverN = 1.0f / totalSamples;

  PathSampler::HyperPoint x, y, z;
  Path xPath, yPath, zPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults, zResults;
  Spectrum f, g;

  // these are the scalar results & reciprocals returned
  // by the Importance object for each Path
  float ix, iy, iz;
  float invIx, invIy, invIz;

  // these are the pdfs & reciprocals of each sample:
  // ix/b, iy/b, iz/b
  float xPdf, yPdf, zPdf;
  float invXPdf, invYPdf, invZPdf;

  // estimate normalization constant
  Path temp;
  float b = mImportance->getNormalizationConstant();
  float invB = 1.0f / b;

  // this pool saves each monte carlo path
  FunctionAllocator saveMonteCarloSample;

  // cline et al, 2005 equation 7
  float ed = b / mMutationsPerSample;

  float edTimesMRecip = 1.0f / (ed*mChainLength);

  // reciprocal of mutations per pixel
  float invMpp = 1.0 / (mMutationsPerSample * mSamplesPerPixel * mSamplesPerPixel);

  // weight each monte carlo sample by the number of samples per pixel
  HilbertSequence walk(0, 1.0f, 0, 1.0f,
                       mFilm->getWidth() * mSamplesPerPixel,
                       mFilm->getHeight() * mSamplesPerPixel);

  // XXX yuck
  KelemenMutator &smallStepper = dynamic_cast<KelemenMutator&>(*mMutator);

  float px, py;
  progress.restart(totalSamples);
  while(walk(px,py, (*mRandomSequence)(), (*mRandomSequence)()))
  {
    // construct a hyperpoint
    PathSampler::constructHyperPoint(*mRandomSequence, x);

    // replace the first two coords with
    // pixel samples
    x[0][0] = px;
    x[0][1] = py;

    // sample a path
    Path temp;
    if(((PathSampler*)smallStepper.getSampler())->constructPath(*mScene, x, temp))
    {
      // safely temp to xPath
      saveMonteCarloSample.freeAll();
      temp.clone(xPath, saveMonteCarloSample);

      // purge all memory alloc'd for temp
      ScatteringDistributionFunction::mPool.freeAll();

      // get a monte carlo estimate
      xResults.clear();
      Spectrum e = smallStepper.evaluate(xPath, xResults);
      ix = (*mImportance)(x, xPath, xResults);
      invIx = 1.0f / ix;
      xPdf = ix * invB;
      invXPdf = b * invIx;

      float a;
      if(ix > 0)
      {
        unsigned int numChains = ifloor((*mRandomSequence)() + ix * edTimesMRecip);
        for(size_t i = 0; i < numChains; ++i)
        {
          // copy the monte carlo sample to y into the local pool
          copyPath(yPath, xPath);
          y = x;
          iy = ix;
          invIy = invIx;
          yPdf = xPdf;
          invYPdf = invXPdf;
          yResults = xResults;
          f = e;

          for(size_t j = 0; j < mChainLength; ++j)
          {
            // mutate & evaluate
            int whichMutation = smallStepper.smallStep(y,yPath,z,zPath);
            if(whichMutation != -1)
            {
              zResults.clear();
              g = mMutator->evaluate(zPath, zResults);

              // compute importance
              iz = (*mImportance)(z, zPath, zResults);
              invIz = 1.0f / iz;
              zPdf = iz * invB;
              invZPdf = b * invIz;
            } // end if
            else
            {
              iz = 0;
            } // end else

            // calculate accept probability
            // we assume the transition pdf ~ 1 for small steps
            a = std::min<float>(1.0f, iz * invIy);

            if(iy > 0)
            {
              // record y
              float yWeight = invMpp * (1.0f - a) * invYPdf;
              for(ResultList::const_iterator r = yResults.begin();
                  r != yResults.end();
                  ++r)
              {
                Spectrum deposit;
                float2 pixel;
                if(r->mEyeLength == 1)
                {
                  unsigned int endOfLightPath = yPath.getSubpathLengths().sum() - r->mLightLength;
                  ::Vector w = yPath[endOfLightPath].mDg.getPoint();
                  w -= yPath[0].mDg.getPoint();
                  yPath[0].mSensor->invert(w, yPath[0].mDg,
                                           pixel[0], pixel[1]);
                } // end if
                else
                {
                  yPath[0].mSensor->invert(yPath[0].mToNext,
                                           yPath[0].mDg,
                                           pixel[0], pixel[1]);
                } // end else

                // each sample contributes (1/spp) * MIS weight * MC weight * f / pdf
                deposit = yWeight * r->mWeight * r->mThroughput / r->mPdf;
                mFilm->deposit(pixel[0], pixel[1], deposit);
              } // end for r
            } // end if

            if(iz > 0)
            {
              // record z
              float zWeight = invMpp * a * invZPdf;
              for(ResultList::const_iterator rz = zResults.begin();
                  rz != zResults.end();
                  ++rz)
              {
                Spectrum deposit;
                float2 pixel;
                if(rz->mEyeLength == 1)
                {
                  unsigned int endOfLightPath = zPath.getSubpathLengths().sum() - rz->mLightLength;
                  ::Vector w = zPath[endOfLightPath].mDg.getPoint();
                  w -= zPath[0].mDg.getPoint();
                  zPath[0].mSensor->invert(w, zPath[0].mDg,
                                           pixel[0], pixel[1]);
                } // end if
                else
                {
                  zPath[0].mSensor->invert(zPath[0].mToNext,
                                           zPath[0].mDg,
                                           pixel[0], pixel[1]);
                } // end else

                // each sample contributes (1/spp) * MIS weight * MC weight * f
                deposit = zWeight * rz->mWeight * rz->mThroughput / rz->mPdf;
                mFilm->deposit(pixel[0], pixel[1], deposit);
              } // end for r
            } // end if

            // accept?
            if((*mRandomSequence)() < a)
            {
              ++mNumAccepted;
              y = z;

              // safely copy the path
              copyPath(yPath, zPath);

              f = g;
              iy = iz;
              invIy = invIz;
              yPdf = zPdf;
              invYPdf = invZPdf;
              yResults = zResults;
            } // end if

            // purge all malloc'd memory for this sample
            ScatteringDistributionFunction::mPool.freeAll();
          } // end for j
        } // end for i
      } // end if
    } // end if

    ++progress;
  } // end while

  // purge the local store
  mLocalPool.freeAll();
} // end EnergyRedistributionRenderer::kernel()

