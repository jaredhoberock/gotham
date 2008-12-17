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
#include "HaltCriterion.h"

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
  // XXX TODO kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  unsigned int totalPixels = film->getWidth() * film->getHeight();
  const TargetPixelSampleCount *halt = dynamic_cast<const TargetPixelSampleCount*>(mHalt.get());
  if(!halt)
  {
    std::cerr << "EnergyRedistributionRenderer::kernel(): This Renderer requires a HaltCriterion of type TargetPixelSampleCount." << std::endl;
    exit(-1);
  } // end if

  size_t xStrata = halt->getXStrata();
  size_t yStrata = halt->getYStrata();
  size_t totalSamples = (xStrata * yStrata) * totalPixels;

  PathSampler::HyperPoint x, y, z;
  Path xPath, yPath, zPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults, zResults;
  Spectrum f, g;

  // these are the scalar results & reciprocals returned
  // by the Importance object for each Path
  float ix, iy, iz;
  float invIx = 0, invIy = 0, invIz = 0;

  // these are the pdfs & reciprocals of each sample:
  // ix/b, iy/b, iz/b
  float xPdf, yPdf, zPdf;
  float invXPdf = 0, invYPdf = 0, invZPdf = 0;

  // estimate normalization constant
  Path temp;
  float b = mImportance->getNormalizationConstant();
  float invB = 1.0f / b;

  // this pool saves each monte carlo path
  FunctionAllocator saveMonteCarloSample;

  // cline et al, 2005 equation 7
  float ed = b / mMutationsPerSample;

  float edTimesMRecip = 1.0f / (ed*mChainLength);

  // weight each monte carlo sample by the number of samples per pixel
  HilbertSequence walk(0, 1.0f, 0, 1.0f,
                       film->getWidth() * xStrata,
                       film->getHeight() * yStrata);

  // XXX yuck
  KelemenMutator &smallStepper = dynamic_cast<KelemenMutator&>(*mMutator);

  float px, py;
  progress.restart(totalSamples);
  while(walk(px,py, (*mRandomSequence)(), (*mRandomSequence)()))
  {
    // construct a hyperpoint
    PathSampler::constructHyperPoint(*mRandomSequence, x);
    ++mNumMonteCarloSamples;

    // replace the first two coords with
    // pixel samples
    x[0][0] = px;
    x[0][1] = py;

    PathToImage mapToImage;

    // sample a path
    Path temp;
    if(((PathSampler*)smallStepper.getSampler())->constructPath(*mScene, getShadingContext(), x, temp))
    {
      // safely temp to xPath
      saveMonteCarloSample.freeAll();
      temp.clone(xPath, saveMonteCarloSample);

      // purge all memory alloc'd for temp
      mShadingContext->freeAll();

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
            if(smallStepper.smallStep(y,yPath,z,zPath))
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

            ++mNumMutations;

            // calculate accept probability
            // we assume the transition pdf ~ 1 for small steps
            a = std::min<float>(1.0f, iz * invIy);

            if(iy > 0)
            {
              // record y
              float yWeight = (1.0f - a) * invYPdf;
              mRecord->record(yWeight, y, yPath, yResults);

              // add to the acceptance image
              // XXX TODO: generalize this to all samplers somehow
              gpcpu::float2 pixel;
              float yu, yv;
              mapToImage(yResults[0], y, yPath, yu, yv);
              mAcceptanceImage.deposit(yu, yv, Spectrum(1.0f - a, 1.0f - a, 1.0f - a));
            } // end if

            if(iz > 0)
            {
              // record z
              float zWeight = a * invZPdf;
              mRecord->record(zWeight, z, zPath, zResults);

              // add to the acceptance image
              // XXX TODO: generalize this to all samplers somehow
              float zu, zv;
              mapToImage(zResults[0], z, zPath, zu, zv);
              mAcceptanceImage.deposit(zu, zv, Spectrum(a, a, a));
              mProposalImage.deposit(zu, zv, Spectrum::white());
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
            mShadingContext->freeAll();

            ++mNumSamples;
          } // end for j
        } // end for i
      } // end if
    } // end if

    ++progress;
  } // end while

  // purge the local store
  mLocalPool.freeAll();
} // end EnergyRedistributionRenderer::kernel()

void EnergyRedistributionRenderer
  ::postRenderReport(const double elapsed) const
{
  Parent::postRenderReport(elapsed);

  std::cout << "Monte Carlo samples: " << mNumMonteCarloSamples << std::endl;
  std::cout << "Mutations: " << mNumMutations << std::endl;
  std::cout << "mutation/MC ratio: " << static_cast<float>(mNumMutations) / mNumMonteCarloSamples << std::endl;
} // end EnergyRedistributionRenderer::postRenderReport()

