/*! \file CudaDebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaDebugRenderer class.
 */

#include "CudaDebugRenderer.h"
#include "../../geometry/Ray.h"
#include "../../records/RenderFilm.h"
#include "../primitives/CudaIntersection.h"
#include "../geometry/CudaDifferentialGeometryVector.h"
#include <vector_functions.h>
#include <stdcuda/cuda_algorithm.h>
#include "../shading/CudaShadingContext.h"
#include "../shading/CudaScatteringDistributionFunction.h"
#include "../primitives/CudaSurfacePrimitiveList.h"
#include "cudaDebugRendererUtil.h"
#include <stdcuda/fill.h>

using namespace stdcuda;

void CudaDebugRenderer
  ::generateHyperPoints(stdcuda::vector_dev<float4> &u0,
                        stdcuda::vector_dev<float4> &u1,
                        const size_t n) const
{
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  gpcpu::size2 imageDim(film->getWidth(),film->getHeight());

  std::vector<float4> tempU0(n), tempU1(n);
  for(size_t i = 0; i != n; ++i)
  {
    // convert thread index to pixel index
    gpcpu::size2 pixelIndex(i % imageDim[0],
                            i / imageDim[0]);

    // convert pixel index to location in unit square
    gpcpu::float2 uv(static_cast<float>(pixelIndex[0]) / imageDim[0],
                     static_cast<float>(pixelIndex[1]) / imageDim[1]); 

    tempU0[i] = make_float4(0,0,0,0);
    tempU1[i] = make_float4(uv[0], uv[1], 0, 0);
  } // end for i

  u0.resize(n);
  u1.resize(n);

  stdcuda::copy(tempU0.begin(), tempU0.end(), u0.begin());
  stdcuda::copy(tempU1.begin(), tempU1.end(), u1.begin());
} // end CudaDebugRenderer::generateHyperPoints()

void CudaDebugRenderer
  ::sampleEyeRays(const device_ptr<const float4> &u0,
                  const device_ptr<const float4> &u1,
                  const device_ptr<float3> &origins,
                  const device_ptr<float3> &directions,
                  const device_ptr<float2> &intervals,
                  const device_ptr<float> &pdfs,
                  const size_t n) const
{
  // get the sensors
  const CudaSurfacePrimitiveList &sensors = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getSensors());

  vector_dev<PrimitiveHandle> prims(n);
  CudaDifferentialGeometryVector dg(n);

  // sample eye points
  sensors.sampleSurfaceArea(u0, &prims[0], dg, pdfs, n);

  // get a list of MaterialHandles
  const CudaSurfacePrimitiveList &primitives = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getPrimitives());
  vector_dev<MaterialHandle> materials(n);
  primitives.getMaterialHandles(device_ptr<const PrimitiveHandle>(&prims[0]),
                                &materials[0],
                                n);

  // evaluate sensor shader
  // allocate space for sensor functions
  vector_dev<CudaScatteringDistributionFunction> sensorFunctions(n);

  // evaluate sensor shader
  CudaShadingContext *context = dynamic_cast<CudaShadingContext*>(mShadingContext.get());
  context->evaluateSensor(&materials[0],
                          dg,
                          &sensorFunctions[0],
                          n);

  // reinterpret u1 into a float3 *
  const float4 *temp2 = u1;
  device_ptr<const float3> u1AsAFloat3(reinterpret_cast<const float3*>(temp2));

  // sample the scattering functions
  vector_dev<bool> delta(n);
  vector_dev<float3> s(n);
  context->sampleUnidirectionalScattering(&sensorFunctions[0],
                                          dg,
                                          u1AsAFloat3,
                                          sizeof(float4),
                                          &s[0],
                                          &directions[0],
                                          sizeof(float3),
                                          &pdfs[0],
                                          &delta[0],
                                          n);

  // copy from differential geometry into originsAndMinT
  stdcuda::copy(dg.mPoints, dg.mPoints + n, origins);

  // set min t to epsilon
  // set max t to infinity
  stdcuda::fill(intervals, intervals + n, make_float2(0.0005f, std::numeric_limits<float>::infinity()));
} // end CudaDebugRenderer::sampleEyeRay()

void CudaDebugRenderer
  ::shade(const device_ptr<const float3> &directions,
          const CudaDifferentialGeometryArray &dg,
          const device_ptr<const PrimitiveHandle> &hitPrims,
          const device_ptr<const bool> &stencil,
          const device_ptr<float3> &results,
          const size_t n) const
{
  CudaShadingContext *context = dynamic_cast<CudaShadingContext*>(mShadingContext.get());

  // get a list of MaterialHandles
  const CudaSurfacePrimitiveList &primitives = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getPrimitives());
  vector_dev<MaterialHandle> materials(n);
  primitives.getMaterialHandles(hitPrims,
                                sizeof(PrimitiveHandle),
                                &stencil[0],
                                &materials[0],
                                n);

  // allocate space for scattering functions
  vector_dev<CudaScatteringDistributionFunction> cf(n);

  // evaluate scattering shader
  context->evaluateScattering(&materials[0],
                              dg,
                              &stencil[0],
                              &cf[0],
                              n);

  // allocate space for emission functions
  vector_dev<CudaScatteringDistributionFunction> ce(n);

  // evaluate emission shader
  context->evaluateEmission(&materials[0],
                            dg,
                            &stencil[0],
                            &ce[0],
                            n);

  // XXX kill these temps
  vector_dev<float3> wo(n);
  flipVectors(&directions[0], &wo[0], n);

  // evaluate scattering bsdf
  vector_dev<float3> scattering(n);
  context->evaluateBidirectionalScattering(&cf[0],
                                           &wo[0],
                                           dg,
                                           &wo[0],
                                           &stencil[0],
                                           &scattering[0],
                                           n);

  // evaluate emission bsdf
  vector_dev<float3> emission(n);
  context->evaluateUnidirectionalScattering(&ce[0],
                                            &wo[0],
                                            dg,
                                            &stencil[0],
                                            &emission[0],
                                            n);

  // sum into results
  sumScatteringAndEmission(&scattering[0],
                           &emission[0],
                           &stencil[0],
                           &results[0],
                           n);
} // end CudaDebugRenderer::shade()

void CudaDebugRenderer
  ::deposit(const size_t batchIdx,
            const size_t threadIdx,
            const Spectrum *results)
{
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  gpcpu::size2 imageDim(film->getWidth(),film->getHeight());
  size_t i = batchIdx * mWorkBatchSize + threadIdx;

  // convert thread index to pixel index
  gpcpu::size2 pixelIndex(i % imageDim[0],
                          i / imageDim[0]);
  film->deposit(pixelIndex[0],pixelIndex[1], results[i]);
} // end CudaDebugRenderer::deposit()

void CudaDebugRenderer
  ::kernel(ProgressCallback &progress)
{
  // XXX TODO: kill this
  // compute the total work
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size_t totalWork = film->getWidth() * film->getHeight();

  // compute the total number of batches we need to complete the work
  size_t numBatches = totalWork / mWorkBatchSize;

  // allocate per-thread data
  stdcuda::vector_dev<float3> origins(totalWork);
  stdcuda::vector_dev<float3> directions(totalWork);
  stdcuda::vector_dev<float2> intervals(totalWork);
  stdcuda::vector_dev<float> pdfsDevice(totalWork);
  stdcuda::vector_dev<float4> u0(totalWork);
  stdcuda::vector_dev<float4> u1(totalWork);
  std::vector<float> pdfs(totalWork);
  stdcuda::vector_dev<CudaIntersection> intersectionsDevice(totalWork);
  CudaDifferentialGeometryVector dg(totalWork);
  stdcuda::vector_dev<PrimitiveHandle> hitPrims(totalWork);

  // init stencil to 1
  stdcuda::vector_dev<bool> stencilDevice(totalWork);
  for(size_t i = 0; i != totalWork; ++i)
  {
    stencilDevice[i] = true;
  } // end for i

  // init results to black
  std::vector<float3> resultsHost(totalWork, make_float3(0,0,0));
  stdcuda::vector_dev<float3> results(totalWork);
  stdcuda::copy(resultsHost.begin(), resultsHost.end(), results.begin());

  // sampleEyeRay + intersect + shade + deposit = 4
  progress.restart(totalWork * 4);

  generateHyperPoints(u0,u1,totalWork);

  for(size_t i = 0; i != numBatches; ++i)
  {
    sampleEyeRays(&u0[i * mWorkBatchSize],
                  &u1[i * mWorkBatchSize],
                  &origins[i * mWorkBatchSize],
                  &directions[i * mWorkBatchSize],
                  &intervals[i * mWorkBatchSize],
                  &pdfsDevice[i * mWorkBatchSize],
                  mWorkBatchSize);

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i
  sampleEyeRays(&u0[numBatches * mWorkBatchSize],
                &u1[numBatches * mWorkBatchSize],
                &origins[numBatches * mWorkBatchSize],
                &directions[numBatches * mWorkBatchSize],
                &intervals[numBatches * mWorkBatchSize],
                &pdfsDevice[numBatches * mWorkBatchSize],
                totalWork % mWorkBatchSize);

  // purge all malloc'd memory for the last partial batch
  mShadingContext->freeAll();
  progress += totalWork % mWorkBatchSize;

  // intersect en masse
  intersect(&origins[0], &directions[0], &intervals[0], dg, &hitPrims[0], &stencilDevice[0], totalWork);
  progress += totalWork;

  // shade
  for(size_t i = 0; i != numBatches; ++i)
  {
    shade(&directions[i * mWorkBatchSize],
          dg + i * mWorkBatchSize,
          &hitPrims[i * mWorkBatchSize],
          &stencilDevice[i * mWorkBatchSize],
          &results[i * mWorkBatchSize],
          mWorkBatchSize);

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i

  // shade the last partial batch
  shade(&directions[numBatches * mWorkBatchSize],
        dg + numBatches * mWorkBatchSize,
        &hitPrims[numBatches * mWorkBatchSize],
        &stencilDevice[numBatches * mWorkBatchSize],
        &results[numBatches * mWorkBatchSize],
        totalWork % mWorkBatchSize);
  progress += totalWork % mWorkBatchSize;

  // purge all malloc'd memory for the last partial batch
  mShadingContext->freeAll();

  // copy results to host
  stdcuda::copy(results.begin(), results.end(), &resultsHost[0]);

  for(size_t i = 0; i != numBatches; ++i)
  {
    for(size_t j = 0; j != mWorkBatchSize; ++j)
    {
      deposit(i,j,(Spectrum*)&resultsHost[0]);
    } // end for j

    progress += mWorkBatchSize;
  } // end for i
  for(size_t j = 0; j != totalWork % mWorkBatchSize; ++j)
    deposit(numBatches,j,(Spectrum*)&resultsHost[0]);
  progress += totalWork % mWorkBatchSize;
} // end CudaDebugRenderer::kernel()

void CudaDebugRenderer
  ::intersect(const device_ptr<const float3> &origins,
              const device_ptr<const float3> &directions,
              const device_ptr<const float2> &intervals,
              CudaDifferentialGeometryArray &dg,
              const device_ptr<PrimitiveHandle> &hitPrims,
              const device_ptr<bool> &stencil,
              const size_t n)
{
  const CudaScene *scene = static_cast<const CudaScene*>(mScene.get());

  // compute the total number of batches we need to complete the work
  size_t numBatches = n / mWorkBatchSize;

  device_ptr<const float3> o = origins;
  device_ptr<const float3> d = directions;
  device_ptr<const float2> i = intervals;
  CudaDifferentialGeometryArray dgPtr = dg;
  device_ptr<PrimitiveHandle> hitPrimsPtr = hitPrims;
  device_ptr<bool> s = stencil;
  device_ptr<const float3> end = o + numBatches * mWorkBatchSize;
  for(;
      o != end;
      o += mWorkBatchSize,
      d += mWorkBatchSize,
      i += mWorkBatchSize,
      dgPtr += mWorkBatchSize,
      hitPrimsPtr += mWorkBatchSize,
      s += mWorkBatchSize)
  {
    scene->intersect(o, d, i, dgPtr, hitPrimsPtr, s, mWorkBatchSize);
  } // end for i

  // do the last partial batch
  scene->intersect(o, d, i, dgPtr, hitPrimsPtr, s, n % mWorkBatchSize);
} // end CudaDebugRenderer::intersect()

