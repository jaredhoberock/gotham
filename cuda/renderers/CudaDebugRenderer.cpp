/*! \file CudaDebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaDebugRenderer class.
 */

#include "CudaDebugRenderer.h"
#include "../../geometry/Ray.h"
#include "../../records/RenderFilm.h"
#include "../primitives/CudaIntersection.h"
#include <vector_functions.h>
#include <stdcuda/cuda_algorithm.h>
#include "../shading/CudaShadingContext.h"
#include "../shading/CudaScatteringDistributionFunction.h"
#include "../primitives/CudaSurfacePrimitiveList.h"
#include "cudaDebugRendererUtil.h"

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
                  const device_ptr<float4> &originsAndMinT,
                  const device_ptr<float4> &directionsAndMaxT,
                  const device_ptr<float> &pdfs,
                  const size_t n) const
{
  // get the sensors
  const CudaSurfacePrimitiveList &sensors = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getSensors());

  vector_dev<PrimitiveHandle> prims(n);
  vector_dev<CudaDifferentialGeometry> dg(n);

  // sample eye points
  sensors.sampleSurfaceArea(u0, &prims[0], &dg[0], pdfs, n);

  // get a list of MaterialHandles
  const CudaSurfacePrimitiveList &primitives = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getPrimitives());
  vector_dev<MaterialHandle> materials(n);
  primitives.getMaterialHandles(&prims[0],
                                &materials[0],
                                n);

  // evaluate sensor shader
  // allocate space for sensor functions
  vector_dev<CudaScatteringDistributionFunction> sensorFunctions(n);

  // evaluate sensor shader
  CudaShadingContext *context = dynamic_cast<CudaShadingContext*>(mShadingContext.get());
  context->evaluateSensor(&materials[0],
                          &dg[0],
                          sizeof(CudaDifferentialGeometry),
                          &sensorFunctions[0],
                          n);

  // reinterpret directionsAndMaxT into a float3 *
  float4 *temp = directionsAndMaxT;
  device_ptr<float3> wo(reinterpret_cast<float3*>(temp));

  // reinterpret u1 into a float3 *
  const float4 *temp2 = u1;
  device_ptr<const float3> u1AsAFloat3(reinterpret_cast<const float3*>(temp2));

  // sample the scattering functions
  vector_dev<bool> delta(n);
  vector_dev<float3> s(n);
  context->sampleUnidirectionalScattering(&sensorFunctions[0],
                                          &dg[0],
                                          sizeof(CudaDifferentialGeometry),
                                          u1AsAFloat3,
                                          sizeof(float4),
                                          &s[0],
                                          sizeof(float3),
                                          wo,
                                          sizeof(float4),
                                          &pdfs[0],
                                          sizeof(float),
                                          &delta[0],
                                          sizeof(bool),
                                          n);

  // copy from differential geometry into originsAndMinT
  // set min t to epsilon
  // set max t to infinity
  sampleEyeRaysEpilogue(&dg[0], &originsAndMinT[0], &directionsAndMaxT[0], n);
} // end CudaDebugRenderer::sampleEyeRay()

void CudaDebugRenderer
  ::shade(device_ptr<const float4> directionsAndMaxT,
          device_ptr<const CudaIntersection> intersectionsDevice,
          device_ptr<const int> stencilDevice,
          device_ptr<float3> results,
          const size_t n) const
{
  CudaShadingContext *context = dynamic_cast<CudaShadingContext*>(mShadingContext.get());

  // get a pointer to the first CudaDifferentialGeometry
  // in the first CudaIntersection in the list
  const void *temp = intersectionsDevice;
  device_ptr<const CudaDifferentialGeometry> firstDg(reinterpret_cast<const CudaDifferentialGeometry*>(temp));

  // get a pointer to the first PrimitiveHandle
  // in the first CudaIntersection in the list
  // the PrimitiveHandle immediately follows the DifferentialGeometry, so
  // index the "2nd" DifferentialGeometry to get this pointer
  temp = &firstDg[1];
  device_ptr<const PrimitiveHandle> firstPrimHandle(reinterpret_cast<const PrimitiveHandle*>(temp));
  
  // get a list of MaterialHandles
  const CudaSurfacePrimitiveList &primitives = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getPrimitives());
  vector_dev<MaterialHandle> materialsDevice(n);
  primitives.getMaterialHandles(firstPrimHandle,
                                sizeof(CudaIntersection),
                                &stencilDevice[0],
                                &materialsDevice[0],
                                n);

  // allocate space for scattering functions
  vector_dev<CudaScatteringDistributionFunction> cf(n);

  // evaluate scattering shader
  context->evaluateScattering(&materialsDevice[0],
                              firstDg,
                              sizeof(CudaIntersection),
                              &stencilDevice[0],
                              &cf[0],
                              n);

  // allocate space for emission functions
  vector_dev<CudaScatteringDistributionFunction> ce(n);

  // evaluate emission shader
  context->evaluateEmission(&materialsDevice[0],
                            firstDg,
                            sizeof(CudaIntersection),
                            &stencilDevice[0],
                            &ce[0],
                            n);

  // XXX kill these temps
  vector_dev<float3> woDevice(n);
  rayDirectionsToWo(&directionsAndMaxT[0],
                    &woDevice[0],
                    n);

  // evaluate scattering bsdf
  vector_dev<float3> scatteringDevice(n);
  context->evaluateBidirectionalScattering(&cf[0],
                                           &woDevice[0],
                                           firstDg,
                                           sizeof(CudaIntersection),
                                           &woDevice[0],
                                           &stencilDevice[0],
                                           &scatteringDevice[0],
                                           n);

  // evaluate emission bsdf
  vector_dev<float3> emissionDevice(n);
  context->evaluateUnidirectionalScattering(&ce[0],
                                            &woDevice[0],
                                            firstDg,
                                            sizeof(CudaIntersection),
                                            &stencilDevice[0],
                                            &emissionDevice[0],
                                            n);

  // sum into results
  sumScatteringAndEmission(&scatteringDevice[0],
                           &emissionDevice[0],
                           &stencilDevice[0],
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
  stdcuda::vector_dev<float4> originsAndMinT(totalWork);
  stdcuda::vector_dev<float4> directionsAndMaxT(totalWork);
  stdcuda::vector_dev<float> pdfsDevice(totalWork);
  stdcuda::vector_dev<float4> u0(totalWork);
  stdcuda::vector_dev<float4> u1(totalWork);
  std::vector<float> pdfs(totalWork);
  stdcuda::vector_dev<CudaIntersection> intersectionsDevice(totalWork);

  // XXX eliminate this
  std::vector<Ray> rays(totalWork);

  // XXX eliminate this
  std::vector<int> stencil(totalWork, 1);

  // init stencil to 1
  stdcuda::vector_dev<int> stencilDevice(totalWork);
  stdcuda::copy(stencil.begin(), stencil.end(), stencilDevice.begin());

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
                  &originsAndMinT[i * mWorkBatchSize],
                  &directionsAndMaxT[i * mWorkBatchSize],
                  &pdfsDevice[i * mWorkBatchSize],
                  mWorkBatchSize);

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i
  sampleEyeRays(&u0[numBatches * mWorkBatchSize],
                &u1[numBatches * mWorkBatchSize],
                &originsAndMinT[numBatches * mWorkBatchSize],
                &directionsAndMaxT[numBatches * mWorkBatchSize],
                &pdfsDevice[numBatches * mWorkBatchSize],
                totalWork % mWorkBatchSize);

  // purge all malloc'd memory for the last partial batch
  mShadingContext->freeAll();
  progress += totalWork % mWorkBatchSize;

  // intersect en masse
  intersect(&originsAndMinT[0], &directionsAndMaxT[0], &intersectionsDevice[0], &stencilDevice[0], totalWork);
  progress += totalWork;

  // shade
  for(size_t i = 0; i != numBatches; ++i)
  {
    shade(&directionsAndMaxT[i * mWorkBatchSize],
          &intersectionsDevice[i * mWorkBatchSize],
          &stencilDevice[i * mWorkBatchSize],
          &results[i * mWorkBatchSize],
          mWorkBatchSize);

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i

  // shade the last partial batch
  shade(&directionsAndMaxT[numBatches * mWorkBatchSize],
        &intersectionsDevice[numBatches * mWorkBatchSize],
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
  ::intersect(device_ptr<const float4> originsAndMinT,
              device_ptr<const float4> directionsAndMaxT,
              device_ptr<CudaIntersection> intersections,
              device_ptr<int> stencil,
              const size_t n)
{
  const CudaScene *scene = static_cast<const CudaScene*>(mScene.get());

  // compute the total number of batches we need to complete the work
  size_t numBatches = n / mWorkBatchSize;

  device_ptr<const float4> o = originsAndMinT;
  device_ptr<const float4> d = directionsAndMaxT;
  device_ptr<CudaIntersection> inter = intersections;
  device_ptr<int> s = stencil;
  device_ptr<const float4> end = o + numBatches * mWorkBatchSize;
  for(;
      o != end;
      o += mWorkBatchSize,
      d += mWorkBatchSize,
      inter += mWorkBatchSize,
      s += mWorkBatchSize)
  {
    scene->intersect(o, d, inter, s, mWorkBatchSize);
  } // end for i

  // do the last partial batch
  scene->intersect(o, d, inter, s, n % mWorkBatchSize);
} // end CudaDebugRenderer::intersect()

