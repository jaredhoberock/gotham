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

using namespace stdcuda;

void CudaDebugRenderer
  ::sampleEyeRay(const size_t batchIdx,
                 const size_t threadIdx,
                 Ray *rays,
                 float *pdfs) const
{
  size_t i = batchIdx * mWorkBatchSize + threadIdx;

  // XXX TODO: kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  gpcpu::size2 imageDim(film->getWidth(),film->getHeight());

  // convert thread index to pixel index
  gpcpu::size2 pixelIndex(i % imageDim[0],
                          i / imageDim[0]);

  // convert pixel index to location in unit square
  gpcpu::float2 uv(static_cast<float>(pixelIndex[0]) / imageDim[0],
                   static_cast<float>(pixelIndex[1]) / imageDim[1]); 

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float &pdf = pdfs[i];
  mScene->getSensors()->sampleSurfaceArea(0, &sensor, pdf);

  // sample a point on the sensor
  DifferentialGeometry dgSensor;
  sensor->sampleSurfaceArea(0,0,0,dgSensor,pdf);

  // generate a Ray
  ScatteringDistributionFunction &s = *mShadingContext->evaluateSensor(sensor->getMaterial(), dgSensor);

  // sample a sensing direction
  bool delta;
  ::Vector d;
  s.sample(dgSensor, uv[0], uv[1], 0.5f, d, pdf, delta);
  rays[i] = Ray(dgSensor.getPoint(), d);
} // end CudaDebugRenderer::sampleEyeRay()

void CudaDebugRenderer
  ::shade(const Ray *rays,
          const float *pdfs,
          device_ptr<const CudaIntersection> intersectionsDevice,
          device_ptr<const int> stencilDevice,
          Spectrum *results,
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

  std::vector< ::Vector> wo(n);
  for(size_t i = 0; i != n; ++i)
  {
    wo[i] = -rays[i].getDirection();
  } // end for i

  // XXX kill these temps
  vector_dev<float3> woDevice(n), wiDevice(n);
  stdcuda::copy(wo.begin(), wo.end(), woDevice.begin());
  stdcuda::copy(woDevice.begin(), woDevice.end(), wiDevice.begin());

  // evaluate scattering bsdf
  vector_dev<float3> scatteringDevice(n);
  context->evaluateBidirectionalScattering(&cf[0],
                                           &woDevice[0],
                                           firstDg,
                                           sizeof(CudaIntersection),
                                           &wiDevice[0],
                                           &stencilDevice[0],
                                           &scatteringDevice[0],
                                           n);

  // XXX kill this temp
  std::vector<Spectrum> scattering(n);
  float3 *hostPtr = reinterpret_cast<float3*>(&scattering[0]);
  stdcuda::copy(scatteringDevice.begin(), scatteringDevice.end(), hostPtr);

  // evaluate emission bsdf
  vector_dev<float3> emissionDevice(n);
  context->evaluateUnidirectionalScattering(&ce[0],
                                            &woDevice[0],
                                            firstDg,
                                            sizeof(CudaIntersection),
                                            &stencilDevice[0],
                                            &emissionDevice[0],
                                            n);

  // XXX kill this temp
  std::vector<Spectrum> emission(n);
  hostPtr = reinterpret_cast<float3*>(&emission[0]);
  stdcuda::copy(emissionDevice.begin(), emissionDevice.end(), hostPtr);

  // sum
  for(size_t i = 0; i != n; ++i)
  {
    if(stencilDevice[i])
    {
      results[i] = scattering[i] + emission[i];
    } // end if
  } // end for i
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
  std::vector<Spectrum> results(totalWork, Spectrum::black());

  // sampleEyeRay + intersect + shade + deposit = 4
  progress.restart(totalWork * 4);

  for(size_t i = 0; i != numBatches; ++i)
  {
    for(size_t j = 0; j != mWorkBatchSize; ++j)
    {
      sampleEyeRay(i,j, &rays[0], &pdfs[0]);
    } // end for j

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i
  for(size_t j = 0; j != totalWork % mWorkBatchSize; ++j)
    sampleEyeRay(numBatches, j, &rays[0], &pdfs[0]);

  // purge all malloc'd memory for the last partial batch
  mShadingContext->freeAll();
  progress += totalWork % mWorkBatchSize;

  // copy rays to device
  // XXX eliminate this
  for(size_t i = 0; i != rays.size(); ++i)
  {
    const Ray &r = rays[i];
    originsAndMinT[i] = make_float4(r.getAnchor()[0],
                                    r.getAnchor()[1],
                                    r.getAnchor()[2],
                                    r.getInterval()[0]);

    directionsAndMaxT[i] = make_float4(r.getDirection()[0],
                                       r.getDirection()[1],
                                       r.getDirection()[2],
                                       r.getInterval()[1]);
  } // end for i

  // intersect en masse
  intersect(&originsAndMinT[0], &directionsAndMaxT[0], &intersectionsDevice[0], &stencilDevice[0], totalWork);
  progress += totalWork;

  // copy back to host
  // XXX eliminate this
  std::vector<Intersection> intersections(totalWork);
  CudaIntersection *hostPtr = reinterpret_cast<CudaIntersection*>(&intersections[0]);
  stdcuda::copy(intersectionsDevice.begin(), intersectionsDevice.end(), hostPtr);
  stdcuda::copy(stencilDevice.begin(), stencilDevice.end(), &stencil[0]);

  for(size_t i = 0; i != numBatches; ++i)
  {
    shade(&rays[i * mWorkBatchSize],
          &pdfs[i * mWorkBatchSize],
          &intersectionsDevice[i * mWorkBatchSize],
          &stencilDevice[i * mWorkBatchSize],
          &results[i * mWorkBatchSize],
          mWorkBatchSize);

    // purge all malloc'd memory for this batch
    mShadingContext->freeAll();

    progress += mWorkBatchSize;
  } // end for i

  // shade the last partial batch
  shade(&rays[numBatches * mWorkBatchSize],
        &pdfs[numBatches * mWorkBatchSize],
        &intersectionsDevice[numBatches * mWorkBatchSize],
        &stencilDevice[numBatches * mWorkBatchSize],
        &results[numBatches * mWorkBatchSize],
        totalWork % mWorkBatchSize);
  progress += totalWork % mWorkBatchSize;

  // purge all malloc'd memory for the last partial batch
  mShadingContext->freeAll();

  for(size_t i = 0; i != numBatches; ++i)
  {
    for(size_t j = 0; j != mWorkBatchSize; ++j)
    {
      deposit(i,j,&results[0]);
    } // end for j

    progress += mWorkBatchSize;
  } // end for i
  for(size_t j = 0; j != totalWork % mWorkBatchSize; ++j)
    deposit(numBatches,j,&results[0]);
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

