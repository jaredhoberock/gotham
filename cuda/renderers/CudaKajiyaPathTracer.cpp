/*! \file CudaKajiyaPathTracer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaKajiyaPathTracer class.
 */

#include "CudaKajiyaPathTracer.h"
#include "kajiyaPathTracerUtil.h"
#include <stdcuda/vector_dev.h>
#include "../../numeric/RandomSequence.h"
#include "../primitives/CudaSurfacePrimitiveList.h"
#include "../primitives/CudaIntersection.h"
#include <stdcuda/cuda_algorithm.h>
#include "../../records/RenderFilm.h"
#include <hilbertsequence/HilbertSequence.h>

using namespace stdcuda;

void CudaKajiyaPathTracer
  ::stratify(const size_t w, const size_t h,
             const device_ptr<float4> &u,
             const size_t n)
{
  std::vector<float4> temp(n);
  stdcuda::copy(u, u + n, temp.begin());

  stratify(w,h,temp);
  stdcuda::copy(temp.begin(), temp.end(), u);
} // end CudaKajiyaPathTracer::stratify()

void CudaKajiyaPathTracer
  ::stratify(const size_t w, const size_t h,
             std::vector<float4> &u)
{
  HilbertSequence seq(0, 1.0f, 0, 1.0f,
                      w, h);
  for(size_t i = 0; i != u.size(); ++i)
  {
    unsigned int x = i % w;
    unsigned int y = i / w;

    //float px, py;
    //seq(px, py, u[i].x, u[i].y);
    u[i].x = static_cast<float>(x) / w;
    u[i].y = static_cast<float>(y) / h;
  } // end for i
} // end CudaKajiyaPathTracer::stratify()

void CudaKajiyaPathTracer
  ::generateHyperPoints(const device_ptr<float4> &u,
                        const size_t n)
{
  static RandomSequence rng;

  std::vector<float4> temp(n);
  for(size_t i = 0; i != n; ++i)
  {
    float u0 = rng();
    float u1 = rng();
    float u2 = rng();
    float u3 = rng();

    temp[i] = make_float4(u0,u1,u2,u3);
  } // end for i

  stdcuda::copy(temp.begin(), temp.end(), u);
} // end CudaKajiyaPathTracer::generateHyperPoints()

void CudaKajiyaPathTracer
  ::kernel(ProgressCallback &progress)
{
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  unsigned int workingSetSize = mWorkBatchSize; 

  // some shorthand
  const CudaSurfacePrimitiveList &sensors = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getSensors());
  const CudaSurfacePrimitiveList &emitters = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getEmitters());
  CudaShadingContext &context = *dynamic_cast<CudaShadingContext*>(mShadingContext.get());
  const CudaSurfacePrimitiveList &primitiveList = dynamic_cast<const CudaSurfacePrimitiveList&>(*mScene->getPrimitives());
  const CudaScene &scene = *static_cast<const CudaScene*>(mScene.get());

  // stencil
  vector_dev<bool> stencil(workingSetSize);

  // pixel locations
  stdcuda::vector_dev<float2> pixelLocations(workingSetSize);
  // XXX delete this
  std::vector<float2> pixelLocationsHost(workingSetSize);

  // random numbers
  stdcuda::vector_dev<float4> u(workingSetSize);

  // the result of each sample
  vector_dev<float3> results(workingSetSize);
  // XXX delete this
  std::vector<float3> hostResults(workingSetSize);

  // path throughput
  vector_dev<float3> throughput(workingSetSize);
  // storage for rays
  vector_dev<float4> originsAndMinT(workingSetSize);
  vector_dev<float4> directionsAndMaxT(workingSetSize);

  // storage for intersections
  vector_dev<CudaIntersection> intersections(workingSetSize);

  // per intersection storage
  vector_dev<PrimitiveHandle> primitives(workingSetSize);
  vector_dev<MaterialHandle> materials(workingSetSize);
  vector_dev<CudaDifferentialGeometry> dg(workingSetSize);
  vector_dev<CudaScatteringDistributionFunction> functions(workingSetSize);
  vector_dev<float3> scattering(workingSetSize);
  vector_dev<float3> wo(workingSetSize);
  vector_dev<bool> delta(workingSetSize);
  vector_dev<float> pdfs(workingSetSize);

  // per light sample storage
  vector_dev<CudaDifferentialGeometry> lightDg(workingSetSize);
  vector_dev<CudaScatteringDistributionFunction> lightFunctions(workingSetSize);
  vector_dev<bool> shadowStencil(workingSetSize);
  vector_dev<float3> wi(workingSetSize);
  vector_dev<float3> emission(workingSetSize);
  vector_dev<float> geometricTerm(workingSetSize);

  progress.restart(workingSetSize);

  // while rays < total
  
  // XXX implement fill or something
  for(size_t i = 0; i != workingSetSize; ++i)
  {
    stencil[i] = 1;
    shadowStencil[i] = 1;
    throughput[i] = make_float3(1,1,1);
    results[i] = make_float3(0,0,0);
  } // end for i
  
  // generate random numbers
  generateHyperPoints(&u[0], u.size());

  // sample the surface of the sensor
  sensors.sampleSurfaceArea(&u[0], &primitives[0], &dg[0], &pdfs[0], workingSetSize);

  // divide throughput by pdf
  divideByPdf(&pdfs[0], &throughput[0], workingSetSize);

  // find each hit primitive's material
  primitiveList.getMaterialHandles(&primitives[0], &materials[0], workingSetSize);

  // evaluate the sensor's shader
  context.evaluateSensor(&materials[0],
                         &dg[0],
                         sizeof(CudaDifferentialGeometry),
                         &functions[0],
                         workingSetSize);

  //   for path length
  size_t pathLength = 1;
  
  // generate more numbers
  generateHyperPoints(&u[0], u.size());

  if(pathLength == 1)
  {
    stratify(film->getWidth(), film->getHeight(), &u[0], workingSetSize);
  }

  // sample distribution function
  if(pathLength == 1)
  {
    // reinterpret directionsAndMaxT into a float3 *
    float4 *temp = &directionsAndMaxT[0];
    device_ptr<float3> wi(reinterpret_cast<float3*>(temp));

    // reinterpret u into a float3 *
    const float4 *temp2 = &u[0];
    device_ptr<const float3> uAsAFloat3(reinterpret_cast<const float3*>(temp2));
    context.sampleUnidirectionalScattering(&functions[0],
                                           &dg[0],
                                           sizeof(CudaDifferentialGeometry),
                                           uAsAFloat3,
                                           sizeof(float4),
                                           &scattering[0],
                                           sizeof(float3),
                                           wi,
                                           sizeof(float4),
                                           &pdfs[0],
                                           sizeof(float),
                                           &delta[0],
                                           sizeof(bool),
                                           workingSetSize);

    // save the first two elements of u to use later
    extractFloat2(&u[0], &pixelLocations[0], workingSetSize);
  } // end if
  else
  {
    // sample bidirectional scattering
  } // end else

  // create rays
  finalizeRays(&dg[0],
               0.0005f,
               std::numeric_limits<float>::infinity(),
               &stencil[0],
               &originsAndMinT[0],
               &directionsAndMaxT[0],
               workingSetSize);

  // find intersection, update stencil
  // XXX we need an in stencil and an out stencil
  scene.intersect(&originsAndMinT[0],
                  &directionsAndMaxT[0],
                  &intersections[0],
                  &stencil[0],
                  workingSetSize);
  
  // flip ray directions to wo
  flipRayDirections(&directionsAndMaxT[0],
                    &stencil[0],
                    &wo[0],
                    workingSetSize);

  // convert solid angle pdf into projected solid angle pdf
  // this is equivalent to multiplying throughput by the cosine term
  // XXX and might also be faster because it does not involve a divide
  toProjectedSolidAnglePdf(&wo[0],
                           &dg[0],
                           &stencil[0],
                           &pdfs[0],
                           workingSetSize);

  // update: throughput *= scattering/pdf
  updateThroughput(&scattering[0],
                   &pdfs[0],
                   &stencil[0],
                   &throughput[0],
                   workingSetSize);

  // get a pointer to the first CudaDifferentialGeometry
  // in the first CudaIntersection in the list
  const void *temp = &intersections[0];
  device_ptr<const CudaDifferentialGeometry> firstDg(reinterpret_cast<const CudaDifferentialGeometry*>(temp));

  // get a pointer to the fist PrimitiveHandle
  // in the first CudaIntersection in the list
  temp = &firstDg[1];
  device_ptr<const PrimitiveHandle> firstPrimHandle(reinterpret_cast<const PrimitiveHandle*>(temp));

  // get the material of the intersected primitive
  primitiveList.getMaterialHandles(firstPrimHandle,
                                   sizeof(CudaIntersection),
                                   &stencil[0],
                                   &materials[0],
                                   workingSetSize);

  // evaluate the surface shader
  context.evaluateScattering(&materials[0],
                             firstDg,
                             sizeof(CudaIntersection),
                             &stencil[0],
                             &functions[0],
                             workingSetSize);

  // always accumulate emission for eye rays
  if(pathLength == 1)
    stdcuda::copy(stencil.begin(), stencil.end(), delta.begin());

  // evaluate emission at the hit point
  context.evaluateEmission(&materials[0],
                           firstDg,
                           sizeof(CudaIntersection),
                           &delta[0],
                           &lightFunctions[0],
                           workingSetSize);

  context.evaluateUnidirectionalScattering(&lightFunctions[0],
                                           &wo[0],
                                           firstDg,
                                           sizeof(CudaIntersection),
                                           &delta[0],
                                           &emission[0],
                                           workingSetSize);

  // result += delta * throughput * emission
  accumulateEmission(&throughput[0], &emission[0], &delta[0], &results[0], workingSetSize);

  size_t numLightSamples = 16;
  for(size_t i = 0; i != numLightSamples; ++i)
  {
    // generate random numbers
    generateHyperPoints(&u[0], u.size());
    
    // sample light
    emitters.sampleSurfaceArea(&u[0], &primitives[0], &lightDg[0], &pdfs[0], workingSetSize);
    
    // get the material of the light
    primitiveList.getMaterialHandles(&primitives[0], &materials[0], workingSetSize);
    
    // create shadow rays pointing from intersection to sampled light point
    createShadowRays(&intersections[0], &lightDg[0], 0.0005f, &stencil[0], &originsAndMinT[0], &directionsAndMaxT[0], workingSetSize);
    
    // trace shadow rays
    scene.shadow(&originsAndMinT[0], &directionsAndMaxT[0], &stencil[0], &shadowStencil[0], workingSetSize);
    
    // evaluate emission shader
    context.evaluateEmission(&materials[0], &lightDg[0], sizeof(CudaDifferentialGeometry), &shadowStencil[0], &lightFunctions[0], workingSetSize);
    
    // turn shadow ray directions into wi
    rayDirectionsToFloat3(&directionsAndMaxT[0], &shadowStencil[0], &wi[0], workingSetSize);
    
    // evaluate scattering function at intersection
    context.evaluateBidirectionalScattering(&functions[0],
                                            &wo[0],
                                            firstDg,
                                            sizeof(CudaIntersection),
                                            &wi[0],
                                            &shadowStencil[0],
                                            &scattering[0],
                                            workingSetSize);

    // flip shadow ray directions
    flipVectors(&wi[0], &shadowStencil[0], workingSetSize);
    //// XXX we should just have a flip vector function
    //flipRayDirections(&directionsAndMaxT[0],
    //                  &shadowStencil[0],
    //                  &wi[0],
    //                  workingSetSize);

    // evaluate emission function at light
    context.evaluateUnidirectionalScattering(&lightFunctions[0],
                                             &wi[0],
                                             &lightDg[0],
                                             &shadowStencil[0],
                                             &emission[0],
                                             workingSetSize);
    
    // evaluate geometric term
    evaluateGeometricTerm(firstDg,
                          sizeof(CudaIntersection),
                          &lightDg[0],
                          &shadowStencil[0],
                          &geometricTerm[0],
                          workingSetSize);

    // result += throughput * scattering * emission * geometricTerm / pdf
    accumulateLightSample(1.0f / numLightSamples,
                          &throughput[0],
                          &scattering[0],
                          &emission[0],
                          &geometricTerm[0],
                          &pdfs[0],
                          &shadowStencil[0],
                          &results[0],
                          workingSetSize);
  } // end for i
  
  // scatter results
  // XXX implement a gpu path for this
  stdcuda::copy(results.begin(), results.end(), hostResults.begin());
  stdcuda::copy(pixelLocations.begin(), pixelLocations.end(), pixelLocationsHost.begin());
  for(size_t i = 0; i != workingSetSize; ++i)
  {
    Spectrum s(hostResults[i].x,
               hostResults[i].y,
               hostResults[i].z);

    film->deposit(pixelLocationsHost[i].x,
                  pixelLocationsHost[i].y,
                  s);
  } // end for i

  progress += workingSetSize;

  //// scale results by 1/spp
  //// XXX fix this later -- we shouldn't need to do this
  //float spp = static_cast<float>(workingSetSize) / (film->getWidth() * film->getHeight());
  //float invSpp = 1.0f / spp;
  //film->scale(invSpp);
} // end CudaKajiyaPathTracer::kernel()

