/*! \file CudaKajiyaPathTracer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaKajiyaPathTracer class.
 */

#include "CudaKajiyaPathTracer.h"
#include "kajiyaPathTracerUtil.h"
#include <stdcuda/vector_dev.h>
#include <stdcuda/fill.h>
#include "../../numeric/RandomSequence.h"
#include "../primitives/CudaSurfacePrimitiveList.h"
#include "../primitives/CudaIntersection.h"
#include <stdcuda/cuda_algorithm.h>
#include "../../records/RenderFilm.h"
#include <hilbertsequence/HilbertSequence.h>

using namespace stdcuda;

void CudaKajiyaPathTracer
  ::generateHyperPoints(const device_ptr<float4> &u,
                        const size_t n)
{
  static RandomSequence rng;

  // cast u to a float *
  // call the rng
  // XXX this is borked at the moment
  //(*mRandomSequence)((float*)u.get(), 4 * n);
  
  // XXX there's about a 10% performance penalty
  //     if we just use the cpu method for generating
  //     random numbers (which isn't broken) on 8800 GTS
  std::vector<float4> temp(n);

  for(size_t i = 0; i != n; ++i)
  {
    temp[i].x = rng();
    temp[i].y = rng();
    temp[i].z = rng();
    temp[i].w = rng();
  } // end for i

  stdcuda::copy(temp.begin(), temp.end(), &u[0]);
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
  vector_dev<unsigned int> component(workingSetSize);
  vector_dev<float> pdfs(workingSetSize);

  // per light sample storage
  vector_dev<CudaDifferentialGeometry> lightDg(workingSetSize);
  vector_dev<CudaScatteringDistributionFunction> lightFunctions(workingSetSize);
  vector_dev<bool> shadowStencil(workingSetSize);
  vector_dev<float3> wi(workingSetSize);
  vector_dev<float3> emission(workingSetSize);
  vector_dev<float> geometricTerm(workingSetSize);

  progress.restart(workingSetSize);

  //size_t spp = 1;
  //size_t spp = 16;
  size_t spp = 100;

  for(size_t pixelSample = 0; pixelSample < spp; ++pixelSample)
  {
    // while rays < total
    
    // initialize
    stdcuda::fill(throughput.begin(), throughput.end(), make_float3(1,1,1));
    stdcuda::fill(results.begin(), results.end(), make_float3(0,0,0));
    stdcuda::fill(stencil.begin(), stencil.end(), true);
    stdcuda::fill(shadowStencil.begin(), shadowStencil.end(), true);
    
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
    for(size_t pathLength = 1; pathLength <= 7; ++pathLength)
    {
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
        // reinterpret directionsAndMaxT into a float3 *
        float4 *temp = &directionsAndMaxT[0];
        device_ptr<float3> wi(reinterpret_cast<float3*>(temp));

        // reinterpret u into a float3 *
        const float4 *temp2 = &u[0];
        device_ptr<const float3> uAsAFloat3(reinterpret_cast<const float3*>(temp2));

        // get a pointer to the first CudaDifferentialGeometry
        // in the first CudaIntersection in the list
        const void *temp3 = &intersections[0];
        device_ptr<const CudaDifferentialGeometry> firstDg(reinterpret_cast<const CudaDifferentialGeometry*>(temp3));

        // sample
        //std::cerr << "before sample(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        context.sampleBidirectionalScattering(&functions[0],
                                              &wo[0],
                                              sizeof(float3),
                                              firstDg,
                                              sizeof(CudaIntersection),
                                              uAsAFloat3,
                                              sizeof(float4),
                                              &stencil[0],
                                              &scattering[0],
                                              sizeof(float3),
                                              wi,
                                              sizeof(float4),
                                              &pdfs[0],
                                              sizeof(float),
                                              &delta[0],
                                              sizeof(bool),
                                              &component[0],
                                              sizeof(unsigned int),
                                              workingSetSize);
        //std::cerr << "after sample(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // copy DifferentialGeometry from the intersections
        // into dg
        // XXX this really really really sucks
        copyDifferentialGeometry(&intersections[0],&dg[0],workingSetSize);
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
      // XXX and might also be slower because it involves a divide
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
      //std::cerr << "before evaluateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      context.evaluateEmission(&materials[0],
                               firstDg,
                               sizeof(CudaIntersection),
                               &delta[0],
                               &lightFunctions[0],
                               workingSetSize);
      //std::cerr << "after evaluateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      //std::cerr << "before evaluateUnidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      context.evaluateUnidirectionalScattering(&lightFunctions[0],
                                               &wo[0],
                                               firstDg,
                                               sizeof(CudaIntersection),
                                               &delta[0],
                                               &emission[0],
                                               workingSetSize);
      //std::cerr << "after evaluateUnidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // result += delta * throughput * emission
      //std::cerr << "before accumulateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      accumulateEmission(&throughput[0], &emission[0], &delta[0], &results[0], workingSetSize);
      //std::cerr << "after accumulateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      size_t numLightSamples = 1;
      //if(pathLength <= 2)
      //  numLightSamples = 16;

      for(size_t i = 0; i != numLightSamples; ++i)
      {
        // generate random numbers
        //std::cerr << "before generateHyperPoints(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        generateHyperPoints(&u[0], u.size());
        //std::cerr << "after generateHyperPoints(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // sample light
        //std::cerr << "before sampleSurfaceArea(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        emitters.sampleSurfaceArea(&u[0], &primitives[0], &lightDg[0], &pdfs[0], workingSetSize);
        //std::cerr << "after sampleSurfaceArea(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // get the material of the light
        //std::cerr << "before getMaterialHandles(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        primitiveList.getMaterialHandles(&primitives[0], &materials[0], workingSetSize);
        //std::cerr << "after getMaterialHandles(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // create shadow rays pointing from intersection to sampled light point
        //std::cerr << "before createShadowRays(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        createShadowRays(&intersections[0], &lightDg[0], 0.0005f, &stencil[0], &originsAndMinT[0], &directionsAndMaxT[0], workingSetSize);
        //std::cerr << "after createShadowRays(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // trace shadow rays
        //std::cerr << "before shadow(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        scene.shadow(&originsAndMinT[0], &directionsAndMaxT[0], &stencil[0], &shadowStencil[0], workingSetSize);
        //std::cerr << "after shadow(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate emission shader
        //std::cerr << "before evaluateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        context.evaluateEmission(&materials[0], &lightDg[0], sizeof(CudaDifferentialGeometry), &shadowStencil[0], &lightFunctions[0], workingSetSize);
        //std::cerr << "after evaluateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // turn shadow ray directions into wi
        //std::cerr << "before rayDirectionsToFloat3(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        rayDirectionsToFloat3(&directionsAndMaxT[0], &shadowStencil[0], &wi[0], workingSetSize);
        //std::cerr << "after rayDirectionsToFloat3(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate scattering function at intersection
        //std::cerr << "before evaluateBidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        context.evaluateBidirectionalScattering(&functions[0],
                                                &wo[0],
                                                firstDg,
                                                sizeof(CudaIntersection),
                                                &wi[0],
                                                &shadowStencil[0],
                                                &scattering[0],
                                                workingSetSize);
        //std::cerr << "after evaluateBidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // flip shadow ray directions
        //std::cerr << "before flipVectors(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        flipVectors(&wi[0], &shadowStencil[0], workingSetSize);
        //std::cerr << "after flipVectors(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // evaluate emission function at light
        //std::cerr << "before evaluateUnidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        context.evaluateUnidirectionalScattering(&lightFunctions[0],
                                                 &wi[0],
                                                 &lightDg[0],
                                                 &shadowStencil[0],
                                                 &emission[0],
                                                 workingSetSize);
        //std::cerr << "after evaluateUnidirectionalScattering(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate geometric term
        //std::cerr << "before evaluateGeometricTerm(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        evaluateGeometricTerm(firstDg,
                              sizeof(CudaIntersection),
                              &lightDg[0], &shadowStencil[0], &geometricTerm[0],
                              workingSetSize);
        //std::cerr << "after evaluateGeometricTerm(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // result += throughput * scattering * emission * geometricTerm / pdf
        //std::cerr << "before accumulateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        accumulateLightSample(1.0f / numLightSamples,
                              &throughput[0],
                              &scattering[0],
                              &emission[0],
                              &geometricTerm[0],
                              &pdfs[0],
                              &shadowStencil[0],
                              &results[0],
                              workingSetSize);
        //std::cerr << "after accumulateEmission(): cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      } // end for i
    } // end for pathLength
    
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

    progress += workingSetSize / spp;
  } // end for spp

  // scale results by 1/spp
  // XXX fix this later -- we shouldn't need to do this
  float invSpp = 1.0f / spp;
  film->scale(invSpp);
  
  //std::cerr << "cuda error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
} // end CudaKajiyaPathTracer::kernel()

void CudaKajiyaPathTracer
  ::setRandomSequence(const boost::shared_ptr<CudaRandomSequence> &r)
{
  mRandomSequence = r;
} // end CudaRandomSequence::setRandomSequence()

