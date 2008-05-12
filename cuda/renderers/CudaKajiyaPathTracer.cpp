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
#include "../geometry/CudaDifferentialGeometryVector.h"
#include <stdcuda/cuda_algorithm.h>
#include "../../records/RenderFilm.h"
#include <hilbertsequence/HilbertSequence.h>
#include <vector_functions.h>

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
  std::vector<float3> hostResults(workingSetSize, make_float3(0,0,0));

  // path throughput
  vector_dev<float3> throughput(workingSetSize);

  // storage for rays
  vector_dev<float3> directions(workingSetSize);
  const float rayEpsilon = 0.00005f;

  // per intersection storage
  CudaDifferentialGeometryVector eyeDg(workingSetSize);
  vector_dev<unsigned int> primitivesOrMaterials(workingSetSize);
  vector_dev<CudaScatteringDistributionFunction> functions(workingSetSize);
  vector_dev<float3> scattering(workingSetSize);
  vector_dev<float3> wo(workingSetSize);
  vector_dev<bool> delta(workingSetSize);
  vector_dev<unsigned int> component(workingSetSize);
  vector_dev<float> pdfs(workingSetSize);

  // per light sample storage
  CudaDifferentialGeometryVector lightDg(workingSetSize);
  vector_dev<CudaScatteringDistributionFunction> lightFunctions(workingSetSize);
  vector_dev<bool> shadowStencil(workingSetSize);
  vector_dev<float3> wi(workingSetSize);
  vector_dev<float3> emission(workingSetSize);
  vector_dev<float> geometricTerm(workingSetSize);
  //std::cerr << "after mallocs: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  progress.restart(workingSetSize);

  //size_t spp = 1;
  //size_t spp = 16;
  //size_t spp = 100;
  size_t spp = 1000;

  for(size_t pixelSample = 0; pixelSample < spp; ++pixelSample)
  {
    // while rays < total
    
    // initialize
    stdcuda::fill(throughput.begin(), throughput.end(), make_float3(1,1,1));
    //std::cerr << "after fill: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    stdcuda::fill(results.begin(), results.end(), make_float3(0,0,0));
    //std::cerr << "after fill: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    stdcuda::fill(stencil.begin(), stencil.end(), true);
    //std::cerr << "after fill: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    stdcuda::fill(shadowStencil.begin(), shadowStencil.end(), true);
    //std::cerr << "after fill: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    // generate random numbers
    generateHyperPoints(&u[0], u.size());
    //std::cerr << "after generateHyperPoints(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // sample the surface of the sensor
    sensors.sampleSurfaceArea(&u[0], &primitivesOrMaterials[0], eyeDg, &pdfs[0], workingSetSize);
    //std::cerr << "after sampleSurfaceArea(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // divide throughput by pdf
    divideByPdf(&pdfs[0], &throughput[0], workingSetSize);
    //std::cerr << "after divideByPdf(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // map the primitive to its material
    primitiveList.getMaterialHandles(&primitivesOrMaterials[0], &stencil[0], workingSetSize);
    //std::cerr << "after getMaterialHandles(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // evaluate the sensor's shader
    context.evaluateSensor(&primitivesOrMaterials[0],
                           eyeDg,
                           &functions[0],
                           workingSetSize);
    //std::cerr << "after evaluateSensor(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // for path length
    //for(size_t pathLength = 3; pathLength <= 3; ++pathLength)
    //for(size_t pathLength = 3; pathLength <= 5; ++pathLength)
    for(size_t pathLength = 3; pathLength <= 7; ++pathLength)
    {
      // generate more numbers
      generateHyperPoints(&u[0], u.size());
      //std::cerr << "after generateHyperPoints(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      if(pathLength == 3)
      {
        stratify(film->getWidth(), film->getHeight(), &u[0], workingSetSize);
        //std::cerr << "after strafity(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      } // end if

      // sample distribution function
      if(pathLength == 3)
      {
        // reinterpret u into a float3 *
        const float4 *temp2 = &u[0];
        device_ptr<const float3> uAsAFloat3(reinterpret_cast<const float3*>(temp2));
        context.sampleUnidirectionalScattering(&functions[0],
                                               eyeDg,
                                               uAsAFloat3,
                                               sizeof(float4),
                                               &scattering[0],
                                               &directions[0],
                                               sizeof(float3),
                                               &pdfs[0],
                                               &delta[0],
                                               workingSetSize);
        //std::cerr << "after sampleUnidirectionalScattering(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // save the first two elements of u to use later
        extractFloat2(&u[0], &pixelLocations[0], workingSetSize);
        //std::cerr << "after extractFloat2(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      } // end if
      else
      {
        // reinterpret u into a float3 *
        const float4 *temp2 = &u[0];
        device_ptr<const float3> uAsAFloat3(reinterpret_cast<const float3*>(temp2));

        // sample
        context.sampleBidirectionalScattering(&functions[0],
                                              &wo[0],
                                              sizeof(float3),
                                              eyeDg,
                                              uAsAFloat3,
                                              sizeof(float4),
                                              &stencil[0],
                                              &scattering[0],
                                              &directions[0],
                                              sizeof(float3),
                                              &pdfs[0],
                                              &delta[0],
                                              &component[0],
                                              workingSetSize);
      } // end else

      // convert solid angle pdf into projected solid angle pdf
      // this is equivalent to multiplying throughput by the cosine term
      // XXX and might also be slower because it involves a divide
      toProjectedSolidAnglePdf(&directions[0],
                               eyeDg.mNormals,
                               &stencil[0],
                               &pdfs[0],
                               workingSetSize);
      //std::cerr << "after toProjectedSolidAnglePdf(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // find intersection, update stencil
      // XXX we need an in stencil and an out stencil
      scene.intersect(eyeDg.mPoints,
                      &directions[0],
                      make_float2(rayEpsilon, std::numeric_limits<float>::infinity()),
                      eyeDg,
                      &primitivesOrMaterials[0],
                      &stencil[0],
                      workingSetSize);
      //std::cerr << "after intersect(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // flip ray directions to wo
      flipVectors(&directions[0], &stencil[0], &wo[0], workingSetSize);
      //std::cerr << "after flipVectors(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // update: throughput *= scattering/pdf
      // XXX we could ensure that throughput is > 0 at this point and modify the stencil
      updateThroughput(&scattering[0],
                       &pdfs[0],
                       &stencil[0],
                       &throughput[0],
                       workingSetSize);
      //std::cerr << "after updateThroughput(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // get the material of the intersected primitive
      primitiveList.getMaterialHandles(&primitivesOrMaterials[0], &stencil[0], workingSetSize);
      //std::cerr << "after getMaterialHandles(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // evaluate the surface shader
      context.evaluateScattering(&primitivesOrMaterials[0], eyeDg, &stencil[0], &functions[0], workingSetSize);
      //std::cerr << "after evaluateScattering(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // always accumulate emission for eye rays
      if(pathLength == 3)
        stdcuda::copy(stencil.begin(), stencil.end(), delta.begin());

      // evaluate emission at the hit point
      context.evaluateEmission(&primitivesOrMaterials[0], eyeDg, &delta[0], &lightFunctions[0], workingSetSize);
      //std::cerr << "after evaluateEmission(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      context.evaluateUnidirectionalScattering(&lightFunctions[0], &wo[0], eyeDg, &delta[0], &emission[0], workingSetSize);
      //std::cerr << "after evaluateUnidirectionalScattering(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      // result += delta * throughput * emission
      accumulateEmission(&throughput[0], &emission[0], &delta[0], &results[0], workingSetSize);
      //std::cerr << "after accumulateEmission(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

      size_t numLightSamples = 1;
      //if(pathLength <= 2)
      //  numLightSamples = 16;

      for(size_t i = 0; i != numLightSamples; ++i)
      {
        // generate random numbers
        generateHyperPoints(&u[0], u.size());
        //std::cerr << "after generateHyperPoints(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // sample light
        emitters.sampleSurfaceArea(&u[0], &stencil[0], &primitivesOrMaterials[0], lightDg, &pdfs[0], workingSetSize);
        //std::cerr << "after sampleSurfaceArea(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // get the material of the light
        primitiveList.getMaterialHandles(&primitivesOrMaterials[0], &stencil[0], workingSetSize);
        //std::cerr << "after getMaterialHandles(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // create shadow rays pointing from intersection to sampled light point
        createShadowRays(eyeDg.mPoints, lightDg.mPoints, 0.0005f, &stencil[0], &directions[0], workingSetSize);
        //std::cerr << "after createShadowRays(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // trace shadow rays
        scene.shadow(eyeDg.mPoints, &directions[0], make_float2(rayEpsilon, 1.0f - rayEpsilon), &stencil[0], &shadowStencil[0], workingSetSize);
        //std::cerr << "after shadow(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate emission shader
        context.evaluateEmission(&primitivesOrMaterials[0], lightDg, &shadowStencil[0], &lightFunctions[0], workingSetSize);
        //std::cerr << "after evaluateEmission(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // turn shadow ray directions into wi
        // XXX this is probably not necessary
        stdcuda::copy(directions.begin(), directions.end(), wi.begin());
        //std::cerr << "after copy(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate scattering function at intersection
        context.evaluateBidirectionalScattering(&functions[0],
                                                &wo[0],
                                                eyeDg,
                                                &wi[0],
                                                &shadowStencil[0],
                                                &scattering[0],
                                                workingSetSize);
        //std::cerr << "after evaluateBidirectionalScattering(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // flip shadow ray directions
        flipVectors(&wi[0], &shadowStencil[0], workingSetSize);
        //std::cerr << "after flipVectors(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

        // evaluate emission function at light
        context.evaluateUnidirectionalScattering(&lightFunctions[0],
                                                 &wi[0],
                                                 lightDg,
                                                 &shadowStencil[0],
                                                 &emission[0],
                                                 workingSetSize);
        //std::cerr << "after evaluateUnidirectionalScattering(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        
        // evaluate geometric term
        evaluateGeometricTerm(eyeDg.mPoints, eyeDg.mNormals, lightDg.mPoints, lightDg.mNormals, &shadowStencil[0], &geometricTerm[0], workingSetSize);
        //std::cerr << "after evaluateGeometricTerm(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;

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
        //std::cerr << "after accumulateLightSample(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
      } // end for i
    } // end for pathLength
    
    // scatter results
    // XXX implementing a gpu path for this is probably not worth it
    //     this takes essentially 0 time
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

