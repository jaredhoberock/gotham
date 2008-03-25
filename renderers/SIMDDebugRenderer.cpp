/*! \file SIMDDebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SIMDDebugRenderer class.
 */

#include "SIMDDebugRenderer.h"
#include "../geometry/Point.h"
#include "../geometry/Vector.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../records/RenderFilm.h"
#include "../shading/Material.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"

using namespace gpcpu;

void SIMDDebugRenderer
  ::sampleEyeRay(const size_t batchIdx,
                 const size_t threadIdx,
                 Ray *rays,
                 float *pdfs) const
{
  size_t i = batchIdx * mWorkBatchSize + threadIdx;

  // XXX TODO: kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size2 imageDim(film->getWidth(),film->getHeight());

  // convert thread index to pixel index
  size2 pixelIndex(i % imageDim[0],
                   i / imageDim[0]);

  // convert pixel index to location in unit square
  float2 uv(static_cast<float>(pixelIndex[0]) / imageDim[0],
            static_cast<float>(pixelIndex[1]) / imageDim[1]); 

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float &pdf = pdfs[i];
  mScene->getSensors()->sampleSurfaceArea(0, &sensor, pdf);

  // sample a point on the sensor
  DifferentialGeometry dgSensor;
  sensor->sampleSurfaceArea(0,0,0,dgSensor,pdf);

  // generate a Ray
  ScatteringDistributionFunction &s = *sensor->getMaterial()->evaluateSensor(dgSensor);

  // sample a sensing direction
  bool delta;
  ::Vector d;
  s.sample(dgSensor, uv[0], uv[1], 0.5f, d, pdf, delta);
  rays[i] = Ray(dgSensor.getPoint(), d);
} // end SIMDDebugRenderer::sampleEyeRay()

void SIMDDebugRenderer
  ::shade(const size_t batchIdx,
          const size_t threadIdx,
          const Ray *rays,
          const float *pdfs,
          const Intersection *intersections,
          const int *stencil,
          Spectrum *results) const
{
  size_t i = batchIdx * mWorkBatchSize + threadIdx;
  if(stencil[i])
  {
    const Intersection &inter = intersections[i];
    PrimitiveHandle prim = inter.getPrimitive();
    const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>((*mScene->getPrimitives())[prim].get());

    Spectrum &L = results[i];
    const ::Vector &d = rays[i].getDirection();

    // evaluate scattering
    ScatteringDistributionFunction *f = sp->getMaterial()->evaluateScattering(inter.getDifferentialGeometry());
    L = f->evaluate(-d,inter.getDifferentialGeometry(),-d);

    // add emission
    ScatteringDistributionFunction *e = sp->getMaterial()->evaluateEmission(inter.getDifferentialGeometry());
    L += e->evaluate(-d, inter.getDifferentialGeometry());
  } // end if
} // end SIMDDebugRenderer::shade()

void SIMDDebugRenderer
  ::deposit(const size_t batchIdx,
            const size_t threadIdx,
            const Spectrum *results)
{
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size2 imageDim(film->getWidth(),film->getHeight());
  size_t i = batchIdx * mWorkBatchSize + threadIdx;


  // convert thread index to pixel index
  size2 pixelIndex(i % imageDim[0],
                   i / imageDim[0]);
  film->deposit(pixelIndex[0],pixelIndex[1], results[i]);
} // end SIMDDebugRenderer::deposit()

void SIMDDebugRenderer
  ::kernel(ProgressCallback &progress)
{
  // XXX TODO: kill this
  // compute the total work
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size_t totalWork = film->getWidth() * film->getHeight();

  // compute the total number of batches we need to complete the work
  size_t numBatches = totalWork / mWorkBatchSize;

  // allocate per-thread data
  std::vector<Ray> rays(totalWork);
  std::vector<float> pdfs(totalWork);
  std::vector<Intersection> intersections(totalWork);

  // init stencil to 1
  std::vector<int> stencil(totalWork, 1);

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
    ScatteringDistributionFunction::mPool.freeAll();

    progress += mWorkBatchSize;
  } // end for i
  for(size_t j = 0; j != totalWork % mWorkBatchSize; ++j)
    sampleEyeRay(numBatches, j, &rays[0], &pdfs[0]);

  // purge all malloc'd memory for the last partial batch
  ScatteringDistributionFunction::mPool.freeAll();
  progress += totalWork % mWorkBatchSize;

  intersect(&rays[0], &intersections[0], &stencil[0]);
  progress += totalWork;

  for(size_t i = 0; i != numBatches; ++i)
  {
    for(size_t j = 0; j != mWorkBatchSize; ++j)
    {
      shade(i,j, &rays[0], &pdfs[0], &intersections[0], &stencil[0], &results[0]);
    } // end for j

    // purge all malloc'd memory for this batch
    ScatteringDistributionFunction::mPool.freeAll();

    progress += mWorkBatchSize;
  } // end for i

  for(size_t j = 0; j != totalWork % mWorkBatchSize; ++j)
  {
    shade(numBatches, j, &rays[0], &pdfs[0], &intersections[0], &stencil[0], &results[0]);
  } // end for j
  progress += totalWork % mWorkBatchSize;

  // purge all malloc'd memory for the last partial batch
  ScatteringDistributionFunction::mPool.freeAll();

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
} // end SIMDDebugRenderer::kernel()

void SIMDDebugRenderer
  ::intersect(Ray *rays,
              Intersection *intersections,
              int *stencil)
{
  // XXX TODO: kill this
  // compute the total work
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size_t totalWork = film->getWidth() * film->getHeight();

  // compute the total number of batches we need to complete the work
  size_t numBatches = totalWork / mWorkBatchSize;

  Ray *r = rays;
  Intersection *inter = intersections;
  int *s = stencil;
  Ray *end = rays + numBatches * mWorkBatchSize;
  for(;
      r != end;
      r += mWorkBatchSize,
      inter += mWorkBatchSize,
      s += mWorkBatchSize)
  {
    mScene->intersect(r, inter, s, mWorkBatchSize);
  } // end for i

  // do the last partial batch
  mScene->intersect(r, inter, s, totalWork % mWorkBatchSize);
} // end SIMDDebugRenderer::intersect()

