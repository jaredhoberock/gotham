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
#include "../primitives/PrimitiveList.h"
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
  const MaterialList &materials = mScene->getMaterials();
  ScatteringDistributionFunction &s = *materials[sensor->getMaterial()]->evaluateSensor(dgSensor);

  // sample a sensing direction
  bool delta;
  ::Vector d;
  s.sample(dgSensor, uv[0], uv[1], 0.5f, d, pdf, delta);
  rays[i] = Ray(dgSensor.getPoint(), d);
} // end SIMDDebugRenderer::sampleEyeRay()

void SIMDDebugRenderer
  ::evaluate(ScatteringDistributionFunction **f,
             const ::Vector *wo,
             const DifferentialGeometry *dg,
             const ::Vector *wi,
             const int *stencil,
             Spectrum *results,
             const size_t n) const
{
  // evaluate the functions into results
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      // bidirectional scattering
      results[i] = f[i]->evaluate(wo[i], dg[i], wi[i]);
    } // end if
  } // end for
} // end SIMDDebugRenderer::evaluate()

void SIMDDebugRenderer
  ::evaluate(ScatteringDistributionFunction **f,
             const ::Vector *wo,
             const DifferentialGeometry *dg,
             const int *stencil,
             Spectrum *results,
             const size_t n) const
{
  // evaluate the functions into results
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      // unidirectional scattering
      results[i] = f[i]->evaluate(wo[i], dg[i]);
    } // end if
  } // end for
} // end SIMDDebugRenderer::evaluate()
                

void SIMDDebugRenderer
  ::shade(const Ray *rays,
          const float *pdfs,
          const Intersection *intersections,
          const int *stencil,
          Spectrum *results,
          const size_t n) const
{
  std::vector<ScatteringDistributionFunction*> f(n);
  std::vector<ScatteringDistributionFunction*> e(n);

  const PrimitiveList<> &primitives = *mScene->getPrimitives();
  const MaterialList &materials = mScene->getMaterials();

  // create the list of MaterialHandles and DifferentialGeometry
  std::vector<MaterialHandle> handles(n);
  std::vector<DifferentialGeometry> dg(n);
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      PrimitiveHandle ph = intersections[i].getPrimitive();
      const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>(primitives[ph].get());
      handles[i] = sp->getMaterial();
      dg[i] = intersections[i].getDifferentialGeometry();
    } // end if
  } // end for i

  // evaluate scattering
  materials.evaluateScattering(&dg[0], &handles[0], stencil, &f[0], n);

  // evaluate emission
  materials.evaluateEmission(&dg[0], &handles[0], stencil, &e[0], n);

  std::vector< ::Vector> wo(n);
  std::vector< ::Vector> wi(n);
  for(size_t i = 0; i != n; ++i)
  {
    wo[i] = -rays[i].getDirection();
    wi[i] = -rays[i].getDirection();
  } // end for i

  // evaluate scattering
  evaluate(&f[0], &wo[0], &dg[0], &wi[0], stencil, &results[0], n);

  // evaluate emission
  std::vector<Spectrum> emission(n, Spectrum::black());
  evaluate(&e[0], &wo[0], &dg[0], stencil, &emission[0], n );

  // add in emission
  for(size_t i = 0; i != n; ++i)
  {
    results[i] += emission[i];
  } // end for i
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

  // intersect en masse
  intersect(&rays[0], &intersections[0], &stencil[0]);
  progress += totalWork;

  for(size_t i = 0; i != numBatches; ++i)
  {
    shade(&rays[i * mWorkBatchSize],
          &pdfs[i * mWorkBatchSize],
          &intersections[i * mWorkBatchSize],
          &stencil[i * mWorkBatchSize],
          &results[i * mWorkBatchSize],
          mWorkBatchSize);

    // purge all malloc'd memory for this batch
    ScatteringDistributionFunction::mPool.freeAll();

    progress += mWorkBatchSize;
  } // end for i

  // shade the last partial batch
  shade(&rays[numBatches * mWorkBatchSize],
        &pdfs[numBatches * mWorkBatchSize],
        &intersections[numBatches * mWorkBatchSize],
        &stencil[numBatches * mWorkBatchSize],
        &results[numBatches * mWorkBatchSize],
        totalWork % mWorkBatchSize);
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

