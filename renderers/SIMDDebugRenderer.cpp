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
  ::kernel(const size_t threadIdx)
{
  // XXX TODO: kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  size2 imageDim(film->getWidth(),film->getHeight());

  // convert thread index to pixel index
  size2 pixelIndex(threadIdx % imageDim[0],
                   threadIdx / imageDim[0]);

  // convert pixel index to location in unit square
  float2 uv(static_cast<float>(pixelIndex[0]) / imageDim[0],
            static_cast<float>(pixelIndex[1]) / imageDim[1]); 

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float pdf;
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
  Ray r = Ray(dgSensor.getPoint(), d);

  Spectrum L(0.0f,0.0f,0.0f);

  // intersect the Scene
  Primitive::Intersection inter;
  if(mScene->intersect(r, inter))
  {
    const Primitive *prim = inter.getPrimitive();

    // evaluate scattering
    ScatteringDistributionFunction *f = static_cast<const SurfacePrimitive*>(prim)->getMaterial()->evaluateScattering(inter.getDifferentialGeometry());
    L = f->evaluate(-d,inter.getDifferentialGeometry(),-d);

    // add emission
    ScatteringDistributionFunction *e = static_cast<const SurfacePrimitive*>(prim)->getMaterial()->evaluateEmission(inter.getDifferentialGeometry());
    L += e->evaluate(-d, inter.getDifferentialGeometry());
  } // end if

  film->deposit(pixelIndex[0],pixelIndex[1],L);
} // end SIMDDebugRenderer::kernel()

