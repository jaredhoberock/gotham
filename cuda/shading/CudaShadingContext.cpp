/*! \file CudaShadingContext.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaShadingContext class.
 */

#include "CudaShadingContext.h"
#include "CudaScatteringDistributionFunction.h"
#include "CudaLambertian.h"
#include "CudaHemisphericalEmission.h"
#include "../include/CudaMaterial.h"
#include "CudaDefaultMaterial.h"
#include "../../shading/Lambertian.h"
#include "../../shading/HemisphericalEmission.h"
#include "evaluateBidirectionalScattering.h"
#include "evaluateUnidirectionalScattering.h"
#include <stdcuda/cuda_algorithm.h>
#include <vector_functions.h>

using namespace stdcuda;

void CudaShadingContext
  ::setMaterials(const boost::shared_ptr<MaterialList> &materials)
{
  mMaterials.reset(new MaterialList());

  // go through the Materials; only add one if it is a CudaMaterial
  // if it is not, create a CudaMaterial in its stead
  for(MaterialList::iterator m = materials->begin();
      m != materials->end();
      ++m)
  {
    if(dynamic_cast<CudaMaterial*>(m->get()) || (*m)->isSensor() || (*m)->isEmitter())
    {
      mMaterials->push_back(*m);
    } // end if
    else
    {
      mMaterials->push_back(boost::shared_ptr<Material>(new CudaDefaultMaterial()));
    } // end else
  } // end for i
} // end CudaShadingContext::setMaterials()

