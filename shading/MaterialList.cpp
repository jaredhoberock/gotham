/*! \file MaterialList.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MaterialList class.
 */

#include "MaterialList.h"

MaterialList
  ::~MaterialList(void)
{
  ;
} // end MaterialList::~MaterialList()

void MaterialList
  ::evaluateScattering(const DifferentialGeometry *dg,
                       const MaterialHandle *handles,
                       const int *stencil,
                       ScatteringDistributionFunction **f,
                       const size_t n) const
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      MaterialHandle h = handles[i];
      f[i] = (*this)[h]->evaluateScattering(dg[i]);
    } // end if
  } // end for i
} // end MaterialList::evaluateScattering()

void MaterialList
  ::evaluateEmission(const DifferentialGeometry *dg,
                     const MaterialHandle *handles,
                     const int *stencil,
                     ScatteringDistributionFunction **f,
                     const size_t n) const
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      MaterialHandle h = handles[i];
      f[i] = (*this)[h]->evaluateEmission(dg[i]);
    } // end if
  } // end for i
} // end MaterialList::evaluateEmission()

