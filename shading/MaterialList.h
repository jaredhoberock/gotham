/*! \file MaterialList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a list of Materials.
 */

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>
#include "Material.h"
#include "MaterialHandle.h"

class MaterialList
  : public std::vector<boost::shared_ptr<Material> >
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef std::vector<boost::shared_ptr<Material> > Parent;

    /*! Null destructor does nothing.
     */
    virtual ~MaterialList(void);

    /*! This method evaluates the scattering of a list of DifferentialGeometry
     *  and MaterialHandles.
     *  \param dg A list of DifferentialGeometries.
     *  \param handles A list of MaterialHandles.
     *  \param stencil This stencil controls which elements of the list get processed.
     *  \param f Newly allocated ScatteringDistributionFunctions are
     *           returned to this list corresponding to each
     *           (DifferentialGeometry, MaterialHandle) pair.
     *  \param n The size of each of the lists.
     */
    virtual void evaluateScattering(const DifferentialGeometry *dg,
                                    const MaterialHandle *handles,
                                    const int *stencil,
                                    ScatteringDistributionFunction **f,
                                    const size_t n) const;

    /*! This method evaluates the emission of a list of DifferentialGeometry and MaterialHandles.
     *  \param dg A list of DifferentialGeometries.
     *  \param handles A list of MaterialHandles.
     *  \param stencil This stencil controls which elements of the list get processed.
     *  \param f Newly allocated ScatteringDistributionFunctions are
     *           returned to this list corresponding to each
     *           (DifferentialGeometry, MaterialHandle) pair.
     *  \param n The size of each of the lists.
     */
    virtual void evaluateEmission(const DifferentialGeometry *dg,
                                  const MaterialHandle *handles,
                                  const int *stencil,
                                  ScatteringDistributionFunction **f,
                                  const size_t n) const;
}; // end MaterialList

