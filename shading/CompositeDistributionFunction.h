/*! \file CompositeDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringDistributionFunction
 *         which is composed of a sum of simpler ScatteringDistributionFunctions.
 */

#ifndef COMPOSITE_DISTRIBUTION_FUNCTION_H
#define COMPOSITE_DISTRIBUTION_FUNCTION_H

#include "ScatteringDistributionFunction.h"
#include <boost/array.hpp>

#define MAX_COMPONENTS 5

class CompositeDistributionFunction
  : public ScatteringDistributionFunction,
    protected boost::array<ScatteringDistributionFunction*,MAX_COMPONENTS>
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent0;

    /*! Null constructor sets mSize to 0 and calls the Parents.
     */
    inline CompositeDistributionFunction(void);

    /*! This method evaluates this CompositeDistributionFunction
     *  for a pair of directions.
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi A vector pointing towards the incoming direction.
     *  \return The bidirectional scattering toward direction wo from wi.
     */
    using Parent0::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method samples this CompositeDistributionFunction given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true if wi was sampled from a delta
     *               distribution; false, otherwise.
     *  \param component This is set to the index of the component which generated
     *         wi.
     *  \return The bidirectional scattering from wi to wo is returned here.
     */
    using Parent0::sample;
    virtual Spectrum sample(const Vector &wo,
                            const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector &wi,
                            float &pdf,
                            bool &delta,
                            ComponentIndex &component) const;

    /*! This method adds a new ScatteringDistributionFunction to this
     *  CompositeDistributionFunction.
     *  \param rhs A pointer to the ScatteringDistributionFunction to add.
     *  \return *this
     */
    inline CompositeDistributionFunction &operator+=(ScatteringDistributionFunction *rhs);

    /*! This method adds a new ScatteringDistributionFunction to this
     *  CompositeDistributionFunction.
     *  \param rhs A reference to the ScatteringDistributionFunction to add.
     *  \return *this
     */
    inline CompositeDistributionFunction &operator+=(ScatteringDistributionFunction &rhs);

    /*! This method clones this CompositeDistributionFunction.
     *  \param allocator The FunctionAllocator to allocate from.
     *  \return a pointer to a newly-allocated clone of this CompositeDistributionFunction;
     *          0, if no memory could be allocated.
     */
    virtual ScatteringDistributionFunction *clone(FunctionAllocator &allocator) const;

    /*! This method returns true if each of its components is specular; false, otherwise.
     *  \return As above.
     */
    virtual bool isSpecular(void) const;

    using Parent0::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component) const;

    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

  protected:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef boost::array<ScatteringDistributionFunction*,MAX_COMPONENTS> Parent1;

    /*! The number of ScatteringDistributionFunctions carried by this
     *  CompositeDistributionFunction.
     */
    size_t mSize;
}; // end CompositeDistributionFunction

#include "CompositeDistributionFunction.inl"

#endif // COMPOSITE_DISTRIBUTION_FUNCTION_H

