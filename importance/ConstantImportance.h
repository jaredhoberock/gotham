/*! \file ConstantImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance class
 *         which assigns a constant importance to all points
 *         in the domain.
 */

#ifndef CONSTANT_IMPORTANCE_H
#define CONSTANT_IMPORTANCE_H

#include "ScalarImportance.h"

class ConstantImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! operator()() method assigns a constant
     *  scalar importance of 1.0 to each point.
     *  \param x The HyperPoint of interest.
     *  \param f The value of the integrand at x.
     *  \return 1.0f
     */
    virtual float operator()(const PathSampler::HyperPoint &x,
                             const Spectrum &f);
}; // end ConstantImportance

#endif // CONSTANT_IMPORTANCE_H

