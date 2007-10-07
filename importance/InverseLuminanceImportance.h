/*! \file InverseLuminanceImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance class
 *         which assigns importance inversely proportional
 *         to luminance.
 */

#ifndef INVERSE_LUMINANCE_IMPORTANCE_H
#define INVERSE_LUMINANCE_IMPORTANCE_H

#include "ScalarImportance.h"

class InverseLuminanceImportance
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
     *  \return 1.0 / f.luminance()
     */
    virtual float operator()(const PathSampler::HyperPoint &x,
                             const Spectrum &f);
}; // end InverseLuminanceImportance

#endif // INVERSE_LUMINANCE_IMPORTANCE_H

