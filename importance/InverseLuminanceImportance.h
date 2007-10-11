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

    /*! This method assigns a constant
     *  scalar importance of 1.0 to each point.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);
}; // end InverseLuminanceImportance

#endif // INVERSE_LUMINANCE_IMPORTANCE_H

