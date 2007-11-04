/*! \file ManualImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which assigns importance proportional
 *         to luminance times a manual coefficient.
 */

#ifndef MANUAL_IMPORTANCE_H
#define MANUAL_IMPORTANCE_H

#include "LuminanceImportance.h"

class ManualImportance
  : public LuminanceImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef LuminanceImportance Parent;

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance by returning its luminance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);
}; // end ManualImportance

#endif // MANUAL_IMPORTANCE_H

