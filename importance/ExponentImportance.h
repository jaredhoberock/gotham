/*! \file ExponentImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which assigns importance equal to luminance
 *         raised to an exponent.
 */

#ifndef EXPONENT_IMPORTANCE_H
#define EXPONENT_IMPORTANCE_H

#include "LuminanceImportance.h"

class ExponentImportance
  : public LuminanceImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef LuminanceImportance Parent;

    /*! Constructor accepts an exponent.
     *  \param k Sets mExponent.
     */
    ExponentImportance(const float k);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance by returning its luminance raised to mExponent.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

  protected:
    /*! The exponent.
     */
    float mExponent;
}; // end ExponentImportance

#endif // EXPONENT_IMPORTANCE_H

