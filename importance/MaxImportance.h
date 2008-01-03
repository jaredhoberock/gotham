/*! \file MaxImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which assigns importance equal to the max over
 *         the Path's throughput's Spectrum.
 */

#ifndef MAX_IMPORTANCE_H
#define MAX_IMPORTANCE_H

#include "ScalarImportance.h"

class MaxImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

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

    /*! This method converts the spectral Monte Carlo throughput
     *  of a PathSampler::Result into scalar importance by returning its luminance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param r The Result of interest.
     *  \return The scalar importance of r.
     */
    static float evaluateImportance(const PathSampler::HyperPoint &x,
                                    const Path &xPath,
                                    const PathSampler::Result &r);

    static float evaluateImportance(const PathSampler::HyperPoint &x,
                                    const Path &xPath,
                                    const std::vector<PathSampler::Result> &results);
}; // end MaxImportance

#endif // MAX_IMPORTANCE_H

