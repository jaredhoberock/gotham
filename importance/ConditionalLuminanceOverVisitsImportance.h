/*! \file ConditionalLuminanceOverVisitsImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance class
 *         whose importance is conditional on the other Path
 *         being considered.
 */

#ifndef CONDITIONAL_LUMINANCE_OVER_VISTS_IMPORTANCE_H
#define CONDITIONAL_LUMINANCE_OVER_VISTS_IMPORTANCE_H

#include "LuminanceOverVisits.h"

class ConditionalLuminanceOverVisitsImportance
  : public LuminanceOverVisits
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef LuminanceOverVisits Parent;

    /*! Constructor accepts whether or not to do interpolation
     *  when looking up visits.
     *  \param doInterpolate Sets mDoInterpolate.
     */
    ConditionalLuminanceOverVisitsImportance(const bool doInterpolate);

    /*! This method assigns scalar importance to Path x given knowledge
     *  of a "competing" Path, y.
     *  \param x The HyperPoint uniquely specifying xPath.
     *  \param xPath The Path of interest.
     *  \param xResults The PathSampler Results resulting from xPath.
     *  \param y The HyperPoint uniquely specifying yPath.
     *  \param yPath The Path "competing" with xPath.
     *  \param yResults The PathSampler Results resulting from yPath.
     *  \return The scalar importance of x, given y.
     */
    float evaluate(const PathSampler::HyperPoint &x,
                   const Path &xPath,
                   const std::vector<PathSampler::Result> &xResults,
                   const PathSampler::HyperPoint &y,
                   const Path &yPath,
                   const std::vector<PathSampler::Result> &yResults);
}; // end ConditionalLuminanceOverVisitsImportance

#endif // CONDITIONAL_LUMINANCE_OVER_VISTS_IMPORTANCE_H

