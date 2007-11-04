/*! \file HierarchicalLuminanceOverVisits.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which acts like LuminanceOverVisits but
 *         uses hierarchical image information.
 */

#ifndef HIERARCHICAL_LUMINANCE_OVER_VISITS_H
#define HIERARCHICAL_LUMINANCE_OVER_VISITS_H

#include "ConditionalLuminanceOverVisitsImportance.h"

class HierarchicalLuminanceOverVisits
  : public ConditionalLuminanceOverVisitsImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ConditionalLuminanceOverVisitsImportance Parent;

    /*! Constructor accepts whether or not to do interpolation
     *  when looking up visits.
     *  \param doInterpolate Sets mDoInterpolate.
     *  \param radius The search radius to use.
     */
    HierarchicalLuminanceOverVisits(const bool doInterpolate,
                                    const float radius);

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

  protected:
    float mRadius;
}; // end HierarchicalLuminanceOverVisits

#endif // HIERARCHICAL_LUMINANCE_OVER_VISITS_H

