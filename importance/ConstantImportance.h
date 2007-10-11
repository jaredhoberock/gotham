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

    /*! This method assigns a constant
     *  scalar importance of 1.0 to each Path.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The sum of 1 over the area product pdfs of Results in results.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);
}; // end ConstantImportance

#endif // CONSTANT_IMPORTANCE_H

