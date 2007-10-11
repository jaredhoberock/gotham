/*! \file MultipleImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which is composed of a number of different
 *         ScalarImportance strategies.
 */

#ifndef MULTIPLE_IMPORTANCE_H
#define MULTIPLE_IMPORTANCE_H

#include "ScalarImportance.h"

class MultipleImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! Null destructor deletes the list of ScalarImportance strategies.
     */
    virtual ~MultipleImportance(void);

    /*! This method is called prior to rendering.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param renderer A reference to the MetropolisRenderer owning this
     *                  ScalarImportance.
     */
    virtual void preprocess(const boost::shared_ptr<RandomSequence> &r,
                            const boost::shared_ptr<const Scene> &scene,
                            const boost::shared_ptr<PathMutator> &mutator,
                            MetropolisRenderer &renderer);

    /*! This method assigns a scalar importance to a Path by choosing randomly
     *  from the list of ScalarImportance functions.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

  protected:
    /*! A RandomSequence to use for choosing among ScalarImportance strategies.
     */
    boost::shared_ptr<RandomSequence> mRandomSequence;

    /*! A list of ScalarImportance strategies.
     */
    std::vector<ScalarImportance*> mStrategies;
}; // end MultipleImportance

#endif // MULTIPLE_IMPORTANCE_H

