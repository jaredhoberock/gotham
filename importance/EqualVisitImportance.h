/*! \file EqualVisitImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class that attempts to force the Metropolis
 *         process to visit each pixel the same number
 *         of times.
 */

#ifndef EQUAL_VISIT_IMPORTANCE_H
#define EQUAL_VISIT_IMPORTANCE_H

#include "ScalarImportance.h"
#include "ConstantImportance.h"
class RenderFilm;

class EqualVisitImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! Constructor accepts whether or not to do interpolation
     *  when looking up visits.
     *  \param doInterpolate Sets mDoInterpolate
     */
    EqualVisitImportance(const bool doInterpolate);

    /*! This method is called prior to rendering.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param renderer A reference to the MetropolisRenderer owning this
     *                  EqualVisitImportance.
     */
    virtual void preprocess(const boost::shared_ptr<RandomSequence> &r,
                            const boost::shared_ptr<const Scene> &scene,
                            const boost::shared_ptr<PathMutator> &mutator,
                            MetropolisRenderer &renderer);

    /*! This method assigns a scalar importance to a Path.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

  protected:
    const RenderFilm *mAcceptance;
    ConstantImportance mConstantImportance;
    bool mDoInterpolate;
}; // end EqualVisitImportance

#endif // EQUAL_VISIT_IMPORTANCE_H

