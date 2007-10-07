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
class RenderFilm;

class EqualVisitImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

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

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param f The spectral Monte Carlo throughput of the Path of interest.
     *  \return The scalar importance of x.
     */
    virtual float operator()(const PathSampler::HyperPoint &x,
                             const Spectrum &f);

  protected:
    const RenderFilm *mAcceptance;
}; // end EqualVisitImportance

#endif // EQUAL_VISIT_IMPORTANCE_H

