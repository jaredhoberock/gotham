/*! \file LuminanceOverVisits.h
 *  \author Jared Hoberoc
 *  \brief Defines the interface to a ScalarImportance class whose
 *         importance is proportional to LuminanceImportance divided by
 *         EqualVisitImportance.
 */

#ifndef LUMINANCE_OVER_VISITS_H
#define LUMINANCE_OVER_VISITS_H

#include "ScalarImportance.h"
#include "LuminanceImportance.h"
#include "EqualVisitImportance.h"

class LuminanceOverVisits
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! Constructor accepts whether or not to do interpolation
     *  when looking up visits.
     *  \param doInterpolate Sets mDoInterpolate.
     */
    LuminanceOverVisits(const bool doInterpolate);

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

    /*! This method assigns a scalar importance to a Path.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     */
    float evaluate(const PathSampler::HyperPoint &x,
                   const Path &xPath,
                   const std::vector<PathSampler::Result> &results);

  protected:
    const RenderFilm *mAcceptance;
    LuminanceImportance mLuminanceImportance;
    bool mDoInterpolate;
}; // end LuminanceOverVisits

#endif // LUMINANCE_OVER_VISITS_H

