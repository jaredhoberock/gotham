/*! \file MetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Renderer
 *         implementing Metropolis light transport.
 */

#ifndef METROPOLIS_RENDERER_H
#define METROPOLIS_RENDERER_H

#include "MonteCarloRenderer.h"
#include <boost/shared_ptr.hpp>
#include "../importance/ScalarImportance.h"
#include "../mutators/PathMutator.h"
#include "../shading/FunctionAllocator.h"
#include "../records/RenderFilm.h"
#include "../records/MipMappedRenderFilm.h"

class MetropolisRenderer
  : public MonteCarloRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MonteCarloRenderer Parent;

    /*! Null constructor does nothing.
     */
    MetropolisRenderer(void);

    /*! Constructor accepts a PathMutator and calls the
     *  null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     */
    MetropolisRenderer(const boost::shared_ptr<RandomSequence> &s,
                       const boost::shared_ptr<PathMutator> &mutator,
                       const boost::shared_ptr<ScalarImportance> &importance);

    /*! Constructor accepts a pointer to a Scene, Record, and PathMutator.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     */
    MetropolisRenderer(boost::shared_ptr<const Scene> &s,
                       boost::shared_ptr<Record> &r,
                       const boost::shared_ptr<RandomSequence> &sequence,
                       const boost::shared_ptr<PathMutator> &m,
                       const boost::shared_ptr<ScalarImportance> &i);

    /*! This method sets mMutator.
     *  \param m Sets mMutator.
     */
    virtual void setMutator(boost::shared_ptr<PathMutator> &m);

    /*! This method sets mImportance.
     *  \param i Sets mImportance.
     */
    virtual void setImportance(boost::shared_ptr<ScalarImportance> &i);

    /*! This method sets Renderer::mScene and calls mMutator.setScene(s).
     *  \param s Sets mScene.
     */
    virtual void setScene(const boost::shared_ptr<const Scene> &s);

    /*! This method calls the Parent's method and also hands the
     *  RandomSequence to mMutator.
     *  \param s Sets mRandomSequence.
     */
    virtual void setRandomSequence(const boost::shared_ptr<RandomSequence> &s);

    /*! This method returns a pointer to mAcceptanceImage.
     *  \return &mAcceptanceImage.
     */
    RenderFilm *getAcceptanceImage(void);

    /*! This method sets the filename of mAcceptanceImage.
     *  \param filename The name of the file to write mAcceptanceImage to.
     */
    void setAcceptanceFilename(const std::string &filename);

    /*! This method sets the filename of mProposalImage.
     *  \param filename The name of the file to write mProposalImage to.
     */
    void setProposalFilename(const std::string &filename);

  protected:
    /*! This method safely copies a Path by cloning its integrands into
     *  mLocalPool.
     *  \param dst The destination Path.
     *  \param src The source Path.
     */
    void copyPath(Path &dst, const Path &src);

    /*! This method coordinates preprocessing tasks prior to rendering.
     */
    virtual void preprocess(void);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method calls the Parent and writes mAcceptanceImage.
     */
    virtual void postprocess(void);

    /*! This method reports render statistics.
     *  \param elapsed The length of the render, in seconds.
     */
    virtual void postRenderReport(const double elapsed) const;

    boost::shared_ptr<PathMutator> mMutator;

    boost::shared_ptr<ScalarImportance> mImportance;

    FunctionAllocator mLocalPool;

    /*! A count of the number of accepted proposals.
     */
    unsigned long mNumAccepted;

    /*! An image of the acceptance rate.
     */
    RenderFilm mAcceptanceImage;

    /*! An image of the proposal rate.
     */
    RenderFilm mProposalImage;
}; // end MetropolisRenderer

#endif // METROPOLIS_RENDERER_H

