/*! \file PathDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MonteCarloRenderer 
 *         which solves the light transport by sampling in the
 *         space of Paths.
 */

#ifndef PATH_DEBUG_RENDERER_H
#define PATH_DEBUG_RENDERER_H

#include "MonteCarloRenderer.h"
#include <boost/shared_ptr.hpp>
#include "../path/PathSampler.h"
#include "../films/RandomAccessFilm.h"

class PathDebugRenderer
  : public MonteCarloRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MonteCarloRenderer Parent;

    /*! Null constructor does nothing.
     */
    PathDebugRenderer(void);

    /*! Constructor accepts a RandomSequence and a PathSampler
     *  and calls the null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param sampler Sets mSampler.
     */
    PathDebugRenderer(const boost::shared_ptr<RandomSequence> &s,
                      const boost::shared_ptr<PathSampler> &sampler);

    /*! Constructor accepts a pointer to a Scene, Film, and PathSampler.
     *  \param s Sets mScene.
     *  \param f Sets mFilm.
     *  \param sequence Sets Parent::mRandomSequence.
     *  \param sampler Sets mSampler.
     */
    PathDebugRenderer(boost::shared_ptr<const Scene>  s,
                      boost::shared_ptr<RenderFilm> f,
                      const boost::shared_ptr<RandomSequence> &sequence,
                      const boost::shared_ptr<PathSampler> &sampler);

    /*! This method sets mSampler.
     *  \param s Sets mSampler.
     */
    virtual void setSampler(const boost::shared_ptr<PathSampler> &s);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    virtual void preprocess(void);
    virtual void postprocess(void);

    boost::shared_ptr<PathSampler> mSampler;

    RandomAccessFilm mSquaredEstimate;
}; // end PathDebugRenderer

#endif // PATH_DEBUG_RENDERER_H

