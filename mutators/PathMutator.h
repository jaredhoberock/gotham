/*! \file PathMutator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         for mutating light Paths.
 */

#ifndef PATH_MUTATOR_H
#define PATH_MUTATOR_H

#include "../path/PathSampler.h"
#include "../numeric/RandomSequence.h"
#include "../primitives/Scene.h"
#include <boost/shared_ptr.hpp>
#include "../path/Path.h"

class PathMutator
{
  public:
    /*! Null constructor does nothing.
     */
    PathMutator(void);

    /*! Constructor accepts a RandomSequence.
     *  \param s Sets mRandomSequence.
     *  \param sampler Sets mSampler.
     */
    PathMutator(const boost::shared_ptr<RandomSequence> &s,
                const boost::shared_ptr<PathSampler> &sampler);

    /*! Constructor accepts a RandomSequence.
     *  \param sampler Sets mSampler.
     */
    PathMutator(const boost::shared_ptr<PathSampler> &sampler);

    /*! Null destructor does nothing.
     */
    inline virtual ~PathMutator(void){;};

    int operator()(const PathSampler::HyperPoint &x,
                   const Path &a,
                   PathSampler::HyperPoint &y,
                   Path &b);

    virtual int mutate(const PathSampler::HyperPoint &x,
                       const Path &a,
                       PathSampler::HyperPoint &y,
                       Path &b) = 0;

    virtual float evaluateTransitionRatio(const unsigned int which,
                                          const PathSampler::HyperPoint &x,
                                          const Path &a,
                                          const float ix,
                                          const PathSampler::HyperPoint &y,
                                          const Path &b,
                                          const float iy) const = 0;

    virtual Spectrum evaluate(const Path &x,
                              std::vector<PathSampler::Result> &results) = 0;

    /*! This method sets mRandomSequence.
     *  \param s Sets mRandomSequence.
     */
    void setRandomSequence(const boost::shared_ptr<RandomSequence> &s);

    /*! This method sets mScene.
     *  \param s Sets mScene.
     */
    void setScene(const boost::shared_ptr<const Scene> &s);

    /*! This method sets mSampler.
     *  \param s Sets mSampler.
     */
    void setSampler(const boost::shared_ptr<PathSampler> &s);

    /*! This method returns a raw const pointer to mSampler.
     *  \return mSampler.get()
     */
    const PathSampler *getSampler(void) const;

    /*! This method is called prior to rendering.
     *  \note The default implementation does nothing.
     */
    virtual void preprocess(void);

    /*! This method is called after rendering.
     *  \note The default implementation does nothing.
     */
    virtual void postprocess(void);

  protected:
    boost::shared_ptr<RandomSequence> mRandomSequence;

    boost::shared_ptr<const Scene> mScene;

    /*! A PathSampler for generating Paths.
     */
    boost::shared_ptr<PathSampler> mSampler;
}; // end PathMutator

#endif // PATH_MUTATOR_H

