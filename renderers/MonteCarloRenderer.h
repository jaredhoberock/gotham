/*! \file MonteCarloRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Renderer
 *         which uses a sequence of pseudo-random
 *         numbers.
 */

#ifndef MONTE_CARLO_RENDERER_H
#define MONTE_CARLO_RENDERER_H

#include "Renderer.h"
#include "../numeric/RandomSequence.h"

class MonteCarloRenderer
  : public Renderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Renderer Parent;

    /*! Null constructor calls the Parent.
     */
    inline MonteCarloRenderer(void);

    /*! Constructor accepts a RandomSequence.
     *  \param s Sets mRandomSequence.
     */
    inline MonteCarloRenderer(const boost::shared_ptr<RandomSequence> &s);

    /*! Constructor accepts a RandomSequence
     *  and calls the Parent.
     *  \param s Sets Parent::mScene.
     *  \param r Sets Parent::mRecord.
     *  \param sequence Sets mRandomSequence.
     */
    inline MonteCarloRenderer(boost::shared_ptr<const Scene> &s,
                              boost::shared_ptr<Record> &r,
                              const boost::shared_ptr<RandomSequence> &sequence);

    /*! This method sets mRandomSequence.
     *  \param s Sets mRandomSequence.
     */
    virtual void setRandomSequence(const boost::shared_ptr<RandomSequence> &s);

    /*! This method intializes mNumSamples to 0.
     */
    virtual void preprocess(void);

    /*! This method scales mRecord by 1 / samples per pixel.
     */
    virtual void postprocess(void);

    /*! This method returns mNumSamples.
     *  \return mNumSamples
     */
    unsigned long getNumSamples(void) const;

  protected:
    /*! A sequence of pseudo-random numbers.
     */
    boost::shared_ptr<RandomSequence> mRandomSequence;

    /*! This counts the total number of samples
     *  generated during kernel().
     */
    unsigned long mNumSamples; 
}; // end MonteCarloRenderer

#include "MonteCarloRenderer.inl"

#endif // MONTE_CARLO_RENDERER_H

