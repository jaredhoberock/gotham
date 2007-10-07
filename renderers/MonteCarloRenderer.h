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
     *  \param f Sets Parent::mFilm.
     *  \param sequence Sets mRandomSequence.
     */
    inline MonteCarloRenderer(boost::shared_ptr<const Scene> &s,
                              boost::shared_ptr<RenderFilm> &f,
                              const boost::shared_ptr<RandomSequence> &sequence);

    /*! This method sets mRandomSequence.
     *  \param s Sets mRandomSequence.
     */
    virtual void setRandomSequence(const boost::shared_ptr<RandomSequence> &s);

  protected:
    /*! A sequence of pseudo-random numbers.
     */
    boost::shared_ptr<RandomSequence> mRandomSequence;
}; // end MonteCarloRenderer

#include "MonteCarloRenderer.inl"

#endif // MONTE_CARLO_RENDERER_H

