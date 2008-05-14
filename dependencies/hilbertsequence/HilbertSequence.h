/*! \file HilbertSequence.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         which outputs a hilbert-ordered sequence
 *         of points in a 2D domain.
 */

#ifndef HILBERT_SEQUENCE_H
#define HILBERT_SEQUENCE_H

#include <stratifiedsequence/StratifiedSequence.h>
#include <hilbertwalk/HilbertWalk.h>
#include <vector>
#include <gpcpu/Vector.h>

class HilbertSequence
  : public StratifiedSequence
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef StratifiedSequence Parent;

    /*! Null constructor does nothing.
     */
    inline HilbertSequence(void);

    inline HilbertSequence(const float xStart, const float xEnd,
                           const float yStart, const float yEnd,
                           const size_t xStrata,
                           const size_t yStrata);

    /*! This method resets this HilbertSequence.
     *  \param xStart The beginning of the sequence of x-coordinates.
     *  \param xEnd The end of the sequence of x-coordinates.
     *  \param yStart The beginning of the sequence of y-coordinates.
     *  \param yEnd The end of the sequence of y-coordinates.
     *  \param xStrata The number of strata in the x dimension.
     *  \param yStrata The number of strata in the y dimension.
     */
    inline void reset(const float xStart, const float xEnd,
                      const float yStart, const float yEnd,
                      const size_t xStrata,
                      const size_t yStrata);

    /*! This method resets the current position of this StratifiedSequence
     *  without modifying any of the parameters of the sequence.
     */
    inline virtual void reset(void);

  protected:
    inline virtual bool advance(void);

    /*! A stack of rectangular blocks remaining to be
     *  traversed.
     */
    std::vector<gpcpu::size4> mBlockStack;

    /*! A HilbertWalk to control the order of the sequence.
     */
    HilbertWalk mWalk;
}; // end HilbertSequence

#include "HilbertSequence.inl"

#endif // HILBERT_SEQUENCE_H

