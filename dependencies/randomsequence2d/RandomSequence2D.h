/*! \file RandomSequence2D.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a random sequence of
 *         points in 2D.
 */

#ifndef RANDOM_SEQUENCE_2D_H
#define RANDOM_SEQUENCE_2D_H

class RandomSequence2D
{
  public:
    /*! Null constructor does nothing.
     */
    inline RandomSequence2D(void);

    inline RandomSequence2D(const float xStart, const float xEnd,
                            const float yStart, const float yEnd);

    /*! Null destructor does nothing.
     */
    inline virtual ~RandomSequence2D(void);

    inline virtual void reset(const float xStart, const float xEnd,
                              const float yStart, const float yEnd);

    inline virtual bool operator()(float &x, float &y,
                                   float z0, float z1);

  protected:
    float mXStart, mDeltaX;
    float mYStart, mDeltaY;
}; // end RandomSequence2D

#include "RandomSequence2D.inl"

#endif // RANDOM_SEQUENCE_2D_H

