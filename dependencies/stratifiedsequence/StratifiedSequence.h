/*! \file StratifiedSequence.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a sequence of
 *         points which forms a stratified set in
 *         2D.
 */

#ifndef STRATIFIED_SEQUENCE_H
#define STRATIFIED_SEQUENCE_H

#include "../randomsequence2d/RandomSequence2D.h"

class StratifiedSequence
  : public RandomSequence2D
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RandomSequence2D Parent;

    /*! Null constructor does nothing.
     */
    inline StratifiedSequence(void);

    inline StratifiedSequence(const float xStart, const float xEnd,
                              const float yStart, const float yEnd,
                              const size_t xStrata,
                              const size_t yStrata);

    inline virtual void reset(const float xStrata, const float xEnd,
                              const float yStrata, const float yEnd,
                              const size_t xStrata,
                              const size_t yStrata);

    /*! This method resets the current position of this StratifiedSequence
     *  without modifying any of the parameters of the sequence.
     */
    inline virtual void reset(void);

    /*! Null destructor does nothing.
     */
    inline virtual ~StratifiedSequence(void);

    inline virtual bool operator()(float &x, float &y);

    inline virtual bool operator()(float &x, float &y,
                                   float xJitter, float yJitter);

  protected:
    inline virtual bool advance(void);

    /*! The number of strata in either dimension.
     */
    size_t mNumStrata[2];

    /*! The row and column index of the current point.
     */
    size_t mCurrentRaster[2];

    /*! The current point.
     */
    float mCurrentPoint[2];

    /*! The origin of the sequence.
     */
    float mOrigin[2];

    /*! The spacing between strata.
     */
    float mStrataSpacing[2];
}; // end StratifiedSequence

#include "StratifiedSequence.inl"

#endif // STRATIFIED_SEQUENCE_H

