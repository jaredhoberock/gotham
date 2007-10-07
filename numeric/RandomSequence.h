/*! \file RandomSequence.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a sequence
 *         of pseudo-random numbers in the unit
 *         interval.
 */

#ifndef RANDOM_SEQUENCE_H
#define RANDOM_SEQUENCE_H

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_01.hpp>

class RandomSequence
  : public boost::uniform_01<boost::random::lagged_fibonacci_01<float, 48, 607, 273>, float>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef boost::uniform_01<boost::random::lagged_fibonacci_01<float, 48, 607, 273>, float> Parent;

    /*! Null constructor seeds mGenerator with 13.
     *  \param seed The seed of the sequence.
     */
    inline RandomSequence(const unsigned int seed = 13u);

  protected:
    /*! \typedef Generator
     *  \brief The type of the variate generator.
     */
    typedef boost::random::lagged_fibonacci_01<float, 48, 607, 273> Generator;
}; // end RandomSequence

#include "RandomSequence.inl"

#endif // RANDOM_SEQUENCE_H

