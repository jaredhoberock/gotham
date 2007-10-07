/*! \file RandomSequence.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RandomSequence.inl.
 */

#include "RandomSequence.h"

RandomSequence
  ::RandomSequence(const unsigned int seed)
    :Parent(Generator(boost::uint32_t(seed)))
{
  ;
} // end RandomSequence::RandomSequence()

