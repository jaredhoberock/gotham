/*! \file MonteCarloRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MonteCarloRenderer class.
 */

#include "MonteCarloRenderer.h"

void MonteCarloRenderer
  ::setRandomSequence(const boost::shared_ptr<RandomSequence> &sequence)
{
  mRandomSequence = sequence;
} // end MonteCarloRenderer::setRandomSequence()

