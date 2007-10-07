/*! \file MonteCarloRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for MonteCarloRenderer.h.
 */

#include "MonteCarloRenderer.h"

MonteCarloRenderer
  ::MonteCarloRenderer(void)
     :Parent()
{
  ;
} // end MonteCarloRenderer::MonteCarloRenderer()

MonteCarloRenderer
  ::MonteCarloRenderer(const boost::shared_ptr<RandomSequence> &sequence)
     :Parent()
{
  setRandomSequence(sequence);
} // end MonteCarloRenderer::MonteCarloRenderer()

MonteCarloRenderer
  ::MonteCarloRenderer(boost::shared_ptr<const Scene> &s,
                       boost::shared_ptr<RenderFilm> &f,
                       const boost::shared_ptr<RandomSequence> &sequence)
     :Parent(s,f)
{
  setRandomSequence(sequence);
} // end MonteCarloRenderer::MonteCarloRenderer()

