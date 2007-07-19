/*! \file Renderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Renderer.h.
 */

#include "Renderer.h"

Renderer
  ::Renderer(void)
    :mScene(),mFilm()
{
  ;
} // end Renderer::Renderer()

Renderer
  ::Renderer(boost::shared_ptr<const Scene> s,
             boost::shared_ptr<RandomAccessFilm> f)
    :mScene(s),mFilm(f)
{
  ;
} // end Renderer::Renderer()

void Renderer
  ::setScene(boost::shared_ptr<const Scene> s)
{
  mScene = s;
} // end Renderer::setScene()

boost::shared_ptr<const Scene> Renderer
  ::getScene(void) const
{
  return mScene;
} // end Renderer::getScene()

void Renderer
  ::setFilm(boost::shared_ptr<RandomAccessFilm> f)
{
  mFilm = f;
} // end Renderer::setFilm()

boost::shared_ptr<const RandomAccessFilm> Renderer
  ::getFilm(void) const
{
  return mFilm;
} // end Renderer::getFilm()

