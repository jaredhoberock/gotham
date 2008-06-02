/*! \file DebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a renderer that shades
 *         eye rays based on surface normal.
 */

#ifndef DEBUG_RENDERER_H
#define DEBUG_RENDERER_H

#include "Renderer.h"

class DebugRenderer
  : public Renderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Renderer Parent;

    /*! Constructor accepts number of pixel strata in each dimension.
     *  \param xStrata Sets mXStrata;
     *  \param yStrata Sets mXStrata;
     */
    DebugRenderer(const size_t xStrata, const size_t yStrata);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! The number of samples to take per pixel in the x dimension.
     */
    size_t mXStrata;

    /*! The number of samples to take per pixel in the y dimension.
     */
    size_t mYStrata;
}; // end DebugRenderer

#endif // DEBUG_RENDERER_H

