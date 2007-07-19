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

    /*! Null constructor does nothing.
     */
    DebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene, Camera, and Film.
     *  \param s Sets mScene.
     *  \param f Sets mFilm.
     */
    DebugRenderer(boost::shared_ptr<const Scene> s,
                  boost::shared_ptr<RandomAccessFilm> f);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void render(ProgressCallback &progress);
}; // end Renderer

#endif // DEBUG_RENDERER_H

