/*! \file LightDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Renderer for debugging
 *         lights.
 */

#ifndef LIGHT_DEBUG_RENDERER_H
#define LIGHT_DEBUG_RENDERER_H

#include "Renderer.h"

class LightDebugRenderer
  : public Renderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Renderer Parent;

    /*! Null constructor does nothing.
     */
    LightDebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene, Camera, and Film.
     *  \param s Sets mScene.
     *  \param f Sets mFilm.
     */
    LightDebugRenderer(boost::shared_ptr<const Scene>  s,
                       boost::shared_ptr<RandomAccessFilm> f);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void render(ProgressCallback &progress);
}; // end LightDebugRenderer

#endif // LIGHT_DEBUG_RENDERER_H

