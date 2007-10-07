/*! \file RenderViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface for a CgViewer
 *         for pricksie applications.
 */

#ifndef RENDER_VIEWER_H
#define RENDER_VIEWER_H

#include "SceneViewer.h"
#include "../films/RenderFilm.h"
#include "../renderers/Renderer.h"
#include <gl++/texture/Texture.h>
#include <boost/shared_ptr.hpp>

class RenderThunk;
class Sampler;
class SurfaceIntegrator;

class RenderViewer
  : public SceneViewer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SceneViewer Parent;

    RenderViewer(void);
    virtual void draw(void);
    virtual void keyPressEvent(KeyEvent *e);
    virtual void init(void);
    virtual void resizeGL(int w, int h);
    virtual void setImage(boost::shared_ptr<RenderFilm> i);
    virtual void setRenderer(boost::shared_ptr<Renderer> r);

  protected:
    virtual void drawScenePreview(void);
    virtual void startRender(void);
    virtual void postSelection(const QPoint &p);

    /*! Render resources
     */
    boost::shared_ptr<RenderFilm> mImage;
    boost::shared_ptr<Renderer> mRenderer;
    Renderer::ProgressCallback mProgress;

    /*! A Texture to upload the image to.
     */
    Texture mTexture;

    /*! Whether or not to draw the preview of the Scene.
     */
    bool mDrawPreview;

    /*! A gamma to apply to the result.
     */
    float mGamma;

    /*! An exposure to apply to the result.
     */
    float mExposure;

    /*! Whether or not to apply tonemapping.
     */
    bool mDoTonemap;
}; // end RenderViewer

#include "RenderViewer.inl"

#endif // RENDER_VIEWER_H

