/*! \file RenderViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface for a CgViewer
 *         for pricksie applications.
 */

#ifndef RENDER_VIEWER_H
#define RENDER_VIEWER_H

#include "SceneViewer.h"
#include "../films/RandomAccessFilm.h"
#include "../renderers/Renderer.h"
#include <texture/Texture.h>
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

    virtual void draw(void);
    virtual void keyPressEvent(QKeyEvent *e);
    virtual void init(void);
    virtual void resizeGL(int w, int h);
    virtual void setCamera(boost::shared_ptr<Camera> c);
    virtual void setImage(boost::shared_ptr<RandomAccessFilm> i);
    virtual void setRenderer(boost::shared_ptr<Renderer> r);

  protected:
    virtual void drawScenePreview(void);
    virtual void startRender(void);
    virtual void drawProgress(const unsigned int i);
    virtual void postSelection(const QPoint &p);

    struct DrawProgress
    {
      inline void operator()(const unsigned int i);
      RenderViewer *mViewer;
      unsigned int mTotalWork;
    };

    /*! Render resources
     */
    boost::shared_ptr<RandomAccessFilm> mImage;
    boost::shared_ptr<Camera> mCamera;
    boost::shared_ptr<Renderer> mRenderer;

    /*! A progress callback.
     */
    Renderer::ProgressCallback mProgress;

    /*! A Texture to upload the image to.
     */
    Texture mTexture;

    /*! Whether or not to draw the preview of the Scene.
     */
    bool mDrawPreview;
}; // end RenderViewer

#include "RenderViewer.inl"

#endif // RENDER_VIEWER_H

