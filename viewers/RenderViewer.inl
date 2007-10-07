/*! \file RenderViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RenderViewer.h.
 */

#include "RenderViewer.h"
#include <qstring.h>
#include "../renderers/MetropolisRenderer.h"
#include "../renderers/EnergyRedistributionRenderer.h"

#include <boost/thread/thread.hpp>
using boost::thread;
using namespace boost;

RenderViewer
  ::RenderViewer(void)
    :Parent()
{
  ;
} // end RenderViewer::RenderViewer()

void RenderViewer
  ::draw(void)
{
  if(!mDrawPreview)
  {
    float *data = reinterpret_cast<float*>(&mImage->raster(0,0));
    GLenum datatype = GL_FLOAT_RGB16_NV;

    mTexture.init(datatype,
                  width(), height(),
                  0,
                  GL_RGB, data);

    float progressScale = static_cast<float>(mProgress.expected_count()) / mProgress.count();

    // choose which program to use
    Program &p = mDoTonemap ? mTexture2DRectTonemapProgram : mTexture2DRectGammaProgram;

    p.bind();

    // compensate for progress if we're doing metropolis
    if(!mDoTonemap
       && dynamic_cast<const MetropolisRenderer*>(mRenderer.get())
       && dynamic_cast<const EnergyRedistributionRenderer*>(mRenderer.get()) == 0)
    {
      p.setUniform1f("scale", powf(2.0f, mExposure) * progressScale);
    } // end if
    else
    {
      p.setUniform1f("scale", powf(2.0f, mExposure));
    } // end else

    p.setUniform1f("gamma", mGamma);
    p.setUniform1f("Ywa", mImage->getMaximumLuminance());

    printGLError(__FILE__, __LINE__);

    p.unbind();

    glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);

    drawTexture(mTexture, p);
    glPopAttrib();
  } // end if
  else
  {
    drawScenePreview();
  } // end else
} // end RenderViewer::draw()

void RenderViewer
  ::drawScenePreview(void)
{
  Parent::draw();
} // end RenderViewer::drawScenePreview()

void RenderViewer
  ::init(void)
{
  glewInit();

  // first init the parent
  Parent::init();

  // init other stuff
  mTexture.create();

  mDrawPreview = true;
  mExposure = 0.0f;
  mGamma = 1.0f;

  mDoTonemap = false;
} // end RenderViewer::init()

void RenderViewer
  ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case Qt::Key_R:
    {
      startRender();
      mDrawPreview = false;
      updateGL();
      break;
    } // end case Qt::Key_R

    case Qt::Key_P:
    {
      mDrawPreview = !mDrawPreview;
      updateGL();
      break;
    } // end case Qt::Key_P

    case '(':
    {
      mGamma -= 0.1f;
      mGamma = std::max(mGamma, 0.1f);

      char msg[32];
      sprintf(msg, "Gamma: %f", mGamma);
      displayMessage(msg);
      updateGL();
      break;
    } // end case (

    case ')':
    {
      mGamma += 0.1f;

      char msg[32];
      sprintf(msg, "Gamma: %f", mGamma);
      displayMessage(msg);
      updateGL();
      break;
    } // end case )

    case '{':
    {
      mExposure -= 1.0f;

      char msg[32];
      sprintf(msg, "Exposure: %f", mExposure);
      displayMessage(msg);
      updateGL();
      break;
    } // end case '{'

    case '[':
    {
      mExposure -= 0.1f;

      char msg[32];
      sprintf(msg, "Exposure: %f", mExposure);
      displayMessage(msg);
      updateGL();
      break;
    } // end case '['

    case '}':
    {
      mExposure += 1.0f;

      char msg[32];
      sprintf(msg, "Exposure: %f", mExposure);
      displayMessage(msg);
      updateGL();
      break;
    } // end case '}'

    case ']':
    {
      mExposure += 0.1f;

      char msg[32];
      sprintf(msg, "Exposure: %f", mExposure);
      displayMessage(msg);
      updateGL();
      break;
    } // end case ']'

    case 'T':
    {
      mDoTonemap = !mDoTonemap;
      updateGL();
      break;
    } // end case 'T'

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end RenderViewer::keyPressEvent()

void RenderViewer
  ::resizeGL(int w, int h)
{
  Parent::resizeGL(w,h);
  mImage->resize(w,h);

  char buffer[32];
  sprintf(buffer, "%d", width());
  std::string message = buffer;
  message += "x";
  sprintf(buffer, "%d", height());
  message += buffer;

  displayMessage(message);
  updateGL();
} // end RenderViewer::resizeGL()

class RenderThunk
{
  public:

  inline RenderThunk(Renderer *renderer,
                     Renderer::ProgressCallback *progress)
    :mRenderer(renderer),mProgress(progress)
  {
    ;
  }

  virtual void run(void)
  {
    (*this)();
  }

  void operator()(void)
  {
    mRenderer->render(*mProgress);
  }

  Renderer *mRenderer;
  Renderer::ProgressCallback *mProgress;
};

void RenderViewer
  ::postSelection(const QPoint &p)
{
  ;
} // end RenderViewer::postSelection()

void RenderViewer
  ::setImage(shared_ptr<RenderFilm> i)
{
  mImage = i;
} // end RenderViewer::setImage()

void RenderViewer
  ::setRenderer(shared_ptr<Renderer> r)
{
  mRenderer = r;
} // end Renderer::setRenderer()

void RenderViewer
  ::startRender(void)
{
  // XXX make RenderThunk refer to a shared_ptr
  RenderThunk render = RenderThunk(mRenderer.get(), &mProgress);
  boost::thread renderThread(render);

  // update the visualization every second
  setAnimationPeriod(1000);

  // update the frame automatically
  startAnimation();
} // end RenderViewer::startRender()

