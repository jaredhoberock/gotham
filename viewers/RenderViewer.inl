/*! \file RenderViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RenderViewer.h.
 */

#include "RenderViewer.h"
#include <qstring.h>
#include "../renderers/MetropolisRenderer.h"
#include "../renderers/EnergyRedistributionRenderer.h"
#include "../records/GpuFilm.h"
#include "../records/GpuFilterFilm.h"

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
    const Texture *texture = &mTexture;
    GpuFilm<RenderFilm> *gpuFilm = dynamic_cast<GpuFilm<RenderFilm> *>(mImage.get());
    if(gpuFilm != 0)
    {
      gpuFilm->renderPendingDeposits();
      texture = &gpuFilm->mTexture;
    } // end if

    const float *data = reinterpret_cast<const float*>(&mImage->raster(0,0));
    GLenum datatype = GL_FLOAT_RGB16_NV;

    mTexture.init(datatype,
                  width(), height(),
                  0,
                  GL_RGB, data);

    // choose which program to use
    Program &p = mDoTonemap ? mTexture2DRectTonemapProgram : mTexture2DRectGammaProgram;

    p.bind();

    // scale by 1/spp if we haven't finished the render
    float scale = 1.0f;
    //if(!mDoTonemap && mProgress.count() < mProgress.expected_count())
    if(mProgress.count() < mProgress.expected_count())
    {
      // XXX DESIGN kill this dynamic_cast somehow
      const MonteCarloRenderer *mc = dynamic_cast<const MonteCarloRenderer*>(mRenderer.get());
      if(mc)
      {
        float spp = mc->getNumSamples();
        spp /= (mImage->getWidth() * mImage->getHeight());
        scale = 1.0f / spp;
      } // end if
    } // end else

    p.setUniform1f("scale", powf(2.0f, mExposure) * scale);
    p.setUniform1f("gamma", mGamma);
    float meanLogL = mImage->getSumLogLuminance() / (mImage->getWidth() * mImage->getHeight());
    float Lwa = expf(meanLogL);
    p.setUniform1f("Lwa", Lwa);
    p.setUniform1f("Lwhite", mImage->getMaximumLuminance());
    p.setUniform1f("a", mMiddleGrey);

    printGLError(__FILE__, __LINE__);

    p.unbind();

    glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);

    drawTexture(*texture, p);
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
  mMiddleGrey = 0.5f;

  // XXX DESIGN need a better way to get a GpuFilm a GL context
  mImage->init();
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
      char msg[32];
      if(mDoTonemap)
      {
        mMiddleGrey -= 0.01f;

        sprintf(msg, "Middle grey: %f", mMiddleGrey);
      } // end if
      else
      {
        mExposure -= 0.1f;

        sprintf(msg, "Exposure: %f", mExposure);
      } // end else

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
      char msg[32];

      if(mDoTonemap)
      {
        mMiddleGrey += 0.01f;

        sprintf(msg, "Middle grey: %f", mMiddleGrey);
      } // end if
      else
      {
        mExposure += 0.1f;

        sprintf(msg, "Exposure: %f", mExposure);
      } // end else

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

