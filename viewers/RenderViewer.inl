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
    :Parent(),mDrawPreview(true),
     mExposure(0.0f),mGamma(1.0f),
     mDoTonemap(false),mMiddleGrey(0.5f)
{
  ;
} // end RenderViewer::RenderViewer()

void RenderViewer
  ::init(void)
{
  // first init the parent
  Parent::init();

  // no state file please
  setStateFileName(Parent::String(""));

  // init other stuff
  mTexture.create();

  // resize
  resize(512,512);

  shared_ptr<const RenderFilm> film = dynamic_pointer_cast<const RenderFilm,const Record>(mRenderer->getRecord());
  if(film.get() != 0)
  {
    resize(film->getWidth(), film->getHeight());
  } // end if
} // end RenderViewer::init()


void RenderViewer
  ::drawFilm(const shared_ptr<const RenderFilm> &f) 
{
  // figure out which texture to use
  const Texture *texture = &mTexture;
  const GpuFilm<RenderFilm> *gpuFilm = dynamic_cast<const GpuFilm<RenderFilm> *>(f.get());
  if(gpuFilm != 0)
  {
    // XXX yuck fix this
    const_cast<GpuFilm<RenderFilm> *>(gpuFilm)->renderPendingDeposits();
    texture = &gpuFilm->mTexture;
  } // end if

  const float *data = reinterpret_cast<const float*>(&f->raster(0,0));
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
  if(mProgress.count() < mProgress.expected_count())
  {
    // XXX DESIGN kill this dynamic_cast somehow
    const MonteCarloRenderer *mc = dynamic_cast<const MonteCarloRenderer*>(mRenderer.get());
    if(mc)
    {
      float spp = mc->getNumSamples();
      spp /= (f->getWidth() * f->getHeight());
      scale = 1.0f / spp;
    } // end if
  } // end else

  p.setUniform1f("scale", powf(2.0f, mExposure) * scale);
  p.setUniform1f("gamma", mGamma);
  float meanLogL = f->getSumLogLuminance() / (f->getWidth() * f->getHeight());
  float Lwa = expf(meanLogL);
  p.setUniform1f("Lwa", Lwa);
  p.setUniform1f("Lwhite", f->getMaximumLuminance());
  p.setUniform1f("a", mMiddleGrey);

  printGLError(__FILE__, __LINE__);

  p.unbind();

  glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);

  drawTexture(*texture, p);
  glPopAttrib();
} // end RenderViewer::draw()

void RenderViewer
  ::drawPhotons(const PhotonMap &photons)
{
  glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);

  // scale by 1/spp if we haven't finished the render
  float scale = (width() * height());
  //if(mProgress.count() < mProgress.expected_count())
  {
    // XXX DESIGN kill this dynamic_cast somehow
    const MonteCarloRenderer *mc = dynamic_cast<const MonteCarloRenderer*>(mRenderer.get());
    if(mc)
    {
      scale /= mc->getNumSamples();
    } // end if
  } // end else

  mPassthroughGammaProgram.bind();
  mPassthroughGammaProgram.setUniform1f("scale", powf(2.0f, mExposure) * scale);
  mPassthroughGammaProgram.setUniform1f("gamma", mGamma);

  // draw dem photons
  glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
  glInterleavedArrays(GL_C3F_V3F, sizeof(Photon), &photons[0].mPower[0]);
  glDrawArrays(GL_POINTS, 0, photons.size());
  glPopClientAttrib();

  mPassthroughGammaProgram.unbind();

  glPopAttrib();
} // end RenderViewer::drawPhotons()

void RenderViewer
  ::draw(void)
{
  if(!mDrawPreview)
  {
    shared_ptr<const Record> rec = mRenderer->getRecord();

    shared_ptr<const RenderFilm> film = dynamic_pointer_cast<const RenderFilm,const Record>(rec);

    if(film.get())
    {
      // draw the film
      drawFilm(film);
    } // end if
    else
    {
      shared_ptr<const PhotonRecord> photons = dynamic_pointer_cast<const PhotonRecord, const Record>(rec);
      if(photons.get())
      {
        // draw the photons
        drawPhotons(*photons);
      } // end if
    } // end else
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
      setGamma(std::max(mGamma - 0.1f, 0.1f));

      char msg[32];
      sprintf(msg, "Gamma: %f", mGamma);
      displayMessage(msg);
      updateGL();
      break;
    } // end case (

    case ')':
    {
      setGamma(mGamma + 0.1f);

      char msg[32];
      sprintf(msg, "Gamma: %f", mGamma);
      displayMessage(msg);
      updateGL();
      break;
    } // end case )

    case '{':
    {
      char msg[32];
      if(mDoTonemap)
      {
        mMiddleGrey -= 1.0f;

        sprintf(msg, "Middle grey: %f", mMiddleGrey);
      } // end if
      else
      {
        mExposure -= 1.0f;

        sprintf(msg, "Exposure: %f", mExposure);
      } // end else

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
      char msg[32];

      if(mDoTonemap)
      {
        mMiddleGrey += 1.0f;

        sprintf(msg, "Middle grey: %f", mMiddleGrey);
      } // end if
      else
      {
        mExposure += 1.0f;

        sprintf(msg, "Exposure: %f", mExposure);
      } // end else

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

  inline virtual ~RenderThunk(void){;};

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
  ::setRenderer(shared_ptr<Renderer> r)
{
  mRenderer = r;

  shared_ptr<const Record> rec = mRenderer->getRecord();
  const RandomAccessFilm *image = dynamic_cast<const RandomAccessFilm*>(rec.get());
  if(image != 0)
  {
    resizeGL(image->getWidth(), image->getHeight());
  } // end if
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

void RenderViewer
  ::setGamma(const float g)
{
  mGamma = g;
} // end RenderViewer::setGamma()

float RenderViewer
  ::getGamma(void) const
{
  return mGamma;
} // end RenderViewer::getGamma()

