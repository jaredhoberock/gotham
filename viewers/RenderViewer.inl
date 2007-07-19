/*! \file RenderViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RenderViewer.h.
 */

#include <GL/glew.h>
#include "RenderViewer.h"
#include <qstring.h>

#include <boost/thread/thread.hpp>
using boost::thread;
using namespace boost;

void RenderViewer
  ::drawProgress(const unsigned int i)
{
  //std::cerr << "RenderViewer::drawProgress(): sample " << i << std::endl;
  std::cerr << "\rRenderViewer::drawProgress(): sample " << i;
} // end RenderViewer::drawProgress()

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

    Program &p = mTexture2DRectProgram;

    glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);

    // XXX this crashes on ubuntu?
    //glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);

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
} // end RenderViewer::init()

void RenderViewer
  ::keyPressEvent(QKeyEvent *e)
{
  switch(e->key())
  {
    case Qt::Key_R:
    {
      startRender();
      updateGL();
      break;
    } // end case Qt::Key_R

    case Qt::Key_P:
    {
      mDrawPreview = !mDrawPreview;
      updateGL();
      break;
    } // end case Qt::Key_P

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

void RenderViewer::DrawProgress
  ::operator()(const unsigned int i)
{
  unsigned int rowWidth = 80;
  unsigned int maxStars = rowWidth - 12;
  unsigned int stars = (maxStars*i) / mTotalWork;

  std::string progressString("\rprogress: [");
  for(int i = 0; i < stars; ++i)
    progressString += "+";
  for(int i = stars; i < maxStars; ++i)
    progressString += " ";
  progressString += "]";

  if(stars == maxStars)
    progressString += "\n";

  std::cerr << progressString;
  boost::thread::yield();
}

void RenderViewer
  ::postSelection(const QPoint &p)
{
  ;
} // end RenderViewer::postSelection()

void RenderViewer
  ::setCamera(shared_ptr<Camera> c)
{
  mCamera = c;
} // end RenderViewer::setCamera()

void RenderViewer
  ::setImage(shared_ptr<RandomAccessFilm> i)
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
  DrawProgress temp;
  temp.mViewer = this;
  // XXX fix this
  temp.mTotalWork = width() * height();

  mProgress = temp;

  // XXX make RenderThunk refer to a shared_ptr
  RenderThunk render = RenderThunk(mRenderer.get(), &mProgress);
  boost::thread renderThread(render);

  // update the visualization every second
  setAnimationPeriod(1000);

  // update the frame automatically
  startAnimation();
} // end RenderViewer::startRender()

