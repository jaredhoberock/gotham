/*! \file GlutViewer.inl
 *  \author Jared Hoberock, Yuntao Jia
 *  \brief Inline file for GlutViewer.h.
 */

#include <string.h>
#include "GlutViewer.h"
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <vector>

GlutViewer* GlutViewer::viewer = 0;

GlutViewer
  ::GlutViewer(void)
{
  // message
  mMessageDieTime = 0;

  // message display
  m_nMsgShiftX = 0;
  m_nMsgShiftY = 0;

  // width & height
  mWidth = 600;
  mHeight = 400;

  // title
  mWindowTitle = "GlutViewer";
  
  // camera
  mCamera.setPosition(gpcpu::float3(0,0,0));
  mCamera.setUpVector(gpcpu::float3(0,1,0));
  mCamera.setViewDirection(gpcpu::float3(0,0,-1));
  mCamera.setFieldOfView(M_PI * 45.0f / 180.0f);
  mCamera.setAspectRatio(static_cast<float>(width()) / height());

  // animation
  mIsAnimating = false;
  mAnimationPeriod = 0;

  viewer = this;

  // initialize glut
  initializeGlut();

  // initialize gl
  initializeGL();
} // end GlutViewer::GlutViewer()

void GlutViewer
  ::initializeGlut(void)
{
  int argc = 0;
  glutInit(&argc,0);

  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100,100);
  glutInitWindowSize(width(),height());

  // create the window
  glutCreateWindow(mWindowTitle.c_str());

  // register callbacks
  glutReshapeFunc(reshapeFunc);
  glutDisplayFunc(displayFunc);  
  glutKeyboardFunc(keyFunc);
} // end GlutViewer::initializeGlut()

void GlutViewer
  ::init(void)
{
  ;
} // end GlutViewer::init()

void GlutViewer
  ::show(void)
{
  // call init() before the main loop
  init();
  glutMainLoop();
} // end GlutViewer::show()

GlutViewer* GlutViewer
  ::getInstance(void)
{
  return viewer;
} // end GlutViewer::getInstance()

void GlutViewer
  ::resize(const int w, const int h)
{
  mWidth = w;
  mHeight = h;

  glutReshapeWindow(mWidth,mHeight);
} // endG GlutViewer::setDimensions()

void GlutViewer
  ::resizeGL(int w, int h)
{
  // the callback simply sets the data
  // members, it does not call resize()
  mWidth = w;
  mHeight = h;
} // end GlutViewer::resizeGL()

GlutViewer
  ::~GlutViewer(void)
{
  ;
} // end GlutViewer::~GlutViewer()

/* static call back function for glut */
void GlutViewer
  ::displayFunc(void)
{
  GlutViewer* pViewer = GlutViewer::getInstance();
  pViewer->render();
} // end GlutViewer::render()

void GlutViewer
  ::render(void)
{
  beginDraw();
  draw();
  endDraw();
} // end GlutViewer::render()

void GlutViewer
  ::animate(void)
{
  ;
} // end GlutViewer::animate()

void GlutViewer
  ::timerFunc(int value)
{
  GlutViewer *pViewer = GlutViewer::getInstance();

  if(pViewer->mIsAnimating)
  {
    pViewer->animate();
    pViewer->render();

    // if we're still animating, ask for another frame
    if(pViewer->mIsAnimating)
    {
      glutTimerFunc(pViewer->mAnimationPeriod, timerFunc, value);
    } // end if
  } // end if
} // end GlutViewer::timerFunc()

void GlutViewer
  ::idleFunc(void)
{
  GlutViewer* pViewer = GlutViewer::getInstance();
  pViewer->animate();
  pViewer->render();
} // end GlutViewer::idleFunc()

void GlutViewer
  ::reshapeFunc(int w, int h)
{
  GlutViewer::getInstance()->resizeGL(w,h);
} // end GlutViewer::reshapeFunc()

void GlutViewer
  ::keyFunc(unsigned char key, int x, int y)
{
  // convert key to uppercase
  unsigned int k = toupper(key);

  // get modifiers
  unsigned int modifiers = glutGetModifiers();

  // convert to Qt-compatible codes
  unsigned int m = 0;
  if(modifiers & GLUT_ACTIVE_SHIFT)
  {
    m |= 0x02000000;
  } // end if
  if(modifiers & GLUT_ACTIVE_CTRL)
  {
    m |= 0x04000000;
  } // end if
  if(modifiers & GLUT_ACTIVE_ALT)
  {
    m |= 0x08000000;
  } // end if

  // create a KeyEvent
  KeyEvent e(k,m);

  GlutViewer::getInstance()->keyPressEvent(&e);
} // end GlutViewer::keyFunc()

void GlutViewer
  ::draw(void)
{
  ;
} // GlutViewer::draw()

void GlutViewer
  ::initializeGL(void)
{
  glPushAttrib(GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_LIGHTING_BIT | GL_POLYGON_BIT | GL_TRANSFORM_BIT
               | GL_VIEWPORT_BIT);

  glClearColor(51.0f / 255.0f,
               51.0f / 255.0f,
               51.0f / 255.0f,
               1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glColor3f(180.0f / 255.0f,
            180.0f / 255.0f,
            180.0f / 255.0f);

  glEnable(GL_COLOR_MATERIAL);
  
  // Enable depth test
  glEnable(GL_DEPTH_TEST);

  // Cull backfacing polygons
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);

  // Enable lighting
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
} // end GlutViewer::initializeGL()

void GlutViewer
  ::beginDraw(void)
{
  // clear buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set up transform
  glPushAttrib(GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);

  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();
  gluPerspective(180.0f * camera()->fieldOfView() / M_PI,
                 camera()->aspectRatio(),
                 camera()->zNear(),
                 1000.0f);

  gpcpu::float3 lookAt = camera()->position() + camera()->viewDirection();
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(camera()->position().x,
            camera()->position().y,
            camera()->position().z,
            lookAt.x,
            lookAt.y,
            lookAt.z,
            camera()->upVector().x,
            camera()->upVector().y,
            camera()->upVector().z);
  
  glPushMatrix();

  // set the viewport
  glViewport(0, 0, width(), height());
} // end GlutViewer::beginDraw()

void GlutViewer
  ::endDraw(void)
{
  glPopMatrix();
  glPopAttrib();
  
  drawText();
  
  glutSwapBuffers();
} // end GlutViewer::endDraw()

void GlutViewer
  ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 0x1b:
    {
      exit(0);
      break;
    } // end case esc

    // the 'enter' key
    case 13:
    {
      // toggle animation
      if(mIsAnimating) stopAnimation();
      else startAnimation();
      break;
    } // end case 'enter'
  } // end switch
} // end GlutViewer::keyPressEvent()

static void strokeString(int x, int y, const char *msg)
{
  glRasterPos2f( (GLfloat)x, (GLfloat)y);
  int len = (int) strlen(msg);
  for(int i = 0; i < len; i++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, msg[i]);
} // end strokeString()

void GlutViewer
  ::drawText(int wndId, const std::string &text)
{
  glutSetWindow(wndId);
  glClear(GL_COLOR_BUFFER_BIT);
  glLineWidth(1);

  std::string temp = text;
  char *pch = strtok(&temp[0], "\n");

  for(int j = 1; pch != 0; j++)
  {
    strokeString(40,  20 + j * 14, pch);
    pch = strtok(0, "\n");
  } // end for j

  glutSwapBuffers();
} // end GlutViewer::drawText()

void GlutViewer
  ::displayMessage(const std::string &message, int delay)
{
  strcpy(m_Msg, message.c_str());
  m_nMsgShiftX = 0;
  m_nMsgShiftY = 0;

  mMessageDieTime = glutGet(GLUT_ELAPSED_TIME) + delay;
} // end GlutViewer::displayMessage()

void GlutViewer
  ::drawText()
{
  // is the message dead yet?
  if(glutGet(GLUT_ELAPSED_TIME) >= mMessageDieTime)
    return;
  
  glPushMatrix();
  glLoadIdentity();
  
  // set matrix
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, width(), height(), 0);
  glLineWidth(1);
  
  glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);
  glDisable(GL_LIGHTING);
  glColor3f(1,1,1);
  strokeString(MESSAGE_INDENT+m_nMsgShiftX, height()-MESSAGE_INDENT+m_nMsgShiftY, m_Msg);
  glPopAttrib();
  
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
} // end GlutViewer::drawText()

void GlutViewer
  ::updateGL(void)
{
  glutPostRedisplay();
} // end GlutViewer::updateGL()

int GlutViewer
  ::width(void) const
{
  return mWidth;
} // end GlutViewer::width()

int GlutViewer
  ::height(void) const
{
  return mHeight;
} // end GlutViewer::height()

bool GlutViewer
  ::animationIsStarted(void) const
{
  return mIsAnimating;
} // end GlutViewer::animationIsStarted()

void GlutViewer
  ::startAnimation(void)
{
  mIsAnimating = true;
  //glutIdleFunc(idleFunc);
  glutTimerFunc(mAnimationPeriod, timerFunc, 0);
} // end GlutViewer::startAnimation()

void GlutViewer
  ::stopAnimation(void)
{
  mIsAnimating = false;
  //glutIdleFunc(0);
} // end GlutViewer::stopAnimation()

void GlutViewer
  ::setStateFileName(const std::string &filename)
{
  ;
} // end GlutViewer::setStateFileName()

void GlutViewer
  ::setSceneBoundingBox(const gpcpu::float3 &min,
                        const gpcpu::float3 &max)
{
  ;
} // end GlutViewer::setSceneBoundingBox()

void GlutViewer
  ::setWindowTitle(const std::string &title)
{
  mWindowTitle = title;
  glutSetWindowTitle(title.c_str());
  glutSetIconTitle(title.c_str());
} // end GlutViewer::setWindowTitle()

void GlutViewer
  ::setAnimationPeriod(int period)
{
  mAnimationPeriod = period;
} // end GlutViewer::setAnimationPeriod()

GlutViewer::Camera
  ::Camera(void)
    :mZNear(0.005f)
{
  ;
} // end Camera::Camera()

GlutViewer::Camera *GlutViewer
  ::camera(void) const
{
  return &mCamera;
} // end GlutViewer::camera()

void GlutViewer::Camera
  ::setPosition(const gpcpu::float3 &pos)
{
  mPosition = pos;
} // end Camera::setPosition()

gpcpu::float3 GlutViewer::Camera
  ::position(void) const
{
  return mPosition;
} // end Camera::position()

void GlutViewer::Camera
  ::setUpVector(const gpcpu::float3 &up)
{
  mUpVector = up;
} // end Camera::setUpVector()

gpcpu::float3 GlutViewer::Camera
  ::upVector(void) const
{
  return mUpVector;
} // end Camera::upVector()

void GlutViewer::Camera
  ::setViewDirection(const gpcpu::float3 &direction)
{
  mViewDirection = direction;
} // end Camera::setViewDirection()

gpcpu::float3 GlutViewer::Camera
  ::viewDirection(void) const
{
  return mViewDirection;
} // end Camera::viewDirection()

void GlutViewer::Camera
  ::setFieldOfView(float fov)
{
  mFieldOfView = fov;
} // end Camera::setFieldOfView()

float GlutViewer::Camera
  ::fieldOfView(void) const
{
  return mFieldOfView;
} // end Camera::fieldOfView()

void GlutViewer::Camera
  ::setAspectRatio(float aspect)
{
  mAspectRatio = aspect;
} // end Camera::setAspectRatio()

float GlutViewer::Camera
  ::aspectRatio(void) const
{
  return mAspectRatio;
} // end Camera::aspectRatio()

float GlutViewer::Camera
  ::zNear(void) const
{
  return mZNear;
} // end Camera::zNear()

