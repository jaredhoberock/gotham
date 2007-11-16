/*! \file GpuFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for GpuFilm.h.
 */

#include "GpuFilm.h"
#include <checkframebufferstatus/checkFramebufferStatus.h>

template<typename ParentFilmType>
  GpuFilm<ParentFilmType>
    ::GpuFilm(void)
      :Parent()
{
  ;
} // end GpuFilm::GpuFilm()

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::resize(const unsigned int width,
             const unsigned int height)
{
  Parent::resize(width,height);

  // resize the Texture if it exists
  if(mTexture.getIdentifier() != 0)
  {
    mTexture.texImage2D(GL_FLOAT_RGB16_NV,
                        Parent::getWidth(), Parent::getHeight(),
                        0, GL_RGB, GL_UNSIGNED_INT, (void*)0);
  } // end if
} // end GpuFilm::resize()

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::init(void)
{
  Parent::init();

  mFramebuffer.create();
  mTexture.create();
  mTexture.texImage2D(GL_FLOAT_RGB16_NV,
                      Parent::getWidth(), Parent::getHeight(),
                      0, GL_RGB, GL_UNSIGNED_INT, (void*)0);
} // end GpuFilm

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::deposit(const float px, const float py,
              const Spectrum &s)
{
  Parent::deposit(px,py,s);

  boost::mutex::scoped_lock lock(mMutex);
  mDepositBuffer.push_back(std::make_pair(gpcpu::float2(px,py), s));
  lock.unlock();
} // end GpuFilm::deposit()

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::renderPendingDeposits(void)
{
  // setup the framebuffer
  glPushAttrib(GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_COLOR_BUFFER_BIT | GL_POINT_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);

  // identity xfrm
  // XXX PERF we don't need this with user shaders
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // set up the projection [0,1)^2
  gluOrtho2D(0, 1, 0, 1);

  glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
  glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

  // set up the viewport
  glViewport(0,0,mTexture.getWidth(),mTexture.getHeight());

  // bind the framebuffer
  mFramebuffer.bind();

  // XXX PERF attach this once in the init or something
  mFramebuffer.attachTexture(mTexture.getTarget(),
                             GL_COLOR_ATTACHMENT0_EXT,
                             mTexture);

  glEnable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glBlendFunc(GL_ONE, GL_ONE);
  glPointSize(1.0f);

  checkFramebufferStatus(__FILE__, __LINE__);

  // XXX FEATURE add a geometry shader to extrude a quad over a patch of pixels
  // XXX FEATURE add a fragment shader implementing a pixel reconstruction filter

  // XXX PERF replace this with a glDrawElements call
  glBegin(GL_POINTS);
  for(size_t i = 0;
      i != mDepositBuffer.size();
      ++i)
  {
    glColor3fv(mDepositBuffer[i].second);
    glVertex2fv(mDepositBuffer[i].first);
  } // end for i
  glEnd();

  // XXX PERF no need to attach/detach each time
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);
  mFramebuffer.unbind();

  // restore matrix state
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glPopAttrib();

  // clear pending deposits
  mDepositBuffer.clear();
} // end GpuFilm::renderPendingDeposits()

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::scale(const typename Parent::Pixel &s)
{
  Parent::scale(s);

  std::cerr << "GpuFilm::scale(): Implement me!" << std::endl;
} // end GpuFilm::scale()

template<typename ParentFilmType>
  void GpuFilm<ParentFilmType>
    ::fill(const typename Parent::Pixel &v)
{
  Parent::fill(v);

  glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
  glClearColor(v[0],v[1],v[2],0);

  glViewport(0,0,mTexture.getWidth(),mTexture.getHeight());

  mFramebuffer.bind();

  // XXX PERF attach this once in the init or something
  mFramebuffer.attachTexture(mTexture.getTarget(), GL_COLOR_ATTACHMENT0_EXT, mTexture);

  glClear(GL_COLOR_BUFFER_BIT);

  // XXX PERF no need to attach/detach each time
  mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);
  mFramebuffer.unbind();

  glPopAttrib();
} // end GpuFilm::fill()

