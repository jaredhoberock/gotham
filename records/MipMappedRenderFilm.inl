/*! \file MipMappedRenderFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for MipMappedRenderFilm.h.
 */

#include "MipMappedRenderFilm.h"
#include <stdio.h>

MipMappedRenderFilm
  ::MipMappedRenderFilm(void)
    :Parent()
{
  ;
} // end MipMappedRenderFilm::MipMappedRenderFilm()

MipMappedRenderFilm
  ::MipMappedRenderFilm(const unsigned int width,
                        const unsigned int height,
                        const std::string &filename)
   :Parent(width,height,filename)
{
  resize(width,height);
} // end MipMappedRenderFilm::MipMappedRenderFilm()

void MipMappedRenderFilm
  ::deposit(const float px, const float py,
            const Spectrum &s)
{
  // deposit into the Parent
  Parent::deposit(px, py, s);

  // deposit into each level of the mipmap
  float scale;
  for(size_t i = 0; i < mMipMap.size(); ++i)
  {
    scale = mMipMap[i].first;
    mMipMap[i].second.deposit(px, py, scale * s);
  } // end for i
} // end MipMappedRenderFilm::deposit()

void MipMappedRenderFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  // first call the Parent
  Parent::resize(width,height);
  
  // clear the mipmap
  mMipMap.clear();

  // add a mip map level while both of w & h are > 0
  unsigned int w = width / 2, h = height / 2;
  float scale = 0.25f;
  unsigned int level = 1;
  while(w > 0 && h > 0)
  {
    mMipMap.resize(mMipMap.size() + 1);
    mMipMap.back().first = scale;
    mMipMap.back().second.resize(w,h);

    w /= 2;
    h /= 2;
    scale *= 0.25f;
    ++level;
  } // end while
} // end MipMappedRenderFilm::resize()

void MipMappedRenderFilm
  ::fill(const Pixel &v)
{
  // call the Parent
  Parent::fill(v);

  // fill each level
  for(size_t i = 0; i < mMipMap.size(); ++i)
  {
    mMipMap[i].second.fill(v);
  } // end for i
} // end MipMappedRenderFilm::fill()

void MipMappedRenderFilm
  ::scale(const Pixel &s)
{
  // call the Parent
  Parent::scale(s);

  // scale each level
  for(size_t i = 0; i < mMipMap.size(); ++i)
  {
    mMipMap[i].second.scale(s);
  } // end for i
} // end MipMappedRenderFilm::scale()

void MipMappedRenderFilm
  ::postprocess(void)
{
  // postprocess the parent first
  Parent::postprocess();

  // postprocess each level
  for(size_t i = 0; i < mMipMap.size(); ++i)
  {
    mMipMap[i].second.postprocess();
  } // end for i
} // end MipMappedRenderFilm::postprocess()

RenderFilm &MipMappedRenderFilm
  ::getMipLevel(const unsigned int i)
{
  if(i == 0) return *this;
  return mMipMap[i-1].second;
} // end MipMappedRenderFilm::getMipLevel()

const RenderFilm &MipMappedRenderFilm
  ::getMipLevel(const unsigned int i) const
{
  if(i == 0) return *this;
  return mMipMap[i-1].second;
} // end MipMappedRenderFilm::getMipLevel()

unsigned int MipMappedRenderFilm
  ::getNumMipLevels(void) const
{
  return mMipMap.size() + 1;
} // end MipMappedRenderFilm::getNumMipLevels()

