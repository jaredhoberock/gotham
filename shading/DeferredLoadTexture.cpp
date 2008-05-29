/*! \file DeferredLoadTexture.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DeferredLoadTexture class.
 */

#include "DeferredLoadTexture.h"

DeferredLoadTexture
  ::DeferredLoadTexture(const char *filename)
    :Parent(),mNeedsLoad(true),mFilename(filename)
{
  ;
} // end DeferredLoadTexture::DeferredLoadTexture()

void DeferredLoadTexture
  ::load(void)
{
  Parent::load(mFilename.c_str());
  mNeedsLoad = false;
} // end DeferredLoadTexture::load()

const Spectrum &DeferredLoadTexture
  ::texRect(const size_t x,
            const size_t y) const
{
  if(mNeedsLoad)
  {
    // yuck
    const_cast<DeferredLoadTexture*>(this)->load();
  } // end if

  return Parent::texRect(x,y);
} // end DeferredLoadTexture::texRect()

