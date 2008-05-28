/*! \file TextureHandle.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an opaque
 *         pointer type for Textures.
 */

#pragma once

#include <string>

typedef unsigned int TextureHandle;

struct TextureParameter
{
  inline TextureParameter(const std::string &alias)
    :mHandle(0),mAlias(alias){}

  inline TextureParameter(const TextureHandle &h)
    :mHandle(h),mAlias(""){}

  operator TextureHandle (void) const {return mHandle;}

  TextureHandle mHandle;

  // filename or other alias for a Texture to use during shading
  // if this is not '' during preprocess, the texture referred to
  // by this alias will be loaded and bound
  std::string mAlias;
}; // end TextureParameter

