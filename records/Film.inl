/*! \file Film.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Film.h.
 */

#include "Film.h"

Film
  ::Film(void)
    :mWidth(0),mHeight(0)
{
  ;
} // end Film::Film()

Film
  ::Film(const unsigned int width,
         const unsigned int height)
    :mWidth(width),mHeight(height)
{
  ;
} // end Film::Film()

void Film
  ::resize(const unsigned int width,
           const unsigned int height)
{
  mWidth = width;
  mHeight = height;
} // end Film::resize()

unsigned int Film
  ::getWidth(void) const
{
  return mWidth;
} // end Film::getWidth()

unsigned int Film
  ::getHeight(void) const
{
  return mHeight;
} // end Film::getHeight()

