/*! \file Film.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         the film of a Camera.
 */

#ifndef FILM_H
#define FILM_H

class Film
{
  public:
    /*! Null constructor calls the Parent.
     */
    inline Film(void);

    /*! Constructor accepts a width and height
     *  for this Film.
     *  \param width The width of this Film in pixels.
     *  \param height The height of this Film in pixels.
     */
    inline Film(const unsigned int width,
                const unsigned int height);

    /*! This method resizes this Film.
     *  \param width The width of this Film in pixels.
     *  \param height The height of this Film in pixels.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);

    /*! This method returns the width of this Film.
     *  \return mWidth
     */
    inline unsigned int getWidth(void) const;

    /*! This method returns the height of this Film.
     *  \return mHeight
     */
    inline unsigned int getHeight(void) const;

  protected:
    /*! The width of this Film in pixels.
     */
    unsigned int mWidth;

    /*! The height of this Film in pixels.
     */
    unsigned int mHeight;
}; // end Film

#include "Film.inl"

#endif // FILM_H

