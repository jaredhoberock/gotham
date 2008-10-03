/*! \file Array2.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a templated 2d array.
 */

#ifndef ARRAY_2_H
#define ARRAY_2_H

#include <vector>
#include <gpcpu/Vector.h>

template<typename Type>
  class Array2
    : protected std::vector<Type>
{
  public:
    /*! \typedef Element
     *  \brief Shorthand.
     */
    typedef Type Element;

    /*! \typedef size2.
     *  \brief Shorthand.
     */
    typedef gpcpu::Vector<size_t,2> size2;

    /*! Null constructor calls the Parent.
     */
    inline Array2(void);

    /*! Constructor accepts a width and height
     *  for this Array2.
     *  \param width The width of this Array2 in Elements.
     *  \param height The height of this Array2 in Elements.
     */
    inline Array2(const size_t width,
                  const size_t height);

    /*! Null destructor does nothing.
     */
    inline virtual ~Array2(void){;};

    /*! This method resizes this Array2.
     *  \param width The width of this Array2 in Elements.
     *  \param height The height of this Array2 in Elements.
     */
    inline virtual void resize(const size_t width,
                               const size_t height);

    /*! This method returns a reference to the Element
     *  at location (u,v)
     *  \param u A parametric coordinate in [0,1).
     *  \param v A parametric coordinate in [0,2).
     *  \return A reference to the Element at (u,v).
     */
    inline Element &element(const float u,
                            const float v);

    /*! This method returns a const reference to the Element
     *  at location (u,v)
     *  \param u A parametric coordinate in [0,1).
     *  \param v A parametric coordinate in [0,2).
     *  \return A reference to the Element at (u,v).
     */
    inline const Element &element(const float u,
                                  const float v) const;

    /*! This method returns a const reference to the Element
     *  at raster location (i,j)
     *  \param i A raster coordinate in [0,mWidth)
     *  \param j A raster coordinate in [0,mHeight)
     *  \return A const reference to the Element at (i,j)
     */
    inline const Element &raster(const size_t i,
                                 const size_t j) const;

    /*! This method returns a reference to the Element
     *  at raster location (i,j)
     *  \param i A raster coordinate in [0,mWidth)
     *  \param j A raster coordinate in [0,mHeight)
     *  \return A const reference to the Element at (i,j)
     */
    inline Element &raster(const size_t i,
                           const size_t j);

    /*! This method converts parametric coordinates to raster coordinates.
     *  \param u A parametric coordinate in [0,1).
     *  \param v A parametric coordinate in [0,0).
     *  \return The raster coordinate of (u,v) in
     *          [0, mDimensions[0]) x [0, mDimensions[1]).
     */
    inline size2 rasterCoordinates(const float u,
                                   const float v) const;

    /*! This method fills this Array2 with the given Element.
     *  \param v The fill value.
     */
    inline void fill(const Element &v);

    /*! This method returns the dimensions of this Array2.
     *  \return mDimensions
     */
    inline const size2 &getDimensions(void) const;

  protected:
    /*! This method maps the given real number in [0,1)
     *  to a column in this RandomAccessFilm.
     *  \param t A number in [0,1).
     *  \return The index of the raster column containing t.
     */
    inline size_t column(const float t) const;

    /*! This method maps the given real number in [0,1)
     *  to an integer column index in this RandomAccessFilm
     *  and also returns the fractional remainder.
     *  \param t A number in [0,1).
     *  \param i The integer part of t's mapping (the column index)
     *           is returned here.
     *  \param frac The fractional part of t's mapping is returned here.
     */
    inline void column(float t, size_t &i, float &frac) const;

    /*! This method maps the given real number in [0,1)
     *  to a row in this RandomAccessFilm.
     *  \param t A number in [0,1).
     *  \return THe index of the raster row containing t.
     */
    inline size_t row(const float t) const;

    /*! This method maps the given real number in [0,1)
     *  to an integer row index in this RandomAccessFilm
     *  and also returns the fractional remainder.
     *  \param t A number in [0,1).
     *  \param i The integer part of t's mapping (the row index)
     *           is returned here.
     *  \param frac The fractional part of t's mapping is returned here.
     */
    inline void row(float t, size_t &i, float &frac) const;

    /*! \typedef
     *  \brief Shorthand.
     */
    typedef std::vector<Element> Parent;
    
    /*! The dimensions of this image
     */
    size2 mDimensions;
}; // end Array2

#include "Array2.inl"

#endif // ARRAY_2_H

