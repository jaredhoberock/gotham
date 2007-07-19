/*! \file Rasterizer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class which
 *         Rasterizes a Surface.
 */

#ifndef RASTERIZER_H
#define RASTERIZER_H

#include <boost/shared_ptr.hpp>

template<typename PrimitiveType>
  class Rasterizer
{
  public:
    /*! Null constructor does nothing.
     */
    inline Rasterizer(void);
    inline Rasterizer(boost::shared_ptr<PrimitiveType> p);
    virtual void operator()(void) = 0;
    inline virtual void setPrimitive(boost::shared_ptr<PrimitiveType> p);

  protected:
    boost::shared_ptr<PrimitiveType> mPrimitive;
}; // end class Rasterizer

#include "Rasterizer.inl"

#endif // RASTERIZER_H

