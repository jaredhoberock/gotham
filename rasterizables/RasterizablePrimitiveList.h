/*! \file RasterizablePrimitiveList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PrimitiveList class
 *         which implements Rasterizable.
 */

#ifndef RASTERIZABLE_PRIMITIVE_LIST_H
#define RASTERIZABLE_PRIMITIVE_LIST_H

#include "../primitives/PrimitiveList.h"
#include "Rasterizable.h"

template<typename PrimitiveListParentType>
  class RasterizablePrimitiveList
    : public PrimitiveListParentType,
      public Rasterizable
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef PrimitiveListParentType Parent0;
    
    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Rasterizable Parent1;

    /*! This method rasterizes this RasterizablePrimitiveList
     *  by calling rasterize() on each Primitive in its list
     *  if it is an instance of Rasterizable.
     */
    inline virtual void rasterize(void);
}; // end RasterizablePrimitiveList

#include "RasterizablePrimitiveList.inl"

#endif // RASTERIZABLE_PRIMITIVE_LIST_H

