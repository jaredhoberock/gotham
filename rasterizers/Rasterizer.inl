/*! \file Rasterizer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Rasterizer.h.
 */

#include "Rasterizer.h"

template<typename PrimitiveType>
  Rasterizer<PrimitiveType>
    ::Rasterizer(void)
{
  ;
} // end Rasterizer::Rasterizer()

template<typename PrimitiveType>
  Rasterizer<PrimitiveType>
    ::Rasterizer(boost::shared_ptr<PrimitiveType> p)
      :mPrimitive(p)
{
  ;
} // end Rasterizer::Rasterizer()

template<typename PrimitiveType>
  void Rasterizer<PrimitiveType>
    ::setPrimitive(boost::shared_ptr<PrimitiveType> p)
{
  mPrimitive = p;
} // end Rasterizer::setPrimitive()

