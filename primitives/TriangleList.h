/*! \file TriangleList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a specialization
 *         of SurfacePrimitiveList which only applies
 *         to SurfacePrimitives which can be triangulated.
 */

#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "SurfacePrimitiveList.h"

class TriangleList
  : public SurfacePrimitiveList
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SurfacePrimitiveList Parent;

    /*! This method adds a new SurfacePrimitive to this
     *  SurfacePrimitiveList.
     *  \param p The SurfacePrimitive to add.
     *  \note If p's Surface is not triangulatable, it is not added to this
     *        TriangleList.
     */
    virtual void push_back(const boost::shared_ptr<Primitive> &p);
}; // end class TriangleList

#endif // TRIANGLE_LIST_H

