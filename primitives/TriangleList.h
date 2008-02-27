/*! \file TriangleList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a specialization
 *         of SurfacePrimitiveList which only applies
 *         to SurfacePrimitives which can be triangulated.
 */

#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

//#include "SurfacePrimitiveList.h"
#include "PrimitiveList.h"

class TriangleList
//  : public SurfacePrimitiveList
  : public PrimitiveList<>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PrimitiveList<> Parent;

    /*! This method adds a new SurfacePrimitive to this
     *  SurfacePrimitiveList.
     *  \param p The SurfacePrimitive to add.
     *  \note If p's Surface is not triangulatable, it is not added to this
     *        TriangleList.
     */
    virtual void push_back(boost::shared_ptr<ListElement> &p);
}; // end class TriangleList

#endif // TRIANGLE_LIST_H

