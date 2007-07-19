/*! \file Scene.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing a scene composed of Primitives.
 */

#ifndef SCENE_H
#define SCENE_H

#include <boost/shared_ptr.hpp>

#include <vector>
#include "Primitive.h"
class BoundingBox;
class SurfacePrimitiveList;

/*! \class Scene
 *  \brief A Scene is a database of Primitives which may be queried for Ray intersection and shading requests.
 */
class Scene
{
  public:
    /*! Null constructor sets mRaysCast to 0.
     */
    inline Scene(void);

    /*! Constructor accepts a Primitive.
     *  \param g Sets mPrimitive.
     */
    inline Scene(boost::shared_ptr<Primitive> p);

    /*! Destructor deletes mPrimitive if it is not 0.
     */
    ~Scene(void);

    /*! This method returns the BoundingBox of mPrimitive.
     *  \param b A BoundingBox bounding mPrimitive is returned here.
     */
    virtual void getBoundingBox(BoundingBox &b) const;

    /*! This method computes the first intersection between the given Ray and this scene, if one exists.
     *  \param r The Ray to intersect.
     *  \param inter If an intersection exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered by r is returned here.
     *  \return true if an intersection exists; false, otherwise.
     *  \note If r is found to intersect this Scene, r's parametric bounds are appropriately updated to accomodate the intersection.
     */
    bool intersect(Ray &r, Primitive::Intersection &inter) const;

    /*! This method computes whether or not an intersection between the given Ray and this scene exists.
     *  \param r The Ray to intersect.
     *  \return true if an intersection exists; false, otherwise.
     */
    bool intersect(const Ray &r) const;

    /*! This method sets mPrimitive.
     *  \param g Sets mPrimitive.
     */
    inline void setPrimitive(boost::shared_ptr<Primitive> g);

    /*! Returns a const pointer to mPrimitive.
     *  \return mPrimitive
     *  XXX Should this return a shared_ptr?
     */
    inline boost::shared_ptr<Primitive> getPrimitive(void) const;

    /*! This method sets mEmitters.
     *  \param e Sets mEmitters.
     */
    inline void setEmitters(boost::shared_ptr<SurfacePrimitiveList> e);

    /*! This method sets mSensors.
     *  \return s Sets mSensors.
     */
    inline void setSensors(boost::shared_ptr<SurfacePrimitiveList> s);

    /*! This method returns mRaysCast.
     *  \return mRaysCast
     */
    inline long unsigned int getRaysCast(void) const;

    /*! This method resets mRaysCast.
     */
    inline void resetRaysCast(void);

    /*! This method returns a const pointer to the
     *  emitters list.
     *  \return mEmitters
     */
    inline const SurfacePrimitiveList *getEmitters(void) const;

    /*! This method returns a const pointer to the sensors list.
     *  \return mSensors
     */
    inline const SurfacePrimitiveList *getSensors(void) const;

  protected:
    /*! A Scene contains the Primitive to be rendered.
     */
    boost::shared_ptr<Primitive> mPrimitive;

    /*! A Scene contains the list of emitters.
     *  XXX Would it make more sense for the
     *      Renderer to own this list?
     */
    boost::shared_ptr<SurfacePrimitiveList> mEmitters;

    /*! A Scene contains a list of sensors.
     *  XXX Would it make more sense for the Renderer
     *      to own this list?
     */
    boost::shared_ptr<SurfacePrimitiveList> mSensors;

    /*! This counts the number of Rays intersected against this Scene.
     */
    mutable long unsigned int mRaysCast;
}; // end class Scene

#include "Scene.inl"

#endif // SCENE_H

