/*! \file Scene.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing a scene composed of Primitives.
 */

#ifndef SCENE_H
#define SCENE_H

#include <boost/shared_ptr.hpp>

#include <vector>
#include "Intersection.h"
#include "Primitive.h"
#include "PrimitiveList.h"
#include "../shading/MaterialList.h"
class BoundingBox;
class SurfacePrimitiveList;

/*! \class Scene
 *  \brief A Scene is a database of Primitives which may be queried for Ray intersection and shading requests.
 */
class Scene
{
  public:
    /*! Null constructor calls resetStatistics().
     */
    inline Scene(void);

    /*! Constructor accepts a Primitive.
     *  \param g Sets mPrimitive.
     */
    inline Scene(boost::shared_ptr<Primitive> p);

    /*! Null destructor does nothing.
     */
    inline virtual ~Scene(void);

    /*! This method returns the BoundingBox of mPrimitive.
     *  \param b A BoundingBox bounding mPrimitive is returned here.
     */
    inline virtual void getBoundingBox(BoundingBox &b) const;

    /*! This method computes the first intersection between the given Ray and this scene, if one exists.
     *  \param r The Ray to intersect.
     *  \param inter If an intersection exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered by r is returned here.
     *  \return true if an intersection exists; false, otherwise.
     *  \note If r is found to intersect this Scene, r's parametric bounds are appropriately updated to accomodate the intersection.
     */
    inline virtual bool intersect(Ray &r, Intersection &inter) const;

    /*! This method provides a SIMD path for ray intersection.
     *  \param rays A list of Rays to intersect.
     *  \param intersections Af an intersection for a Ray exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered is returned here.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of lists rays, intersections, and stencil.
     */
    inline virtual void intersect(Ray *rays,
                                  Intersection *intersections,
                                  int *stencil,
                                  const size_t n) const;

    /*! This method computes whether or not an intersection between the given Ray and this scene exists.
     *  \param r The Ray to intersect.
     *  \return true if an intersection exists; false, otherwise.
     */
    inline virtual bool intersect(const Ray &r) const;

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
     *  \param s Sets mSensors.
     */
    inline void setSensors(boost::shared_ptr<SurfacePrimitiveList> s);

    /*! This method sets mSurfaces.
     *  \param s Sets mSurfaces.
     */
    inline void setSurfaces(boost::shared_ptr<SurfacePrimitiveList> s);

    /*! This method sets mPrimitives.
     *  \param p Sets mPrimitives.
     */
    inline void setPrimitives(boost::shared_ptr<PrimitiveList<> > p);

    /*! This method sets mMaterials.
     *  \param m Sets mMaterials.
     */
    inline void setMaterials(const boost::shared_ptr<MaterialList> p);

    /*! This method returns mRaysCast.
     *  \return mRaysCast
     */
    inline long unsigned int getRaysCast(void) const;

    /*! This method sets mRaysCast.
     *  \param r Sets mRaysCast.
     */
    inline void setRaysCast(const long unsigned int r) const;

    /*! This method returns mShadowRaysCast.
     *  \return mShadowRaysCast.
     */
    inline long unsigned int getShadowRaysCast(void) const;

    /*! This method returns mBlockedShadowRays.
     *  \return mBlockedShadowRays.
     */
    inline long unsigned int getBlockedShadowRays(void) const;

    /*! This method resets this Scene's statistics.
     */
    inline void resetStatistics(void);

    /*! This method returns a const pointer to the
     *  emitters list.
     *  \return mEmitters
     */
    inline const SurfacePrimitiveList *getEmitters(void) const;

    /*! This method returns a const pointer to the sensors list.
     *  \return mSensors
     */
    inline const SurfacePrimitiveList *getSensors(void) const;

    /*! This method returns a const pointer to the
     *  surfaces list.
     *  \return mSurfaces
     */
    inline const SurfacePrimitiveList *getSurfaces(void) const;

    /*! This method returns a const pointer to the primitives list.
     *  \return mPrimitives
     */
    inline const PrimitiveList<> *getPrimitives(void) const;

    /*! This method returns a const reference to the materials list.
     *  \return mMaterials
     */
    inline const MaterialList &getMaterials(void) const;

    /*! This method is called immediately prior to rendering
     *  and calls mPrimitive->finalize()
     */
    inline virtual void preprocess(void);

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

    /*! A Scene contains a list of surfaces.
     */
    boost::shared_ptr<SurfacePrimitiveList> mSurfaces;

    /*! A Scene contains a list of all Primitives.
     */
    boost::shared_ptr<PrimitiveList<> > mPrimitives;

    /*! A Scene contains a list of all Materials.
     */
    boost::shared_ptr<MaterialList> mMaterials;

    /*! This counts the number of Rays intersected against this Scene.
     */
    mutable long unsigned int mRaysCast;

    /*! This counts the number of shadow Rays intersected against this Scene.
     */
    mutable long unsigned int mShadowRaysCast;

    /*! This counts the number of shadow Rays intersected against this Scene that were blocked.
     */
    mutable long unsigned int mBlockedShadowRays;
}; // end class Scene

#include "Scene.inl"

#endif // SCENE_H

