/*! \file SurfacePrimitiveList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a specialization
 *         of PrimitiveList which only applies to
 *         SurfacePrimitives.
 */

#ifndef SURFACE_PRIMITIVE_LIST_H
#define SURFACE_PRIMITIVE_LIST_H

#include "PrimitiveList.h"
#include "SurfacePrimitive.h"
#include <aliastable/AliasTable.h>

class SurfacePrimitiveList
  : public PrimitiveList<SurfacePrimitive>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PrimitiveList<SurfacePrimitive> Parent;

    /*! Null constructor sets mSurfaceArea to zero.
     */
    SurfacePrimitiveList(void);

    /*! This method samples the surface area of this SurfacePrimitiveList.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param u3 A fourth real number in [0,1).
     *  \param surf A pointer to the Surface at the sampled Point is returned here.
     *  \param dg The DifferentialGeometry at the sampled Point is
     *            returned here.
     *  \param pdf The surface area pdf at dg is returned here.
     *  \return true if this SurfacePrimitiveList's surface area could be sampled;
     *          false, otherwise.
     */
    bool sampleSurfaceArea(const float u0,
                           const float u1,
                           const float u2,
                           const float u3,
                           const SurfacePrimitive **surf,
                           DifferentialGeometry &dg,
                           float &pdf) const;

    /*! This method evaluates the surface area measure pdf of choosing a point
     *  on a surface of interest from this SurfacePrimitiveList.
     *  \param surf The SurfacePrimitive of interest.
     *  \param dg The DifferentialGeometry on surf at the point of interest.
     *  \param The surface area measure pdf of choosing (surf, dg) from this
     *         SurfacePrimitiveList.
     */
    float evaluateSurfaceAreaPdf(const SurfacePrimitive *surf,
                                 const DifferentialGeometry &dg) const;

    /*! This method samples the surfaces area of this SurfacePrimitiveList.
     *  \param u A real number in [0,1).
     *  \prim A pointer to a SurfacePrimitive uniformly sampled from the
     *        surface area of this SurfacePrimitiveList is returned here.
     *  \param pdf The probability of sampling prim is returned here.
     *  \return true if this SurfacePrimitiveList's surface area could be sampled;
     *          false, otherwise.
     */
    bool sampleSurfaceArea(const float u,
                           const SurfacePrimitive **prim,
                           float &pdf) const;

    /*! This method evaluates the surface area measure pdf of choosing
     *  the given SurfacePrimitive from this SurfacePrimitiveList.
     *  \param prim The SurfacePrimitive of interest.
     *  \return The surface area measure pdf of choosing prim.
     */
    float evaluateSurfaceAreaPdf(const SurfacePrimitive *prim) const;

    /*! This method adds a new SurfacePrimitive to this
     *  SurfacePrimitiveList.
     *  \param p The SurfacePrimitive to add.
     */
    virtual void push_back(boost::shared_ptr<SurfacePrimitive> &p);

    /*! This method finalizes this SurfacePrimitiveList.
     */
    virtual void finalize(void);

  protected:
    /*! This method builds mSurfaceAreaPdf.
     */
    void buildSurfaceAreaPdf(void);

    /*! The sum of surface areas of the SurfacePrimitives
     *  in this SurfacePrimitiveList.
     */
    float mSurfaceArea;

    /*! One over the surface area of the SurfacePrimitives
     *  in this SurfacePrimitiveList.
     */
    float mOneOverSurfaceArea;

    typedef AliasTable<const SurfacePrimitive*,float> PrimTable;
    PrimTable mSurfaceAreaPdf;
}; // end SurfacePrimitiveList

#endif // SURFACE_PRIMITIVE_LIST_H

