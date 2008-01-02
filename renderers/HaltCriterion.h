/*! \file HaltCriterion.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         which determines when a MonteCarloRenderer
 *         should halt.
 */

#ifndef HALT_CRITERION_H
#define HALT_CRITERION_H

#include <cstddef>
#include "../api/Gotham.h"

#include "Renderer.h"
class MonteCarloRenderer;

class HaltCriterion
{
  public:
    inline virtual ~HaltCriterion(void){;};

    /*! This method returns true when mRenderer needs
     *  to stop sampling.
     *  \return As above.
     */
    virtual bool operator()(void) = 0;

    /*! This method intializes this HaltCriterion.
     *  \param r Sets mRenderer.
     *  \param p Sets mProgress.
     */
    virtual void init(const MonteCarloRenderer *r,
                      Renderer::ProgressCallback *p);

    /*! This method returns mRenderer.
     *  \return mRenderer.
     */
    const MonteCarloRenderer *getRenderer(void) const;

    /*! This method acts as a factory for creating
     *  HaltCriterion objects.
     *  \param attr An AttributeMap describing the parameters of the render.
     *  \return A new HaltCriterion object.
     */
    static HaltCriterion *createCriterion(const Gotham::AttributeMap &attr);

  protected:
    const MonteCarloRenderer *mRenderer;

    Renderer::ProgressCallback *mProgress;
}; // end HaltCriterion

class TargetCriterion
  : public HaltCriterion
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef HaltCriterion Parent;

    /*! \typedef Target
     *  \brief A big number. For portability.
     */
    typedef long unsigned int Target;

    /*! This method intializes this HaltCriterion.
     *  \param r Sets mRenderer.
     *  \param p Sets mProgress and sets mProgress's expected count.
     */
    virtual void init(const MonteCarloRenderer *r,
                      Renderer::ProgressCallback *p);

    /*! This method sets mTarget.
     *  \param t Sets mTarget.
     */
    void setTarget(const Target t);

    /*! This method returns mTarget.
     *  \return mTarget
     */
    Target getTarget(void) const;

  protected:
    Target mTarget;

    /*! This counter holds the value of the previous
     *  call to operator()().
     */
    Target mPrevious;
}; // end TargetCriterion

class TargetSampleCount
  : public TargetCriterion
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TargetCriterion Parent;

    /*! This method returns true when mRenderer has taken
     *  equal to or greater than Parent::mTarget samples.
     */
    virtual bool operator()(void);
}; // end TargetSampleCount

class TargetRayCount
  : public TargetCriterion
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TargetCriterion Parent;

    /*! Constructor accepts a target ray count.
     *  \param t Sets Parent::mTarget.
     */
    TargetRayCount(const Target r);   

    /*! This method returns true when mRenderer has cast
     *  equal to or gretaer than Parent::mTarget rays.
     */
    virtual bool operator()(void);
}; // end TargetRayCount

#endif // HALT_CRITERION_H

