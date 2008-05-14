/*! \file HilbertWalk.inl
 *  \author Jared Hoberock
 *  \brief Inline file for HilbertWalk.h.
 */

#include "HilbertWalk.h"

HilbertWalk
  ::HilbertWalk(void)
{
  ;
} // end HilbertWalk::HilbertWalk()

HilbertWalk
  ::HilbertWalk(const size_t n,
                const size_t x0,
                const size_t y0)
{
  init(n,x0,y0);
} // end HilbertWalk::HilbertWalk()

void HilbertWalk
  ::init(const size_t n,
         const size_t x0,
         const size_t y0)
{
  mCurrentPosition = gpcpu::size2(x0,y0);
  mWalkDirection = gpcpu::int2(0,1);
  mStack.clear();
  mStack.push_back(std::make_pair(R, n));
} // end HilbertWalk::init()

void HilbertWalk
  ::f(void)
{
  mCurrentPosition += mWalkDirection;
} // end HilbertWalk::f()

void HilbertWalk
  ::p(void)
{
  // left turn
  if(mWalkDirection[1] == 1)
  {
    mWalkDirection = gpcpu::int2(-1,0);
  } // end if
  else if(mWalkDirection[0] == -1)
  {
    mWalkDirection = gpcpu::int2(0,-1);
  } // end else if
  else if(mWalkDirection[1] == -1)
  {
    mWalkDirection = gpcpu::int2(1,0);
  } // end else if
  else if(mWalkDirection[0] == 1)
  {
    mWalkDirection = gpcpu::int2(0,1);
  } // end else if
} // end HilbertWalk::p()

void HilbertWalk
  ::m(void)
{
  // right turn
  if(mWalkDirection[1] == 1)
  {
    mWalkDirection = gpcpu::int2(1,0);
  } // end if
  else if(mWalkDirection[0] == 1)
  {
    mWalkDirection = gpcpu::int2(0,-1);
  } // end else if
  else if(mWalkDirection[1] == -1)
  {
    mWalkDirection = gpcpu::int2(-1,0);
  } // end else if
  else if(mWalkDirection[0] == -1)
  {
    mWalkDirection = gpcpu::int2(0,1);
  } // end else if
} // end HilbertWalk::m()

void HilbertWalk
  ::l(const size_t n)
{
  if(n == 0) return;
  mStack.push_back(std::make_pair(P,0));
  mStack.push_back(std::make_pair(R,n-1));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(M,0));
  mStack.push_back(std::make_pair(L,n-1));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(L,n-1));
  mStack.push_back(std::make_pair(M,0));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(R,n-1));
  mStack.push_back(std::make_pair(P,0));
} // end HilbertWalk::l()

void HilbertWalk
  ::r(const size_t n)
{
  if(n == 0) return;
  mStack.push_back(std::make_pair(M,0));
  mStack.push_back(std::make_pair(L,n-1));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(P,0));
  mStack.push_back(std::make_pair(R,n-1));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(R,n-1));
  mStack.push_back(std::make_pair(P,0));
  mStack.push_back(std::make_pair(F,0));
  mStack.push_back(std::make_pair(L,n-1));
  mStack.push_back(std::make_pair(M,0));
} // end HilbertWalk::r()

bool HilbertWalk
  ::operator()(size_t &x, size_t &y)
{
  if(mStack.empty()) return false;

  x = mCurrentPosition[0];
  y = mCurrentPosition[1];

  std::pair<TurtleCommand, size_t> command;

  while(!mStack.empty())
  {
    command = mStack.back();
    mStack.pop_back();

    switch(command.first)
    {
      case F:
        f();
        break;

      case P:
        p();
        break;

      case M:
        m();
        break;

      case L:
        l(command.second);
        break;

      case R:
        r(command.second);
        break;
    } // end switch()

    if(command.first == F)
    {
      break;
    } // end if
  } // end while

  return true;
} // end HilbertWalk::operator()

