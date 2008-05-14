/*! \file bsp.h
 *  \author Nate A. Carr & Jared Hoberock
 *  \brief Defines the interface to a BSP-tree
 *         for ray/triangle intersection built using
 *         the Surface Area Heuristic.
 */

#ifndef __MGL_BSP__H
#define __MGL_BSP__H

#include "blockAllocator.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>


#define BSP_TP 0.6f
#define BSP_TK1 0.1f
#define BSP_TK2 0.1f
#define BSP_TK3 0.1f
#define BSP_MA  0.85f
#define BSP_MB  0.85f

#define EPSILON 0.00005f

typedef int I32;
typedef unsigned int U32;

template<class T, typename Real>
class bspNode
{
  public:
    bspNode(){}
    // SET AND GET CHILDREN NODES
    inline void setChildren(bspNode<T,Real> *pNode){_pChildren = pNode;};
    inline bspNode<T, Real> *getLeftChild()const{return _pChildren;}
    inline bspNode<T, Real> *getRightChild()const{ return getLeftChild()+1;}
    // SET AND THE AXIS 0-x,1-y,2-z,3-leaf
    inline void setAxis(I32 plane){ _axis = plane; }
    inline I32 getAxis()const{ return _axis; }
    inline bool isLeaf()const{ return (_axis<0);}
    // SET THE SPLIT PLANE LOCATION
    inline void setSplit(Real split){ _splitPlane = split;}
    inline Real getSplit()const{ return _splitPlane;}
    // SET A POINTER TO THE ELEMENTS IN THE LEAF NODE 
    inline void setElements(T *pElements){ _pElements = pElements; }
    inline T  *getElements()const{ return _pElements;}
    // SET THE NUMBER OF ELEMENTS
    inline void setElementCount(U32 elements){_elementCount = elements;}
    inline U32 getElementCount()const{ return _elementCount;}
  private:
    I32 _axis;
    union{
      Real  _splitPlane;
      U32  _elementCount;
    };
    union{
      bspNode<T, Real> *_pChildren;
      T *_pElements;
    };
};


template<class T, typename P, typename Real = float>
class bspTree
{
  public:
    bspTree(){ root.setElements(0);};
    // BUILD THE BSP TREE
    template<class BOUNDS,class IT>
      void buildTree(IT pBegin,IT pEnd,BOUNDS& boundsFunc);
    // PERFORM RAY INTERSECTION
    template<class INTERSECT> 
      bool intersect(const P& O,const P& D, Real minT, Real maxT, INTERSECT& intrsct) const;
    // PERFORM SHADOW RAY INTERSECTION
    template<class SHADOW>
      bool shadow(const P& O,const P& D, Real minT, Real maxT, SHADOW& intrsct) const;

    // FRONT TO BACK TRAVERSAL
    template<class VISITOR>
      void frontToBackTraversal(const P& eye, VISITOR& visitor);

    // BACK TO FRONT TRAVERSAL
    template<class VISITOR>
      void backToFrontTraversal(const P& eye, VISITOR& visitor);

    template<class VISITOR>
      void backToFrontTraversalVoxel(const P& eye, VISITOR& visitor);

    template<class VISITOR>
      void backToFrontTraversalVoxel(const P& eye, VISITOR& visitor) const;

  private:
    // FIND SPLITTING PLANE TO BEST PARTITION OBJECTS ALONG A GIVEN AXIS
    template<class BOUNDS,class IT>
      Real calcSplitValue(I32 axis,
          IT pBegin,IT pEnd,
          const P& minBounds,
          const P& maxBounds,
          BOUNDS& boundsFunc,
          Real *pSplit);
    // MAKE NODE A LEAF NODE
    template<class IT>
      void makeLeaf(bspNode<T, Real> *pNode,IT pBegin,IT pEnd);
    // BSP COST FUNCTIONS
    Real unSplitCostFunction(U32 count);
    Real splitCostFunction(I32 axis,const P& minBounds,const P& maxBounds,
        Real split,
        U32 leftCount,U32 rightCount,U32 bothCount);
    // TREE CONSTRUCTION
    template<class BOUNDS,class IT>
      void constructTree(bspNode<T, Real> *pNode,
          const P& minBounds,
          const P& maxBounds,
          IT pBegin,IT pEnd,
          BOUNDS& boundsFunc);
    // MEMBERS
    bspNode<T, Real> root;
    P          _minBounds,_maxBounds;
    // ALLOCATORS FOR NODE DATA
    blockAllocator< bspNode<T, Real> , 1024 > _nodeAllocator;
    blockAllocator< T , 256 > _elementAllocator;
};




template<class T,typename P, typename Real>
class bspStackNode{
  public:
    const bspNode<T, Real> *_pNode;
    P _b;//vec3f _a;
    Real   _minT;
    Real   _maxT;
};

template<class T, typename P, typename Real>
template<class SHADOWINTERSECT>
bool bspTree<T,P,Real>
::shadow(const P &o, const P &d,
         Real minT, Real maxT,
         SHADOWINTERSECT &inter) const
{
  P invDir;

  // SLABS METHOD FOR INTERSECTING RAY WITH BOX
  for(U32 i=0;i<3;++i)
  {
    if(d[i] != 0)
    {
      invDir[i] = static_cast<Real>(1.0)/d[i];
      Real tNear = (_minBounds[i] - o[i]) * invDir[i];
      Real tFar = (_maxBounds[i] - o[i]) * invDir[i];
      if( tNear > tFar ) 
        std::swap(tNear,tFar);

      minT = std::max(tNear, minT);
      maxT = std::min(tFar, maxT);

      if(minT > maxT )
      {
        return false;
      } // end if
    } // end if
    else
    {
      if( o[i] < _minBounds[i] || o[i] > _maxBounds[i] )
      {
        return false;
      } // end if

      invDir[i] = std::numeric_limits<Real>::infinity();
    } // end else
  } // end for i

  // initialize todo stack
  bspStackNode<T,P,Real> todo[50];
  unsigned int todoPos = 0;
  const bspNode<T,Real> *node= &root;

  while(node != 0)
  {
    if(!node->isLeaf())
    {
      // process interior node

      // compute parametric distance along ray to split plane
      I32 axis = node->getAxis();
      Real tPlane = (node->getSplit() - o[axis]);
      tPlane *= invDir[axis];

      // get node children pointers for ray
      const bspNode<T,Real> *firstChild = node->getLeftChild(), *secondChild = firstChild + 1;
      bool aboveLeft = o[axis] > node->getSplit();
      if(aboveLeft)
      {
        std::swap(firstChild, secondChild);
      } // end else

      // advance to next child, possibly enqueue another child
      if(tPlane > maxT || tPlane < 0.0)
      {
        node = firstChild;
      } // end if
      else if(tPlane < minT)
      {
        node = secondChild;
      } // end else if
      else
      {
        // enqueue secondChild in todo list
        todo[todoPos]._pNode = secondChild;
        //todo[todoPos]._minT = tPlane;
        //todo[todoPos]._maxT = maxT;
        ++todoPos;

        node = firstChild;
        //maxT = tPlane;
      } // end else
    } // end if
    else
    {
      // check for intersections inside leaf
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      if(nPrimitives)
      {
        if(inter(o,d,pE,pE + nPrimitives, minT, maxT))
        {
          return true;
        } // end if
      } // end if

      // grab next node to process from todo
      if(todoPos > 0)
      {
        --todoPos;
        node = todo[todoPos]._pNode;
        //minT = todo[todoPos]._minT;
        //maxT = todo[todoPos]._maxT;
      } // end if
      else
      {
        break;
      } // end else
    } // end else
  } // end while

  return false;
} // end bspTree::intersect()

template<class T, typename P, typename Real>
template<class INTERSECT>
bool bspTree<T,P,Real>
::intersect(const P &o, const P &d,
    Real minT, Real maxT,
    INTERSECT &inter) const
{
  P invDir;

  // SLABS METHOD FOR INTERSECTING RAY WITH BOX
  for(U32 i=0;i<3;++i)
  {
    if(d[i] != 0)
    {
      invDir[i] = static_cast<Real>(1.0)/d[i];
      Real tNear = (_minBounds[i] - o[i]) * invDir[i];
      Real tFar = (_maxBounds[i] - o[i]) * invDir[i];
      if( tNear > tFar ) 
        std::swap(tNear,tFar);

      minT = std::max(tNear, minT);
      maxT = std::min(tFar, maxT);

      if(minT > maxT )
      {
        return false;
      } // end if
    } // end if
    else
    {
      if( o[i] < _minBounds[i] || o[i] > _maxBounds[i] )
      {
        return false;
      } // end if

      invDir[i] = std::numeric_limits<Real>::infinity();
    } // end else
  } // end for i

  // initialize todo stack
  bspStackNode<T,P,Real> todo[50];
  unsigned int todoPos = 0;
  const bspNode<T,Real> *node= &root;

  while(node != 0)
  {
    if(!node->isLeaf())
    {
      // process interior node

      // compute parametric distance along ray to split plane
      I32 axis = node->getAxis();
      Real tPlane = (node->getSplit() - o[axis]);
      tPlane *= invDir[axis];

      // get node children pointers for ray
      const bspNode<T,Real> *firstChild = node->getLeftChild(), *secondChild = firstChild + 1;
      bool aboveLeft = o[axis] > node->getSplit();
      if(aboveLeft)
      {
        std::swap(firstChild, secondChild);
      } // end else

      // advance to next child, possibly enqueue another child
      if(tPlane > maxT || tPlane < 0.0)
      {
        node = firstChild;
      } // end if
      else if(tPlane < minT)
      {
        node = secondChild;
      } // end else if
      else
      {
        // enqueue secondChild in todo list
        todo[todoPos]._pNode = secondChild;
        todo[todoPos]._minT = tPlane;
        todo[todoPos]._maxT = maxT;
        ++todoPos;

        node = firstChild;
        maxT = tPlane;
      } // end else
    } // end if
    else
    {
      // check for intersections inside leaf
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      if(nPrimitives)
      {
        if(inter(o,d,pE,pE + nPrimitives, minT, maxT))
        {
          return true;
        } // end if
      } // end if

      // grab next node to process from todo
      if(todoPos > 0)
      {
        --todoPos;
        node = todo[todoPos]._pNode;
        minT = todo[todoPos]._minT;
        maxT = todo[todoPos]._maxT;
      } // end if
      else
      {
        break;
      } // end else
    } // end else
  } // end while

  return false;
} // end bspTree::intersect()

template<class T, typename P, typename Real>
  template<class VISITOR>
void bspTree<T,P,Real>::frontToBackTraversal(const P& eye, VISITOR& visitor)
{
  typedef bspNode<T,Real> Node;

  std::vector<Node *> stack;
  stack.push_back(&root);
  while(!stack.empty())
  {
    // pop the back
    const Node *node = stack.back();

    if(node->isLeaf())
    {
      // visit the leaves
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      for(U32 i = 0;
          i != nPrimitives;
          ++i, ++pE)
      {
        visitor(*pE);
      } // end for

      stack.pop_back();
    } // end if
    else
    {
      Node *nearNode = node->getLeftChild();
      Node *farNode = nearNode + 1;

      // sort the children
      // if the eye is on the left side of the split plane,
      // the left child is near
      // otherwise the right child is near

      if(eye[node->getAxis()] > node->getSplit())
      {
        std::swap(nearNode, farNode);
      } // end if

      // pop the node off the stack
      stack.pop_back();

      // push the far node
      stack.push_back(farNode);

      // push the near node
      stack.push_back(nearNode);
    } // end else
  } // end while
} // end bspTree::frontToBackTraversal()

template<class T, typename P, typename Real>
  template<class VISITOR>
void bspTree<T,P,Real>::backToFrontTraversalVoxel(const P& eye, VISITOR& visitor)
{
  typedef bspNode<T,Real> Node;
  typedef std::pair<P, P> Voxel;
  typedef std::pair<Node *, Voxel> StackEntry;

  std::vector<StackEntry> stack;
  Voxel v(_minBounds, _maxBounds);
  stack.push_back(std::make_pair(&root, v));
  while(!stack.empty())
  {
    // pop the back
    const Node *node = stack.back().first;
    const Voxel &v = stack.back().second;

    if(node->isLeaf())
    {
      // visit the leaves
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      for(U32 i = 0;
          i != nPrimitives;
          ++i, ++pE)
      {
        visitor(*pE, v.first, v.second);
      } // end for

      stack.pop_back();
    } // end if
    else
    {
      Node *nearNode = node->getLeftChild();
      Node *farNode = nearNode + 1;

      // split v into left & right halves
      Voxel nearV = v;
      nearV.second[node->getAxis()] = node->getSplit();
      Voxel farV = v;
      farV.first[node->getAxis()] = node->getSplit();

      // sort the children
      // if the eye is on the left side of the split plane,
      // the left child is near
      // otherwise the right child is near

      if(eye[node->getAxis()] > node->getSplit())
      {
        std::swap(nearNode, farNode);
        std::swap(nearV, farV);
      } // end if

      // pop the node off the stack
      stack.pop_back();

      // push the near node
      stack.push_back(std::make_pair(nearNode, nearV));

      // push the far node
      stack.push_back(std::make_pair(farNode, farV));
    } // end else
  } // end while
} // end bspTree::backToFrontTraversalVoxel()

template<class T, typename P, typename Real>
  template<class VISITOR>
void bspTree<T,P,Real>::backToFrontTraversalVoxel(const P& eye, VISITOR& visitor) const
{
  typedef bspNode<T,Real> Node;
  typedef std::pair<P, P> Voxel;
  typedef std::pair<const Node *, Voxel> StackEntry;

  std::vector<StackEntry> stack;
  Voxel v(_minBounds, _maxBounds);
  stack.push_back(std::make_pair(&root, v));
  while(!stack.empty())
  {
    // pop the back
    const Node *node = stack.back().first;
    const Voxel &v = stack.back().second;

    if(node->isLeaf())
    {
      // visit the leaves
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      for(U32 i = 0;
          i != nPrimitives;
          ++i, ++pE)
      {
        visitor(*pE, v.first, v.second);
      } // end for

      stack.pop_back();
    } // end if
    else
    {
      Node *nearNode = node->getLeftChild();
      Node *farNode = nearNode + 1;

      // split v into left & right halves
      Voxel nearV = v;
      nearV.second[node->getAxis()] = node->getSplit();
      Voxel farV = v;
      farV.first[node->getAxis()] = node->getSplit();

      // sort the children
      // if the eye is on the left side of the split plane,
      // the left child is near
      // otherwise the right child is near

      if(eye[node->getAxis()] > node->getSplit())
      {
        std::swap(nearNode, farNode);
        std::swap(nearV, farV);
      } // end if

      // pop the node off the stack
      stack.pop_back();

      // push the near node
      stack.push_back(std::make_pair(nearNode, nearV));

      // push the far node
      stack.push_back(std::make_pair(farNode, farV));
    } // end else
  } // end while
} // end bspTree::backToFrontTraversalVoxel()

template<class T, typename P, typename Real>
  template<class VISITOR>
void bspTree<T,P,Real>::backToFrontTraversal(const P& eye, VISITOR& visitor)
{
  typedef bspNode<T,Real> Node;

  std::vector<Node *> stack;
  stack.push_back(&root);
  while(!stack.empty())
  {
    // pop the back
    const Node *node = stack.back();

    if(node->isLeaf())
    {
      // visit the leaves
      U32 nPrimitives = node->getElementCount();
      T *pE = node->getElements();
      for(U32 i = 0;
          i != nPrimitives;
          ++i, ++pE)
      {
        visitor(*pE);
      } // end for

      stack.pop_back();
    } // end if
    else
    {
      Node *nearNode = node->getLeftChild();
      Node *farNode = nearNode + 1;

      // sort the children
      // if the eye is on the left side of the split plane,
      // the left child is near
      // otherwise the right child is near

      if(eye[node->getAxis()] > node->getSplit())
      {
        std::swap(nearNode, farNode);
      } // end if

      // pop the node off the stack
      stack.pop_back();

      // push the near node
      stack.push_back(nearNode);

      // push the far node
      stack.push_back(farNode);
    } // end else
  } // end while
} // end bspTree::backToFrontTraversal()

template<class T,typename P, typename Real>
  template<class BOUNDS,class IT>
void bspTree<T,P,Real>::buildTree(IT pBegin,IT pEnd,BOUNDS& boundsFunc)
{
  IT pE = pBegin;
  Real inf = std::numeric_limits<Real>::infinity();
  _minBounds[0] = inf; _minBounds[1] = inf; _minBounds[2] = inf;
  _maxBounds[0] = -inf; _maxBounds[1] = -inf; _maxBounds[2] = -inf;

  while(pE!=pEnd){
    for(U32 i=0;i<3;i++){
      Real minVal = boundsFunc(i,true,*pE);
      Real maxVal = boundsFunc(i,false,*pE);
      if(_minBounds[i] > minVal)
        _minBounds[i] = minVal;
      if(_maxBounds[i] < maxVal)
        _maxBounds[i] = maxVal;
    }
    ++pE;
  }
  for(U32 i=0;i<3;++i)
  {
    // always widen the bounding box
    // this ensures that axis-aligned primitives always
    // lie strictly within the bounding box
    _minBounds[i] -= EPSILON;
    _maxBounds[i] += EPSILON;
  }

  constructTree(&root,
      _minBounds,_maxBounds,
      pBegin,pEnd,
      boundsFunc);
}

  template<class T, typename P, typename Real>
Real bspTree<T,P,Real>::unSplitCostFunction(U32 count)
{
  Real TK1 = BSP_TK1; // COST OF ACCESSING NODE
  Real TP  = BSP_TP; // COST OF INTERSECTING A PRIMITIVE
  return TK1 + count * TP;
}

  template<class T, typename P, typename Real>
Real bspTree<T,P,Real>::splitCostFunction(I32 axis,
    const P& minBounds,
    const P& maxBounds,
    Real split,
    U32 leftCount,U32 rightCount,U32 bothCount)
{
  //dimension of bounding box
  P dim = maxBounds - minBounds;
  // Expectance to hit node
  Real ENH = dim[0]*dim[1] + dim[0]*dim[2] + dim[1]*dim[2];
  // percentage split
  Real r = (split - minBounds[axis])/dim[axis];
  //
  Real EPH = 0,temp = 0;
  switch(axis){
    case 0: // Expectance to hit intersecting plane
      EPH = dim[1]*dim[2];
      temp = dim[0]*(dim[1] + dim[2]) / ENH;
      break;
    case 1: // Expectance to hit intersecting plane
      EPH = dim[0]*dim[2];
      temp = dim[1]*(dim[0] + dim[2]) / ENH;
      break;
    case 2: // Expectance to hit intersecting plane
      EPH = dim[0]*dim[1];
      temp = dim[2]*(dim[0] + dim[1]) / ENH;
      break;
  }
  // Chance of hitting intersecting plane
  Real CPH = EPH / ENH;
  //Real CaH = (r*dim._x*dim._y + r*dim._x*dim._z + dim._y*dim._z ) / EPH   - CPH;
  Real CaH = r*temp;
  //Real CbH = ((1.0-r)*dim._x*dim._y + (1.0-r)*dim._x*dim._z + dim._y*dim._z ) / EPH   - CPH;
  Real CbH = (1-r)*temp;
  //
  Real TP = BSP_TP;  // time to intersect a primitive
  Real TK1 = BSP_TK1; // COST OF ACCESSING NODE
  Real TK2 = BSP_TK2; // COST OF TESTING RAY SEGMENT AGAINST PLANE
  Real TK3 = BSP_TK3;// COST OF RETRIEVING A PARTICULAR CHILD
  Real Ma=BSP_MA;Real Mb=BSP_MA;
  Real cost = TK1 + TK2 + (1+CPH)*TK3 +
    TP * ( bothCount +
        (CPH*0.5f) * (leftCount*(1+Ma) + rightCount*(1+Mb)) +
        CaH*leftCount + CbH*rightCount);
  return cost;
}

template<typename Real>
class bspIntervalNode
{
  public:
    bspIntervalNode(){}
    bspIntervalNode(Real pos,bool begin):_pos(pos),_begin(begin){}
    Real  _pos;
    bool _begin;
    bool operator<(const bspIntervalNode& node)const{
      return _pos<node._pos;
    }
};

template<class T,typename P,typename Real>
  template<class BOUNDS,class IT>
Real bspTree<T,P,Real>::calcSplitValue(I32 axis,
    IT pBegin,IT pEnd,
    const P& minBounds,
    const P& maxBounds,
    BOUNDS& boundsFunc,
    Real *pSplit)
{
  // printf("split==>");
  U32 Pab = 0;
  U32 Pa = 0;
  U32 Pb = (U32)(pEnd - pBegin);

  std::vector<bspIntervalNode<Real> > intvlList;
  while( pBegin != pEnd ){
    Real bmin = boundsFunc(axis,true,*pBegin);
    Real bmax = boundsFunc(axis,false,*pBegin);
    intvlList.push_back( bspIntervalNode<Real>(bmin,true) );
    intvlList.push_back( bspIntervalNode<Real>(bmax,false) );
    if( bmin < minBounds[axis] ){
      ++Pab; --Pb;
    }
    ++pBegin;
  }
  std::sort(intvlList.begin(),intvlList.end());
  Real minCost = std::numeric_limits<Real>::infinity();

  Real split,epsilon = EPSILON;
  for(U32 i=0;i<intvlList.size();++i){
    Real cost=minCost;
    if( intvlList[i]._begin ){
      split = intvlList[i]._pos-epsilon;
      if( split >= minBounds[axis] && split <= maxBounds[axis]){
        cost = splitCostFunction(axis,minBounds,maxBounds,split,
            Pa,Pb,Pab);
      }
      ++Pab;--Pb;
    }
    else{
      --Pab;++Pa;
      split = intvlList[i]._pos+epsilon;
      if( split >= minBounds[axis] && split <= maxBounds[axis] ){
        cost = splitCostFunction(axis,minBounds,maxBounds,split,
            Pa,Pb,Pab);
      }
    }
    if( cost < minCost ){
      minCost = cost;
      *pSplit = split;
    }
  }
  if( *pSplit <= minBounds[axis] || *pSplit >=maxBounds[axis] )
    minCost = std::numeric_limits<Real>::infinity();
  // printf("<==split");
  return minCost;
}



template<class T, typename P, typename Real>
  template<class IT>
void bspTree<T,P,Real>::makeLeaf(bspNode<T,Real> *pNode,
    IT pBegin,IT pEnd)
{
  //printf("make leaf==>");
  U32 elements = (U32)(pEnd-pBegin);

  //printf("Leaf %i\n",elements);
  // make leaf node
  //T *pE = _elementAllocator.alloc(elements);
  T *pE = (T*)malloc(elements * sizeof(T));
  pNode->setAxis(-1);
  pNode->setElements(pE);
  pNode->setElementCount(elements);
  while( pBegin != pEnd ){
    *pE = *pBegin;
    ++pE; ++pBegin;
  }
  //printf("<==make leaf");
}


template<class T,typename P,typename Real>
  template<class BOUNDS,class IT>
void bspTree<T,P,Real>::constructTree(bspNode<T,Real> *pNode,
    const P& minBounds,const P& maxBounds,
    IT pBegin,IT pEnd,
    BOUNDS& boundsFunc)
{
  U32 elements = static_cast<unsigned int>(pEnd-pBegin);
  if( elements == 0 ){
    makeLeaf(pNode,pBegin,pEnd);
    return;
  }
  Real bestSplit = 0; 
  I32 splitAxis = -1;
  Real minCost = unSplitCostFunction(elements); 
  for(I32 i=0;i<3;++i){
    Real split; 
    Real cost = calcSplitValue(i,pBegin,pEnd,minBounds,maxBounds,boundsFunc,&split);
    if( cost < minCost ){
      splitAxis = i;
      bestSplit    = split;
      minCost  = cost;
    }
  }
  if( splitAxis >= 0 ){
    bspNode<T,Real> *pNodes = _nodeAllocator.alloc(2);
    // SET THE NODE
    pNode->setChildren(pNodes);
    pNode->setAxis(splitAxis);
    pNode->setSplit(bestSplit);
    //Real temp = pNode->getSplit();
    // SPLIT AND RECURSE "left"
    std::vector<T> nodes;
    nodes.reserve(elements);
    for(IT pE = pBegin;pE != pEnd;++pE){
      Real val = boundsFunc(splitAxis,true,*pE);
      if( val < bestSplit )
        nodes.push_back(*pE);
    }
    P mid = maxBounds; mid[splitAxis] = bestSplit;
    constructTree(pNodes,minBounds,mid,nodes.begin(),nodes.end(),boundsFunc);
    // SPLIT AND RECURSE "right" 
    nodes.clear();nodes.reserve(elements);
    for(IT pE = pBegin;pE != pEnd;++pE){
      Real val = boundsFunc(splitAxis,false,*pE);
      if( val>bestSplit )
        nodes.push_back(*pE);
    }
    mid = minBounds; mid[splitAxis] = bestSplit;
    constructTree(pNodes+1,mid,maxBounds,nodes.begin(),nodes.end(),boundsFunc);
  }
  else{
    makeLeaf(pNode,pBegin,pEnd);
  }
}


#endif

