#ifndef __MGL_BLOCK_ALLOCATOR__H
#define __MGL_BLOCK_ALLOCATOR__H

//#include "../mgl.h"
#include <vector>

typedef unsigned int U32;

template<class T,U32 BLOCKSIZE>
class blockAllocator
{
public:
	blockAllocator():_pS(0){}
	~blockAllocator(){clear();}

	T *alloc(U32 count){ 
		if( count > BLOCKSIZE )
			return 0;
		if( !_pS || ( _sizeLeft < count ) ){
			_pS = new T[BLOCKSIZE];
			_allocList.push_back(_pS);
			_sizeLeft = BLOCKSIZE;
		}
		T *pTemp = _pS;
		_sizeLeft -= count;
		_pS += count;
		return pTemp;	
	}

	void clear(){
		typename std::vector<T*>::iterator ppS;
		for(ppS=_allocList.begin();ppS!=_allocList.end();++ppS){
			delete [] (*ppS);
		}
		_allocList.clear();
		_pS = 0;
	}
private:
	T *_pS;
	U32 _sizeLeft;
	std::vector<T*> _allocList;
};

#endif
