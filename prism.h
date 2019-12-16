#pragma once

#include "hitable.h"
#include "rectangle.h"

// TODO technically this only works for rectangular prisms so name is a misnomer
// cube order below:
//			5
//	3	4	1	2
//			6

class prism : public hitable {
public:
	__device__ prism();
	__device__ prism(vec3 cen, vec3 t, vec3 fr, float height, float width, float depth, material* mater) : center(cen), top(t), front(fr), mat_ptr(mater) {
		top.make_unit_vector();
		front.make_unit_vector();
		side = cross(top, front);
		side.make_unit_vector();
		// TODO may need to make additional functions for rectangle and triangle (ie. like the vec3 class)
		faces[0] = rectangle(front, top, side, center + front * (depth / 2), width, height, mater);
		faces[1] = rectangle(side, top, -front, center + side * (width / 2), depth, height, mater);
		faces[2] = rectangle(-front, top, -side, center - front * (depth / 2), width, height, mater);
		faces[3] = rectangle(-side, top, front, center - side * (width / 2), depth, height, mater);
		faces[4] = rectangle(top, -front, side, center + top * (height / 2), width, depth, mater);
		faces[5] = rectangle(-top, front, side, center - top * (height / 2), width, depth, mater);
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	vec3 top;
	vec3 front;
	vec3 side;
	rectangle faces[6];
	material* mat_ptr;
};

__device__ bool prism::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	for (int i = 0; i < 6; i++) {
		if (faces[i].hit(r, tmin, tmax, rec)) {
			return true;
		}
	}
	return false;
}