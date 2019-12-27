#pragma once

#include "hitable.h"
#include "triangle.h"

class tetrahedral : public hitable {
public:
	__device__ tetrahedral();
	__device__ tetrahedral(vec3 cen, vec3 t, vec3 fr, float l_radius, material* mater) : center(cen), top(t), front(fr), mat_ptr(mater) {
		top.make_unit_vector();
		front.make_unit_vector();
		side = cross(top, front);
		side.make_unit_vector();
		float length = l_radius / sqrt(3.0);
		// TODO may need to make additional functions for rectangle and triangle (ie. like the vec3 class)
		faces[0] = triangle(cen - top * length + front * length - side * length, cen + top * length + front * length + side * length, cen + top * length - front * length - side * length, mater);
		faces[1] = triangle(cen - top * length - front * length + side * length, cen + top * length - front * length - side * length, cen + top * length + front * length + side * length, mater);
		faces[2] = triangle(cen + top * length + front * length + side * length, cen - top * length + front * length - side * length, cen - top * length - front * length + side * length, mater);
		faces[3] = triangle(cen + top * length - front * length - side * length, cen - top * length - front * length + side * length, cen - top * length + front * length - side * length, mater);
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	vec3 top;
	vec3 front;
	vec3 side;
	triangle faces[4];
	material* mat_ptr;
};

__device__ bool tetrahedral::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	hit_record temp = rec;
	float min_t = FLT_MAX;
	for (int i = 0; i < 4; i++) {
		if (faces[i].hit(r, tmin, tmax, temp)) {
			if (temp.t < min_t) {
				min_t = temp.t;
				rec = temp;
			}
		}
	}
	if (min_t == FLT_MAX) return false;
	return true;
}