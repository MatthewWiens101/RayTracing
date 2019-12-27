#pragma once

#include "triangle.h"

// TODO technically this works for all quadrilaterals so rectangle is misnomer
// always assumes unit vectors are passed for directional vectors
// always fill clockwise to ensure that normal points outwards

class rectangle : public hitable {
public:
	__device__ rectangle() : normal(vec3()), origin(vec3()), t1(triangle()), t2(triangle()), mat_ptr(nullptr) {};
	__device__ rectangle(vec3 n, vec3 t, vec3 c, vec3 o, float width, float height, material* mater) : normal(n), origin(o), mat_ptr(mater) {
		t1 = triangle(n, o - t * (height / 2) - c * (width / 2), o - t * (height / 2) + c * (width / 2), o + t * (height / 2) - c * (width / 2), mater);
		t2 = triangle(n, o + t * (height / 2) + c * (width / 2), o + t * (height / 2) - c * (width / 2), o - t * (height / 2) + c * (width / 2), mater);
	};
	__device__ rectangle(vec3 p1, vec3 p2, vec3 p3, vec3 p4, material* mater) : mat_ptr(mater) {
		vec3 origin = (p1 + p2 + p3 + p4) / 4;
		vec3 normal = cross(p2 - p1, p4 - p1);
		normal.make_unit_vector();
		t1 = triangle(normal, p1, p2, p4, mater);
		t2 = triangle(normal, p3, p4, p2, mater);
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 normal;
	vec3 origin;
	triangle t1;
	triangle t2;
	material* mat_ptr;
};

__device__ bool rectangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	if (t1.hit(r, t_min, t_max, rec)) {
		return true;
	}
	if (t2.hit(r, t_min, t_max, rec)) {
		return true;
	}
	return false;
}