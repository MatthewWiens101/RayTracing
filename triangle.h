#pragma once

#include "hitable.h"

// always assumes unit vectors are passed for directional vectors
// always fill clockwise to ensure that normal points outwards

class triangle : public hitable {
public:
	__device__ triangle() : normal(vec3()), origin(vec3()), vert1(vec3()), vert2(vec3()), vert3(vec3()), mat_ptr(nullptr) {};
	__device__ triangle(vec3 n, vec3 p1, vec3 p2, vec3 p3, material* mater) : normal(n), vert1(p1), vert2(p2), vert3(p3), mat_ptr(mater) {
		vec3 origin = (p1 + p2 + p3) / 3;
	};
	__device__ triangle(vec3 p1, vec3 p2, vec3 p3, material* mater) : vert1(p1), vert2(p2), vert3(p3), mat_ptr(mater) {
		vec3 normal = cross(p3 - p1, p2 - p1);
		normal.make_unit_vector();
		vec3 origin = (p1 + p2 + p3) / 3;
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 normal;
	vec3 origin;
	vec3 vert1;
	vec3 vert2;
	vec3 vert3;
	material* mat_ptr;
};

// TODO warning there is something very wrong with this function
// Depending on whether vert1 is the large angle of an obtruse triangle, the rendering will either work or not
// This is shown if vert1 is replaced by vert2 or vert3 as the origin
// For now, ensuring that vert1 is the large angle ensures it will work
// However it is not known if this will work for equilateral or acute triangles
// This is some logic error that is not checking something properly, but what?
__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 u = vert2 - vert1;
	vec3 v = vert3 - vert1;
	vec3 oc = vert1 - r.origin();
	float t = dot(normal, oc) / dot(normal, r.direction());
	vec3 intersect = r.point_at_parameter(t);
	vec3 P = intersect - vert1;
	float us = dot(P, u) / u.squared_length();
	float vs = dot(P, v) / v.squared_length();
	if ((0 <= us) && (us <= 1) && (0 <= vs) && (vs <= 1) && (us + vs <= 1)) {
		if (t < t_max && t > t_min) {
			rec.t = t;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = normal;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}