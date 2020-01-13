#pragma once

#include "hitable.h"
#include "triangle.h"

class icosahedron : public hitable {
public:
	__device__ icosahedron();
	__device__ icosahedron(vec3 cen, vec3 t, vec3 fr, float r, material* mater) : center(cen), top(t), front(fr), mat_ptr(mater) {
		top.make_unit_vector();
		front.make_unit_vector();
		side = cross(top, front);
		side.make_unit_vector();
		const float radius = r;
		vec3 p[12];
		//const float l = 2 * radius / sin(1.04720) * cos(1.205935);
		const float upper_vert = radius * cos(1.10710923);
		const float upper_radius = radius * cos(0.4636871);
		p[0] = center + radius * top;
		p[1] = center + upper_vert * top + upper_radius * front;
		p[2] = center + upper_vert * top + upper_radius * cos(1.256637) * front + upper_radius * sin(1.256637) * side;
		p[3] = center + upper_vert * top + upper_radius * cos(2.513274) * front + upper_radius * sin(2.513274) * side;
		p[4] = center + upper_vert * top + upper_radius * cos(2.513274) * front - upper_radius * sin(2.513274) * side;
		p[5] = center + upper_vert * top + upper_radius * cos(1.256637) * front - upper_radius * sin(1.256637) * side;

		p[6] = center - upper_vert * top + upper_radius * cos(0.628319) * front - upper_radius * sin(0.628319) * side;
		p[7] = center - upper_vert * top + upper_radius * cos(1.884956) * front - upper_radius * sin(1.884956) * side;
		p[8] = center - upper_vert * top - upper_radius * front;
		p[9] = center - upper_vert * top + upper_radius * cos(1.884956) * front + upper_radius * sin(1.884956) * side;
		p[10] = center - upper_vert * top + upper_radius * cos(0.628319) * front + upper_radius * sin(0.628319) * side;
		p[11] = center - radius * top;
		// TODO may need to make additional functions for rectangle and triangle (ie. like the vec3 class)
		faces[0] = triangle(p[0], p[1], p[2], mater);
		faces[1] = triangle(p[0], p[2], p[3], mater);
		faces[2] = triangle(p[0], p[3], p[4], mater);
		faces[3] = triangle(p[0], p[4], p[5], mater);
		faces[4] = triangle(p[0], p[5], p[1], mater);

		faces[5] = triangle(p[2], p[1], p[10], mater);
		faces[6] = triangle(p[3], p[2], p[9], mater);
		faces[7] = triangle(p[4], p[3], p[8], mater);
		faces[8] = triangle(p[5], p[4], p[7], mater);
		faces[9] = triangle(p[1], p[5], p[6], mater);

		faces[10] = triangle(p[1], p[6], p[10], mater);
		faces[11] = triangle(p[2], p[10], p[9], mater);
		faces[12] = triangle(p[3], p[9], p[8], mater);
		faces[13] = triangle(p[4], p[8], p[7], mater);
		faces[14] = triangle(p[5], p[7], p[6], mater);

		faces[15] = triangle(p[11], p[10], p[6], mater);
		faces[16] = triangle(p[11], p[6], p[7], mater);
		faces[17] = triangle(p[11], p[7], p[8], mater);
		faces[18] = triangle(p[11], p[8], p[9], mater);
		faces[19] = triangle(p[11], p[9], p[10], mater);
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	vec3 top;
	vec3 front;
	vec3 side;
	triangle faces[20];
	material* mat_ptr;
};

__device__ bool icosahedron::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	hit_record temp = rec;
	float min_t = FLT_MAX;
	for (int i = 0; i < 20; i++) {
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