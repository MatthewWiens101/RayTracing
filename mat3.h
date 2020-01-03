#pragma once
#include "vec3.h"

class mat3 {
public:
	__host__ __device__ mat3() {}
	__host__ __device__ mat3(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7, float e8) { 
		e[0] = e0; e[1] = e1; e[2] = e2; 
		e[3] = e3; e[4] = e4; e[5] = e5; 
		e[6] = e6; e[7] = e7; e[8] = e8;
	}
	__host__ __device__ mat3(vec3 a, vec3 b, vec3 c) {
		e[0] = a.x(); e[1] = a.y(); e[2] = a.z();
		e[3] = b.x(); e[4] = b.y(); e[5] = b.z();
		e[6] = c.x(); e[7] = c.y(); e[8] = c.z();
	}
	__host__ __device__ inline float ind(int i, int j) const { return e[i*3 + j]; }

	__host__ __device__ inline const mat3& operator+() const { return *this; }
	__host__ __device__ inline mat3 operator-() const { return mat3(-e[0], -e[1], -e[2], -e[3], -e[4], -e[5], -e[6], -e[7], -e[8]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; }
	__host__ __device__ inline float operator()(unsigned i, unsigned j) const { return e[i*3 + j]; }
	__host__ __device__ inline float& operator()(unsigned i, unsigned j) { return e[i*3 + j]; }

	__host__ __device__ inline mat3& operator+=(const mat3& m2);
	__host__ __device__ inline mat3& operator-=(const mat3& m2);
	__host__ __device__ inline mat3& operator*=(const mat3& m2);
	__host__ __device__ inline mat3& operator/=(const mat3& m2);
	__host__ __device__ inline mat3& operator*=(const float t);
	__host__ __device__ inline mat3& operator/=(const float t);
	__host__ __device__ inline bool operator==(const mat3& m);
	__host__ __device__ inline bool operator!=(const mat3& m);

	__host__ __device__ inline float det() const {
		return e[0]*(e[4] * e[8] - e[5] * e[7]) - e[1] * (e[3] * e[8] - e[5] * e[6]) + e[2] * (e[3] * e[7] - e[4] * e[6]);
	}
	__host__ __device__ inline void invert();

	// e[0] e[1] e[2]
	// e[3] e[4] e[5]
	// e[6] e[7] e[8]
	float e[9];
};

inline std::istream& operator>>(std::istream& is, mat3& t) {
	is >> t.e[0] >> t.e[1] >> t.e[2] >> t.e[3] >> t.e[4] >> t.e[5] >> t.e[6] >> t.e[7] >> t.e[8];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const mat3& t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2] << "\n" << t.e[3] << " " << t.e[4] << " " << t.e[5] << "\n" << t.e[6] << " " << t.e[7] << " " << t.e[8] << "\n";
	return os;
}

__host__ __device__ inline mat3 operator+(const mat3& m1, const mat3& m2) {
	return mat3(
		m1.e[0] + m2.e[0], m1.e[1] + m2.e[1], m1.e[2] + m2.e[2],
		m1.e[3] + m2.e[3], m1.e[4] + m2.e[4], m1.e[5] + m2.e[5],
		m1.e[6] + m2.e[6], m1.e[7] + m2.e[7], m1.e[8] + m2.e[8]
		);
}

__host__ __device__ inline mat3 operator-(const mat3& m1, const mat3& m2) {
	return mat3(
		m1.e[0] - m2.e[0], m1.e[1] - m2.e[1], m1.e[2] - m2.e[2],
		m1.e[3] - m2.e[3], m1.e[4] - m2.e[4], m1.e[5] - m2.e[5],
		m1.e[6] - m2.e[6], m1.e[7] - m2.e[7], m1.e[8] - m2.e[8]
	);
}

__host__ __device__ inline mat3 operator*(const mat3& m1, const mat3& m2) {
	return mat3(
		m1.e[0] * m2.e[0], m1.e[1] * m2.e[1], m1.e[2] * m2.e[2],
		m1.e[3] * m2.e[3], m1.e[4] * m2.e[4], m1.e[5] * m2.e[5],
		m1.e[6] * m2.e[6], m1.e[7] * m2.e[7], m1.e[8] * m2.e[8]
	);
}

__host__ __device__ inline mat3 operator/(const mat3& m1, const mat3& m2) {
	return mat3(
		m1.e[0] / m2.e[0], m1.e[1] / m2.e[1], m1.e[2] / m2.e[2],
		m1.e[3] / m2.e[3], m1.e[4] / m2.e[4], m1.e[5] / m2.e[5],
		m1.e[6] / m2.e[6], m1.e[7] / m2.e[7], m1.e[8] / m2.e[8]
	);
}

__host__ __device__ inline mat3 operator*(float t, const mat3& m) {
	return mat3(
		m.e[0] * t, m.e[1] * t, m.e[2] * t,
		m.e[3] * t, m.e[4] * t, m.e[5] * t,
		m.e[6] * t, m.e[7] * t, m.e[8] * t
	);
}

__host__ __device__ inline mat3 operator/(const mat3& m, float t) {
	float k = 1 / t;
	return mat3(
		m.e[0] * k, m.e[1] * k, m.e[2] * k,
		m.e[3] * k, m.e[4] * k, m.e[5] * k,
		m.e[6] * k, m.e[7] * k, m.e[8] * k
	);
}

__host__ __device__ inline mat3 operator*(const mat3& m, float t) {
	return mat3(
		m.e[0] * t, m.e[1] * t, m.e[2] * t,
		m.e[3] * t, m.e[4] * t, m.e[5] * t,
		m.e[6] * t, m.e[7] * t, m.e[8] * t
	);
}

__host__ __device__ inline mat3 dot(const mat3& m1, const mat3& m2) {
	float e0 = m1.e[0] * m2.e[0] + m1.e[1] * m2.e[3] + m1.e[2] * m2.e[6];
	float e1 = m1.e[0] * m2.e[1] + m1.e[1] * m2.e[4] + m1.e[2] * m2.e[7];
	float e2 = m1.e[0] * m2.e[2] + m1.e[1] * m2.e[5] + m1.e[2] * m2.e[8];
	float e3 = m1.e[3] * m2.e[0] + m1.e[4] * m2.e[3] + m1.e[5] * m2.e[6];
	float e4 = m1.e[3] * m2.e[1] + m1.e[4] * m2.e[4] + m1.e[5] * m2.e[7];
	float e5 = m1.e[3] * m2.e[2] + m1.e[4] * m2.e[5] + m1.e[5] * m2.e[8];
	float e6 = m1.e[6] * m2.e[0] + m1.e[7] * m2.e[3] + m1.e[8] * m2.e[6];
	float e7 = m1.e[6] * m2.e[1] + m1.e[7] * m2.e[4] + m1.e[8] * m2.e[7];
	float e8 = m1.e[6] * m2.e[2] + m1.e[7] * m2.e[5] + m1.e[8] * m2.e[8];
	return mat3(e0, e1, e2, e3, e4, e5, e6, e7, e8);
}

__host__ __device__ inline vec3 operator*(const mat3& m, const vec3& v) {
	float e0 = m.e[0] * v.e[0] + m.e[1] * v.e[1] + m.e[2] * v.e[2];
	float e1 = m.e[3] * v.e[0] + m.e[4] * v.e[1] + m.e[5] * v.e[2];
	float e2 = m.e[6] * v.e[0] + m.e[7] * v.e[1] + m.e[8] * v.e[2];
	return vec3(e0, e1, e2);
}

__host__ __device__ inline vec3 operator*(const vec3& v, const mat3& m) {
	float e0 = m.e[0] * v.e[0] + m.e[3] * v.e[1] + m.e[6] * v.e[2];
	float e1 = m.e[1] * v.e[0] + m.e[4] * v.e[1] + m.e[7] * v.e[2];
	float e2 = m.e[2] * v.e[0] + m.e[5] * v.e[1] + m.e[8] * v.e[2];
	return vec3(e0, e1, e2);
}

__host__ __device__ inline mat3& mat3::operator+=(const mat3& m) {
	e[0] += m.e[0];
	e[1] += m.e[1];
	e[2] += m.e[2];
	e[3] += m.e[3];
	e[4] += m.e[4];
	e[5] += m.e[5];
	e[6] += m.e[6];
	e[7] += m.e[7];
	e[8] += m.e[8];
	return *this;
}

__host__ __device__ inline mat3& mat3::operator*=(const mat3& m) {
	e[0] *= m.e[0];
	e[1] *= m.e[1];
	e[2] *= m.e[2];
	e[3] *= m.e[3];
	e[4] *= m.e[4];
	e[5] *= m.e[5];
	e[6] *= m.e[6];
	e[7] *= m.e[7];
	e[8] *= m.e[8];
	return *this;
}

__host__ __device__ inline mat3& mat3::operator/=(const mat3& m) {
	e[0] /= m.e[0];
	e[1] /= m.e[1];
	e[2] /= m.e[2];
	e[3] /= m.e[3];
	e[4] /= m.e[4];
	e[5] /= m.e[5];
	e[6] /= m.e[6];
	e[7] /= m.e[7];
	e[8] /= m.e[8];
	return *this;
}

__host__ __device__ inline mat3& mat3::operator-=(const mat3& m) {
	e[0] -= m.e[0];
	e[1] -= m.e[1];
	e[2] -= m.e[2];
	e[3] -= m.e[3];
	e[4] -= m.e[4];
	e[5] -= m.e[5];
	e[6] -= m.e[6];
	e[7] -= m.e[7];
	e[8] -= m.e[8];
	return *this;
}

__host__ __device__ inline mat3& mat3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	e[3] *= t;
	e[4] *= t;
	e[5] *= t;
	e[6] *= t;
	e[7] *= t;
	e[8] *= t;
	return *this;
}

__host__ __device__ inline mat3& mat3::operator/=(const float t) {
	float k = 1.0 / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	e[3] *= k;
	e[4] *= k;
	e[5] *= k;
	e[6] *= k;
	e[7] *= k;
	e[8] *= k;
	return *this;
}

__host__ __device__ inline bool mat3::operator==(const mat3& m) {
	if (e[0] != m.e[0]) return false;
	if (e[1] != m.e[1]) return false;
	if (e[2] != m.e[2]) return false;
	if (e[3] != m.e[3]) return false;
	if (e[4] != m.e[4]) return false;
	if (e[5] != m.e[5]) return false;
	if (e[6] != m.e[6]) return false;
	if (e[7] != m.e[7]) return false;
	if (e[8] != m.e[8]) return false;
	return true;
}

__host__ __device__ inline bool mat3::operator!=(const mat3& m) {
	if (e[0] == m.e[0]) return false;
	if (e[1] == m.e[1]) return false;
	if (e[2] == m.e[2]) return false;
	if (e[3] == m.e[3]) return false;
	if (e[4] == m.e[4]) return false;
	if (e[5] == m.e[5]) return false;
	if (e[6] == m.e[6]) return false;
	if (e[7] == m.e[7]) return false;
	if (e[8] == m.e[8]) return false;
	return true;
}

__host__ __device__ void mat3::invert() {
	float d = (*this).det();
	mat3 copy = *this;
	e[0] = copy.e[4] * copy.e[8] - copy.e[5] * copy.e[7];
	e[1] = -copy.e[3] * copy.e[8] + copy.e[5] * copy.e[6];
	e[2] = copy.e[3] * copy.e[7] - copy.e[4] * copy.e[6];
	e[3] = -copy.e[1] * copy.e[8] + copy.e[2] * copy.e[7];
	e[4] = copy.e[0] * copy.e[8] - copy.e[2] * copy.e[6];
	e[5] = -copy.e[0] * copy.e[7] + copy.e[1] * copy.e[6];
	e[6] = copy.e[1] * copy.e[5] - copy.e[4] * copy.e[2];
	e[7] = -copy.e[0] * copy.e[5] + copy.e[2] * copy.e[3];
	e[8] = copy.e[0] * copy.e[4] - copy.e[1] * copy.e[3];
	*this /= d;
}

__host__ __device__ inline mat3 inverse(const mat3& m) {
	float d = m.det();
	float e0 = m.e[4] * m.e[8] - m.e[5] * m.e[7];
	float e1 = -m.e[3] * m.e[8] + m.e[5] * m.e[6];
	float e2 = m.e[3] * m.e[7] - m.e[4] * m.e[6];
	float e3 = -m.e[1] * m.e[8] + m.e[2] * m.e[7];
	float e4 = m.e[0] * m.e[8] - m.e[2] * m.e[6];
	float e5 = -m.e[0] * m.e[7] + m.e[1] * m.e[6];
	float e6 = m.e[1] * m.e[5] - m.e[4] * m.e[2];
	float e7 = -m.e[0] * m.e[5] + m.e[2] * m.e[3];
	float e8 = m.e[0] * m.e[4] - m.e[1] * m.e[3];
	return mat3(e0, e1, e2, e3, e4, e5, e6, e7, e8) / d;
}

__host__ __device__ inline mat3 rotation_matrix(vec3 u, float theta) {
	u.make_unit_vector();
	float e0 = cos(theta) + u.e[0] * u.e[0] * (1 - cos(theta));
	float e1 = u.e[0] * u.e[1] * (1 - cos(theta)) - u.e[2] * sin(theta);
	float e2 = u.e[0] * u.e[2] * (1 - cos(theta)) + u.e[1] * sin(theta);
	float e3 = u.e[1] * u.e[0] * (1 - cos(theta)) + u.e[2] * sin(theta);
	float e4 = cos(theta) + u.e[1] * u.e[1] * (1 - cos(theta));
	float e5 = u.e[1] * u.e[2] * (1 - cos(theta)) - u.e[0] * sin(theta);
	float e6 = u.e[2] * u.e[0] * (1 - cos(theta)) - u.e[1] * sin(theta);
	float e7 = u.e[2] * u.e[1] * (1 - cos(theta)) + u.e[0] * sin(theta);
	float e8 = cos(theta) + u.e[2] * u.e[2] * (1 - cos(theta));
	return mat3(e0, e1, e2, e3, e4, e5, e6, e7, e8);
}
